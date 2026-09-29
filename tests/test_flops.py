import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.train import CONFIG_DIR, CONFIG_NAME
from walrus.utils.flops import (
    BACKWARD_MULTIPLIER,
    FlopAccountant,
    GRANULAR_FLOP_KEYS,
    ModelShape,
    analytic_token_shape,
    conv_out_size,
    decoder_stage_shapes,
    encoder_shapes_from_tokens,
    forward_flops,
    supports_flop_accounting,
    training_step_flops,
)

ENCODER_KERNEL1 = (8, 8, 8)
ENCODER_KERNEL2 = (4, 4, 4)
# The decoder reverses the roles of the two base kernels (vstride_decoder.py __init__).
DECODER_KERNEL1 = ENCODER_KERNEL2
DECODER_KERNEL2 = ENCODER_KERNEL1


# (resolution, boundary conditions, token grid) as measured by running the model,
# walrus_tokens_and_data_pipeline.md 1.2.
@pytest.mark.parametrize(
    "resolution,periodic,expected",
    [
        ((256, 256, 1), (True, True, True), (32, 32, 1)),
        ((512, 512, 1), (True, True, True), (32, 32, 1)),
        ((128, 384, 1), (True, True, True), (32, 32, 1)),
        ((256, 512, 1), (True, True, True), (32, 32, 1)),
        ((256, 256, 1), (False, False, True), (33, 33, 1)),
        ((1024, 256, 1), (False, False, True), (33, 33, 1)),
        # Mixed boundaries add a token only on the non-periodic axis.
        ((512, 512, 1), (True, False, True), (32, 33, 1)),
        ((64, 64, 64), (True, True, True), (16, 16, 16)),
    ],
)
def test_analytic_token_shape(resolution, periodic, expected):
    assert analytic_token_shape(resolution, periodic) == expected


@pytest.mark.parametrize("axis_resolution", [128, 256, 512, 1024])
@pytest.mark.parametrize("is_periodic", [True, False])
def test_decoder_inverts_encoder(axis_resolution, is_periodic):
    """The decoder must land back exactly on the encoder's padded input grid.

    This is the invariant that pins `decoder_stage_shapes`: the circular pre-pad on
    periodic axes and the post-crop have to cancel, stage by stage, or the model could
    not reconstruct the field at its input resolution. 384-pixel axes are excluded
    because the model itself does not satisfy it there — see the test below.
    """
    (stride1, stride2) = choose_kernel_size_deterministic((axis_resolution,))[0]
    tokens = 32 + (0 if is_periodic else 1)

    padded_in, mid = encoder_shapes_from_tokens(
        (tokens,), (stride1,), (stride2,), ENCODER_KERNEL1[:1], ENCODER_KERNEL2[:1]
    )
    # Forward: the encoder's two padding=0 convs reach the token grid.
    assert conv_out_size(padded_in[0], ENCODER_KERNEL1[0], stride1) == mid[0]
    assert conv_out_size(mid[0], ENCODER_KERNEL2[0], stride2) == tokens

    # Backward: the decoder swaps the stride pair and must retrace those two shapes.
    _, dec_mid = decoder_stage_shapes(
        (tokens,), DECODER_KERNEL1[:1], (stride2,), (is_periodic,)
    )
    _, dec_out = decoder_stage_shapes(
        dec_mid, DECODER_KERNEL2[:1], (stride1,), (is_periodic,)
    )
    assert dec_mid == mid
    assert dec_out == padded_in


def test_stride_six_axis_does_not_round_trip():
    """A 384-pixel periodic axis decodes to 394, not the 398 it was encoded from.

    `adaptive_conv_transpose` pads by `(k - s) // s` but crops by `k - s`, so the two
    only cancel when `s` divides `k - s`. Stride 6 (from `patch_dict[12]`, i.e. any
    384-pixel axis — `turbulent_radiative_layer_2D` has one) leaves `(8 - 6) // 6 == 0`
    padding against a 2-wide crop per side. This is the case the source flags with
    "TODO (mm) - This might get messed up for future non-divisible cases"
    ([vstride_decoder.py:225](../walrus/models/decoders/vstride_decoder.py#L225)).

    It is asserted here so the accountant's numbers stay pinned to what the model does
    rather than to what the geometry ought to be: the FLOP count is verified exact
    against FlopCounterMode on that dataset regardless.
    """
    padded_in, mid = encoder_shapes_from_tokens(
        (32,), (6,), (2,), ENCODER_KERNEL1[:1], ENCODER_KERNEL2[:1]
    )
    assert (padded_in, mid) == ((398,), (66,))

    _, dec_mid = decoder_stage_shapes((32,), DECODER_KERNEL1[:1], (2,), (True,))
    _, dec_out = decoder_stage_shapes(dec_mid, DECODER_KERNEL2[:1], (6,), (True,))
    assert dec_mid == mid
    assert dec_out == (394,)


def test_periodic_axes_cost_more_than_the_token_grid_implies():
    """Circular padding makes the transposed conv run on extra positions.

    Missing this term is what made an earlier hand-derivation undercount the decoder on
    periodic datasets while matching exactly on non-periodic ones.
    """
    periodic_padded, _ = decoder_stage_shapes((32,), (4,), (2,), (True,))
    open_padded, _ = decoder_stage_shapes((32,), (4,), (2,), (False,))
    assert periodic_padded == (34,)
    assert open_padded == (32,)


def test_gradient_checkpointing_surcharge():
    shape = ModelShape(
        hidden_dim=64,
        mlp_dim=256,
        processor_blocks=4,
        intermediate_dim=32,
        encoder_kernel1=ENCODER_KERNEL1,
        encoder_kernel2=ENCODER_KERNEL2,
        decoder_kernel1=DECODER_KERNEL1,
        decoder_kernel2=DECODER_KERNEL2,
        gradient_checkpointing_freq=0,
    )
    forward = {
        "encoder": 10.0,
        "encoder_proj1": 6.0,
        "encoder_proj2": 4.0,
        "processor": 100.0,
        "decoder": 20.0,
        "decoder_proj1": 8.0,
        "decoder_proj2": 12.0,
        "total": 130.0,
    }

    off = training_step_flops(forward, shape)
    assert off["recompute"] == 0
    assert off["total"] == pytest.approx(130.0 * (1 + BACKWARD_MULTIPLIER))
    # Every granular key follows the same forward*(1+BACKWARD_MULTIPLIER) rule as
    # "processor" -- no recompute folded in, so it's directly comparable/summable with
    # cumulative_processor_flops.
    for k in GRANULAR_FLOP_KEYS:
        assert off[k] == pytest.approx(forward[k] * (1 + BACKWARD_MULTIPLIER))

    # freq=1 checkpoints every block, so the whole model is recomputed once.
    every = training_step_flops(
        forward, ModelShape(**{**shape.__dict__, "gradient_checkpointing_freq": 1})
    )
    assert every["recompute"] == pytest.approx(130.0)

    # freq=2 checkpoints half the blocks; encoder and decoder are always recomputed.
    half = training_step_flops(
        forward, ModelShape(**{**shape.__dict__, "gradient_checkpointing_freq": 2})
    )
    assert half["recompute"] == pytest.approx(10.0 + 20.0 + 50.0)


def test_unsupported_models_are_detected():
    """Baseline wrappers run through the same Trainer but have no encoder/processor/
    decoder structure, so they must be recognised as unaccountable rather than crash."""

    class NotIsotropic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

    assert not supports_flop_accounting(NotIsotropic())


def _measured_subtrees(counter):
    """FlopCounterMode totals for the encoder and decoder subtrees."""
    encoder = decoder = 0
    for module, ops in counter.get_flop_counts().items():
        parts = module.split(".")
        if len(parts) != 3:
            continue
        if parts[1] == "embed":
            encoder += sum(ops.values())
        elif parts[1] == "debed":
            decoder += sum(ops.values())
    return encoder, decoder


def test_encoder_decoder_flops_match_flop_counter(default_conf):
    """The closed form must equal FlopCounterMode exactly on a real forward pass.

    Only the encoder and decoder are compared: they are convolutions, which are
    registered ops on a CPU build, whereas `FullAttention`'s spatial attention dispatches
    to a CPU SDPA kernel that has no registered formula and is silently counted as zero
    (walrus_flops_analysis.ipynb 4). The processor terms are verified on GPU instead, by
    LeonardoRunScripts/flop_counter_backend_test.py.
    """
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    overrides.append("model/processor/space_mixing=full_spatial_attention")
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()

        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])

    geometry = accountant._observed
    assert geometry is not None, "encoder hook never fired"
    predicted = forward_flops(accountant.shape, geometry)
    measured_encoder, measured_decoder = _measured_subtrees(counter)

    assert predicted["encoder"] == measured_encoder
    assert predicted["decoder"] == measured_decoder


def test_accountant_accumulates_per_microbatch(default_conf):
    """Totals must come from the dataset actually drawn, not shape x step count."""
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()
        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    with torch.no_grad():
        for _ in range(3):
            model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
            accountant.record_microbatch("dummy")

    metrics = accountant.epoch_metrics(elapsed_seconds=60.0)
    assert metrics["epoch_flops"] > 0
    assert metrics["flops_per_microbatch"] == pytest.approx(metrics["epoch_flops"] / 3)
    assert metrics["cumulative_flops"] == pytest.approx(metrics["epoch_flops"])
    assert 0 < metrics["enc_dec_flop_fraction"] <= 1
    assert metrics["tokens_seen"] > 0
    assert metrics["gpu_hours"] == pytest.approx(60.0 / 3600.0)

    # A second epoch must add to, not replace, the running totals.
    accountant.reset_epoch()
    with torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
        accountant.record_microbatch("dummy")
    second = accountant.epoch_metrics(elapsed_seconds=60.0)
    assert second["cumulative_flops"] > metrics["cumulative_flops"]


def test_accountant_reports_granular_proj_flops(default_conf):
    """The proj1/proj2 encoder/decoder breakdown must reach `epoch_metrics` -- what
    `training.py:1189` merges into `train_logs` and logs to W&B -- on the same
    forward+backward basis as `cumulative_processor_flops`, so the two are directly
    comparable/summable, while staying reconcilable with the pure-forward
    `enc_dec_flop_fraction` already reported."""
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()
        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    entries = []
    with torch.no_grad():
        for _ in range(2):
            model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
            entries.append(accountant.record_microbatch("dummy"))

    metrics = accountant.epoch_metrics(elapsed_seconds=60.0)
    for k in GRANULAR_FLOP_KEYS:
        # step_{k}, not forward_{k}: same forward+backward basis as step_processor.
        expected_step = sum(e[f"step_{k}"] for e in entries)
        assert metrics[f"epoch_{k}_flops"] == pytest.approx(expected_step)
        assert metrics[f"epoch_{k}_flops"] > 0
        assert metrics[f"cumulative_{k}_flops"] == pytest.approx(
            metrics[f"epoch_{k}_flops"]
        )
        # The actual point of this test: each granular figure is exactly
        # forward*(1+BACKWARD_MULTIPLIER), the identical rule "processor" follows, so
        # it's on the same basis as cumulative_processor_flops -- not some other,
        # incompatible scaling.
        expected_forward = sum(e[f"forward_{k}"] for e in entries)
        assert metrics[f"epoch_{k}_flops"] == pytest.approx(
            expected_forward * (1.0 + BACKWARD_MULTIPLIER)
        )

    # Summing all five forward+backward figures (processor + the four proj stages)
    # must reproduce the same total "useful" (non-recompute) FLOPs epoch_flops counts,
    # since gradient_checkpointing_freq=0 in this fixture (no recompute surcharge).
    processor_step = sum(e["step_processor"] for e in entries)
    granular_step = sum(
        metrics[f"epoch_{k}_flops"] for k in GRANULAR_FLOP_KEYS
    )
    assert metrics["epoch_flops"] == pytest.approx(processor_step + granular_step)

    # Still reconcilable with the pure-forward enc_dec_flop_fraction already reported:
    # divide the backward multiplier back out before comparing.
    granular_forward = granular_step / (1.0 + BACKWARD_MULTIPLIER)
    fwd_total = sum(e["forward_total"] for e in entries)
    assert metrics["enc_dec_flop_fraction"] == pytest.approx(
        granular_forward / fwd_total, rel=1e-6
    )

    # A second epoch must add to, not replace, the cumulative granular totals too.
    accountant.reset_epoch()
    with torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
        accountant.record_microbatch("dummy")
    second = accountant.epoch_metrics(elapsed_seconds=60.0)
    for k in GRANULAR_FLOP_KEYS:
        assert second[f"cumulative_{k}_flops"] > metrics[f"cumulative_{k}_flops"]
