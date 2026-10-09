"""Validate walrus/utils/flops.py on one real batch from each PT9 pretraining dataset.

SCALING_STRATEGY_2D.md §5.1: must pass before any Tier 0 job is submitted. It is the PT9
counterpart of ../validate_flops_real_data.py, which loops over the smallest7 mixture, and
reuses its method: compare the closed-form encoder/decoder FLOPs with FlopCounterMode on a
CPU forward pass. Processor terms are not checked here, because CPU SDPA has no registered
FLOP formula; ../flop_counter_backend_test.py covers them on GPU.

Unlike that script, this one builds the datamodule from the §4.1 substrate itself
(pt9_substrate.SUBSTRATE: data=Leonardo_pretrain9_2d, override_dimensionality=3,
n_steps_input=6, the acoustic density transforms). Each dataset is therefore checked with
exactly the geometry, field list and encoder routing that training will use. Only the
model is shrunk (2 blocks) to keep the CPU forward pass quick; the encoder and decoder are
unaffected by processor depth.

It also prints the time to read one window per dataset. That is a cold single read on a
login node, not a data-wait measurement, but it flags a pathological dataset before
GATE A does.

Usage (login node, ~2-5 min, no GPU):
    module purge; module load python; module load cuda/12.2
    source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
    python LeonardoRunScripts/PT9_2d/validate_flops_pt9.py
Exit code 0 = every dataset exact, 1 = a mismatch.
"""

import sys
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from pt9_substrate import PRETRAIN_DATASETS, SUBSTRATE, WALRUS_DIR
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.utils.flops import FlopAccountant, analytic_token_shape, forward_flops

OVERRIDES = SUBSTRATE + [
    "model.hidden_dim=768",
    "model.processor_blocks=2",
    "model.processor.space_mixing.mlp_dim=2048",
    "logger=none",
    "distribution=local",
    "data_workers=1",
    "name=flops_validation_pt9",
    "automatic_setup=False",
]


def subtree_flops(counter):
    """FlopCounterMode totals for the encoder and decoder subtree roots (embed.<d>, debed.<d>)."""
    enc = dec = 0
    for module, ops in counter.get_flop_counts().items():
        parts = module.split(".")
        if len(parts) == 3 and parts[1] == "embed":
            enc += sum(ops.values())
        elif len(parts) == 3 and parts[1] == "debed":
            dec += sum(ops.values())
    return enc, dec


def main():
    with initialize_config_dir(config_dir=f"{WALRUS_DIR}/configs", version_base=None):
        cfg = compose(config_name="config", overrides=OVERRIDES)
    # Exactly as train.py::train builds them.
    dm = instantiate(
        cfg.data.module_parameters,
        world_size=1,
        rank=0,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    mixed = dm.train_dataset
    n_states = max(mixed.field_to_index_map.values()) + 1
    model = instantiate(cfg.model, n_states=n_states).eval()
    fmt = ChannelsFirstWithTimeFormatter()

    names = [d.metadata.dataset_name for d in mixed.sub_dsets]
    problems = []
    if sorted(names) != sorted(PRETRAIN_DATASETS):
        problems.append(f"datamodule loaded {sorted(names)}")
    if n_states != 67:
        problems.append(f"n_states = {n_states}, expected 67")
    if list(model.embed.keys()) != ["3"]:
        problems.append(f"encoder keys {list(model.embed.keys())}, expected ['3']")

    hdr = (f"{'dataset':<36}{'resolution':>16}{'fields':>7}{'tokens obs':>16}"
           f"{'tok rule':>9}{'encoder':>9}{'decoder':>9}{'read s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(names):
        t0 = time.perf_counter()
        batch = mixed[max(mixed.offsets[i], 0)]  # first window of dataset i; batched (B=1)
        read_s = time.perf_counter() - t0
        x, state_labels, bcs = fmt.process_input(
            batch, causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta", train=True,
        )[0][:3]

        accountant = FlopAccountant(model, world_size=1)
        counter = FlopCounterMode(model, depth=3, display=False)
        with counter, torch.no_grad():
            model(x, state_labels, bcs.tolist(), metadata=batch["metadata"])
        geom = accountant._observed
        accountant.detach()  # else the next dataset's accountant shares the encoder hooks
        fwd = forward_flops(accountant.shape, geom)
        enc_m, dec_m = subtree_flops(counter)

        res = tuple(int(s) for s in x.shape[3:])
        padded = res + (1,) * (len(geom.encoder_in_shape) - len(res))
        tok_ok = geom.token_shape == analytic_token_shape(padded, geom.periodic)
        enc_ok, dec_ok = fwd["encoder"] == enc_m, fwd["decoder"] == dec_m
        if not (tok_ok and enc_ok and dec_ok):
            problems.append(f"{name}: tokens {tok_ok}, encoder {fwd['encoder']} vs {enc_m}, "
                            f"decoder {fwd['decoder']} vs {dec_m}")
        print(f"{name:<36}{str(res):>16}{geom.n_fields_out:>7}{str(geom.token_shape):>16}"
              f"{'ok' if tok_ok else 'FAIL':>9}{'exact' if enc_ok else 'FAIL':>9}"
              f"{'exact' if dec_ok else 'FAIL':>9}{read_s:>8.2f}")

    print()
    if problems:
        print("RESULT: MISMATCH")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print(f"RESULT: ALL EXACT ({len(names)} datasets, n_states={n_states}, encoder keys ['3'])")


if __name__ == "__main__":
    main()
