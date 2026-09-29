"""Analytical FLOP accounting for `IsotropicModel` (SCALING_STRATEGY.md §5.1).

The closed form below is validated to the FLOP — not to a few percent — against
`torch.utils.flop_counter.FlopCounterMode` in
`demo_notebooks/walrus_flops_analysis.ipynb` (§3-§6 for the processor, §12 for the
encoder/decoder at real dataset resolutions). It is the *primary* instrument: unlike
`FlopCounterMode` it is backend-independent, costs nothing to evaluate, and is defined
for shapes that have never been run.

Three things make WALRUS FLOPs harder to price than a plain ViT's, and all three are
handled explicitly here rather than approximated:

1. **Per-dataset channel counts.** The encoder convolves only over the fields the
   *current* dataset has (`proj1.weight[:, field_indices]`, `vstride_encoder.py:349`),
   not over the global `n_states`. Assuming the latter overestimates encoder cost by
   1.4x-3.1x across the shapes this study uses, so the counts are read per micro-batch
   from the tensors the model actually receives.
2. **Adaptive strides.** The token grid is held at a fixed count per axis by varying the
   encoder stride with input resolution, so resolution enters cost only through the
   intermediate grid between the two conv stages — a step function, not a smooth scaling.
3. **Periodic boundaries change the decoder's work.** `adaptive_conv_transpose` circularly
   pads periodic axes before the transposed conv and crops afterwards
   (`vstride_decoder.py:213-238`), so the conv runs on more positions than the token grid
   implies. This is the term an earlier hand-derivation missed.

Micro-batch FLOPs are *not* constant across the data mixture even though token count is
(§3.2): they span ~1.03x at fullDims and ~1.95x at `320/10/1280` across the seven 2D
datasets. `FlopAccountant` therefore accumulates per micro-batch using the dataset
actually drawn, never `per-shape-cost x steps`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from the_well.data.datasets import BoundaryCondition

logger = logging.getLogger(__name__)

# `IsotropicModel._encoder_forward` appends the fixed indices [2, 0, 1]
# (closed_boundary/open_boundary/bias_correction) to state_labels before the encoder,
# and the decoder does not reconstruct them.
N_BC_CHANNELS = 3

# Backward is two matmuls per forward matmul (grad wrt input, grad wrt weight).
BACKWARD_MULTIPLIER = 2.0

# The four proj1/proj2 sub-stages `FlopAccountant` tracks individually, on top of the
# coarse encoder/processor/decoder split (SCALING_STRATEGY.md 5.1). `forward_flops`
# reports these as pure forward-pass FLOPs; `training_step_flops` then applies the same
# `(1 + BACKWARD_MULTIPLIER)` rule to them as it does to "processor" (no recompute
# folded in, same as "processor"), so `FlopAccountant.cumulative_{k}_flops` is on the
# same forward+backward basis as `cumulative_processor_flops` and the two can be summed
# or compared directly. `enc_dec_flop_fraction` is unaffected -- it's still built from
# the pure-forward `forward_encoder`/`forward_decoder`, a separate accumulator.
GRANULAR_FLOP_KEYS = ("encoder_proj1", "encoder_proj2", "decoder_proj1", "decoder_proj2")


def conv_out_size(in_size: int, kernel: int, stride: int) -> int:
    """Strided-conv output size along one axis.

    Walrus's `learned_pad` path pads the input in the patch jitterer and then convolves
    with `padding=0`, so no padding term appears here.
    """
    if in_size == 1:
        return 1  # adaptive_conv collapses the kernel on singleton axes
    return (in_size - kernel) // stride + 1


def _effective_kernel(
    kernel: Sequence[int], in_shape: Sequence[int]
) -> Tuple[int, ...]:
    """`adaptive_conv` sums the kernel down to size 1 on singleton axes."""
    return tuple(1 if s == 1 else k for k, s in zip(kernel, in_shape))


def conv_stage_flops(
    in_shape: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    c_in: int,
    c_out: int,
    batch: int,
    transposed: bool = False,
) -> int:
    """FLOPs for one adaptive strided (or transposed) conv stage.

    A forward conv costs one multiply-add per *output* position; a transposed conv costs
    one per *input* position. `in_shape` must already include any padding the conv
    actually sees.
    """
    eff_kernel = _effective_kernel(kernel, in_shape)
    if transposed:
        n_positions = math.prod(in_shape)
    else:
        n_positions = math.prod(
            conv_out_size(s, k, st) for s, k, st in zip(in_shape, eff_kernel, stride)
        )
    return 2 * batch * n_positions * math.prod(eff_kernel) * c_in * c_out


def space_mixing_flops(
    hidden_dim: int, mlp_dim: int, n_space: int, t_times_b: int
) -> Dict[str, int]:
    """One `FullAttention` spatial block, by submodule."""
    return {
        "fused_ff_qkv": 2 * t_times_b * n_space * hidden_dim * (mlp_dim + 3 * hidden_dim),
        "attention": 4 * t_times_b * (n_space**2) * hidden_dim,
        "attn_out": 2 * t_times_b * n_space * hidden_dim * hidden_dim,
        "ff_out": 2 * t_times_b * n_space * (mlp_dim // 2) * hidden_dim,
    }


def time_mixing_flops(
    hidden_dim: int, n_time: int, b_times_n_space: int
) -> Dict[str, int]:
    """One `AxialTimeAttention` temporal block, by submodule."""
    return {
        "input_head": 2 * (n_time * b_times_n_space) * hidden_dim * (3 * hidden_dim),
        "attention": 4 * b_times_n_space * (n_time**2) * hidden_dim,
        "output_head": 2 * (n_time * b_times_n_space) * hidden_dim * hidden_dim,
    }


def channel_mixing_flops(
    hidden_dim: int, channel_mlp_dim: Optional[int], n_tokens: int
) -> int:
    """0 when channel mixing is `Identity` (the default), else a 2-layer per-token MLP."""
    if channel_mlp_dim is None:
        return 0
    return 4 * n_tokens * hidden_dim * channel_mlp_dim


@dataclass(frozen=True)
class ModelShape:
    """The shape knobs and fixed architecture constants that set cost."""

    hidden_dim: int
    mlp_dim: int
    processor_blocks: int
    intermediate_dim: int
    encoder_kernel1: Tuple[int, ...]
    encoder_kernel2: Tuple[int, ...]
    decoder_kernel1: Tuple[int, ...]
    decoder_kernel2: Tuple[int, ...]
    causal_in_time: bool = False
    gradient_checkpointing_freq: int = 0
    channel_mlp_dim: Optional[int] = None

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelShape":
        """Read the shape off a live (possibly distribution-wrapped) model."""
        model = unwrap_model(model)
        encoder = next(iter(model.embed.values()))
        decoder = next(iter(model.debed.values()))
        block = model.blocks[0]
        channel_mixing = getattr(block, "channel_mixing", None)
        return cls(
            hidden_dim=model.hidden_dim,
            mlp_dim=block.space_mixing.mlp_dim,
            processor_blocks=len(model.blocks),
            intermediate_dim=encoder.inner_dim,
            encoder_kernel1=tuple(encoder.base_kernel1),
            encoder_kernel2=tuple(encoder.base_kernel2),
            decoder_kernel1=tuple(decoder.base_kernel1),
            decoder_kernel2=tuple(decoder.base_kernel2),
            causal_in_time=model.causal_in_time,
            gradient_checkpointing_freq=model.gradient_checkpointing_freq,
            channel_mlp_dim=(
                None
                if channel_mixing is None or isinstance(channel_mixing, nn.Identity)
                else getattr(channel_mixing, "hidden_features", None)
            ),
        )


@dataclass(frozen=True)
class BatchGeometry:
    """Everything about one micro-batch that affects its FLOP count.

    `encoder_in_shape` is the spatial shape *after* the patch jitterer's padding, i.e.
    what the encoder's first conv actually sees. Being frozen and hashable, this doubles
    as the cache key in `FlopAccountant`.
    """

    encoder_in_shape: Tuple[int, ...]
    stride1: Tuple[int, ...]
    stride2: Tuple[int, ...]
    periodic: Tuple[bool, ...]
    n_fields_in: int
    n_fields_out: int
    n_time_in: int
    batch_size: int
    token_shape: Optional[Tuple[int, ...]] = None
    dataset_name: str = ""

    def encoder_mid_shape(self, kernel1: Sequence[int]) -> Tuple[int, ...]:
        eff = _effective_kernel(kernel1, self.encoder_in_shape)
        return tuple(
            conv_out_size(s, k, st)
            for s, k, st in zip(self.encoder_in_shape, eff, self.stride1)
        )

    def encoder_out_shape(self, kernel1, kernel2) -> Tuple[int, ...]:
        mid = self.encoder_mid_shape(kernel1)
        eff = _effective_kernel(kernel2, mid)
        return tuple(
            conv_out_size(s, k, st) for s, k, st in zip(mid, eff, self.stride2)
        )

    @property
    def n_spatial_tokens(self) -> int:
        if self.token_shape is None:
            raise ValueError("token_shape unknown; call resolve_tokens() first")
        return math.prod(self.token_shape)


def decoder_stage_shapes(
    in_shape: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int],
    periodic: Sequence[bool],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return `(padded_in, out)` per axis for one `adaptive_conv_transpose` stage.

    Periodic axes are circularly padded by `(k - s) // s` on each side before the
    transposed conv and cropped by `k - s` on each side after
    ([vstride_decoder.py:210-251](../models/decoders/vstride_decoder.py)). The padding is
    what makes the conv cost more than the unpadded grid suggests; the crop is what makes
    the two stages round-trip exactly back onto the encoder's input grid.
    """
    padded, out = [], []
    for size, k, s, is_periodic in zip(in_shape, kernel, stride, periodic):
        if size == 1:
            padded.append(1)
            out.append(1)
            continue
        pad_in = (k - s) // s if is_periodic else 0
        size_padded = size + 2 * pad_in
        size_out = (size_padded - 1) * s + k
        if is_periodic:
            size_out -= 2 * (k - s)
        padded.append(size_padded)
        out.append(size_out)
    return tuple(padded), tuple(out)


def forward_flops(shape: ModelShape, geom: BatchGeometry) -> Dict[str, int]:
    """Forward-pass FLOPs for one micro-batch, split encoder/processor/decoder.

    Also returns the per-stage breakdown within the encoder and decoder (`proj1`/`proj2`,
    the two `adaptive_conv`/`adaptive_conv_transpose` stages either side of the token
    grid) and within the processor (`space_mixing`/`time_mixing`/`channel_mixing`, each
    itself a dict by submodule from `space_mixing_flops`/`time_mixing_flops`). These are
    additive, not a replacement: `encoder == encoder_proj1 + encoder_proj2`, etc., so
    existing callers reading the coarse keys (`training_step_flops`, the FlopCounterMode
    cross-check in `tests/test_flops.py`) are unaffected.
    """
    tb = geom.n_time_in * geom.batch_size

    mid_shape = geom.encoder_mid_shape(shape.encoder_kernel1)
    token_shape = geom.token_shape or geom.encoder_out_shape(
        shape.encoder_kernel1, shape.encoder_kernel2
    )
    encoder_proj1 = conv_stage_flops(
        geom.encoder_in_shape,
        shape.encoder_kernel1,
        geom.stride1,
        geom.n_fields_in,
        shape.intermediate_dim,
        batch=tb,
    )
    encoder_proj2 = conv_stage_flops(
        mid_shape,
        shape.encoder_kernel2,
        geom.stride2,
        shape.intermediate_dim,
        shape.hidden_dim,
        batch=tb,
    )
    encoder = encoder_proj1 + encoder_proj2

    n_space = math.prod(token_shape)
    space = space_mixing_flops(shape.hidden_dim, shape.mlp_dim, n_space, tb)
    time_ = time_mixing_flops(
        shape.hidden_dim, geom.n_time_in, geom.batch_size * n_space
    )
    channel = channel_mixing_flops(
        shape.hidden_dim, shape.channel_mlp_dim, tb * n_space
    )
    space_total = shape.processor_blocks * sum(space.values())
    time_total = shape.processor_blocks * sum(time_.values())
    channel_total = shape.processor_blocks * channel
    processor = space_total + time_total + channel_total

    # The decoder re-expands only the last time step unless the model is causal in time
    # (`isotropic_model.py:397-398`).
    t_out = geom.n_time_in if shape.causal_in_time else 1
    # The decoder swaps the encoder's per-axis stride pair (`vstride_decoder.py:261`).
    dec_stride1, dec_stride2 = geom.stride2, geom.stride1
    padded1, mid1 = decoder_stage_shapes(
        token_shape, shape.decoder_kernel1, dec_stride1, geom.periodic
    )
    padded2, _ = decoder_stage_shapes(
        mid1, shape.decoder_kernel2, dec_stride2, geom.periodic
    )
    decoder_proj1 = conv_stage_flops(
        padded1,
        shape.decoder_kernel1,
        dec_stride1,
        shape.hidden_dim,
        shape.intermediate_dim,
        batch=t_out * geom.batch_size,
        transposed=True,
    )
    decoder_proj2 = conv_stage_flops(
        padded2,
        shape.decoder_kernel2,
        dec_stride2,
        shape.intermediate_dim,
        geom.n_fields_out,
        batch=t_out * geom.batch_size,
        transposed=True,
    )
    decoder = decoder_proj1 + decoder_proj2

    return {
        "encoder": encoder,
        "encoder_proj1": encoder_proj1,
        "encoder_proj2": encoder_proj2,
        "processor": processor,
        "processor_space_mixing": space_total,
        "processor_time_mixing": time_total,
        "processor_channel_mixing": channel_total,
        "decoder": decoder,
        "decoder_proj1": decoder_proj1,
        "decoder_proj2": decoder_proj2,
        "total": encoder + processor + decoder,
    }


def training_step_flops(forward: Dict[str, int], shape: ModelShape) -> Dict[str, float]:
    """Forward + backward FLOPs for one micro-batch, including recompute.

    Gradient checkpointing re-runs part of the forward during backward. That changes
    *measured* FLOPs without changing *useful* FLOPs, and the two arms of the §8
    head-to-head need not share the setting, so the surcharge is reported separately
    rather than folded in silently. The encoder and decoder are checkpointed whenever
    any checkpointing is on; processor block `i` is checkpointed when
    `i % freq == 0` (`isotropic_model.py:126-135`).

    `processor` and each of `GRANULAR_FLOP_KEYS` follow the same rule: forward FLOPs
    times `(1 + BACKWARD_MULTIPLIER)`, with the checkpointing recompute surcharge left
    out and reported only in the lump `recompute`/`total` above -- so every per-submodule
    figure this function returns is on the same basis and can be summed or compared
    directly (`FlopAccountant`'s `cumulative_processor_flops` and
    `cumulative_{encoder,decoder}_proj{1,2}_flops` all come from here).
    """
    freq = shape.gradient_checkpointing_freq
    if freq > 0:
        n_checkpointed = math.ceil(shape.processor_blocks / freq)
        recompute = (
            forward["encoder"]
            + forward["decoder"]
            + forward["processor"] * n_checkpointed / shape.processor_blocks
        )
    else:
        recompute = 0.0

    useful = forward["total"] * (1.0 + BACKWARD_MULTIPLIER)
    result = {
        "useful": useful,
        "recompute": float(recompute),
        "total": useful + recompute,
        "processor": forward["processor"] * (1.0 + BACKWARD_MULTIPLIER),
    }
    for k in GRANULAR_FLOP_KEYS:
        result[k] = forward[k] * (1.0 + BACKWARD_MULTIPLIER)
    return result


def analytic_token_shape(
    resolution: Sequence[int], periodic: Sequence[bool]
) -> Tuple[int, ...]:
    """Token grid Walrus produces for a dataset, without running it.

    `choose_kernel_size_deterministic` fixes the token count per axis (32 when at most
    two spatial axes are non-singleton, 16 when all three are), and the patch jitterer's
    boundary padding adds exactly one more token on each *non*-periodic axis — periodic
    axes get circular padding that cancels. Verified against the model in
    `walrus_tokens_and_data_pipeline.md` §1.2.

    Used for planning and for the unique-token denominator; the accountant itself
    observes the real grid instead.
    """
    non_singleton = sum(1 for r in resolution if r != 1)
    base = 512 // 16 if non_singleton <= 2 else 256 // 16
    return tuple(
        1 if r == 1 else base + (0 if p else 1)
        for r, p in zip(resolution, periodic)
    )


def encoder_shapes_from_tokens(
    token_shape: Sequence[int],
    stride1: Sequence[int],
    stride2: Sequence[int],
    kernel1: Sequence[int],
    kernel2: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Invert the encoder to `(padded_in_shape, mid_shape)` for a wanted token grid.

    The patch jitterer pads the input to whatever makes the two `padding=0` convs land
    on the target token count, so the padded shape is recovered by inverting
    `conv_out_size` twice rather than by modelling the padding rule. For planning shapes
    that have never been run; `FlopAccountant` observes the real shape instead.
    """
    mid, padded = [], []
    for tokens, s1, s2, k1, k2 in zip(token_shape, stride1, stride2, kernel1, kernel2):
        if tokens == 1:
            mid.append(1)
            padded.append(1)
            continue
        mid_size = (tokens - 1) * s2 + k2
        mid.append(mid_size)
        padded.append((mid_size - 1) * s1 + k1)
    return tuple(padded), tuple(mid)


def unique_model_tokens(datamodule: Any) -> Optional[int]:
    """`Σ_datasets n_trajectories × n_steps × N_spatial` for the training mixture.

    This is `T_unique_model` from SCALING_STRATEGY.md §3.2 — the denominator of the
    repetition factor. Returns `None` if the datamodule does not expose per-dataset
    metadata in the expected form, in which case the repetition factor is simply not
    logged.
    """
    dataset = getattr(datamodule, "train_dataset", None)
    sub_dsets = getattr(dataset, "sub_dsets", None)
    if not sub_dsets:
        return None

    total = 0
    for sub in sub_dsets:
        meta = sub.metadata
        periodic = [
            str(bc).upper() == BoundaryCondition.PERIODIC.name
            for bc in meta.boundary_condition_types
        ]
        # Pad to the resolution's rank; unspecified axes are treated as non-periodic,
        # which only ever costs one extra token per axis.
        periodic += [False] * (len(meta.spatial_resolution) - len(periodic))
        n_spatial = math.prod(
            analytic_token_shape(meta.spatial_resolution, periodic)
        )
        steps = sum(
            n_traj * n_steps
            for n_traj, n_steps in zip(
                meta.n_trajectories_per_file, meta.n_steps_per_trajectory
            )
        )
        total += steps * n_spatial
    return total


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP wrappers to reach the `IsotropicModel` underneath."""
    for attr in ("_fsdp_wrapped_module", "module"):
        while hasattr(model, attr) and not hasattr(model, "blocks"):
            model = getattr(model, attr)
    return model


def supports_flop_accounting(model: nn.Module) -> bool:
    """Whether this is an encoder/processor/decoder model the formula describes.

    The baseline wrappers in `walrus/models/baseline_wrappers.py` are not, and run
    through the same `Trainer`.
    """
    unwrapped = unwrap_model(model)
    return all(
        hasattr(unwrapped, attr) for attr in ("embed", "debed", "blocks", "hidden_dim")
    )


def _periodic_flags(bcs: Any, n_axes: int) -> Tuple[bool, ...]:
    """Per-axis periodicity from a `(n_dims, 2)` boundary-condition tensor.

    Axis order matches the spatial axes, as `adaptive_conv_transpose` assumes.
    """
    flags = []
    for axis in range(n_axes):
        try:
            flags.append(int(bcs[axis][0]) == BoundaryCondition.PERIODIC.value)
        except (IndexError, TypeError):
            flags.append(False)
    return tuple(flags)


class FlopAccountant:
    """Accumulates per-micro-batch FLOPs and token counts over a training epoch.

    Geometry is observed from the model rather than re-derived from the input
    resolution: a forward pre-hook on the encoder records the padded input shape, the
    strides actually chosen, the boundary conditions, and the real per-dataset channel
    count, and a forward hook records the resulting token grid. Everything downstream is
    exact arithmetic on those observations. The FLOP evaluation itself is cached on the
    geometry, so only the first micro-batch of each `(dataset, resolution, shape)`
    combination costs anything.

    Counts are per rank. `max_samples` caps micro-batches per rank, so global compute is
    `world_size x` the per-rank figure; `epoch_metrics` reports both, and reduces across
    ranks exactly when a process group exists.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        unique_model_tokens: Optional[int] = None,
    ):
        self.shape = ModelShape.from_model(model)
        self.world_size = world_size
        self.unique_model_tokens = unique_model_tokens
        self._cache: Dict[BatchGeometry, Dict[str, float]] = {}
        self._observed: Optional[BatchGeometry] = None
        self._handles: list = []
        self.cumulative_flops = 0.0
        self.cumulative_processor_flops = 0.0
        self.cumulative_granular_flops = {k: 0.0 for k in GRANULAR_FLOP_KEYS}
        self.cumulative_tokens = 0
        self.reset_epoch()
        self._attach(model)

    def _attach(self, model: nn.Module) -> None:
        unwrapped = unwrap_model(model)
        for encoder in unwrapped.embed.values():
            self._handles.append(
                encoder.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            )
            self._handles.append(
                encoder.register_forward_hook(self._post_hook, with_kwargs=False)
            )

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, module, args, kwargs):
        # SpaceBagAdaptiveDVstrideEncoder.forward(x, field_indices, bcs, metadata, *,
        # random_kernel). `field_indices` here is state_labels with the 3 BC-flag
        # indices already appended by IsotropicModel._encoder_forward.
        x, field_indices = args[0], args[1]
        bcs = args[2] if len(args) > 2 else kwargs.get("bcs")
        random_kernel = kwargs.get("random_kernel")
        if random_kernel is None:
            return
        spatial = tuple(int(s) for s in x.shape[3:])
        n_axes = len(spatial)
        n_fields_in = int(field_indices.numel())
        self._observed = BatchGeometry(
            encoder_in_shape=spatial,
            stride1=tuple(int(random_kernel[i][0]) for i in range(n_axes)),
            stride2=tuple(int(random_kernel[i][1]) for i in range(n_axes)),
            periodic=_periodic_flags(bcs, n_axes),
            n_fields_in=n_fields_in,
            n_fields_out=n_fields_in - N_BC_CHANNELS,
            n_time_in=int(x.shape[0]),
            batch_size=int(x.shape[1]),
        )

    def _post_hook(self, module, args, output):
        if self._observed is None:
            return
        x_out = output[0] if isinstance(output, tuple) else output
        token_shape = tuple(int(s) for s in x_out.shape[3:])
        self._observed = BatchGeometry(
            **{**self._observed.__dict__, "token_shape": token_shape}
        )

    def reset_epoch(self) -> None:
        self._epoch_flops = 0.0
        self._epoch_processor_flops = 0.0
        self._epoch_forward_total = 0.0
        self._epoch_enc_dec_flops = 0.0
        self._epoch_granular_flops = {k: 0.0 for k in GRANULAR_FLOP_KEYS}
        self._epoch_tokens = 0
        self._epoch_microbatches = 0

    def record_microbatch(self, dataset_name: str = "") -> Optional[Dict[str, float]]:
        """Account for the micro-batch that was just run through the model.

        Returns `None` if no forward pass has been observed yet (e.g. a fake pass in
        distributed validation), so callers can treat that as a no-op.
        """
        geom = self._observed
        if geom is None:
            return None
        if dataset_name:
            geom = BatchGeometry(**{**geom.__dict__, "dataset_name": dataset_name})

        entry = self._cache.get(geom)
        if entry is None:
            forward = forward_flops(self.shape, geom)
            step = training_step_flops(forward, self.shape)
            entry = {
                "forward_total": float(forward["total"]),
                "forward_encoder": float(forward["encoder"]),
                "forward_processor": float(forward["processor"]),
                "forward_decoder": float(forward["decoder"]),
                "step_total": step["total"],
                "step_processor": step["processor"],
                "tokens": float(
                    geom.batch_size * geom.n_time_in * geom.n_spatial_tokens
                ),
            }
            for k in GRANULAR_FLOP_KEYS:
                entry[f"forward_{k}"] = float(forward[k])
                entry[f"step_{k}"] = float(step[k])
            self._cache[geom] = entry
            logger.debug(
                "FLOP geometry %s: %.3f GF/micro-batch forward, enc+dec %.1f%%",
                dataset_name or geom.encoder_in_shape,
                entry["forward_total"] / 1e9,
                100
                * (entry["forward_encoder"] + entry["forward_decoder"])
                / entry["forward_total"],
            )

        self._epoch_flops += entry["step_total"]
        self._epoch_processor_flops += entry["step_processor"]
        self._epoch_forward_total += entry["forward_total"]
        self._epoch_enc_dec_flops += entry["forward_encoder"] + entry["forward_decoder"]
        for k in GRANULAR_FLOP_KEYS:
            # step_{k}, not forward_{k}: same forward+backward basis as
            # _epoch_processor_flops (step_processor), so cumulative_{k}_flops is
            # directly comparable/summable with cumulative_processor_flops.
            self._epoch_granular_flops[k] += entry[f"step_{k}"]
        self._epoch_tokens += int(entry["tokens"])
        self._epoch_microbatches += 1
        return entry

    def epoch_metrics(self, elapsed_seconds: float) -> Dict[str, float]:
        """Per-epoch compute metrics for the training log (SCALING_STRATEGY.md §5.1).

        Must be called on every rank: it all-reduces when a process group exists, so the
        totals are exact rather than `rank 0 x world_size`.
        """
        granular_local = [self._epoch_granular_flops[k] for k in GRANULAR_FLOP_KEYS]
        local = torch.tensor(
            [
                self._epoch_flops,
                self._epoch_processor_flops,
                self._epoch_forward_total,
                self._epoch_enc_dec_flops,
                float(self._epoch_tokens),
                float(self._epoch_microbatches),
                *granular_local,
            ],
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            total = local.clone().to(_reduce_device())
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            total = total.cpu()
        else:
            total = local * self.world_size

        (
            epoch_flops,
            proc_flops,
            fwd_total,
            enc_dec,
            tokens,
            microbatches,
            *granular_total,
        ) = (t.item() for t in total)
        granular: Dict[str, float] = dict(zip(GRANULAR_FLOP_KEYS, granular_total))

        self.cumulative_flops += epoch_flops
        self.cumulative_processor_flops += proc_flops
        self.cumulative_tokens += int(tokens)
        for k in GRANULAR_FLOP_KEYS:
            self.cumulative_granular_flops[k] += granular[k]

        metrics = {
            "flops_per_microbatch": epoch_flops / microbatches if microbatches else 0.0,
            "epoch_flops": epoch_flops,
            "cumulative_flops": self.cumulative_flops,
            "cumulative_processor_flops": self.cumulative_processor_flops,
            "tokens_seen": float(self.cumulative_tokens),
            "enc_dec_flop_fraction": enc_dec / fwd_total if fwd_total else 0.0,
            "gpu_hours": elapsed_seconds * self.world_size / 3600.0,
        }
        for k in GRANULAR_FLOP_KEYS:
            metrics[f"epoch_{k}_flops"] = granular[k]
            metrics[f"cumulative_{k}_flops"] = self.cumulative_granular_flops[k]
        if self.unique_model_tokens:
            metrics["repetition_factor"] = (
                self.cumulative_tokens / self.unique_model_tokens
            )
        return metrics


def _reduce_device() -> torch.device:
    """NCCL cannot reduce CPU tensors; Gloo can."""
    backend = torch.distributed.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
