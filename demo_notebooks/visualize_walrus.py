#!/usr/bin/env python
"""
Export Walrus IsotropicModel (and its sub-blocks) to ONNX for viewing in
ai-edge-model-explorer.

The full IsotropicModel.forward is hard to trace end-to-end (it takes a
WellMetadata dataclass, does dataclasses.replace, python control flow on
tensor shapes, torch.utils.checkpoint, the patch jitterer, .item() calls, ...).
So this script exports the pieces that matter and trace cleanly:

  processor_block   one SpaceTimeSplitBlock  (FullAttention + AxialTimeAttention
                    + Identity channel-mix) -- the unit repeated `processor_blocks`
                    times; this is ~86% of the model's params
  space_attention   the FullAttention sub-block alone
  time_attention    the AxialTimeAttention sub-block alone
  encoder           SpaceBagAdaptiveDVstrideEncoder (patchify + dim-raise convs)
  decoder           AdaptiveDVstrideDecoder (mirror)
  isotropic_full    the whole model via a tensor-only wrapper -- EXPECTED TO FAIL
                    (IsotropicModel.forward takes a WellMetadata dataclass, a list
                    of boundary conditions, does dataclasses.replace + shape-
                    dependent python branching + torch.utils.checkpoint). The five
                    pieces above already cover the whole architecture: the
                    processor is just `processor_block` x processor_blocks.

A TINY config is used (hidden_dim=64, processor_blocks=2, ...) so the graphs
render fast and stay readable. The *structure* is identical to paper scale --
only the tensor sizes shrink. See `module_tree.txt` (torchinfo) for the full
nn.Module hierarchy + per-module param counts (no ONNX needed).

Run (on the Leonardo login node, which has internet):

    module purge; module load python; module load cuda/12.2
    source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
    pip install "onnx==1.16.2"           # 1.22+ drags in numpy 2 and breaks env1
    python demo_notebooks/visualize_walrus.py

Then view:

    pip install ai-edge-model-explorer
    model_explorer demo_notebooks/model_viz/processor_block.onnx --no_open_in_browser --port 8080
    # ssh -L 8080:localhost:8080 <leonardo>   then open http://localhost:8080

or `scp demo_notebooks/model_viz/*.onnx` to your laptop and run model_explorer there.
"""

from __future__ import annotations

import contextlib
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

import walrus.models.isotropic_model as _im
from walrus.models.shared_utils import position_biases as _pb
from walrus.models.shared_utils.normalization import RMSGroupNorm


@contextlib.contextmanager
def export_friendly():
    """A few ops in Walrus don't survive the TorchScript→ONNX exporter. Swap them
    for math-equivalent plain-op versions just while tracing, then restore."""
    saved = {}

    # 1) F.rms_norm has no ONNX lowering at opset <= 18.
    saved["rms"] = RMSGroupNorm.forward

    def rms_forward(self, x):
        dims = list(x.shape[2:])
        x = x.view(x.shape[0], self.num_groups, -1, *dims)
        red = tuple(range(3, x.ndim))  # rms over trailing (spatial) axes, per group
        x = x * torch.rsqrt(x.pow(2).mean(dim=red, keepdim=True) + self.eps)
        x = x.reshape(x.shape[0], -1, *dims)
        if self.weight is not None:
            x = x * self.weight[(slice(None),) + (None,) * len(dims)]
        return x

    RMSGroupNorm.forward = rms_forward

    # 2) RelativePositionBias does `values[..., mask] = -inf` (bool index_put),
    #    which the exporter emits as a malformed node. Same result via torch.where.
    saved["rpb"] = _pb.RelativePositionBias.compute_bias

    def compute_bias(self, qlen, klen, bc=0):
        dev = self.relative_attention_bias.weight.device
        ctx = torch.arange(qlen, dtype=torch.long, device=dev)[:, None]
        mem = torch.arange(klen, dtype=torch.long, device=dev)[None, :]
        rel = mem - ctx
        bucket = self._relative_position_bucket(
            rel, bidirectional=self.bidirectional, num_buckets=self.num_buckets
        ).to(dev)
        vals = self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)
        if not self.bidirectional:
            causal = torch.ones(qlen, klen, dtype=torch.bool, device=dev).triu(1)
            vals = torch.where(causal, torch.full_like(vals, float("-inf")), vals)
        return vals

    _pb.RelativePositionBias.compute_bias = compute_bias

    # 3) choose_kernel_size_deterministic indexes a dict with traced tensors.
    #    Pin it to the standard [8,4] scale per axis for the full-model export.
    saved["cks"] = _im.choose_kernel_size_deterministic
    _im.choose_kernel_size_deterministic = lambda x_shape: tuple((8, 4) for _ in x_shape)

    try:
        yield
    finally:
        RMSGroupNorm.forward = saved["rms"]
        _pb.RelativePositionBias.compute_bias = saved["rpb"]
        _im.choose_kernel_size_deterministic = saved["cks"]

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo_notebooks" / "model_viz"
OUT.mkdir(parents=True, exist_ok=True)

OPSET = 17
N_STATES = 8          # size of the (fake) field-index map
N_PRESENT = 5         # "real" fields in the batch; +3 boundary/bias slots = 8
HIDDEN = 64
INNER = 32
BLOCKS = 2
HEADS = 4
T, B = 4, 1           # timesteps, batch
GRID = (8, 8, 1)      # token grid after the encoder (Gh, Gw, Gd)
RES = (64, 64, 1)     # raw field resolution fed to the encoder


def build_model() -> nn.Module:
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "walrus" / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                "server=leonardo",
                "model=isotropic_model",
                "model/processor/space_mixing=full_spatial_attention",
                f"model.hidden_dim={HIDDEN}",
                f"model.intermediate_dim={INNER}",
                f"model.processor_blocks={BLOCKS}",
                f"model.groups={HEADS}",
                f"model.processor.space_mixing.num_heads={HEADS}",
                f"model.processor.time_mixing.num_heads={HEADS}",
                "model.causal_in_time=True",
                "model.override_dimensionality=0",
                "model.jitter_patches=False",
                "++model.use_periodic_fixed_jitter=True",
                "++model.input_field_drop=0.0",
                # the rest of `config.yaml`'s groups still need to resolve
                "data=Leonardo_active_matter",
                "trainer=defaults",
                "optimizer=adam",
                "lr_scheduler=inv_sqrt_w_sqrt_ramps",
                "distribution=local",
                "experiment=defaults",
                "checkpoint=defaults",
                "finetuning_mods=defaults",
            ],
        )
    from hydra.utils import instantiate

    model = instantiate(cfg.model, n_states=N_STATES)
    model.eval()
    return model


class FirstOut(nn.Module):
    """Blocks return `(tensor, [])`; the trailing empty list breaks the ONNX
    exporter's output-shape pass. Keep only the tensor."""

    def __init__(self, mod: nn.Module, n_args: int = 1):
        super().__init__()
        self.mod = mod
        self.n_args = n_args

    def forward(self, *args):
        out = self.mod(*args[: self.n_args])
        return out[0] if isinstance(out, (tuple, list)) else out


class EncoderWrap(nn.Module):
    """SpaceBagAdaptiveDVstrideEncoder with field_indices / random_kernel fixed."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc
        fi = list(range(3, 3 + N_PRESENT)) + [2, 0, 1]  # real slots + BC slots
        self.register_buffer("field_indices", torch.tensor(fi, dtype=torch.long))
        self.random_kernel = [[8, 4], [8, 4], [8, 4]]

    def forward(self, x):  # x: (T, B, N_PRESENT+3, *RES)
        out, _ = self.enc(x, self.field_indices, None, None, random_kernel=self.random_kernel)
        return out


class DecoderWrap(nn.Module):
    """AdaptiveDVstrideDecoder with state_labels / stage_info / bcs fixed."""

    def __init__(self, dec: nn.Module):
        super().__init__()
        self.dec = dec
        sl = list(range(3, 3 + N_PRESENT)) + [2, 0, 1]
        self.register_buffer("state_labels", torch.tensor(sl, dtype=torch.long))
        self.stage_info = {"random_kernel": [[8, 4], [8, 4], [8, 4]]}
        self.bcs = [[2, 2], [2, 2], [2, 2]]

    def forward(self, x):  # x: (T, B, HIDDEN, *GRID)
        return self.dec(x, self.state_labels, self.bcs, self.stage_info, None)


def targets(model: nn.Module):
    dim_key = next(iter(model.embed.keys()))
    blk = model.blocks[0]
    x_block = torch.randn(T, B, HIDDEN, *GRID)
    x_space = torch.randn(T * B, HIDDEN, *GRID)
    bcs_dummy = torch.zeros(3, 2, dtype=torch.long)
    x_enc = torch.randn(T, B, N_PRESENT + 3, *RES)
    x_dec = torch.randn(T, B, HIDDEN, *GRID)

    return [
        ("processor_block", FirstOut(blk, n_args=2), (x_block, bcs_dummy)),
        ("space_attention", FirstOut(blk.space_mixing, n_args=2), (x_space, bcs_dummy)),
        ("time_attention", FirstOut(blk.time_mixing, n_args=1), (x_block,)),
        ("encoder", EncoderWrap(model.embed[dim_key]), (x_enc,)),
        ("decoder", DecoderWrap(model.debed[dim_key]), (x_dec,)),
    ]


def export_one(name: str, mod: nn.Module, args, results: dict):
    path = OUT / f"{name}.onnx"
    try:
        mod.eval()
        with torch.no_grad(), export_friendly():
            torch.onnx.export(
                mod, args, str(path),
                opset_version=OPSET, do_constant_folding=True,
                input_names=[f"in{i}" for i in range(len(args))],
                dynamo=False,
            )
        results[name] = f"OK   -> {path.relative_to(REPO)}"
    except Exception as e:  # noqa: BLE001
        results[name] = f"FAIL ({type(e).__name__}: {str(e).splitlines()[0][:140]})"
        (OUT / f"{name}.error.txt").write_text(traceback.format_exc())


def try_full_model(model: nn.Module, results: dict):
    """Best-effort: tensor-only wrapper around IsotropicModel.forward."""
    try:
        from dataclasses import dataclass

        @dataclass
        class FakeMeta:
            n_spatial_dims: int = 2
            dataset_name: str = "fake"

        n_real = N_PRESENT
        sl = torch.tensor(list(range(3, 3 + n_real)), dtype=torch.long)
        bcs = torch.full((1, 2, 2), 2, dtype=torch.long)  # (B, n_dims, 2), wall

        class FullWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):  # x: (T, B, n_real, H, W)
                out = self.m(x, sl, bcs, FakeMeta(), train=False)
                return out[0] if isinstance(out, tuple) else out

        x = torch.randn(T, B, n_real, RES[0], RES[1])
        export_one("isotropic_full", FullWrap(model), (x,), results)
    except Exception as e:  # noqa: BLE001
        results["isotropic_full"] = f"FAIL ({type(e).__name__}: {e})"
        (OUT / "isotropic_full.error.txt").write_text(traceback.format_exc())


def main():
    torch.manual_seed(0)
    print(f"building tiny IsotropicModel (hidden_dim={HIDDEN}, blocks={BLOCKS}) ...")
    model = build_model()

    # always-works text artifact: the module tree + per-module param counts
    try:
        from torchinfo import summary

        s = summary(model, depth=7, verbose=0,
                    col_names=("num_params", "params_percent"))
        (OUT / "module_tree.txt").write_text(str(s))
        print(f"wrote {(OUT / 'module_tree.txt').relative_to(REPO)}")
    except Exception as e:  # noqa: BLE001
        print(f"torchinfo summary failed: {e}")

    results: dict[str, str] = {}
    for name, mod, args in targets(model):
        print(f"exporting {name} ...")
        export_one(name, mod, args, results)
    print("exporting isotropic_full (best effort) ...")
    try_full_model(model, results)

    print("\n=== ONNX export results ===")
    for name, status in results.items():
        print(f"  {name:16s} {status}")
    print(f"\nfiles in {OUT.relative_to(REPO)}/")
    print("view:  model_explorer <file>.onnx --no_open_in_browser --port 8080")


if __name__ == "__main__":
    main()
