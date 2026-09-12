"""Tier 0 / §5.1 acceptance criterion 1: is `FlopCounterMode` usable on Leonardo?

`walrus_flops_analysis.ipynb` §4 found that `FlopCounterMode` silently records **zero**
FLOPs for `FullAttention`'s spatial attention. `F.scaled_dot_product_attention(q, k, v)` with
no mask dispatched, on that CPU build, to `aten._scaled_dot_product_flash_attention_for_cpu`,
which is not in `torch.utils.flop_counter.flop_registry` -- so its FLOPs vanish. On the debug
model the dropped term was 25x larger than everything the counter did record for that module.

That matters more than it first looks: the dropped term is the `O(N_spatial**2)` one, which is
exactly the term SCALING_STRATEGY.md §3.1 needs in order to price 2D against 3D micro-batches.

CUDA *should* dispatch to a registered kernel (flash / efficient / cudnn) and therefore should
not have this problem -- but "should" is not a measurement, and two things can change the answer:

  * **dtype.** Flash attention on A100 wants fp16/bf16. Every run in this study sets
    `trainer.enable_amp=False`, i.e. fp32/TF32, where SDPA may instead fall back to the *math*
    backend. Math decomposes into ordinary matmuls, which *are* registered -- so fp32 might
    actually count correctly for a reason unrelated to the fused kernels. Worth knowing, since
    §5.2 keeps `enable_amp=True` on the table as a pre-Tier-1 lever.
  * **shape.** Backend selection depends on head_dim and sequence length, and this study spans
    head_dim 20 (`hidden_dim=320`) to 88 (`hidden_dim=1408`).

So this sweeps dtype x shape rather than taking a single reading.

The verdict is binary and gates §5.1:
  COUNTED  -> `FlopCounterMode` may be used as the cross-check on the analytical formula.
  DROPPED  -> the analytical formula (validated exactly in the notebook) is the only instrument,
              and any FlopCounterMode number in this study must be corrected by hand.

Why this builds the attention modules directly instead of a whole `IsotropicModel`: which SDPA
kernel fires is decided entirely by the module and its input tensor's shape/dtype/device. Going
through Hydra + the datamodule would add an HDF5 dependency and a dozen failure modes unrelated
to the question, for a gate that needs to be unambiguous. The module kwargs below mirror exactly
what `SpaceTimeSplitBlock` passes down (space_time_split.py:31-44) with this study's overrides.

Run:  sbatch LeonardoRunScripts/flop_counter_backend_test.sh
      (or `python flop_counter_backend_test.py --cpu` to reproduce the notebook's CPU finding)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F  # noqa: F401  (documents the call under test)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

try:  # internal name; report it if present, but never let it break the gate
    from torch.utils.flop_counter import flop_registry
except ImportError:  # pragma: no cover
    flop_registry = {}

from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention

# --------------------------------------------------------------------------- formulas
# Copied verbatim from walrus_flops_analysis.ipynb §5 and §6, where each was checked
# against FlopCounterMode to the FLOP. One multiply-accumulate == 2 FLOPs.


def space_mixing_flops(hidden_dim, mlp_dim, n_space, t_times_b):
    """FLOPs for one FullAttention (spatial) block, by submodule."""
    return {
        "fused_ff_qkv": 2 * t_times_b * n_space * hidden_dim * (mlp_dim + 3 * hidden_dim),
        "attention": 4 * t_times_b * (n_space**2) * hidden_dim,
        "attn_out": 2 * t_times_b * n_space * hidden_dim * hidden_dim,
        "ff_out": 2 * t_times_b * n_space * (mlp_dim // 2) * hidden_dim,
    }


def time_mixing_flops(hidden_dim, t, b_times_n_space):
    """FLOPs for one AxialTimeAttention (temporal) block, by submodule."""
    return {
        "input_head": 2 * (t * b_times_n_space) * hidden_dim * (3 * hidden_dim),
        "attention": 4 * b_times_n_space * (t**2) * hidden_dim,
        "output_head": 2 * (t * b_times_n_space) * hidden_dim * hidden_dim,
    }


# --------------------------------------------------------------------------- dispatch probe
class SDPARecorder(TorchDispatchMode):
    """Records which `aten::_scaled_dot_product_*` overload actually fires."""

    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if "scaled_dot_product" in name:
            self.ops.append(name)
        return func(*args, **(kwargs or {}))


# --------------------------------------------------------------------------- cases
# (label, hidden_dim, mlp_dim, n_spatial) -- num_heads=16 and n_steps_input=6 are pinned
# on every run in this study (SCALING_STRATEGY.md §4.1), batch_size=2 per rank.
SHAPES = [
    ("fullDims      1408/5632, 2D", 1408, 5632, 1024),
    ("Tier2 small    320/1280, 2D", 320, 1280, 1024),
    ("fullDims      1408/5632, 3D", 1408, 5632, 4096),
]

# --quick: the smallest shape only. The 3D case is a 4096x4096 attention matrix over
# 16 heads x 12 (t*b) -- fine on an A100, minutes-to-hours on a login node's CPU. Use
# --quick to sanity-check the machinery interactively; the real gate runs all of SHAPES.
QUICK_SHAPES = [("Tier2 small    320/1280, 2D", 320, 1280, 1024)]
NUM_HEADS, T_STEPS, BATCH = 16, 6, 2
CAUSAL_IN_TIME = True  # model.causal_in_time=True on every run -> rel_pos_bias is not None

# Classification tolerance, as a fraction of the analytic attention term.
TOL = 0.01


def classify(residual, attention_flops):
    """COUNTED / DROPPED / AMBIGUOUS, robust to a differing-but-valid FLOP convention."""
    if attention_flops == 0:
        return "N/A"
    if abs(residual) <= TOL * attention_flops:
        return "COUNTED"
    if abs(residual - attention_flops) <= TOL * attention_flops:
        return "DROPPED"
    return "AMBIGUOUS"


def run_case(kind, hidden_dim, mlp_dim, n_spatial, device, dtype, autocast):
    """One (module, shape, dtype) reading. Returns a record dict."""
    torch.manual_seed(0)

    if kind == "space":
        # SpaceTimeSplitBlock passes hidden_dim/drop_path/gradient_checkpointing/norm_layer;
        # num_heads and mlp_dim come from the space_mixing config + this study's overrides.
        mod = FullAttention(hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_heads=NUM_HEADS)
        side = round(n_spatial ** (1 / 3)) if n_spatial == 4096 else round(n_spatial**0.5)
        shape = (
            (T_STEPS * BATCH, hidden_dim, side, side, side)
            if n_spatial == 4096
            else (T_STEPS * BATCH, hidden_dim, side, side, 1)
        )
        args = (torch.randn(*shape, device=device, dtype=dtype), [])
        hand = space_mixing_flops(hidden_dim, mlp_dim, n_spatial, T_STEPS * BATCH)
    else:
        mod = AxialTimeAttention(
            hidden_dim=hidden_dim, num_heads=NUM_HEADS, causal_in_time=CAUSAL_IN_TIME
        )
        side = round(n_spatial ** (1 / 3)) if n_spatial == 4096 else round(n_spatial**0.5)
        shape = (
            (T_STEPS, BATCH, hidden_dim, side, side, side)
            if n_spatial == 4096
            else (T_STEPS, BATCH, hidden_dim, side, side, 1)
        )
        args = (torch.randn(*shape, device=device, dtype=dtype),)
        hand = time_mixing_flops(hidden_dim, T_STEPS, BATCH * n_spatial)

    mod = mod.to(device=device, dtype=dtype).eval()

    ctx = (
        torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
        if autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), ctx:
        rec = SDPARecorder()
        with rec:
            _ = mod(*args)
        with FlopCounterMode(display=False) as fc:
            _ = mod(*args)
        measured = fc.get_total_flops()

    hand_total = sum(hand.values())
    residual = hand_total - measured
    return {
        "module": "FullAttention (space)" if kind == "space" else "AxialTimeAttention (time)",
        "hidden_dim": hidden_dim,
        "n_spatial": n_spatial,
        "dtype": "bf16-autocast" if autocast else str(dtype).replace("torch.", ""),
        "hand_total": hand_total,
        "attention_term": hand["attention"],
        "measured": measured,
        "residual": residual,
        "residual_over_attn": (residual / hand["attention"]) if hand["attention"] else None,
        "verdict": classify(residual, hand["attention"]),
        "sdpa_ops": sorted(set(rec.ops)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force CPU (reproduces the notebook)")
    ap.add_argument("--quick", action="store_true",
                    help="smallest shape only; for an interactive sanity check, not the gate")
    ap.add_argument("--out", default=None, help="write JSON record here")
    a = ap.parse_args()

    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    print(f"torch {torch.__version__} | device={device}")
    if device == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}")
        print(f"TF32  matmul={torch.backends.cuda.matmul.allow_tf32} "
              f"cudnn={torch.backends.cudnn.allow_tf32}")
    reg = sorted(str(k) for k in flop_registry if "scaled_dot_product" in str(k))
    print("\nSDPA overloads with a registered FLOP formula in this torch build:")
    for r in reg:
        print(f"  {r}")
    print()

    dtype_cases = [(torch.float32, False)]
    if device == "cuda":
        dtype_cases.append((torch.float32, True))  # bf16 autocast, the enable_amp=True lever

    shapes = QUICK_SHAPES if a.quick else SHAPES
    if a.quick:
        print("!! --quick: smallest shape only. This is NOT the §5.1 gate.\n")

    records = []
    for label, hidden_dim, mlp_dim, n_spatial in shapes:
        for dtype, autocast in dtype_cases:
            for kind in ("space", "time"):
                try:
                    r = run_case(kind, hidden_dim, mlp_dim, n_spatial, device, dtype, autocast)
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    r = {"module": kind, "hidden_dim": hidden_dim, "n_spatial": n_spatial,
                         "dtype": "bf16-autocast" if autocast else "float32",
                         "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                r["shape_label"] = label
                records.append(r)

    hdr = (f"{'shape':30}{'module':26}{'dtype':16}{'measured':>16}"
           f"{'analytic':>16}{'resid/attn':>12}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        if r["verdict"] == "ERROR":
            print(f"{r['shape_label']:30}{r['module']:26}{r['dtype']:16}"
                  f"{'-':>16}{'-':>16}{'-':>12}  ERROR: {r['error']}")
            continue
        ratio = r["residual_over_attn"]
        print(f"{r['shape_label']:30}{r['module']:26}{r['dtype']:16}"
              f"{r['measured']:>16,}{r['hand_total']:>16,}"
              f"{(f'{ratio:+.3f}' if ratio is not None else '-'):>12}  {r['verdict']}")

    print("\nSDPA overloads actually dispatched:")
    for op in sorted({o for r in records for o in r.get("sdpa_ops", [])}):
        print(f"  {op}")

    verdicts = {r["verdict"] for r in records}
    tag = "PARTIAL (--quick)" if a.quick else "GATE"
    print("\n" + "=" * 78)
    if verdicts == {"COUNTED"}:
        print(f"{tag}: PASS - FlopCounterMode counts attention in every case tested.")
        print("      It may be used as the cross-check on the analytical formula (§5.1).")
        rc = 0
    elif "DROPPED" in verdicts:
        print(f"{tag}: FAIL - FlopCounterMode silently drops attention in at least one case.")
        print("      The analytical formula is the ONLY instrument; every FlopCounterMode")
        print("      number in this study must be hand-corrected. See the per-row verdicts:")
        for r in records:
            if r["verdict"] == "DROPPED":
                print(f"        DROPPED: {r['shape_label']} / {r['module']} / {r['dtype']}")
        rc = 1
    else:
        print(f"{tag}: INCONCLUSIVE - verdicts {sorted(verdicts)}. Inspect the raw numbers above;")
        print("      an AMBIGUOUS row usually means a valid-but-different FLOP convention in")
        print("      the registered kernel, not a dropped term.")
        rc = 2
    print("=" * 78)

    out = a.out or os.environ.get("FLOP_GATE_OUT")
    if out:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "registered_sdpa_overloads": reg,
            "records": records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
