# ruff: noqa: E501
"""Compute params / disk / projected GPU-h for the SCALING_STRATEGY.md 7.1 grid sweep
and for the WALRUS authors' own grid (ExampleRunScripts/gen_grid_search.py), so the two
can be compared against each other and against the 200M anchor.

Pure arithmetic -- no model, no data, no GPU, no torch. Two formulas, both taken from
code that is already validated elsewhere in the repo:

- Parameter count: `analytic_param_count` from
  demo_notebooks/walrus_example_2_ArchitectureAndDimensions.ipynb 11, which that
  notebook checks bit-for-bit against sum(p.numel()) across a 17-config sweep.
- FLOP proxy: `flop_proxy` from demo_notebooks/leonardo_job_cost_estimator.ipynb 3,
  the same proxy its cost model is regressed on.

GPU-h is a *projection*, not a measurement -- GATE A (5.2) has not reported, so both
regimes 10 prices are given: FLOP-proportional (scaled off fullDims's measured
~7.8 GPU-h/epoch) and the dataloader floor (every shape pinned at that same number).

Usage:
    python shape_cost_table.py            # prints markdown tables + writes the CSV
"""

import csv
import itertools
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(SCRIPT_DIR, "shape_cost_table.csv")

# --- 4.1 pinned substrate ----------------------------------------------------------
INTERMEDIATE_DIM = 352
N_STATES = 67           # pinned by data/field_index_map_override=full_well_field_index
NUM_HEADS = 16
# include_d = [3] under override_dimensionality=3 (2.2a fix, 2026-09-18): =0 was a
# no-op (the code only overrides when >0), so include_d silently defaulted to [2,3]
# and every run before this fix built a dead, never-used 3D encoder/decoder.
# The fix first used =2, which crashes: IsotropicModel.forward takes its dim_key from
# metadata.n_spatial_dims, already 3 for every dataset (pad_cartesian_data_to_d), so
# include_d=[2] leaves no '3' key to look up. Corrected to =3 on 2026-09-22. The key is
# a bucket label, not a shape -- both pairs are built with spatial_dims=3 -- so this is
# nominal only: verified bit-identical weights and output vs. a working =2, and
# N_DIM_KEYS (and every count below) is unchanged.
N_DIM_KEYS = 1
NUM_BUCKETS = 32        # T5 relative position bias
N_STEPS_INPUT = 6
BATCH_SIZE = 2

# --- cost-model constants ----------------------------------------------------------
BYTES_PER_PARAM = 12.4  # sharded AdamW checkpoint, 4.3 (ratio itself unaffected by
                         # the 2.2a fix -- bytes/param, not a fixed byte count)
# Measured pre-2.2a-fix (i.e. against the old, n_dim_keys=2 fullDims with the dead 3D
# branch still built). Since encoder+decoder is a small share of fullDims either way
# (14% before, 7% after), this is a slight overestimate now, not yet re-measured.
FULLDIMS_GPU_H_PER_EPOCH = 7.8   # measured, 5.2
RETAINED_CKPTS = 3      # 4.3's own planning assumption for a sweep run
SCRATCH_HEADROOM_GB = 704        # measured live; 4.3 quotes ~680

FULLDIMS = (1408, 40, 5632)
ANCHOR_200M = (768, 12, 2048)

# --- the two grids -----------------------------------------------------------------
OUR_GRID = {
    "hidden_dim": [320, 384, 464, 560],
    "processor_blocks": [6, 8, 10, 12],
    "mlp_dim": [768, 928, 1120, 1344],
}

# ExampleRunScripts/gen_grid_search.py. Its two-pass generation (27 combos with no
# "options_last" value, then the 37 with at least one) emits exactly the full 4x4x4.
AUTHORS_GRID = {
    "hidden_dim": [768, 928, 1088, 1360],
    "processor_blocks": [6, 8, 10, 12],
    "mlp_dim": [416, 512, 608, 768],
}


def param_count(hidden_dim, processor_blocks, mlp_dim):
    """Returns (encoder, processor, decoder, total)."""
    H, L, M, I, nh = hidden_dim, processor_blocks, mlp_dim, INTERMEDIATE_DIM, NUM_HEADS
    space = H + H * (M + 3 * H) + (M + 3 * H) + (M // 2) * H + H + H * H + 4 * (H // nh)
    time_ = H + 3 * H * H + 3 * H + H * H + H + 4 * (H // nh) + NUM_BUCKETS * nh
    processor = (space + time_) * L
    encoder = (I * N_STATES * 512 + I + H * I * 64 + H) * N_DIM_KEYS
    decoder = (H * I * 64 + I + I * N_STATES * 512 + N_STATES) * N_DIM_KEYS
    return encoder, processor, decoder, encoder + processor + decoder


def flop_proxy(hidden_dim, processor_blocks, mlp_dim, n_tokens=64):
    """Relative forward cost; absolute scale cancels in every ratio below.

    Unlike `param_count`, this proxy's `encdec` term does NOT scale with `N_DIM_KEYS` --
    it's a simplified proxy calibrated against historical measured step times (all of
    which ran the pre-2.2a-fix, n_dim_keys=2 config), not the exact closed form in
    walrus/utils/flops.py. So GPU-h/epoch figures below are a slight overestimate for
    the corrected (n_dim_keys=1) runs -- conservative, not dangerously wrong, since
    encoder+decoder was already a minority of FLOP cost at most shapes -- but not
    re-calibrated against any post-fix measurement either. Treat as provisional like
    every other GPU-h figure in this study pending GATE A (SCALING_STRATEGY.md 5.2).
    """
    H, T, N, I = float(hidden_dim), float(N_STEPS_INPUT), float(n_tokens), float(INTERMEDIATE_DIM)
    space = 4.0 * N * T * H**2 + 2.0 * N**2 * T * H
    time_ = 4.0 * N * T * H**2 + 2.0 * T**2 * N * H
    mlp_f = 2.0 * N * T * H * float(mlp_dim)
    encdec = 2.0 * (2.0 * N * T * (I * H + I * 64.0))
    return float(processor_blocks) * (space + time_ + mlp_f) + encdec


def describe(shape):
    H, L, M = shape
    enc, proc, dec, total = param_count(H, L, M)
    proxy = flop_proxy(H, L, M)
    return {
        "hidden_dim": H, "processor_blocks": L, "mlp_dim": M,
        "params_total_M": total / 1e6,
        "params_processor_M": proc / 1e6,
        "encdec_share_pct": (enc + dec) / total * 100,
        "ckpt_GB": total * BYTES_PER_PARAM / 1e9,
        "gpu_h_per_epoch_flopprop": FULLDIMS_GPU_H_PER_EPOCH * proxy / flop_proxy(*FULLDIMS),
        "gpu_h_per_epoch_floor": FULLDIMS_GPU_H_PER_EPOCH,
        "vs_anchor_params": total / param_count(*ANCHOR_200M)[3],
        "vs_anchor_flops": proxy / flop_proxy(*ANCHOR_200M),
    }


def grid_points(grid):
    return [
        (h, l, m) for h, l, m in itertools.product(
            grid["hidden_dim"], grid["processor_blocks"], grid["mlp_dim"])
    ]


def summarise(name, grid, epochs):
    pts = [describe(p) for p in grid_points(grid)]
    tot_flopprop = sum(p["gpu_h_per_epoch_flopprop"] for p in pts) * epochs
    tot_floor = sum(p["gpu_h_per_epoch_floor"] for p in pts) * epochs
    disk = sum(p["ckpt_GB"] for p in pts) * RETAINED_CKPTS
    return {
        "name": name, "n_points": len(pts), "epochs": epochs,
        "params_min_M": min(p["params_total_M"] for p in pts),
        "params_max_M": max(p["params_total_M"] for p in pts),
        "gpu_h_flopprop": tot_flopprop, "gpu_h_floor": tot_floor,
        "disk_GB": disk,
        "vs_anchor_params_min": min(p["vs_anchor_params"] for p in pts),
        "vs_anchor_params_max": max(p["vs_anchor_params"] for p in pts),
        "points": pts,
    }


def md_row(p, epochs):
    return (f"| {p['hidden_dim']}/{p['processor_blocks']}/{p['mlp_dim']} "
            f"| {p['params_total_M']:.0f} M | {p['params_processor_M']:.0f} M "
            f"| {p['encdec_share_pct']:.0f} % | {p['ckpt_GB']:.1f} GB "
            f"| {p['gpu_h_per_epoch_flopprop']:.2f} | {p['gpu_h_per_epoch_flopprop'] * epochs:.0f} "
            f"| {p['vs_anchor_params']:.2f}x | {p['vs_anchor_flops']:.2f}x |")


HEADER = ("| shape `H/L/M` | total params | processor params | enc+dec share | ckpt size "
          "| GPU-h/epoch (FLOP-prop.) | GPU-h @ {e} ep | vs anchor (params) | vs anchor (FLOPs) |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def corners_and_extremes(grid, epochs):
    """The 8 corners plus the mid-ish point, which is all the doc needs inline."""
    H, L, M = grid["hidden_dim"], grid["processor_blocks"], grid["mlp_dim"]
    picks = list(itertools.product([H[0], H[-1]], [L[0], L[-1]], [M[0], M[-1]]))
    picks.append((H[1], L[1], M[1]))
    return [describe(p) for p in picks]


def main():
    anchor = describe(ANCHOR_200M)
    full = describe(FULLDIMS)
    print(f"200M anchor  {ANCHOR_200M}: {anchor['params_total_M']:.1f} M params, "
          f"{anchor['ckpt_GB']:.1f} GB ckpt, {anchor['gpu_h_per_epoch_flopprop']:.2f} GPU-h/epoch")
    print(f"fullDims     {FULLDIMS}: {full['params_total_M']:.1f} M params, "
          f"{full['ckpt_GB']:.1f} GB ckpt, {full['gpu_h_per_epoch_flopprop']:.2f} GPU-h/epoch\n")

    ours = summarise("our grid (7.1)", OUR_GRID, epochs=40)
    theirs = summarise("authors' grid (gen_grid_search.py)", AUTHORS_GRID, epochs=40)

    for s in (ours, theirs):
        print(f"{s['name']}: {s['n_points']} points, "
              f"{s['params_min_M']:.0f}-{s['params_max_M']:.0f} M params "
              f"({s['vs_anchor_params_min']:.2f}x-{s['vs_anchor_params_max']:.2f}x the anchor)")
        print(f"   GPU-h @ {s['epochs']} ep: {s['gpu_h_flopprop']:.0f} (FLOP-prop.) / "
              f"{s['gpu_h_floor']:.0f} (floor)")
        print(f"   disk at {RETAINED_CKPTS} retained ckpts/run: {s['disk_GB']:.0f} GB "
              f"({s['disk_GB'] / SCRATCH_HEADROOM_GB * 100:.0f} % of {SCRATCH_HEADROOM_GB} GB headroom)\n")

    for s, grid in ((ours, OUR_GRID), (theirs, AUTHORS_GRID)):
        print(f"\n### {s['name']} -- corners\n")
        print(HEADER.format(e=s["epochs"]))
        for p in corners_and_extremes(grid, s["epochs"]):
            print(md_row(p, s["epochs"]))

    fields = list(describe(ANCHOR_200M).keys())
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grid"] + fields)
        w.writeheader()
        for label, grid in (("ours", OUR_GRID), ("authors", AUTHORS_GRID)):
            for p in grid_points(grid):
                w.writerow({"grid": label, **describe(p)})
        w.writerow({"grid": "reference_200M_anchor", **anchor})
        w.writerow({"grid": "reference_fullDims", **full})
    print(f"\nWrote {CSV_OUT}")


if __name__ == "__main__":
    main()
