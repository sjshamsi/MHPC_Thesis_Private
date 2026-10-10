"""
Generate the SCALING_STRATEGY_2D.md §5.2 roofline runs (GATE A) on the PT9 mixture.

GATE A asks whether cost is FLOP-proportional or dataloader-bound as the model shrinks.
If small models are dataloader-bound, the grid sweep gets no cheaper than fullDims per
epoch, and the plan's contingency applies. Every point runs the §4.1 substrate
(pt9_substrate.py) for 3 epochs at N1; only shape and data_workers vary, so each point
is directly comparable to every later tier's runs.

Main family (fit sec/epoch = a + b*FLOPs across it):
    p9_roofline_run_0.sh   fullDims      1408/40/5632  dw10
    p9_roofline_run_1.sh   200M anchor    768/12/2048  dw10
    p9_roofline_run_2.sh   small          464/10/1120  dw10
    p9_roofline_run_3.sh   small                       dw8
    p9_roofline_run_4.sh   small                       dw16

Paired staging test (replaces the old roofline_staged.sh, which staged the whole mixture;
that is infeasible here, as the PT9 train splits alone are ~5.8 TB with Euler):
    p9_roofline_pair_unstaged.sh   small, gray_scott + planetswe only, read from $WORK
    p9_roofline_pair_staged.sh     identical, but both datasets first copied to
                                   node-local /scratch_local

The pair is a cheap-encoder-tier dataset (gray_scott, 128^2) plus a >=512-tier one
(planetswe, 256x512), 330 GB in total. The staging effect is the *difference* between the
two arms; neither arm's sec/epoch is comparable with the main family's, since each
samples a 2-dataset mixture. The staged arm checks free space on /scratch_local first
and aborts cleanly if the pair does not fit. Node-local capacity on booster nodes is not
advertised by SLURM (TmpDisk=0), so that check is the only guard.

Report per-dataset data-wait time from these runs, not just aggregate sec/epoch: Euler's
~37 MB windows are drawn 2/9 of the time from ~212 GB files, and a per-dataset I/O floor
would be invisible in the aggregate (SCALING_STRATEGY_2D.md §5.2). W&B only receives the
aggregate `avg_data_loading_time`; the per-dataset split exists only in the stdout lines
the trainer prints every `log_interval` microbatches (dataset name + data/fwd/bw seconds).
Hence log_interval=10, the upstream default: a 1-in-10 sample, ~2 400 lines per run, and
unbiased, since the dataset drawn does not depend on the batch index.
parse_roofline_logs.py turns those logs into the GATE A table.

Timing excludes validation: with ++trainer.debug_mode=True (pt9_substrate.py) every
validation is capped, and the parser measures the train phase of each epoch only.

Submission (Euler download complete; validate_flops_pt9.py passed 2026-10-02):
    python gen_pt9_roofline_runs.py
    bash run_p9_roofline_all.sh      # main five; runs ../preflight_pt9.py first
    bash run_p9_roofline_pair.sh     # the pair, together, and NOT in the same burst as
                                     # the main five (shared-filesystem contention would
                                     # bias the unstaged arm)
Watch the first job's log for the per-microbatch lines before leaving the rest
unattended; then, once all have finished:
    python parse_roofline_logs.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pt9_substrate import (  # noqa: E402
    EXPERIMENT_DIR,
    PREFIX,
    PRETRAIN_DATASETS,
    SUBSTRATE,
    check_composes,
    render_script,
    shape_overrides,
    write_run_all,
    write_script,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR = "PT9_2d/Roofline/gen_pt9_roofline_runs.py"
WANDB_PROJECT = f"{PREFIX}_roofline"

EPOCHS = 3
BASE_DATA_WORKERS = 10
SMALL_SHAPE_DW_VARIANTS = [8, 16]

# Time caps are generous relative to the smallest7-era estimate (~1.95 h/epoch at
# fullDims), because Euler's reads may lift the per-epoch cost; auto-resume covers any
# overflow, but a cut-off costs a queue cycle.
SHAPES = [
    {"tag": "fullDims", "hidden_dim": 1408, "depth": 40, "mlp": 5632, "gc_freq": 2, "time": "10:00:00"},
    {"tag": "200Manchor", "hidden_dim": 768, "depth": 12, "mlp": 2048, "gc_freq": 0, "time": "05:00:00"},
    {"tag": "small", "hidden_dim": 464, "depth": 10, "mlp": 1120, "gc_freq": 0, "time": "05:00:00"},
]
SMALL = SHAPES[2]

PAIR_DATASETS = ["gray_scott_reaction_diffusion", "planetswe"]
DATASET_ROOT = "/leonardo_work/ICT26_MHPC_0/sshamsi/datasets"


def run_overrides(job_name, shape, data_workers):
    return (
        SUBSTRATE
        + shape_overrides(shape["hidden_dim"], shape["depth"], shape["mlp"], shape["gc_freq"])
        + [
            f"name={job_name}",
            f'logger.wandb_project_name="{WANDB_PROJECT}"',
            "trainer.log_interval=10",  # per-dataset timing sample; see the docstring
            f"trainer.max_epoch={EPOCHS}",
            f"data_workers={data_workers}",
            "auto_resume=$AUTO_RESUME",
        ]
    )


def shape_tag(shape, data_workers):
    return (
        f"H{shape['hidden_dim']}_L{shape['depth']}_M{shape['mlp']}_E{EPOCHS}_dw{data_workers}"
    )


# Drop the other seven pretraining datasets with Hydra's `~key` deletion, so the pair runs
# the exact PT9 config minus seven entries rather than a separately maintained file.
# Single-quoted so bash never attempts tilde expansion on them.
PAIR_DELETIONS = [
    f"'~data.module_parameters.well_dataset_info.{d}'"
    for d in PRETRAIN_DATASETS
    if d not in PAIR_DATASETS
]

STAGING_BLOCK = """
SRC={dataset_root}
STAGE_DIR=/scratch_local/${{SLURM_JOB_ID}}/datasets
DATASETS=({datasets})
# Remove the staged copy however the job ends, including on failure under set -e.
trap 'rm -rf "/scratch_local/${{SLURM_JOB_ID}}"' EXIT

# Abort cleanly, rather than filling the node disk, if the pair does not fit.
NEEDED=$(du -sb "${{DATASETS[@]/#/$SRC/}}" | awk '{{s += $1}} END {{print s}}')
mkdir -p "$STAGE_DIR"
AVAILABLE=$(df -B1 --output=avail "$STAGE_DIR" | tail -1)
echo "Staging needs $((NEEDED / 1024**3)) GiB; /scratch_local has $((AVAILABLE / 1024**3)) GiB free"
if (( NEEDED * 11 / 10 > AVAILABLE )); then
    echo "Not enough node-local space for the paired staging test; aborting." >&2
    exit 1
fi

STAGE_START=$(date +%s)
pids=()
for d in "${{DATASETS[@]}}"; do
    cp -r "$SRC/$d" "$STAGE_DIR/$d" &
    pids+=($!)
done
# Wait on each copy by PID: a bare `wait` returns 0 even if a copy failed, which would
# let training run on a half-staged dataset.
for p in "${{pids[@]}}"; do
    wait "$p"
done
STAGE_END=$(date +%s)
# Exclude this from the sec/epoch comparison: it is paid once, not per epoch.
echo "STAGING_SECONDS: $((STAGE_END - STAGE_START))"
du -sh "$STAGE_DIR"
"""

def verify(overrides, datasets, shape):
    """Compose the script's overrides and check the things that differ between scripts."""
    cfg = check_composes(overrides)
    resolved = sorted(cfg.data.module_parameters.well_dataset_info.keys())
    assert resolved == sorted(datasets), resolved
    m = cfg.model
    assert (m.hidden_dim, m.processor_blocks, m.processor.space_mixing.mlp_dim) == (
        shape["hidden_dim"], shape["depth"], shape["mlp"]
    )
    assert cfg.trainer.debug_mode and cfg.trainer.max_epoch == EPOCHS


def main():
    main_lines, pair_lines = [], []

    points = [(s, BASE_DATA_WORKERS) for s in SHAPES] + [
        (SMALL, dw) for dw in SMALL_SHAPE_DW_VARIANTS
    ]
    for i, (shape, dw) in enumerate(points):
        job_name = f"{PREFIX}_roofline_{shape['tag']}_{shape_tag(shape, dw)}"
        overrides = run_overrides(job_name, shape, dw)
        verify(overrides, PRETRAIN_DATASETS, shape)
        body = render_script(
            job_name=job_name,
            time=shape["time"],
            description=(
                f"# SCALING_STRATEGY_2D.md 5.2 (GATE A): shape "
                f"{shape['hidden_dim']}/{shape['depth']}/{shape['mlp']}, {EPOCHS} epochs,\n"
                f"# data_workers={dw}, on the PT9 mixture. One point of the sec/epoch = a + b*FLOPs fit."
            ),
            generator=GENERATOR,
            overrides=overrides,
        )
        filename = f"{PREFIX}_roofline_run_{i}.sh"
        write_script(SCRIPT_DIR, filename, job_name, body)
        main_lines.append(f"sbatch {filename}")

    for arm in ("unstaged", "staged"):
        job_name = f"{PREFIX}_roofline_pairGSPS_{arm}_{shape_tag(SMALL, BASE_DATA_WORKERS)}"
        overrides = run_overrides(job_name, SMALL, BASE_DATA_WORKERS) + PAIR_DELETIONS
        pre = ""
        if arm == "staged":
            overrides.append('data.well_base_path="$STAGE_DIR/"')
            pre = STAGING_BLOCK.format(
                dataset_root=DATASET_ROOT, datasets=" ".join(PAIR_DATASETS)
            )
        verify(overrides, PAIR_DATASETS, SMALL)
        body = render_script(
            job_name=job_name,
            time="06:00:00",
            description=(
                f"# SCALING_STRATEGY_2D.md 5.2, paired staging test, {arm} arm: roofline-small shape\n"
                f"# on gray_scott + planetswe only. Compare only with the other arm of the pair."
            ),
            generator=GENERATOR,
            overrides=overrides,
            pre_train=pre,
        )
        filename = f"{PREFIX}_roofline_pair_{arm}.sh"
        write_script(SCRIPT_DIR, filename, job_name, body)
        pair_lines.append(f"sbatch {filename}")

    write_run_all(SCRIPT_DIR, f"run_{PREFIX}_roofline_all.sh", main_lines)
    write_run_all(SCRIPT_DIR, f"run_{PREFIX}_roofline_pair.sh", pair_lines, PAIR_DATASETS)
    print(
        f"Generated {len(main_lines)} main roofline scripts + 2 paired staging scripts "
        f"in {SCRIPT_DIR} (all compose); logs under {EXPERIMENT_DIR}/slurm/"
    )


if __name__ == "__main__":
    main()
