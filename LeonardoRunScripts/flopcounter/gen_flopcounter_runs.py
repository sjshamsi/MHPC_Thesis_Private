"""
Generate three tiny debug-queue runs that test the FLOP counter on GPU.

They build ONE training run (one experiment folder, one W&B run) across three separate SLURM
jobs, so the test covers what the PT9 noise-floor chains rely on:

    fc_run_1.sh   auto_resume=False  max_epoch=2   epochs 1-2  (always a fresh run_idx / W&B id)
    fc_run_2.sh   auto_resume=True   max_epoch=4   epochs 3-4  (resumes run 1)
    fc_run_3.sh   auto_resume=True   max_epoch=6   epochs 5-6  (resumes run 2)

The three scripts differ only in those two keys. Submit them yourself, one after the other,
each once the previous has finished (boost_qos_dbg allows 2 submitted jobs per user):

    sbatch fc_run_1.sh      # wait until it finishes
    sbatch fc_run_2.sh      # wait
    sbatch fc_run_3.sh      # wait, then on the login node:
    python check_flopcounter.py

Setup: the PT9 substrate (pt9_substrate.SUBSTRATE: fsdp, Leonardo_pretrain9_2d, causal,
n_steps_input=6, batch_size=2, ...) on 1 node x 4 GPUs, with a tiny model (256/4/512) and
gradient checkpointing every 2nd block, so the recompute term is non-zero.
max_samples=40 keeps each epoch to 40 micro-batches per rank. A checkpoint is written every
epoch, so every step_N/metadata.pt can be checked. Validation is pushed out
(val_frequency=1000); each job still runs the capped end-of-job validation and test.

Expected results: python ../../scripts/count_WALRUS_flops.py fc_run_3.sh --exact
Checker: check_flopcounter.py (W&B history, checkpoints and estimator must agree).
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "PT9_2d"))
from pt9_substrate import (  # noqa: E402
    EXPERIMENT_DIR,
    PRETRAIN_DATASETS,
    SUBSTRATE,
    check_composes,
    render_script,
    shape_overrides,
    write_script,
)

GENERATOR = "flopcounter/gen_flopcounter_runs.py"
WANDB_PROJECT = "flopcounter_test"
HIDDEN_DIM, DEPTH, MLP, GC_FREQ = 256, 4, 512, 2
RUN_NAME = f"fc_H{HIDDEN_DIM}_L{DEPTH}_M{MLP}_gc{GC_FREQ}"
MAX_SAMPLES = 40
DATA_WORKERS = 8
TIME_CAP = "00:30:00"  # boost_qos_dbg maximum
# (script number, auto_resume, max_epoch)
RUNS = [(1, False, 2), (2, True, 4), (3, True, 6)]


def overrides_for(auto_resume: bool, max_epoch: int) -> list:
    return (
        SUBSTRATE
        + shape_overrides(HIDDEN_DIM, DEPTH, MLP, GC_FREQ)
        + [
            f"name={RUN_NAME}",
            f'logger.wandb_project_name="{WANDB_PROJECT}"',
            f"data.module_parameters.max_samples={MAX_SAMPLES}",
            "checkpoint.checkpoint_frequency=1",
            "trainer.val_frequency=1000",
            "trainer.rollout_val_frequency=1000",
            f"data_workers={DATA_WORKERS}",
            f"trainer.max_epoch={max_epoch}",
            f"auto_resume={auto_resume}",
        ]
    )


def main():
    for number, auto_resume, max_epoch in RUNS:
        job_name = f"fc_run_{number}"
        overrides = overrides_for(auto_resume, max_epoch)
        cfg = check_composes(overrides)
        assert sorted(cfg.data.module_parameters.well_dataset_info) == sorted(PRETRAIN_DATASETS)
        assert cfg.distribution.distribution_type == "fsdp"
        assert (cfg.model.hidden_dim, cfg.model.processor_blocks) == (HIDDEN_DIM, DEPTH)
        assert cfg.model.gradient_checkpointing_freq == GC_FREQ
        assert (cfg.trainer.max_epoch, cfg.auto_resume) == (max_epoch, auto_resume)
        assert cfg.checkpoint.checkpoint_frequency == 1
        first = max_epoch - 1
        body = render_script(
            job_name=job_name,
            time=TIME_CAP,
            description=(
                f"# FLOP-counter GPU test, job {number} of 3: epochs {first}-{max_epoch} of one run\n"
                f"# ({RUN_NAME}, W&B project {WANDB_PROJECT}). Submit fc_run_1/2/3.sh one after\n"
                f"# the other, each after the previous finished; then run check_flopcounter.py."
            ),
            generator=GENERATOR,
            overrides=overrides,
        )
        # pt9_substrate's header is for the normal queue; this test belongs in the debug queue.
        assert body.count("#SBATCH --qos=normal\n") == 1
        body = body.replace("#SBATCH --qos=normal\n", "#SBATCH --qos=boost_qos_dbg\n")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    print(f"Generated fc_run_1..3.sh in {SCRIPT_DIR} (all compose); logs under {EXPERIMENT_DIR}/slurm/")


if __name__ == "__main__":
    main()
