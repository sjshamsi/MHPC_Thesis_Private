"""
Generate the SCALING_STRATEGY_2D.md §5.4 noise-floor runs on the PT9 mixture.

Three runs of the 200M-anchor shape (768/12/2048) on the §4.1 substrate
(pt9_substrate.py), identical apart from their names. This branch has no `cfg.seed`, so
each run initialises and samples from OS-entropy randomness. The spread of their
validation metrics is epsilon_noise, the run-to-run noise every shape comparison in the
study is judged against (Choshen §4).

All three run with the same max_epoch=100, and all three are read at the same epoch E.
The doc's earlier sequencing (run 0 to 100, fix max_epoch = E, then run runs 1-2
capped at E) would compare run 0's un-cooled epoch E with runs 1-2's cooled epoch E.
With cooldown_epochs=10 the LR schedule differs from epoch E-10 onward, so the "spread"
would include a schedule difference. inv_sqrt_w_sqrt_ramps is length-independent until
the cooldown begins at max_epoch - 10 = 90, so for any E <= 90 an uncapped run's epoch-E
checkpoint and metrics are exactly what they would have been with a longer run. Cancel
runs 1-2 once they pass E; keep run 0 running as the saturation pilot (§7.0).

Seed 0's curve doubles as §7.0's small-model saturation pilot, which is what E is chosen
from (together with the cooldown-fraction and repetition bounds).

checkpoint_frequency=10 (not the default 20): §5.4 also finetunes each run's epoch-E
checkpoint on the four held-out targets to get epsilon_noise for f_down, and E is not
known yet. Every 10 epochs gives a usable checkpoint for any E that is a multiple of 10.
At ~1.8 GB per checkpoint this is ~18 GB per run.

Wall clock. 100 epochs will not fit in one 24 h job: there is no 200M-anchor timing yet,
but fullDims took 1.92 h per training epoch at N1 on smallest7, and the anchor is ~10x
cheaper per microbatch if compute-bound, so ~0.2-0.5 h/epoch depending on the data floor,
i.e. 20-50 h. The submit wrapper therefore queues each script CHAIN times with
`--dependency=singleton`: SLURM runs at most one job per name at a time, so the links run
back to back and each auto-resumes from the last checkpoint. A link that starts after the
run has already reached max_epoch resumes, skips the epoch loop (training.py:1317) and
exits after a capped test pass (~10 min of one node). To stop a run early (runs 1-2
once past E), cancel the whole chain by name: `scancel -n <job_name>`.

End-of-run validation is capped like every other (++trainer.debug_mode=True,
pt9_substrate.py), so all three runs' metrics at E are the same logged quantity.

Submission (Euler download complete; validate_flops_pt9.py passed 2026-10-02):
    python gen_pt9_noise_floor_runs.py
    bash run_p9_noise_floor_all.sh    # all three runs, each as a CHAIN-link chain
                                      # (../preflight_pt9.py runs first)
    # or, budget-conservative: run run 0's chain alone first (copy its lines from the
    # wrapper), the others later; identical results, because the schedules are identical.
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
GENERATOR = "PT9_2d/NoiseFloor/gen_pt9_noise_floor_runs.py"
WANDB_PROJECT = f"{PREFIX}_noise_floor"

HIDDEN_DIM, DEPTH, MLP, GC_FREQ = 768, 12, 2048, 0
EPOCHS = 100  # ceiling, not the settled max_epoch; see the docstring
RUNS = 3
TIME_CAP = "24:00:00"
DATA_WORKERS = 10
CHAIN = 3  # 24 h links per run; surplus links exit within minutes (see the docstring)


def main():
    lines = []
    for i in range(RUNS):
        job_name = f"{PREFIX}_noise_floor_H{HIDDEN_DIM}_L{DEPTH}_M{MLP}_E{EPOCHS}_run{i}"
        overrides = (
            SUBSTRATE
            + shape_overrides(HIDDEN_DIM, DEPTH, MLP, GC_FREQ)
            + [
                f"name={job_name}",
                f'logger.wandb_project_name="{WANDB_PROJECT}"',
                "trainer.log_interval=200",
                f"trainer.max_epoch={EPOCHS}",
                f"data_workers={DATA_WORKERS}",
                "checkpoint.checkpoint_frequency=10",
                "auto_resume=$AUTO_RESUME",
            ]
        )
        cfg = check_composes(overrides)
        assert sorted(cfg.data.module_parameters.well_dataset_info.keys()) == sorted(PRETRAIN_DATASETS)
        assert (cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (EPOCHS, 10)
        assert cfg.trainer.debug_mode
        body = render_script(
            job_name=job_name,
            time=TIME_CAP,
            description=(
                f"# SCALING_STRATEGY_2D.md 5.4 (noise floor): shape {HIDDEN_DIM}/{DEPTH}/{MLP},\n"
                f"# max_epoch={EPOCHS}, run {i}, on the PT9 mixture. Identical to the other two\n"
                f"# p9_noise_floor_run_*.sh except for the name (no seed on this branch); read all three at the same epoch."
            ),
            generator=GENERATOR,
            overrides=overrides,
        )
        filename = f"{PREFIX}_noise_floor_run_{i}.sh"
        write_script(SCRIPT_DIR, filename, job_name, body)
        lines += [f"sbatch --dependency=singleton {filename}"] * CHAIN

    write_run_all(SCRIPT_DIR, f"run_{PREFIX}_noise_floor_all.sh", lines)
    print(
        f"Generated {RUNS} noise-floor scripts in {SCRIPT_DIR} (all compose; "
        f"{CHAIN}-link chains); "
        f"logs under {EXPERIMENT_DIR}/slurm/"
    )


if __name__ == "__main__":
    main()
