"""
Generate the SCALING_STRATEGY_2D.md §5.4 noise-floor runs on the PT9 mixture.

Three runs of the 200M-anchor shape (768/12/2048) on the §4.1 substrate
(pt9_substrate.py), identical except for `seed` (0, 1, 2). The spread of their
validation metrics is epsilon_noise, the run-to-run noise every shape comparison in the
study is judged against (Choshen §4).

All three run with the same max_epoch=100, and all three are read at the same epoch E.
The doc's earlier sequencing (run seed 0 to 100, fix max_epoch = E, then run seeds 1-2
capped at E) would compare seed 0's un-cooled epoch E with seeds 1-2's cooled epoch E.
With cooldown_epochs=10 the LR schedule differs from epoch E-10 onward, so the "spread"
would include a schedule difference. inv_sqrt_w_sqrt_ramps is length-independent until
the cooldown begins at max_epoch - 10 = 90, so for any E <= 90 an uncapped run's epoch-E
checkpoint and metrics are exactly what they would have been with a longer run. Cancel
seeds 1-2 once they pass E; keep seed 0 running as the saturation pilot (§7.0).

Seed 0's curve doubles as §7.0's small-model saturation pilot, which is what E is chosen
from (together with the cooldown-fraction and repetition bounds).

checkpoint_frequency=10 (not the default 20): §5.4 also finetunes each seed's epoch-E
checkpoint on the four held-out targets to get epsilon_noise for f_down, and E is not
known yet. Every 10 epochs gives a usable checkpoint for any E that is a multiple of 10.
At ~1.8 GB per checkpoint this is ~18 GB per run.

Submission:
    python gen_pt9_noise_floor_runs.py
    bash run_p9_noise_floor_all.sh    # all three at once (preflight_pt9.py runs first)
    # or, budget-conservative: sbatch p9_noise_floor_run_0.sh alone first, the others later;
    # both give identical results because the three schedules are identical.
100 epochs exceeds the 24 h wall-clock cap: resubmit each script with
`sbatch --dependency=singleton <script>` and auto_resume continues the run.
"""

import os

from pt9_substrate import (
    EXPERIMENT_DIR,
    PREFIX,
    SUBSTRATE,
    render_script,
    shape_overrides,
    write_run_all,
    write_script,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR = "PT9_2d/gen_pt9_noise_floor_runs.py"
WANDB_PROJECT = f"{PREFIX}_noise_floor"

HIDDEN_DIM, DEPTH, MLP, GC_FREQ = 768, 12, 2048, 0
EPOCHS = 100  # ceiling, not the settled max_epoch; see the docstring
SEEDS = [0, 1, 2]
TIME_CAP = "24:00:00"
DATA_WORKERS = 10


def main():
    lines = []
    for i, seed in enumerate(SEEDS):
        job_name = f"{PREFIX}_noise_floor_H{HIDDEN_DIM}_L{DEPTH}_M{MLP}_E{EPOCHS}_seed{seed}"
        overrides = (
            SUBSTRATE
            + shape_overrides(HIDDEN_DIM, DEPTH, MLP, GC_FREQ)
            + [
                f"name={job_name}",
                f'logger.wandb_project_name="{WANDB_PROJECT}"',
                "trainer.log_interval=200",
                f"seed={seed}",
                f"trainer.max_epoch={EPOCHS}",
                f"data_workers={DATA_WORKERS}",
                "checkpoint.checkpoint_frequency=10",
                "auto_resume=$AUTO_RESUME",
            ]
        )
        body = render_script(
            job_name=job_name,
            time=TIME_CAP,
            description=(
                f"# SCALING_STRATEGY_2D.md 5.4 (noise floor): shape {HIDDEN_DIM}/{DEPTH}/{MLP},\n"
                f"# max_epoch={EPOCHS}, seed={seed}, on the PT9 mixture. Identical to the other two\n"
                f"# p9_noise_floor_run_*.sh except for `seed=`; read all three at the same epoch."
            ),
            generator=GENERATOR,
            overrides=overrides,
        )
        filename = f"{PREFIX}_noise_floor_run_{i}.sh"
        write_script(SCRIPT_DIR, filename, job_name, body)
        lines.append(f"sbatch {filename}")

    write_run_all(SCRIPT_DIR, f"run_{PREFIX}_noise_floor_all.sh", lines)
    print(
        f"Generated {len(SEEDS)} noise-floor scripts in {SCRIPT_DIR}; "
        f"logs under {EXPERIMENT_DIR}/slurm/"
    )


if __name__ == "__main__":
    main()
