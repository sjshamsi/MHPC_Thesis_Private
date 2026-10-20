"""
GPU stop/continue check on the flopsonly branch, with a per-sample draw log.

Tiny runs on the PT9 substrate (320/6/768, max_samples=100, 2 epochs, 1 node x 4 GPUs, fsdp), sized
for the debug QOS (boost_qos_dbg: 30 min, 2 running jobs per user):
  A1, A2  train epochs 1-2 straight.
  B1, B2  train epochs 1-2, set the epoch-2 checkpoint aside (step_2_first), point `last` back at
          step_1 and launch again: the second launch auto-resumes and redoes epoch 2, as a chain
          link does after a job is killed past epoch 1.

flopsonly has no `seed`: every launch starts from OS-entropy randomness. So A and B start from
different weights, and B's resume draws different data from its first attempt. compare.py reports
the distances; flopsonly_repetition.ipynb interprets them and the draw logs.

Every run exports WALRUS_DRAW_LOG=<run>/draws, so each DataLoader worker writes one CSV row per
training sample (walrus/data/inflated_dataset.py::_log_draws). TORCH_DISTRIBUTED_DEBUG stays off: it
breaks the FSDP checkpoint load on resume. A watchdog dumps every process's Python stack if the log
goes quiet (the dataloader fork hang, mydocs/dataloader_fork_hang/README.md on flopcount).

    python gen_pt9_determinism_runs.py
    for r in A1 B1 A2 B2; do sbatch p9_flopsonly_$r.sh; done
    python compare.py   # on the login node, after all four finish
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pt9_substrate import (  # noqa: E402
    EXPERIMENT_DIR,
    PREFIX,
    SUBSTRATE,
    TORCHRUN,
    check_composes,
    render_script,
    shape_overrides,
    write_script,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR = "PT9_2d/Determinism/gen_pt9_determinism_runs.py"
RUN_ROOT = f"{EXPERIMENT_DIR}/determinism_flopsonly"  # compare.py and the notebook read <RUN_ROOT>/<run>
RUNS = ["A1", "A2", "B1", "B2"]
WATCHDOG_IDLE = 300  # s without log output before the stack dump; above start-up and validation


def pre_train(run, job_name):
    log = f"{EXPERIMENT_DIR}/slurm/{job_name}/$SLURM_JOB_ID.out"
    return f'''
export WALRUS_DRAW_LOG={RUN_ROOT}/{run}/draws
export PYTHONFAULTHANDLER=1
mkdir -p {RUN_ROOT}/{run}
git -C {os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))} rev-parse --abbrev-ref HEAD | grep -qx flopsonly || \\
    {{ echo "main tree is not on flopsonly" >&2; exit 1; }}
# Watchdog: if the log stops growing for {WATCHDOG_IDLE} s, SIGABRT the training workers (not
# torchrun) so faulthandler prints every thread's Python stack to .err.
(
    last=-1; idle=0
    while sleep 30; do
        size=$(stat -c %s "{log}" 2>/dev/null || echo 0)
        if [ "$size" = "$last" ]; then idle=$((idle + 30)); else idle=0; last=$size; fi
        if [ "$idle" -ge {WATCHDOG_IDLE} ]; then
            echo "WATCHDOG: no log output for ${{idle}} s, dumping stacks (SIGABRT)" >&2
            for pid in $(pgrep -u "$USER" -f train.py); do
                grep -qa torchrun /proc/$pid/cmdline || kill -ABRT "$pid" 2>/dev/null || true
            done
            idle=0
        fi
    done
) &
WATCHDOG_PID=$!
'''


def overrides(run):
    return SUBSTRATE + shape_overrides(320, 6, 768, 0) + [
        f"name={PREFIX}_flopsonly_{run}",
        f'logger.wandb_project_name="{PREFIX}_determinism"',
        "trainer.max_epoch=2",
        "data.module_parameters.max_samples=100",
        "trainer.max_rollout_steps=2",        # rollout val/test: ~16 min each at 60 steps
        "trainer.short_validation_length=2",  # validation never touches the weights
        "data_workers=10",
        "checkpoint.checkpoint_frequency=1",
        f"folder_override={RUN_ROOT}/{run}",
        "auto_resume=True",
    ]


def main():
    for run in RUNS:
        ovr = overrides(run)
        cfg = check_composes(ovr)
        assert "seed" not in cfg, "flopsonly has no seed key"
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.data.module_parameters.max_samples) == (320, 2, 100)
        job_name = f"{PREFIX}_flopsonly_{run}"
        post = ""
        if run.startswith("B"):
            ck = f"{RUN_ROOT}/{run}/checkpoints"
            lines = " \\\n".join(f"            {o}" for o in ovr)
            post = (
                "\n# Emulate a job killed after epoch 1: keep the epoch-2 result aside, resume from step_1.\n"
                f"mv {ck}/step_2 {ck}/step_2_first\n"
                f"ln -sfn {ck}/step_1 {ck}/last\n"
                f"ln -sfn {ck}/step_1 {ck}/best\n"
                + TORCHRUN.format(overrides=lines)
            )
        post += "\nkill $WATCHDOG_PID 2>/dev/null || true\n"
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=f"# flopsonly stop/continue check, run {run}: see {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=pre_train(run, job_name),
            post_train=post,
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        print(write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body))


if __name__ == "__main__":
    main()
