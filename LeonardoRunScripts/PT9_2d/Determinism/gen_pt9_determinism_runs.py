"""
Generate the GPU check that a resumed epoch reproduces the uninterrupted one.

Two tiny runs on the PT9 substrate (320/6/768, max_samples=100, 2 epochs, seed 0, 1 node, fsdp),
sized for the debug QOS (boost_qos_dbg: 30 min, 2 jobs per user, so A and B can run at once):

  A  trains epochs 1-2 straight.
  B  trains epochs 1-2, sets its epoch-2 checkpoint aside (step_2_first), points `last` back
     at step_1 and runs again with the same config: the second run auto-resumes and redoes
     epoch 2, exactly what a chain link does after a job is killed past epoch 1.

On GPU, same-seed runs are not bitwise identical (non-deterministic kernels), so compare.py
judges B's resumed epoch-2 weights against the float-noise floor |A - B_first|. If the
resumed epoch drew different data, its distance from B_first would be far above that floor.

Time budget (from the 200M-anchor roofline log): start-up ~1.5 min, ~0.7 s/step, and at the last
epoch the substrate's rollout validation and the rollout test pass each take ~16 min with
max_rollout_steps=60. Validation runs after training and never touches the weights, so it is cut
to max_rollout_steps=2 and short_validation_length=2. B (two launches, three epochs) then needs
~12 min.

Diagnostics (a first B run hung in epoch 2 with no error; mydocs/determinism_code_changes.md): both
scripts set TORCH_DISTRIBUTED_DEBUG=DETAIL (cross-rank collective consistency checks), a PG_TIMEOUT
collective timeout (WALRUS_PG_TIMEOUT_SEC, walrus/utils/distribution_utils.py; covers NCCL
and gloo), the NCCL flight recorder (per-rank collective trace dumped on timeout into the
run folder), and PYTHONFAULTHANDLER=1 with a watchdog that SIGABRTs the training workers if
the log is silent for WATCHDOG_IDLE seconds, so every thread's Python stack lands in .err.

    python gen_pt9_determinism_runs.py
    sbatch p9_determinism_A.sh; sbatch p9_determinism_B.sh
    python compare.py            # on the login node, after both finish

fulldet (p9_fulldet_check.sh): check_disjoint.py builds the real PT9 training sampler under fsdp
and under HSDP (2 groups of 2 GPUs) and checks 2 epochs of windows across all ranks, reading no
data; then a 1-epoch tiny training as a smoke test (one epoch has no epoch boundary, so it cannot
hit the fork-after-async-save hang). It runs the code of the tree holding this generator.
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
RUN_ROOT = f"{EXPERIMENT_DIR}/determinism"  # compare.py reads <RUN_ROOT>/<run>/checkpoints
WATCHDOG_IDLE = 300  # s without log output; above start-up (~1.5 min, longer at fullDims) and validation
PG_TIMEOUT = 600  # s; must exceed WATCHDOG_IDLE, so the stack dump fires before the collective error


def diagnostics(run, job_name, pg_timeout=True, detail=True):
    log = f"{EXPERIMENT_DIR}/slurm/{job_name}/$SLURM_JOB_ID.out"
    timeout = f"export WALRUS_PG_TIMEOUT_SEC={PG_TIMEOUT}\n" if pg_timeout else ""
    # DETAIL wraps every process group in _ProcessGroupWrapper, which broke the DCP checkpoint load
    # on resume (old_B_1/2: "Backend nccl does not support allgather_into_tensor_coalesced").
    debug = "export TORCH_DISTRIBUTED_DEBUG=DETAIL\n" if detail else ""
    return f'''
# --- Hang diagnostics -----------------------------------------------------------------
{debug}{timeout}export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE={RUN_ROOT}/{run}/nccl_trace_rank_
export PYTHONFAULTHANDLER=1
mkdir -p {RUN_ROOT}/{run}
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


# Fork-after-async-save hang test (see mydocs/determinism_code_changes.md): one launch, several
# epoch boundaries, each with the async `last` save in flight while the next epoch forks its
# workers. checkpoint_frequency=0 keeps one rolling `last` checkpoint on disk.
# (name, hidden, depth, mlp, gc_freq, max_samples, epochs); sized from the roofline timings
# (~0.7 s/step anchor, ~3.5 s/step fullDims) to finish within the 30 min debug QOS.
HANG_TESTS = [
    ("hang_anchor", 768, 12, 2048, 0, 100, 6),
    ("hang_fullDims", 1408, 40, 5632, 2, 30, 6),
]


# The same tiny config on the code before the determinism changes (commit 656c00c), checked out as
# a git worktree: does the hang need those changes? WALRUS_PG_TIMEOUT_SEC doesn't exist there, so
# collectives keep their 30 min timeout and only the stack-dump watchdog applies.
OLD_COMMIT = "656c00c"
OLD_TREE = "/leonardo/home/userexternal/sshamsi0/MHPC_Thesis_656c00c"
REPO_TREE = "/leonardo/home/userexternal/sshamsi0/MHPC_Thesis"
# The worktree was removed after the hang investigation; its scripts stay as last generated.
HAVE_OLD = os.path.isdir(OLD_TREE)
OLD_TESTS = ["old_tiny_1", "old_tiny_2"]  # 320/6/768, 6 epochs: 5 exposed boundaries each
# The same two jobs on the current code: with OLD_TESTS, the commit is the only difference.
# No WALRUS_PG_TIMEOUT_SEC: 656c00c ignores it, so both groups build the process group the same way.
NEW_TESTS = ["new_tiny_1", "new_tiny_2"]
# B's exact script (2 epochs, checkpoint every epoch, then set `step_2` aside and relaunch to
# resume) on the 656c00c worktree. The old_tiny jobs differ from B in length and in having no
# resume launch; these two do not, so any difference from B is the code version (or the job name).
OLD_B_TESTS = ["old_B_1", "old_B_2"]
# The same B script without TORCH_DISTRIBUTED_DEBUG=DETAIL, so the resume launch can load its checkpoint:
# on the old code, and on the current code (the first GPU resume result for compare.py).
B_NODETAIL = [("old_B_nd_1", True), ("old_B_nd_2", True), ("cur_B_nd_1", False), ("cur_B_nd_2", False)]


# fulldet check: the repo tree holding this generator (a worktree on branch fulldet).
FULLDET_TREE = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
FULLDET_MESHES = [["distribution=fsdp"], ["distribution=hsdp", "distribution.local_size=2"]]


def compose_in(tree, ovr):
    from hydra import compose, initialize_config_dir

    # Undo shell syntax as pt9_substrate.check_composes does.
    clean = [o.replace("$AUTO_RESUME", "True").replace("'", "").replace('"', "") for o in ovr]
    with initialize_config_dir(config_dir=f"{tree}/walrus/configs", version_base=None):
        return compose(config_name="config", overrides=clean)


def overrides(run, shape=(320, 6, 768, 0), max_samples=100, epochs=2, ckpt_freq=1):
    folder = f"{RUN_ROOT}/{run}"
    return SUBSTRATE + shape_overrides(*shape) + [
        f"name={PREFIX}_determinism_{run}",
        f'logger.wandb_project_name="{PREFIX}_determinism"',
        "seed=0",
        f"trainer.max_epoch={epochs}",
        f"data.module_parameters.max_samples={max_samples}",
        "trainer.max_rollout_steps=2",        # rollout val/test: ~16 min each at 60 steps
        "trainer.short_validation_length=2",  # validation never touches the weights
        "data_workers=10",
        f"checkpoint.checkpoint_frequency={ckpt_freq}",
        f"folder_override={folder}",
        "auto_resume=True",
    ]


def main():
    for run in ("A", "B"):
        ovr = overrides(run)
        cfg = check_composes(ovr)
        assert (cfg.seed, cfg.trainer.max_epoch, cfg.data.module_parameters.max_samples) == (0, 2, 100)
        assert (cfg.trainer.max_rollout_steps, cfg.trainer.short_validation_length) == (2, 2)
        job_name = f"{PREFIX}_determinism_{run}"
        post = ""
        if run == "B":
            ck = f"{RUN_ROOT}/B/checkpoints"
            lines = " \\\n".join(f"            {o}" for o in ovr)
            post = (
                f"\n# Emulate a job killed after epoch 1: keep the epoch-2 result aside, resume from step_1.\n"
                f"mv {ck}/step_2 {ck}/step_2_first\n"
                f"ln -sfn {ck}/step_1 {ck}/last\n"
                f"ln -sfn {ck}/step_1 {ck}/best\n"
                + TORCHRUN.format(overrides=lines)
            )
        post += "\nkill $WATCHDOG_PID 2>/dev/null || true\n"
        body = render_script(
            job_name=f"{PREFIX}_determinism_{run}",
            time="00:30:00",
            description=f"# Determinism check, run {run}: see {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(run, job_name),
            post_train=post,
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{PREFIX}_determinism_{run}.sh", f"{PREFIX}_determinism_{run}", body)
    for name, h, depth, mlp, gc, ms, epochs in HANG_TESTS:
        ovr = overrides(name, (h, depth, mlp, gc), ms, epochs, ckpt_freq=0)
        cfg = check_composes(ovr)
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (h, epochs, 0)
        job_name = f"{PREFIX}_determinism_{name}"
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=f"# Fork-after-async-save hang test at {h}/{depth}/{mlp}: see {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(name, job_name),
            post_train="\nkill $WATCHDOG_PID 2>/dev/null || true\n",
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    for name in OLD_TESTS if HAVE_OLD else []:
        ovr = overrides(name, (320, 6, 768, 0), 100, 6, ckpt_freq=1)
        cfg = compose_in(OLD_TREE, ovr)
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (320, 6, 1)
        job_name = f"{PREFIX}_determinism_{name}"
        guard = (f"export PYTHONPATH={OLD_TREE}\n"
                 f"cd {OLD_TREE}/walrus\n"
                 f"git -C {OLD_TREE} rev-parse --short HEAD | grep -q '^{OLD_COMMIT}' || "
                 f"{{ echo 'worktree is not at {OLD_COMMIT}' >&2; exit 1; }}\n"
                 f"python -c \"import walrus; assert walrus.__file__.startswith('{OLD_TREE}/'), walrus.__file__\"\n")
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=f"# Fork-after-async-save hang test on commit {OLD_COMMIT} (before the determinism changes): see {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(name, job_name) + guard,
            post_train="\nkill $WATCHDOG_PID 2>/dev/null || true\n",
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    for name in OLD_B_TESTS if HAVE_OLD else []:
        ovr = overrides(name)  # B's defaults: tiny shape, max_samples=100, 2 epochs, checkpoint every epoch
        cfg = compose_in(OLD_TREE, ovr)
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (320, 2, 1)
        job_name = f"{PREFIX}_determinism_{name}"
        ck = f"{RUN_ROOT}/{name}/checkpoints"
        lines = " \\\n".join(f"            {o}" for o in ovr)
        guard = (f"export PYTHONPATH={OLD_TREE}\n"
                 f"cd {OLD_TREE}/walrus\n"
                 f"git -C {OLD_TREE} rev-parse --short HEAD | grep -q '^{OLD_COMMIT}' || "
                 f"{{ echo 'worktree is not at {OLD_COMMIT}' >&2; exit 1; }}\n"
                 f"python -c \"import walrus; assert walrus.__file__.startswith('{OLD_TREE}/'), walrus.__file__\"\n")
        post = (f"\n# Emulate a job killed after epoch 1: keep the epoch-2 result aside, resume from step_1.\n"
                f"mv {ck}/step_2 {ck}/step_2_first\n"
                f"ln -sfn {ck}/step_1 {ck}/last\n"
                f"ln -sfn {ck}/step_1 {ck}/best\n"
                + TORCHRUN.format(overrides=lines)
                + "\nkill $WATCHDOG_PID 2>/dev/null || true\n")
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=f"# B's exact script on commit {OLD_COMMIT} (before the determinism changes): see {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(name, job_name, pg_timeout=False) + guard,
            post_train=post,
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    for name, old in B_NODETAIL:
        if old and not HAVE_OLD:
            continue
        ovr = overrides(name)
        tree = OLD_TREE if old else REPO_TREE
        cfg = compose_in(tree, ovr)
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (320, 2, 1)
        job_name = f"{PREFIX}_determinism_{name}"
        ck = f"{RUN_ROOT}/{name}/checkpoints"
        lines = " \\\n".join(f"            {o}" for o in ovr)
        guard = ""
        if old:
            guard = (f"export PYTHONPATH={OLD_TREE}\n"
                     f"cd {OLD_TREE}/walrus\n"
                     f"git -C {OLD_TREE} rev-parse --short HEAD | grep -q '^{OLD_COMMIT}' || "
                     f"{{ echo 'worktree is not at {OLD_COMMIT}' >&2; exit 1; }}\n"
                     f"python -c \"import walrus; assert walrus.__file__.startswith('{OLD_TREE}/'), walrus.__file__\"\n")
        post = (f"\n# Emulate a job killed after epoch 1: keep the epoch-2 result aside, resume from step_1.\n"
                f"mv {ck}/step_2 {ck}/step_2_first\n"
                f"ln -sfn {ck}/step_1 {ck}/last\n"
                f"ln -sfn {ck}/step_1 {ck}/best\n"
                + TORCHRUN.format(overrides=lines)
                + "\nkill $WATCHDOG_PID 2>/dev/null || true\n")
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=(f"# B's exact script, DETAIL off, on {'commit ' + OLD_COMMIT if old else 'the current code'}: "
                         f"see {GENERATOR}."),
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(name, job_name, pg_timeout=False, detail=False) + guard,
            post_train=post,
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    for name in NEW_TESTS:
        ovr = overrides(name, (320, 6, 768, 0), 100, 6, ckpt_freq=1)
        cfg = check_composes(ovr)
        assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.checkpoint.checkpoint_frequency) == (320, 6, 1)
        job_name = f"{PREFIX}_determinism_{name}"
        body = render_script(
            job_name=job_name,
            time="00:30:00",
            description=f"# Control for the {OLD_COMMIT} hang test: same config on the current code. See {GENERATOR}.",
            generator=GENERATOR,
            overrides=ovr,
            pre_train=diagnostics(name, job_name, pg_timeout=False),
            post_train="\nkill $WATCHDOG_PID 2>/dev/null || true\n",
        )
        assert body.count("#SBATCH --qos=normal") == 1
        body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
        write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    name = "fulldet_check"
    job_name = f"{PREFIX}_{name}"
    ovr = overrides(name, max_samples=100, epochs=1)
    cfg = compose_in(FULLDET_TREE, ovr)
    assert (cfg.model.hidden_dim, cfg.trainer.max_epoch, cfg.data.module_parameters.max_samples) == (320, 1, 100)
    guard = (f"export PYTHONPATH={FULLDET_TREE}\n"
             f"cd {FULLDET_TREE}/walrus\n"
             "python -c 'from walrus.data.mixed_dset_sampler import epoch_plan'  # fulldet code\n")
    checks = ""
    for mesh in FULLDET_MESHES:
        # The real PT9 epoch length: the check only draws indices, it trains nothing.
        o = [x for x in overrides(name) if not x.startswith("distribution=")] + mesh
        o = [x for x in o if not x.startswith("data.module_parameters.max_samples=")]
        cfg = compose_in(FULLDET_TREE, o)
        assert cfg.data.module_parameters.max_samples == 2000, cfg.data.module_parameters.max_samples
        lines = " \\\n".join(f"            {x}" for x in o)
        checks += TORCHRUN.format(overrides=lines).replace("train.py \\", f"{SCRIPT_DIR}/check_disjoint.py \\")
    body = render_script(
        job_name=job_name,
        time="00:30:00",
        description=f"# fulldet: cross-rank disjoint windows (fsdp + HSDP), then a 1-epoch smoke run: see {GENERATOR}.",
        generator=GENERATOR,
        overrides=ovr,
        pre_train=diagnostics(name, job_name, pg_timeout=False, detail=False) + guard + checks,
        post_train="\nkill $WATCHDOG_PID 2>/dev/null || true\n",
    )
    assert body.count("check_disjoint.py") == 2 and body.count("#SBATCH --qos=normal") == 1
    body = body.replace("#SBATCH --qos=normal", "#SBATCH --qos=boost_qos_dbg")
    write_script(SCRIPT_DIR, f"{job_name}.sh", job_name, body)
    old = (f"{len(OLD_TESTS)} {OLD_COMMIT} tests, " if HAVE_OLD else f"(no {OLD_COMMIT} worktree: old-code scripts left as they are), ")
    print(f"Generated {PREFIX}_determinism_{{A,B}}.sh, {len(HANG_TESTS)} hang tests, {old}"
          f"{len(NEW_TESTS)} current-code controls and {job_name}.sh in {SCRIPT_DIR}; "
          f"runs write to {RUN_ROOT}/")


if __name__ == "__main__":
    main()
