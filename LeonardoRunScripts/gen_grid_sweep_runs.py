# ruff: noqa: E501
"""Generate the SCALING_STRATEGY.md 7.1 Tier 2 grid sweep: a 3-axis, 64-point grid at
small compute, used to locate x⁽⁰⁾ -- a single architecture on the Pareto-optimal
frontier for small compute -- which the Tier 3 star sweep (gen_star_search_runs.py)
then anchors its exponent fits to.

*** TEMPORARY / PLACEHOLDER -- DO NOT SUBMIT AS-IS ***

This generator is scaffolding, written before three upstream results are in. Everything
marked "PENDING" below is a real placeholder, not a considered final value -- edit it
once the dependency resolves, per SCALING_STRATEGY.md's own sequencing (Section 10):

    T0 roofline (5.2) --[GATE A]--> T1 HP reduction (6) --> T2 grid sweep (this file)

- PENDING GATE A (5.2, roofline jobs submitted, not yet returned): whether the full
  64-point/40-epoch grid below is affordable, or the dataloader-bound contingency
  applies (shrink to 3x3x3=27 points, cut to 25 epochs, stage data to node-local
  NVMe). TIME_CAP is also a guess until 5.2 gives a real per-epoch number at these
  shapes -- it is currently sized off fullDims's historical ~7.8 GPU-h/epoch scaled
  down by a guessed factor, not a measurement.
- PENDING Tier 1 (6, strategy written, not yet run): LR and WD below are the current
  repo-wide pin (2e-4 / 1e-4), used here provisionally. If Tier 1's stability check at
  the (320, 6, 768) corner -- the same shape this grid's own floor touches -- finds
  that pin unstable, it must change here too before submitting.
- PENDING 5.2's data_workers sweep: DATA_WORKERS below is the existing repo default
  (10), not informed by which of {8, 10, 16} the roofline data_workers variation
  actually preferred.
- PENDING §7.0 (new): EPOCHS=40 below was picked by loose analogy to SoViT, not derived
  for WALRUS -- same issue as the star sweep's own placeholder (60 in the doc, 101 as
  currently hardcoded in gen_star_search_runs.py; the two already disagree with each
  other and with this file). §7.0 derives the real number from the LR-warmup floor, the
  cooldown-fraction check, an as-yet-unrun small-model saturation pilot
  (`noise_floor_run_0.sh`, run long specifically for this), and the data-repetition
  ceiling. Do not finalize EPOCHS here until §7.0 resolves -- and note
  `gen_star_search_runs.py` is also still centred on the 200M anchor rather than
  fullDims and missing the §4.1 substrate pins (the n_states override, ...),
  per §2.3 -- it needs its own pass, not just a duration fix.

Usage (once the above are resolved and the placeholders below are edited for real):
    python gen_grid_sweep_runs.py
    ./submit.sh grid_run_0.sh    # submit one at a time, or:
    bash run_grid_sweep_all.sh   # submit every generated job
"""

import os
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_DIR = "/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs"
WALRUS_DIR = "/leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus"
VENV_ACTIVATE = "/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate"
WANDB_PROJECT = "walrus_leonardo_grid_sweep"

# ============================================================================
# PENDING GATE A (SCALING_STRATEGY.md 5.2) -- the full spec if cost is
# FLOP-proportional. If the dataloader-bound contingency applies instead, replace
# with the 3x3x3=27 shrunk grid and EPOCHS=25 (5.2's own contingency text).
# All hidden_dim values divisible by 16 for num_heads=16 and groups=16 (4.1).
# ============================================================================
GRID: Dict[str, List[int]] = {
    "hidden_dim": [320, 384, 464, 560],
    "processor_blocks": [6, 8, 10, 12],
    "mlp_dim": [768, 928, 1120, 1344],
}
EPOCHS = 40

# PENDING Tier 1 (SCALING_STRATEGY.md 6): the repo-wide pin, used provisionally.
LR = "2.e-4"
WD = "1.e-4"

# PENDING 5.2's data_workers variation result.
DATA_WORKERS = 10

# PENDING 5.2 (no real per-epoch measurement at these shapes yet -- this is a guess,
# not a budget). fullDims costs ~7.8 GPU-h/epoch historically; these shapes are much
# smaller (84M-143M params vs. fullDims's 1.29B), so guessing a generous 10x margin
# under that per-epoch figure for 40 epochs. Revisit once 5.2 actually reports.
TIME_CAP = "06:00:00"

HEADS = 16
GC_FREQ = 0  # matches every other small-shape script in this study (84M-143M params)

# model.override_dimensionality=3 -> include_d=[3]: exactly one encoder/decoder pair.
# Must be >0 (=0 builds a second, never-trained pair) and must be 3, not 2: forward()
# takes its dim_key from metadata.n_spatial_dims, which the datamodule has already set to
# 3 for every dataset (inflated_dataset.py, pad_cartesian_data_to_d), so =2 dies with
# KeyError '3'. The '2'/'3' keys are bucket labels, not shapes -- both pairs are built
# with spatial_dims=3 -- so on this 2D-only mixture =3 is bit-identical to a working =2.
# GUARDRAIL: only valid while the mixture is single-dimensionality. Adding a 3D dataset
# would silently share one encoder across both dimensionalities rather than crashing;
# that case needs =0, which dispatches on the actual tensor shape.
SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output={experiment_dir}/slurm/{job_name}/%j.out
#SBATCH --error={experiment_dir}/slurm/{job_name}/%j.err

# SCALING_STRATEGY.md 7.1 (Tier 2 grid sweep -> x^(0)): shape
# {hidden_dim}/{depth}/{mlp}. *** PLACEHOLDER VALUES -- see the module docstring in
# gen_grid_sweep_runs.py for what is still pending before this should be submitted. ***
# Generated by gen_grid_sweep_runs.py -- do not hand-edit, edit the generator and
# regenerate instead.

set -euo pipefail

export OMP_NUM_THREADS=${{SLURM_CPUS_ON_NODE}}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR={experiment_dir}
export AUTO_RESUME=${{AUTO_RESUME:-True}}

module purge
module load python
module load cuda/12.2
source {venv_activate}

cd {walrus_dir}

srun python -u `which torchrun` \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --nproc_per_node=$SLURM_GPUS_PER_NODE \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \\
        train.py \\
            hydra.run.dir=outputs/\\${{now:%Y-%m-%d}}/$SLURM_JOB_ID \\
            distribution=fsdp \\
            model=isotropic_model \\
            name={job_name} \\
            trainer=defaults \\
            server=leonardo \\
            data=Leonardo_smallest7_2d \\
            data/field_index_map_override=full_well_field_index \\
            optimizer=adam \\
            optimizer.lr={lr} \\
            optimizer.weight_decay={wd} \\
            logger.wandb_project_name="{wandb_project}" \\
            trainer.enable_amp=False \\
            trainer.grad_acc_steps=4 \\
            trainer.clip_gradient=10 \\
            trainer.prediction_type=delta \\
            trainer.log_interval=100 \\
            model.gradient_checkpointing_freq={gc_freq} \\
            data.module_parameters.batch_size=2 \\
            data.module_parameters.n_steps_input=6 \\
            data.module_parameters.n_steps_output=1 \\
            data.module_parameters.min_dt_stride=1 \\
            data.module_parameters.max_dt_stride=5 \\
            data.module_parameters.max_samples=2000 \\
            model.projection_dim=48 \\
            model.intermediate_dim=352 \\
            model.groups=16 \\
            model.override_dimensionality=3 \\
            model.hidden_dim={hidden_dim} \\
            model.processor_blocks={depth} \\
            model.drop_path=0.05 \\
            model/processor/space_mixing=full_spatial_attention \\
            model.processor.space_mixing.mlp_dim={mlp} \\
            model.processor.space_mixing.num_heads={heads} \\
            model.processor.time_mixing.num_heads={heads} \\
            model.causal_in_time=True \\
            model.jitter_patches=True \\
            lr_scheduler=inv_sqrt_w_sqrt_ramps \\
            trainer.short_validation_length=20 \\
            trainer.val_frequency=2 \\
            trainer.rollout_val_frequency=5 \\
            trainer.max_rollout_steps=60 \\
            trainer.video_validation=False \\
            trainer.max_epoch={epochs} \\
            data_workers={data_workers} \\
            checkpoint.checkpoint_frequency=25 \\
            auto_resume=$AUTO_RESUME \\
            checkpoint=defaults \\
            experiment=defaults \\
            finetuning_mods=defaults \\
            ++trainer.skip_spectral_metrics=True \\
            ++experiment_dir={experiment_dir}
"""


def main():
    submissions = []
    count = 0
    for hidden_dim in GRID["hidden_dim"]:
        for depth in GRID["processor_blocks"]:
            for mlp in GRID["mlp_dim"]:
                job_name = f"grid{count}_H{hidden_dim}_L{depth}_M{mlp}"
                body = SBATCH_TEMPLATE.format(
                    job_name=job_name,
                    time=TIME_CAP,
                    experiment_dir=EXPERIMENT_DIR,
                    venv_activate=VENV_ACTIVATE,
                    walrus_dir=WALRUS_DIR,
                    wandb_project=WANDB_PROJECT,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    mlp=mlp,
                    lr=LR,
                    wd=WD,
                    gc_freq=GC_FREQ,
                    heads=HEADS,
                    epochs=EPOCHS,
                    data_workers=DATA_WORKERS,
                )
                script_path = os.path.join(SCRIPT_DIR, f"grid_run_{count}.sh")
                with open(script_path, "w") as f:
                    f.write(body)
                os.chmod(script_path, 0o755)
                os.makedirs(os.path.join(EXPERIMENT_DIR, "slurm", job_name), exist_ok=True)
                submissions.append(f"./submit.sh grid_run_{count}.sh")
                count += 1

    run_all_path = os.path.join(SCRIPT_DIR, "run_grid_sweep_all.sh")
    with open(run_all_path, "w") as f:
        f.write("#!/bin/bash\n" + "\n".join(submissions) + "\n")
    os.chmod(run_all_path, 0o755)

    print(f"Generated {count} grid_run_N.sh scripts + run_grid_sweep_all.sh in {SCRIPT_DIR}")
    print("*** PLACEHOLDER VALUES -- see the module docstring for what is still pending. ***")


if __name__ == "__main__":
    main()
