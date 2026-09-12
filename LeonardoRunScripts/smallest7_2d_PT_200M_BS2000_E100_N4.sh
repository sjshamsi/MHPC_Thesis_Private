#!/bin/bash -l
#SBATCH --job-name=smallest7_2d_PT_200M_BS2000_E100_N4
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# ~200M-parameter architecture pretrained FROM SCRATCH on the same 7-dataset
# Leonardo_smallest7_2d mixture as smallest7_2d_PT_BS2000_E100_N4.sh (all 2D:
# active_matter, planetswe, acoustic_scattering_discontinuous,
# gray_scott_reaction_diffusion, turbulent_radiative_layer_2D, helmholtz_staircase,
# viscoelastic_instability).
#
# Filename decodes as: smallest7_2d dataset, PT = PreTraining, 200M = the shrunk
# architecture below (as opposed to the ~1.3B paper-scale
# smallest7_2d_PT_BS2000_E100_N* scripts this one mirrors), BS2000 =
# data.module_parameters.max_samples=2000, E100 = trainer.max_epoch=100,
# N4 = --nodes=4.
#
# This script IS smallest7_2d_PT_BS2000_E100_N4.sh (see that file for the full
# upstream-derivation rationale) with exactly four settings changed to shrink the
# architecture from paper scale down to ~200M params:
#   - model.hidden_dim=768, not 1408 - the processor's token width.
#   - model.processor.space_mixing.mlp_dim=2048, not unset. Unset would default to
#     FullAttention's own 4*hidden_dim (3072 at hidden_dim=768); pinned explicitly
#     here instead so it doesn't just track hidden_dim.
#   - model.processor_blocks=12, not 40 - number of stacked SpaceTimeSplitBlocks.
#   - model.gradient_checkpointing_freq=0, not 2 - gradient checkpointing trades
#     recompute for activation memory. It earns its keep at the paper-scale 40-block
#     depth to fit in GPU memory; this 12-block/hidden_dim=768 model doesn't have
#     that memory pressure, so checkpointing is turned off to avoid paying the
#     recompute cost for nothing.
# logger.wandb_project_name is also changed, to "smallest7_2d_PT_200M", so these runs
# group separately in W&B from the paper-scale smallest7_2d_PT_BS2000_E100_N* family.
# Every other model.*/data.*/trainer.*/optimizer/lr_scheduler/checkpoint/experiment/
# finetuning_mods setting - including num_heads=16 for both space_mixing and
# time_mixing, intermediate_dim=352, groups=16, projection_dim=48, drop_path=0.05,
# causal_in_time/jitter_patches/override_dimensionality - is unchanged.
#
# ~200M is the analytical parameter count (encoder + processor + decoder) for this
# exact configuration: encoder 58.8M + processor 85.1M + decoder 58.8M ~ 202.6M,
# from walrus_param_count_analytical() in
# demo_notebooks/walrus_example_2_ArchitectureAndDimensions.ipynb (walked through in
# demo_notebooks/walrus_architecture_notes.md section 11).

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
export AUTO_RESUME=${AUTO_RESUME:-True}

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py \
            hydra.run.dir=outputs/\${now:%Y-%m-%d}/$SLURM_JOB_ID \
            distribution=fsdp \
            model=isotropic_model \
            name=smallest7_2d_PT_200M_BS2000_E100_N4 \
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            server=leonardo \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="smallest7_2d_PT" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=0 \
            trainer.log_interval=200 \
            trainer.clip_gradient=10 \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.hidden_dim=768 \
            model.groups=16 \
            model.processor_blocks=12 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.num_heads=16 \
            model.processor.space_mixing.mlp_dim=2048 \
            model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True \
            model.jitter_patches=True \
            data.module_parameters.max_samples=2000 \
            trainer.short_validation_length=20 \
            trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.val_frequency=2 \
            trainer.rollout_val_frequency=5 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            trainer.prediction_type="delta" \
            data=Leonardo_smallest7_2d \
            trainer.max_epoch=100 \
            data_workers=10 \
            model.override_dimensionality=0 \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0.0 \
            ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
            trainer.video_validation=False
