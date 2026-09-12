#!/bin/bash -l
#SBATCH --job-name=smallest7_2d_PT_BS2000_E100_N4
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

# Paper-scale architecture, pretrained FROM SCRATCH (not finetuned) on the 7-dataset
# Leonardo_smallest7_2d mixture (all 2D: active_matter, planetswe,
# acoustic_scattering_discontinuous, gray_scott_reaction_diffusion,
# turbulent_radiative_layer_2D, helmholtz_staircase, viscoelastic_instability).
#
# Filename decodes as: smallest7_2d dataset, PT = PreTraining (as opposed to FT =
# FineTuning), BS2000 = data.module_parameters.max_samples=2000 ("Time in fake
# epochs" - already Leonardo_smallest7_2d.yaml's own default, restated explicitly
# here to match the filename), E100 = trainer.max_epoch=100, N1 = --nodes=1.
#
# Since this is PT (pretraining from scratch, not finetuning), the actual hyperparameter
# values are based on upstream's own canonical pretraining launch script,
# walrus/run_scripts/pretrain_example_distributed_walrus.sh - not fullDims_FT_ActiveMatter.sh
# (a finetuning config) or fullDims_6DS.sh, though the latter turns out to already be a close
# Leonardo-adapted copy of that same upstream script. Only the following are deliberately
# changed from upstream's exact values:
#   - server=leonardo, module loads, venv path, WANDB_MODE=offline/WANDB_DIR, experiment_dir:
#     Leonardo-specific environment adaptations (upstream's Flatiron cluster has outbound
#     internet and its own module/venv setup, so none of this is in the upstream script).
#   - distribution=fsdp, not upstream's hsdp: upstream ran across 24 nodes, where hsdp shards
#     within each node and replicates data-parallel-style across nodes. For this single-node
#     (N1) run, hsdp's node-replication dimension degenerates to 1 anyway, so plain fsdp is the
#     correct single-node equivalent, not a value actually being changed.
#   - data=Leonardo_smallest7_2d, not upstream's all_2_3d (the full 19-dataset mixture) - we
#     don't have most of those datasets downloaded on this cluster; smallest7_2d is what's
#     actually available and 2D-only, as requested.
#   - trainer.max_epoch=100, not upstream's 201: E100 was requested explicitly.
#   - trainer.video_validation=False (upstream doesn't set this at all, so it uses
#     trainer=defaults' own True): no working FFmpeg compilation assumed on this cluster,
#     matching fullDims_FT_ActiveMatter.sh's identical reasoning for the same setting.
#   - --nodes=1/--gpus-per-node=4/--cpus-per-gpu=8/--time=24:00:00/--qos=normal/--account=...:
#     Leonardo SLURM scheduling, requested N1 and 24h (upstream's own 24-node/72h job on a
#     different scheduler has no equivalent set of flags to copy).
# Everything else - optimizer=adam lr=2.e-4, lr_scheduler=inv_sqrt_w_sqrt_ramps,
# trainer.grad_acc_steps=4/log_interval=200/clip_gradient=10/short_validation_length=20/
# max_rollout_steps=60/val_frequency=10/rollout_val_frequency=10, all model.* architecture
# and jitter/dt-stride/prediction_type settings, checkpoint=defaults/experiment=defaults,
# finetuning_mods=defaults, data.module_parameters.batch_size/n_steps_input/n_steps_output/
# min_dt_stride/max_dt_stride/max_samples=2000 (BS2000) - is copied verbatim from
# pretrain_example_distributed_walrus.sh.

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
            name=smallest7_2d_PT_BS2000_E100_N4 \   
            trainer=defaults \
            trainer.grad_acc_steps=4 \
            server=leonardo \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="smallest7_2d_PT" \
            trainer.enable_amp=False \
            model.gradient_checkpointing_freq=2 \
            trainer.log_interval=200 \
            trainer.clip_gradient=10 \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.hidden_dim=1408 \
            model.groups=16 \
            model.processor_blocks=40 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.num_heads=16 \
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
