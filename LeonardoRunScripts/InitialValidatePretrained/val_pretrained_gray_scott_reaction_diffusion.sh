#!/bin/bash -l
#SBATCH --job-name=val_pretrained_gray_scott_reaction_diffusion
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Zero-shot evaluation of released pretrained WALRUS checkpoint on gray_scott_reaction_diffusion
#
# Sibling of val_pretrained_activematter.sh - identical protocol, only name= and
# data= change (and --time, see WALLTIME below). See that script's header for the
# full rationale behind each override; in brief:
#
# Every model.*/data.* override is copied from the pretrained checkpoint's training
# config at /leonardo_scratch/fast/ICT26_MHPC/sshamsi/walrus_pretrained/extended_config.yaml:
#   - finetuning_mods=defaults (empty), not finetuning_mods=all
#   - data/field_index_map_override=full_well_field_index, not Leonardo_gray_scott_reaction_diffusion.yaml's
#     default of bc_only_override
#   - data.module_parameters.{n_steps_input,min_dt_stride,max_dt_stride}=6,1,5 and
#     trainer.max_rollout_steps=200 match extended_config.yaml's trained values
#     (eval inherits the data protocol exactly as trained - cf. upstream's
#     walrus/run_scripts/eval_onegpu_example_walrus.sh, which never touches these)
#   - +data.module_parameters.max_rollout_steps=200 is ALSO set (MixedWellDataModule
#     has its own separate default of 100 at walrus/data/multidatamodule.py#L58 that
#     otherwise silently caps every rollout at 100 regardless of trainer.max_rollout_steps)
#   - model.drop_path=0.05 is the trained value (a no-op in eval mode either way)
# All other model.* overrides must match the checkpoint's shape/behavior and are copied as-is.
#
# validate() ignores trainer.max_epoch/val_frequency/rollout_val_frequency entirely
# (always runs the full/untruncated last-epoch validation+test pass, see
# walrus/trainer/training.py::Trainer.validate), so those aren't set here.
#
# NOTE: this validates ONE dataset (gray_scott_reaction_diffusion). Trainer.validate() loops over every
# dataset in whatever `data=` mixture you pass (one DataLoader per dataset, see
# walrus/data/multidatamodule.py::build_loaders_from_dset_list); Leonardo_gray_scott_reaction_diffusion.yaml
# is a single-dataset config, so only gray_scott_reaction_diffusion is validated + tested + rolled out here.
# distribution=hsdp + local_size=4 (all 4 GPUs in ONE sharding group), NOT ddp/4 and NOT
# local/1-GPU. validation_loop parallelises by handing WHOLE datasets to sync-groups
# round-robin (rank_assignment = dataset_idx % num_sync_groups,
# walrus/trainer/training.py#L671); under ddp each GPU is its own group, so a single-dataset
# data= config starves ranks 1-3 at the end-of-dataset dist.barrier() and gloo aborts the
# job ~30 min after rank 0's slowest phase overshoots (job 55241860 died this way at
# ~32 min). local/1-GPU fixes that for the other datasets, but gray_scott is far too heavy
# for one GPU: ~12.5h for the one-step valid pass ALONE (job 54978305), ~30h for full
# valid+test, over the 24h QOS cap. hsdp/local_size=4 makes num_sync_groups collapse to 1,
# so all 4 ranks enter this one dataset together and validation_loop's DistributedSampler
# shards it 4 ways with balanced load - real ~4x speedup, barriers balanced. validate()
# has no mid-run checkpointing - a timeout loses the whole pass.
#
# WALLTIME: hsdp/4 gives a real ~4x shard of this one dataset. Measured single-GPU in job
# 54978305: ~12.5h one-step valid + ~2.6h rollout valid = ~15h valid, ~30h valid+test;
# under hsdp/4 that is ~7.5h. 16:00:00 leaves generous headroom; if it still times out the
# dataset must be split (validate() has no valid-only / test-only flag).
#
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs

module purge
module load python
module load cuda/12.2
export PATH="/leonardo/home/userexternal/sshamsi0/.local/opt/ffmpeg-9.0.1/bin:$PATH"  # self-built ffmpeg+libx264 for trainer.video_validation=True (cluster ffmpeg is --disable-libx264; see scripts/build_ffmpeg.sh)
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
            distribution=hsdp \
            server=leonardo \
            model=isotropic_model \
            name=val_pretrained_gray_scott_reaction_diffusion \
            data=Leonardo_gray_scott_reaction_diffusion \
            data/field_index_map_override=full_well_field_index \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            +data.module_parameters.max_rollout_steps=200 \
            model.hidden_dim=1408 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.groups=16 \
            model.processor_blocks=40 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.num_heads=16 \
            model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True \
            model.override_dimensionality=0 \
            model.jitter_patches=True \
            model.gradient_checkpointing_freq=2 \
            ++model.use_periodic_fixed_jitter=True \
            ++model.input_field_drop=0.0 \
            trainer=defaults \
            trainer.max_rollout_steps=200 \
            trainer.prediction_type="delta" \
            trainer.video_validation=True \
            ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults \
            experiment=validate_leonardo \
            checkpoint=validate_full_mode_leonardo \
            auto_resume=False \
            data_workers=10 \
            logger.wandb_project_name="val_pretrained" \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
