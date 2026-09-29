#!/bin/bash -l
#SBATCH --job-name=roofline_staged_H464_L10_M1120_E3_dw10
#SBATCH --time=06:00:00
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

# SCALING_STRATEGY.md 5.2 (GATE A), 6th point: the roofline-small shape (464/10/1120),
# with the Leonardo_smallest7_2d mixture staged to this node's local NVMe
# (/scratch_local) before training, instead of read live from $WORK over the network
# filesystem. Everything else is identical to roofline_run_2.sh (the un-staged
# data_workers=10 point at this shape) -- storage location is the only thing that
# changes, mirroring storage_bench_walrus.sh's comparison discipline.
#
# The mixture is ~653 GB (active_matter 50G, planetswe 178G,
# acoustic_scattering_discontinuous 151G, gray_scott_reaction_diffusion 152G,
# turbulent_radiative_layer_2D 6.7G, helmholtz_staircase 51G,
# viscoelastic_instability 64G), so staging is not free -- it is timed
# separately (STAGING_SECONDS below) and must be excluded from the sec/epoch fit,
# the same way build.py already drops the first logged interval of every epoch as
# warm-up. Only epoch 2 and 3's train-phase duration are commensurate with the other
# roofline points' epoch 2/3.
#
# Hand-written rather than generated (gen_roofline_runs.py's template does not cover
# this): the staging loop is genuinely different script logic, not just a parameter
# substitution.

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

SRC=/leonardo_work/ICT26_MHPC_0/sshamsi/datasets
STAGE_DIR=/scratch_local/${SLURM_JOB_ID}/datasets
DATASETS=(active_matter planetswe acoustic_scattering_discontinuous
          gray_scott_reaction_diffusion turbulent_radiative_layer_2D
          helmholtz_staircase viscoelastic_instability)

echo "Staging ${#DATASETS[@]} datasets to $STAGE_DIR ..."
STAGE_START=$(date +%s)
mkdir -p "$STAGE_DIR"
for d in "${DATASETS[@]}"; do
    cp -r "$SRC/$d" "$STAGE_DIR/$d" &
done
wait
STAGE_END=$(date +%s)
echo "STAGING_SECONDS: $((STAGE_END - STAGE_START))"
du -sh "$STAGE_DIR"

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
            name=roofline_staged_H464_L10_M1120_E3_dw10 \
            trainer=defaults \
            server=leonardo \
            data=Leonardo_smallest7_2d \
            data/field_index_map_override=full_well_field_index \
            ++data.well_base_path="$STAGE_DIR/" \
            optimizer=adam \
            optimizer.lr=2.e-4 \
            logger.wandb_project_name="walrus_leonardo_roofline" \
            trainer.enable_amp=False \
            trainer.grad_acc_steps=4 \
            trainer.clip_gradient=10 \
            trainer.prediction_type=delta \
            trainer.log_interval=50 \
            model.gradient_checkpointing_freq=0 \
            data.module_parameters.batch_size=2 \
            data.module_parameters.n_steps_input=6 \
            data.module_parameters.n_steps_output=1 \
            data.module_parameters.min_dt_stride=1 \
            data.module_parameters.max_dt_stride=5 \
            data.module_parameters.max_samples=2000 \
            model.projection_dim=48 \
            model.intermediate_dim=352 \
            model.groups=16 \
            model.override_dimensionality=3 \
            model.hidden_dim=464 \
            model.processor_blocks=10 \
            model.drop_path=0.05 \
            model/processor/space_mixing=full_spatial_attention \
            model.processor.space_mixing.mlp_dim=1120 \
            model.processor.space_mixing.num_heads=16 \
            model.processor.time_mixing.num_heads=16 \
            model.causal_in_time=True \
            model.jitter_patches=True \
            seed=0 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps \
            trainer.short_validation_length=20 \
            trainer.val_frequency=2 \
            trainer.rollout_val_frequency=5 \
            trainer.max_rollout_steps=60 \
            trainer.video_validation=False \
            trainer.max_epoch=3 \
            data_workers=10 \
            auto_resume=$AUTO_RESUME \
            checkpoint=defaults \
            experiment=defaults \
            finetuning_mods=defaults \
            ++trainer.skip_spectral_metrics=True \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs

rm -rf "/scratch_local/${SLURM_JOB_ID}"
