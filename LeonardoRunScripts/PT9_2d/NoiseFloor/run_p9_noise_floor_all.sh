#!/bin/bash -l
set -euo pipefail
cd "$(dirname "$0")"
module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
# Refuse to submit until every file these runs read opens and the config composes.
python ../preflight_pt9.py
sbatch --dependency=singleton p9_noise_floor_run_0.sh
sbatch --dependency=singleton p9_noise_floor_run_0.sh
sbatch --dependency=singleton p9_noise_floor_run_0.sh
sbatch --dependency=singleton p9_noise_floor_run_1.sh
sbatch --dependency=singleton p9_noise_floor_run_1.sh
sbatch --dependency=singleton p9_noise_floor_run_1.sh
sbatch --dependency=singleton p9_noise_floor_run_2.sh
sbatch --dependency=singleton p9_noise_floor_run_2.sh
sbatch --dependency=singleton p9_noise_floor_run_2.sh
