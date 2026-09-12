#!/bin/bash -l
#SBATCH --job-name=flop_counter_backend_test
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Tier 0 / §5.1 acceptance criterion 1 (SCALING_STRATEGY.md): the binary gate deciding
# whether `torch.utils.flop_counter.FlopCounterMode` is usable on Leonardo at all.
#
# walrus_flops_analysis.ipynb §4 showed the counter silently records ZERO FLOPs for
# FullAttention's spatial attention on a CPU build, because
# `F.scaled_dot_product_attention(q, k, v)` with no mask dispatches to
# `aten::_scaled_dot_product_flash_attention_for_cpu`, which has no registered FLOP formula.
# The dropped term was 25x larger than everything the counter did record for that module,
# and it is the O(N_spatial^2) term that §3.1 needs to price 2D against 3D micro-batches.
#
# CUDA should dispatch to a registered kernel and so should not have this problem. This job
# measures that rather than assuming it, sweeping dtype x shape because backend selection
# depends on both (see the module docstring in flop_counter_backend_test.py):
#   - fp32 (enable_amp=False, what every run in this study uses) and bf16 autocast
#     (enable_amp=True, the lever §5.2 keeps open until Tier 1)
#   - hidden_dim 1408 (head_dim 88) and 320 (head_dim 20), 2D (1024 tokens) and 3D (4096)
#
# Exit code doubles as the gate: 0 = PASS (counter usable as a cross-check),
# 1 = FAIL (analytical formula is the only instrument), 2 = INCONCLUSIVE.
#
# One GPU and a couple of minutes of compute; sized to boost_qos_dbg (30 min cap).
# Needs no dataset and no Hydra config - it builds the two attention modules directly, so
# the gate cannot fail for reasons unrelated to the question it is asking.
#
# Usage: ./submit.sh flop_counter_backend_test.sh
# Reproduce the notebook's CPU finding on a login node instead:
#   python LeonardoRunScripts/flop_counter_backend_test.py --cpu

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis

RESULT_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/flop_gate
mkdir -p "$RESULT_DIR"

# A FAIL verdict (exit 1) is a valid measurement, not a broken job -- don't let
# `set -e` mark the SLURM job FAILED and muddy sacct. Capture and report instead.
set +e
srun python -u LeonardoRunScripts/flop_counter_backend_test.py \
    --out "$RESULT_DIR/flop_counter_backend_${SLURM_JOB_ID}.json"
GATE_RC=$?
set -e

echo
case "$GATE_RC" in
    0) echo "VERDICT: PASS - FlopCounterMode is usable as a cross-check (§5.1 criterion 1 met)." ;;
    1) echo "VERDICT: FAIL - counter drops attention; the analytical formula is the only instrument." ;;
    2) echo "VERDICT: INCONCLUSIVE - inspect the per-row numbers above." ;;
    *) echo "VERDICT: ERROR - the test itself did not complete (exit $GATE_RC)."; exit "$GATE_RC" ;;
esac
echo "Record: $RESULT_DIR/flop_counter_backend_${SLURM_JOB_ID}.json"
