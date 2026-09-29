#!/bin/bash
# Download The Well datasets through a CINECA data mover, using the per-dataset curl configs
# written by gen_well_curl_configs.py (scripts/well_curl_configs/<dataset>.cfg).
#
# Why a data mover: login nodes kill processes after 10 min of CPU time (TLS decryption of
# hundreds of GB exceeds that), compute nodes have no internet, and lrd_all_serial caps jobs
# at 4 h. Data movers have no CPU limit. They have no shell either -- only curl/wget/rsync/...
# -- so this script runs on the LOGIN node and hands one curl command per dataset to the data
# mover over ssh. Login-node -> data-mover ssh uses host-based auth (no 2FA certificate needed);
# that does NOT work from inside a batch job, so run this interactively, inside tmux.
#
# The bytes flow internet -> data mover -> Lustre directly; the local ssh client only carries
# the progress meter, so it stays far below the login node's CPU limit.
#
# Usage (login node, inside tmux, env1 venv active so the post-download --verify can run):
#   ./download_well_via_datamover.sh shear_flow                   # one dataset
#   ./download_well_via_datamover.sh shear_flow rayleigh_benard   # several, sequentially
#   DM_HOST=dmover2.leonardo.cineca.it ./download_well_via_datamover.sh ...  # pin a data mover
#   PARALLEL_MAX=2 ./download_well_via_datamover.sh ...           # fewer concurrent files
#                                                                  # (only if the mover's curl >= 7.66;
#                                                                  #  data.leonardo.cineca.it's is 7.61.1
#                                                                  #  and always downloads sequentially)
#
# Interrupted? Rerun the same command: --continue-at - resumes each file from its current size.
# On a rerun, files that are already complete can make curl report a range error (the server
# answers 416 to a resume request past EOF). That is expected; the --verify step at the end is
# the authoritative completeness check, not curl's exit code.
#
# Lustre notes (CINECA "Lustre Best Practice"):
#   - Never run two copies of this for the same dataset at once: two curls appending to the
#     same file is exactly the "multiple processes writing one file region" contention case.
#   - Don't monitor progress with du / find / ls -lR on the dataset tree (metadata-heavy);
#     watch curl's meter, `cinQuota`, or rerun the --verify step.
#   - Files inherit $WORK's default progressive layout (1 OST < 10 GB, 2 OSTs to 100 GB, 4 above),
#     the same layout as the 8 datasets already downloaded. See gen_well_curl_configs.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="$SCRIPT_DIR/well_curl_configs"
DM_HOST="${DM_HOST:-data.leonardo.cineca.it}"
PARALLEL_MAX="${PARALLEL_MAX:-4}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset> [<dataset> ...]" >&2
    echo "Available configs:" >&2
    for cfg in "$CFG_DIR"/*.cfg; do echo "  $(basename "$cfg" .cfg)" >&2; done
    exit 1
fi

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
    echo "WARNING: not inside tmux/screen -- a dropped login session kills the transfer" >&2
    echo "(rerunning resumes it, but multi-hour downloads should run under tmux)." >&2
fi

# --parallel needs curl >= 7.66 (Sept 2019). CINECA's data movers run RHEL8's stock curl
# (7.61.1, Sept 2018) which predates it and errors out on the unknown flag. Detect this once
# up front instead of failing mid-download, and fall back to sequential transfer -- config
# files with multiple url/output pairs still download every file one after another without
# --parallel, just without the concurrency.
REMOTE_CURL_VERSION="$(ssh -x "$USER@$DM_HOST" curl --version 2>/dev/null | head -1 | awk '{print $2}')"
PARALLEL_ARGS=()
if [ -n "$REMOTE_CURL_VERSION" ] && \
   [ "$(printf '%s\n7.66.0\n' "$REMOTE_CURL_VERSION" | sort -V | head -1)" = "7.66.0" ]; then
    PARALLEL_ARGS=(--parallel --parallel-max "$PARALLEL_MAX")
    echo "$DM_HOST: curl $REMOTE_CURL_VERSION supports --parallel (max $PARALLEL_MAX)" >&2
else
    echo "$DM_HOST: curl ${REMOTE_CURL_VERSION:-unknown} lacks --parallel -- downloading sequentially" >&2
fi

for ds in "$@"; do
    cfg="$CFG_DIR/$ds.cfg"
    if [ ! -f "$cfg" ]; then
        echo "No config for '$ds' at $cfg -- run gen_well_curl_configs.py first." >&2
        exit 1
    fi
    echo "=== $ds via $DM_HOST -- started $(date '+%F %T') ==="
    # Absolute --config path: the data mover has a fake /home and no $WORK/$HOME variables.
    # --fail: never save an HTTP error page under a .hdf5 name.
    ssh -xt "$USER@$DM_HOST" curl \
        --config "$cfg" \
        --create-dirs \
        --continue-at - \
        --fail \
        --retry 5 --retry-delay 30 \
        "${PARALLEL_ARGS[@]}" \
        || echo "curl exited non-zero for $ds (see above) -- rerun to resume." >&2
    echo "=== $ds -- finished $(date '+%F %T') ==="

    if python -c "import yaml" 2>/dev/null; then
        python "$SCRIPT_DIR/gen_well_curl_configs.py" --verify --dataset "$ds" || true
    else
        echo "(skipping --verify: activate the env1 venv to enable it)" >&2
    fi
done
