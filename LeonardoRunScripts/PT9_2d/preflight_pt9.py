"""Refuse to submit PT9 runs until the data and config are actually ready.

Checks, in about a minute on a login node:
1. Every train/valid/test HDF5 file of the requested datasets opens. A file that is still
   downloading fails here ("truncated file") instead of an hour into a training job.
2. The §4.1 substrate composes: the data config resolves to exactly the nine pretraining
   datasets (the blindness post-condition of SCALING_STRATEGY_2D.md §5.3), and the pinned
   field map gives n_states = 67.

Usage:
    python preflight_pt9.py                      # all nine pretraining datasets
    python preflight_pt9.py planetswe gray_scott_reaction_diffusion
Exit code 0 = ready; 1 = not ready (reasons printed).
"""

import glob
import sys

import h5py
from hydra import compose, initialize_config_dir

from pt9_substrate import PRETRAIN_DATASETS, SUBSTRATE, WALRUS_DIR

DATASET_ROOT = "/leonardo_work/ICT26_MHPC_0/sshamsi/datasets"


def check_files(datasets):
    problems = []
    for d in datasets:
        for split in ("train", "valid", "test"):
            files = sorted(glob.glob(f"{DATASET_ROOT}/{d}/data/{split}/*.hdf5"))
            if not files:
                problems.append(f"{d}/{split}: no files")
            for f in files:
                try:
                    with h5py.File(f, "r"):
                        pass
                except OSError as e:
                    problems.append(f"{f}: {str(e).splitlines()[0][:120]}")
    return problems


def check_config():
    problems = []
    with initialize_config_dir(config_dir=f"{WALRUS_DIR}/configs", version_base=None):
        cfg = compose(config_name="config", overrides=SUBSTRATE)
    resolved = sorted(cfg.data.module_parameters.well_dataset_info.keys())
    if resolved != sorted(PRETRAIN_DATASETS):
        problems.append(f"data config resolves to {resolved}, expected {sorted(PRETRAIN_DATASETS)}")
    n_states = max(cfg.data.field_index_map_override.values()) + 1
    if n_states != 67:
        problems.append(f"n_states = {n_states}, expected 67")
    return problems


def main():
    datasets = sys.argv[1:] or PRETRAIN_DATASETS
    unknown = [d for d in datasets if d not in PRETRAIN_DATASETS]
    if unknown:
        print(f"Not PT9 pretraining datasets: {unknown}")
        sys.exit(1)
    problems = check_files(datasets) + check_config()
    if problems:
        print("PT9 preflight FAILED -- do not submit:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print(f"PT9 preflight passed: {len(datasets)} datasets readable, config composes.")


if __name__ == "__main__":
    main()
