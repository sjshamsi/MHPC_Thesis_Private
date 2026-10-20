"""Weight distances for the flopsonly stop/continue check (see gen_pt9_determinism_runs.py).

Run on the login node after all four jobs finish:
    python compare.py
It converts each epoch-2 checkpoint to a flat float32 array, one fresh process per checkpoint, then
computes relative distances ||x - y|| / ||y|| into <RUN_ROOT>/weight_distances.json, which
flopsonly_repetition.ipynb reads. Login-node processes are killed at 600 CPU-seconds, and reading
one array costs up to ~50 s of kernel time there, so no single process handles more than 6.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_pt9_determinism_runs import RUN_ROOT  # noqa: E402

OUT = f"{RUN_ROOT}/weights"
CKPTS = {  # label -> checkpoint dir; all are epoch-2 weights
    "A1": f"{RUN_ROOT}/A1/checkpoints/step_2",
    "A2": f"{RUN_ROOT}/A2/checkpoints/step_2",
    "B1_first": f"{RUN_ROOT}/B1/checkpoints/step_2_first",
    "B1_resumed": f"{RUN_ROOT}/B1/checkpoints/step_2",
    "B2_first": f"{RUN_ROOT}/B2/checkpoints/step_2_first",
    "B2_resumed": f"{RUN_ROOT}/B2/checkpoints/step_2",
}
PAIRS = {  # name -> (x, y), and what the distance measures
    "B1 resumed vs B1 first": ("B1_resumed", "B1_first"),  # same step_1 start, different epoch-2 data
    "B2 resumed vs B2 first": ("B2_resumed", "B2_first"),
    "A1 vs A2": ("A1", "A2"),  # different init and different data
    "B1 first vs B2 first": ("B1_first", "B2_first"),
    "A1 vs B1 first": ("A1", "B1_first"),
    "A2 vs B2 first": ("A2", "B2_first"),
}


def convert(label):
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    with tempfile.TemporaryDirectory() as tmp:
        dcp_to_torch_save(CKPTS[label], f"{tmp}/w.pt")
        model = torch.load(f"{tmp}/w.pt", weights_only=False)["app"]["model"]
    np.save(f"{OUT}/{label}.npy", torch.cat([v.flatten().float() for _, v in sorted(model.items())]).numpy())


def dist(x, y, chunk=1 << 22):
    num = den = 0.0
    for i in range(0, len(x), chunk):
        a, b = np.asarray(x[i:i + chunk], np.float64), np.asarray(y[i:i + chunk], np.float64)
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return (num / den) ** 0.5


def distances(names):
    labels = sorted({lab for n in names for lab in PAIRS[n]})
    arrays = {lab: np.load(f"{OUT}/{lab}.npy") for lab in labels}
    return {n: dist(arrays[PAIRS[n][0]], arrays[PAIRS[n][1]]) for n in names}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "convert":
        convert(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "distances":
        print(json.dumps(distances(sys.argv[2:])))
    else:
        os.makedirs(OUT, exist_ok=True)
        for label in CKPTS:
            if not os.path.exists(f"{OUT}/{label}.npy"):
                subprocess.run([sys.executable, __file__, "convert", label], check=True)
        result = {}
        for chunk in (list(PAIRS)[i:i + 3] for i in range(0, len(PAIRS), 3)):  # <= 6 arrays per process
            out = subprocess.run([sys.executable, __file__, "distances", *chunk], check=True,
                                 capture_output=True, text=True).stdout
            result.update(json.loads(out.strip().splitlines()[-1]))
        json.dump(result, open(f"{RUN_ROOT}/weight_distances.json", "w"), indent=1)
        for name, d in result.items():
            print(f"{name:<26}{d:.3e}")
