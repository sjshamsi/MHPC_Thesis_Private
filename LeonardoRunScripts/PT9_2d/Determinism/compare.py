"""Judge the determinism check (see gen_pt9_determinism_runs.py). Run on the login node."""

import sys
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_pt9_determinism_runs import RUN_ROOT  # noqa: E402


def weights(step_dir):
    with tempfile.TemporaryDirectory() as tmp:
        dcp_to_torch_save(step_dir, f"{tmp}/w.pt")
        model = torch.load(f"{tmp}/w.pt", weights_only=False)["app"]["model"]
    return torch.cat([v.flatten().double() for _, v in sorted(model.items())])


def dist(x, y):
    return float((x - y).norm() / y.norm())


a = weights(f"{RUN_ROOT}/A/checkpoints/step_2")
first = weights(f"{RUN_ROOT}/B/checkpoints/step_2_first")
resumed = weights(f"{RUN_ROOT}/B/checkpoints/step_2")
floor, d_res, d_res_a = dist(a, first), dist(resumed, first), dist(resumed, a)
print(f"float-noise floor   |A - B_first|       = {floor:.3e}")
print(f"resume              |B_resumed - B_first| = {d_res:.3e}")
print(f"resume vs A         |B_resumed - A|     = {d_res_a:.3e}")
ok = d_res <= 2 * floor + 1e-7
print("PASS: the resumed epoch matches the uninterrupted one within float noise" if ok else
      "FAIL: the resumed epoch differs by more than float noise -- it drew different data")
sys.exit(0 if ok else 1)
