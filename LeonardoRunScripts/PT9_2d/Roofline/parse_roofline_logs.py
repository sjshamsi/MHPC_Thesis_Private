"""Turn the PT9 roofline runs' slurm logs into the GATE A timing table (SCALING_STRATEGY_2D.md §5.2).

W&B gets only aggregate per-epoch timings. The per-dataset split that GATE A needs, i.e.
whether one dataset (Euler's ~37 MB windows) sets a data-loading floor, exists only in
the per-microbatch lines the trainer prints every `log_interval` microbatches:

    [2026-..][walrus.trainer.training][INFO] - Epoch    2, Batch 11/2000, Rank   0, SyncStep: False:
        Data: planetswe, loss ..., total_time 0.31s, data 0.0021s, fwd 0.12s, bw 0.20s, opt 0.000s

For each run (one directory of .out files under <logs>/slurm/, resubmissions included)
this prints:
- train seconds per epoch: from the first rank's "Epoch N/M: starting training" to the
  last rank's "Epoch N/M: training loss", so validation and checkpointing are excluded.
  Epoch 1 is reported separately (worker spin-up, allocator warm-up) and left out of the mean.
- per dataset: the number of sampled microbatches, and the median and p90 of the data-wait
  seconds. Batch 1 of each epoch (worker spin-up) and all of epoch 1 are excluded.

Only `data` (CPU wait on the loader) and epoch wall time are trustworthy: fwd/bw are not
CUDA-synchronised, so they mostly measure kernel launch time, and they are printed only
as a rough guide. The GATE A fit sec/epoch = a + b*FLOPs takes FLOPs from W&B
(`epoch_flops`, logged by FlopAccountant) after `wandb sync`.

Usage:
    python parse_roofline_logs.py                     # every p9_roofline_* run
    python parse_roofline_logs.py p9_roofline_small   # runs whose name contains this
"""

import glob
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

LOG_ROOT = "/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm"

TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]"
EPOCH_RE = re.compile(TS + r".*Epoch (\d+)/\d+: (starting training|training loss)")
BATCH_RE = re.compile(
    r"Epoch\s+(\d+), Batch (\d+)/\d+, Rank\s+(\d+).*?\n\s*Data: (\S+)\s*, .*?"  # name is padded to 32
    r"data ([\d.]+)s, fwd ([\d.]+)s, bw ([\d.]+)s"
)


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def p90(xs):
    return sorted(xs)[int(0.9 * (len(xs) - 1))]


def parse_run(run_dir):
    starts, ends = {}, {}
    per_dset = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(run_dir, "*.out"))):
        text = open(path, errors="replace").read()
        for ts, epoch, what in EPOCH_RE.findall(text):
            t, e = parse_ts(ts), int(epoch)
            if what == "starting training":
                starts[e] = min(starts.get(e, t), t)
            else:
                ends[e] = max(ends.get(e, t), t)
        for epoch, batch, _rank, dset, data, fwd, bw in BATCH_RE.findall(text):
            if int(epoch) == 1 or int(batch) == 1:
                continue
            for k, v in (("data", data), ("fwd", fwd), ("bw", bw)):
                per_dset[dset][k].append(float(v))
    epoch_s = {e: (ends[e] - starts[e]).total_seconds() for e in starts if e in ends}
    return epoch_s, per_dset


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "p9_roofline_"
    runs = sorted(d for d in glob.glob(os.path.join(LOG_ROOT, f"*{pattern}*")) if os.path.isdir(d))
    if not runs:
        sys.exit(f"No run directories matching *{pattern}* under {LOG_ROOT}")

    print(f"{'run':<58}{'epoch 1 s':>10}{'mean s/ep (2+)':>16}{'GPU-h/ep':>10}")
    print("-" * 94)
    details = []
    for run in runs:
        name = os.path.basename(run)
        epoch_s, per_dset = parse_run(run)
        later = [s for e, s in sorted(epoch_s.items()) if e > 1]
        e1 = f"{epoch_s[1]:.0f}" if 1 in epoch_s else "-"
        mean = f"{statistics.mean(later):.0f}" if later else "-"
        gpuh = f"{statistics.mean(later) * 4 / 3600:.2f}" if later else "-"  # N1 = 4 GPUs
        print(f"{name:<58}{e1:>10}{mean:>16}{gpuh:>10}")
        details.append((name, per_dset))

    for name, per_dset in details:
        if not per_dset:
            continue
        print(f"\n{name}: per-dataset microbatch timing (sampled every log_interval; epoch 1 excluded)")
        print(f"  {'dataset':<36}{'n':>6}{'data med':>10}{'data p90':>10}{'fwd med*':>10}{'bw med*':>10}")
        for dset, t in sorted(per_dset.items(), key=lambda kv: -statistics.median(kv[1]["data"])):
            print(
                f"  {dset:<36}{len(t['data']):>6}{statistics.median(t['data']):>10.4f}"
                f"{p90(t['data']):>10.4f}{statistics.median(t['fwd']):>10.3f}"
                f"{statistics.median(t['bw']):>10.3f}"
            )
    print("\n* not CUDA-synchronised: launch time, not GPU time. Only `data` and s/epoch are reliable.")


if __name__ == "__main__":
    main()
