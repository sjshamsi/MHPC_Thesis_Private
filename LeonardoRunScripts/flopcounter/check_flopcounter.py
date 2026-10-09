"""Check the FLOP-counter GPU test after fc_run_1/2/3.sh have all finished.

Compares three independent records of the same 6-epoch run:
  * what W&B received: the offline history in logs/wandb/offline-run-*-<run id>/run-*.wandb,
    one directory per SLURM job, merged by epoch;
  * the checkpoints: checkpoints/step_N/metadata.pt["flop_accountant"];
  * the analytical estimator: scripts/count_WALRUS_flops.py fc_run_3.sh --exact, per epoch
    (replays the sampler's dataset draws; same formulas as FlopAccountant).

Checks (relative tolerance 1e-9):
  A  one W&B run id across 3 offline runs; epochs 1..6 each logged once; jobs start at 1, 3, 5
  B  cumulative = previous cumulative + this epoch, for every epoch incl. across resumes
  C  per-epoch FLOPs, tokens, proj stages, processor, enc+dec fraction, FLOPs/micro-batch
     equal the estimator's exact values
  D  recompute (cumulative minus proj stages minus processor) equals the estimator's, and > 0
  E  each checkpoint's saved counters equal W&B's cumulative values at that epoch
  F  repetition_factor = tokens_seen / T_unique_model (registry, per-axis BCs)

Usage (login node, walrus venv):  python LeonardoRunScripts/flopcounter/check_flopcounter.py
Exit 0 = all PASS.
"""

import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR / "LeonardoRunScripts" / "PT9_2d"))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from gen_flopcounter_runs import RUNS, overrides_for  # noqa: E402
from pt9_substrate import EXPERIMENT_DIR, check_composes  # noqa: E402

from walrus.utils.experiment_utils import get_experiment_name  # noqa: E402
from walrus.utils.flops import GRANULAR_FLOP_KEYS  # noqa: E402

RTOL = 1e-9
LAST_EPOCH = RUNS[-1][2]
EPOCHS = list(range(1, LAST_EPOCH + 1))
JOB_STARTS = [max_epoch - 1 for _, _, max_epoch in RUNS]  # 1, 3, 5
SCRIPT = SCRIPT_DIR / f"fc_run_{RUNS[-1][0]}.sh"

results = []


def check(group, name, ok, detail=""):
    results.append((group, name, bool(ok), detail))


def close(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)


# ------------------------------------------------------------------ locate the run
def experiment_folder() -> Path:
    cfg = check_composes(overrides_for(True, LAST_EPOCH))
    base = Path(EXPERIMENT_DIR) / get_experiment_name(cfg)
    if not base.is_dir():
        sys.exit(f"no experiment folder {base}: has fc_run_1.sh run yet?")
    runs = sorted((p for p in base.iterdir() if p.name.isnumeric()), key=lambda p: int(p.name))
    if not runs:
        sys.exit(f"no run folders under {base}")
    return runs[-1]


def read_wandb_history(run_id: str):
    """[(offline run dir, [history rows])] in job order, from the offline .wandb files."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    out = []
    for path in sorted(glob.glob(f"{EXPERIMENT_DIR}/wandb/offline-run-*-{run_id}/run-*.wandb")):
        ds = datastore.DataStore()
        ds.open_for_scan(path)
        rows, ids = [], set()
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = pb.Record()
            rec.ParseFromString(data)
            kind = rec.WhichOneof("record_type")
            if kind == "run":
                ids.add(rec.run.run_id)
            elif kind == "history":
                rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json)
                             for i in rec.history.item})
        out.append((path, ids, rows))
    return out


def estimator(epoch: int) -> dict:
    cmd = [sys.executable, "-W", "ignore", str(REPO_DIR / "scripts" / "count_WALRUS_flops.py"),
           str(SCRIPT), "--exact", "--json", "--start-epoch", str(epoch), "--end-epoch", str(epoch)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_DIR).stdout
    return json.loads(out[out.index("{"):])


# ------------------------------------------------------------------ main
def main():
    folder = experiment_folder()
    run_id = (folder / "wandb_run_id.txt").read_text().strip()
    print(f"experiment folder  {folder}\nW&B run id         {run_id}\n")

    # --- A: one W&B run across three jobs
    jobs = read_wandb_history(run_id)
    check("A", "3 offline runs (one per job)", len(jobs) == 3, f"found {len(jobs)}")
    check("A", "all carry the folder's run id", all(ids == {run_id} for _, ids, _ in jobs),
          str([sorted(ids) for _, ids, _ in jobs]))
    by_epoch, starts = {}, []
    for _, _, rows in jobs:
        train_rows = [r for r in rows if "epoch_flops" in r]
        if train_rows:
            starts.append(min(int(r["epoch"]) for r in train_rows))
        for r in train_rows:
            by_epoch.setdefault(int(r["epoch"]), []).append(r)
    check("A", f"epochs {EPOCHS[0]}..{EPOCHS[-1]} each logged exactly once",
          sorted(by_epoch) == EPOCHS and all(len(v) == 1 for v in by_epoch.values()),
          f"logged {sorted((e, len(v)) for e, v in by_epoch.items())}")
    check("A", f"jobs start at epochs {JOB_STARTS}", starts == JOB_STARTS, f"started at {starts}")
    if sorted(by_epoch) != EPOCHS:
        return report()
    log = {e: by_epoch[e][0] for e in EPOCHS}

    # --- B: continuity of the cumulative counters, including across resumes
    for key, per_epoch in [("cumulative_flops", "epoch_flops")] + [
        (f"cumulative_{k}_flops", f"epoch_{k}_flops") for k in GRANULAR_FLOP_KEYS
    ]:
        prev, bad = 0.0, []
        for e in EPOCHS:
            if not close(log[e][key], prev + log[e][per_epoch]):
                bad.append(e)
            prev = log[e][key]
        check("B", f"{key}[e] = [e-1] + {per_epoch}[e]", not bad, f"breaks at epochs {bad}" if bad else "")
    toks = [log[e]["tokens_seen"] for e in EPOCHS]
    check("B", "tokens_seen increases every epoch", all(b > a for a, b in zip([0] + toks, toks)), str(toks))

    # --- C/D: accuracy against the estimator's exact replay
    exp = {e: estimator(e) for e in EPOCHS}
    prev_tok, prev_proc = 0.0, 0.0
    for e in EPOCHS:
        j = exp[e]["job"]
        got_tokens = log[e]["tokens_seen"] - prev_tok
        got_proc = log[e]["cumulative_processor_flops"] - prev_proc
        exp_enc, exp_dec = j["flops_encoder"], j["flops_decoder"]
        got_enc = log[e]["epoch_encoder_proj1_flops"] + log[e]["epoch_encoder_proj2_flops"]
        got_dec = log[e]["epoch_decoder_proj1_flops"] + log[e]["epoch_decoder_proj2_flops"]
        rows = [
            ("epoch_flops", log[e]["epoch_flops"], j["flops_total"]),
            ("tokens", got_tokens, j["tokens_seen"]),
            ("encoder (proj1+proj2)", got_enc, exp_enc),
            ("decoder (proj1+proj2)", got_dec, exp_dec),
            ("processor", got_proc, j["flops_processor"]),
            ("enc_dec_flop_fraction", log[e]["enc_dec_flop_fraction"], j["forward_enc_dec_fraction"]),
            ("flops_per_microbatch", log[e]["flops_per_microbatch"], j["flops_total"] / j["microbatches"]),
        ]
        for name, got, want in rows:
            check("C", f"epoch {e}: {name}", close(got, want), f"logged {got:.9e} expected {want:.9e}")
        recompute = log[e]["epoch_flops"] - got_enc - got_dec - got_proc
        check("D", f"epoch {e}: recompute", close(recompute, j["flops_recompute"]) and recompute > 0,
              f"logged {recompute:.9e} expected {j['flops_recompute']:.9e}")
        prev_tok, prev_proc = log[e]["tokens_seen"], log[e]["cumulative_processor_flops"]

    # --- E: checkpoints carry the same counters
    for e in EPOCHS:
        meta_file = folder / "checkpoints" / f"step_{e}" / "metadata.pt"
        if not meta_file.exists():
            check("E", f"step_{e} metadata", False, f"missing {meta_file}")
            continue
        saved = torch.load(meta_file, weights_only=False).get("flop_accountant")
        ok = saved is not None and close(saved["cumulative_flops"], log[e]["cumulative_flops"]) \
            and saved["cumulative_tokens"] == int(log[e]["tokens_seen"]) \
            and close(saved["cumulative_processor_flops"], log[e]["cumulative_processor_flops"])
        check("E", f"step_{e} metadata == W&B cumulative", ok, "no flop_accountant key" if saved is None else "")

    # --- F: repetition factor
    unique = exp[LAST_EPOCH]["job"]["unique_model_tokens"]
    for e in EPOCHS:
        check("F", f"epoch {e}: repetition_factor", close(log[e]["repetition_factor"], log[e]["tokens_seen"] / unique),
              f"logged {log[e]['repetition_factor']:.6e}, tokens/unique {log[e]['tokens_seen'] / unique:.6e}")

    print(f"\n{'epoch':>6}{'epoch_flops (TF)':>18}{'expected (TF)':>15}{'tokens_seen':>14}{'cumulative (TF)':>17}")
    for e in EPOCHS:
        print(f"{e:>6}{log[e]['epoch_flops'] / 1e12:>18.4f}{exp[e]['job']['flops_total'] / 1e12:>15.4f}"
              f"{log[e]['tokens_seen']:>14,.0f}{log[e]['cumulative_flops'] / 1e12:>17.4f}")
    return report()


def report():
    print()
    for group, name, ok, detail in results:
        print(f"  {group}  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""))
    n_fail = sum(not ok for _, _, ok, _ in results)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'} ({len(results)} checks)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    os.chdir(REPO_DIR)
    main()
