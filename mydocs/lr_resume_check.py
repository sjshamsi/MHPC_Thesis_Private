"""Replay train.py's LR-scheduler handling: uninterrupted run vs. resume after epoch E, with and
without the resume fix. Plain torch, no data or GPU. See determinism_code_changes.md §2.5.

    python mydocs/lr_resume_check.py      # walrus venv, from the repo root
"""
import torch
from walrus.optim.schedulers import InverseSqrtLinearRamps, InverseSqrtLinearWarmupSqrtCooldown

BASE, MAX_EPOCH = 2e-4, 100
KW = dict(max_epochs=MAX_EPOCH, warmup_epochs=10, cooldown_epochs=10, warmup_lr_factor=0.1, cooldown_lr_factor=0.001)

def opt():
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=BASE)

def uninterrupted(cls):
    o = opt(); s = cls(o, **KW, last_epoch=-1)
    lrs = {}
    for epoch in range(1, MAX_EPOCH + 1):
        lrs[epoch] = o.param_groups[0]["lr"]     # lr this epoch trains with
        s.step()                                  # end of train_one_epoch
    return lrs, o, s

def resumed(cls, E, fix):
    # Job 1: epochs 1..E, then the checkpoint (saved after the epoch's scheduler.step()).
    o = opt(); s = cls(o, **KW, last_epoch=-1)
    for epoch in range(1, E + 1):
        s.step()
    saved = o.state_dict()
    # Job 2: train.py resume path.
    o2 = opt(); o2.load_state_dict(saved)
    start_epoch = E + 1; last_epoch = start_epoch - 1
    resumed_lrs = [g["lr"] for g in o2.param_groups]
    s2 = cls(o2, **KW, last_epoch=max(-1, last_epoch - 1))
    if fix and last_epoch > 0:
        for g, lr in zip(o2.param_groups, resumed_lrs): g["lr"] = lr
        s2._last_lr = resumed_lrs
    lrs = {}
    for epoch in range(start_epoch, MAX_EPOCH + 1):
        lrs[epoch] = o2.param_groups[0]["lr"]; s2.step()
    return lrs

for cls in (InverseSqrtLinearRamps, InverseSqrtLinearWarmupSqrtCooldown):
    ref, _, _ = uninterrupted(cls)
    print(f"\n{cls.__name__}")
    print(f"  {'resume after':>12} | {'ref lr next ep':>14} | {'no fix: ratio next / at ep 100':>30} | {'with fix: max |ratio-1|':>22}")
    for E in (1, 5, 10, 11, 50, 90, 95):
        nf, wf = resumed(cls, E, False), resumed(cls, E, True)
        r_next, r_end = nf[E + 1] / ref[E + 1], nf[MAX_EPOCH] / ref[MAX_EPOCH]
        dev = max(abs(wf[e] / ref[e] - 1) for e in wf)
        print(f"  {E:>12} | {ref[E+1]:14.3e} | {r_next:13.4f} / {r_end:13.4f} | {dev:22.1e}")
