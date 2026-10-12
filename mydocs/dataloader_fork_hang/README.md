# DataLoader worker hang after an async checkpoint save (investigation, Oct 2026)

**Status: diagnosed; fix proposed, not merged.** §8a describes a zero-cost fix: fork the next
epoch's workers before the async save starts. It is saved as `fork_before_save.patch` and passes the
CPU tests, but it isn't in `walrus/` and hasn't been tested on GPU yet.

## 1. Summary

The GPU determinism check (`LeonardoRunScripts/PT9_2d/Determinism/`, run **B**) hung 3 times out
of 3 at the start of epoch 2. A diagnostic run captured every process's stack. The cause is a
**fork-while-threads-running deadlock**:

1. At the end of an epoch, `CheckPointer.save_if_necessary` saves the checkpoint with
   `dcp.async_save` (`walrus/trainer/checkpoints.py:401`). It stages the state synchronously,
   then writes it from a **background thread**.
2. Under fsdp, a checkpoint is saved **every epoch**: the `last` checkpoint is appended
   unconditionally (`checkpoints.py:367-369`), whatever `checkpoint_frequency` says.
3. The next epoch starts about 1 s later. `iter(dataloader)` (`walrus/trainer/training.py:1067`)
   **forks** `data_workers` (here 10) new worker processes while the save thread is still
   running (§4.4).
4. The save thread does CPU tensor work on the rank's OpenMP thread pool (`OMP_NUM_THREADS =
   $SLURM_CPUS_ON_NODE` = 32 in every PT9 script). A forked child inherits the OpenMP runtime's
   memory, including any internal lock a busy pool thread holds at that instant, but not the
   threads themselves.
5. The child's first OpenMP call, `torch.set_num_threads(1)` at
   `torch/utils/data/_utils/worker.py:259`, blocks forever on that lock.
6. The worker never serves its batches, and its rank waits forever in `DataLoader.__next__`
   (no DataLoader timeout is set). The other ranks wait in FSDP's `all_gather` for that rank,
   and the job hangs. NCCL's timeout in these runs is **30 min** (§6), so without diagnostics
   it produces no error before a 30-min job limit.

The mechanism itself is upstream WALRUS + PyTorch behaviour: async checkpoint saving plus forked
DataLoader workers. It is **timing-dependent**, and it has only been seen in the tiny model:
- **Small model (320/6/768):** it struck tiny-model runs repeatedly on the current code.
- **Large models:** 0 of 20 exposed epoch boundaries at 200 M and 1.2 B parameters, over two
  rounds (§7, §7f).
- **Earlier production runs:** 0 of roughly 145 boundaries (§6).

**The code before the determinism changes did not hang, so those changes are implicated (§7e, §7f).**
- **Current code:** B's exact script hung in **7 of 8** runs, with and without
  `TORCH_DISTRIBUTED_DEBUG=DETAIL`, and the stack dumps show the same frozen worker every time.
- **Commit `656c00c`:** the same script completed **4 of 4**, and the two runs without `DETAIL`
  also resumed successfully.
- **Chance:** if both versions hung equally often, all 7 hangs falling on the current code would
  happen with probability C(8,7)/C(12,7) = 8/792 ≈ 1.0 % (§7f).

That is not the whole picture. On the current code, other tiny-model runs did *not* hang: A 0 of 3,
and the 6-epoch controls 0 of 20 boundaries. A's first launch is also script-identical to B's, and in
round 2 a B run passed its first launch for the first time (§7f). So "the current code" alone isn't
the trigger, and the mechanism linking the changes to the hang is unknown. Earlier sections (§7b,
§7c) concluded the changes were not the cause; §7e and §7f supersede that.

**The hang rate also changes from day to day (§8b).** On 2026-10-12, unfixed `flopcount`, with
code identical to the runs that hung and the same save/fork overlap, passed B's script **8 of 8**.
Branch `fulldet` passed 3 of 3 the same day (its README §7g). Across days, B's script hung 7 of 8
times on 2026-10-09/11 and 0 of 11 on 2026-10-12. So the code-version contrast in §7f is confounded
by date. Only §7e's same-day contrast (2026-10-10: current code 2 of 2 hung, `656c00c` 0 of 4)
stands.

---

## 2. Timeline of the runs

All runs: PT9 substrate (`LeonardoRunScripts/PT9_2d/pt9_substrate.py`), 1 node × 4 GPUs, fsdp,
`data_workers=10`, seed 0, debug QOS (`boost_qos_dbg`, 30 min). A and B are identical except
that B, after its first launch, sets `step_2` aside and resumes (the determinism check).

| job | run | shape | diagnostics | node | outcome | logs here? |
|---|---|---|---|---|---|---|
| 59843236 | A | 320/6/768 | none | lrdn0382 | COMPLETED, 7 min 48 s | no (deleted in the first cleanup) |
| 59843238 | B | 320/6/768 | none | lrdn0788 | **hung** at epoch 2 batch 1 of launch 1 from 23:04:40; no error for 24 min; killed at the 30 min limit | no (deleted); key facts in §4.1 |
| 59845924 / 59845925 | A / B | 320/6/768 | first attempt | — | crashed at start-up in under 90 s: a bug in the new `WALRUS_PG_TIMEOUT_SEC` code (it captured the `elif` chain in `configure_distribution`; fixed, all branches checked) | no (deleted) |
| 59846208 | A | 320/6/768 | DETAIL, 180 s timeout, 420 s watchdog | lrdn1085 | COMPLETED, 7 min 59 s | `slurm_logs/59846208_A_completed.*` |
| 59846210 | B | 320/6/768 | DETAIL, 180 s timeout, 420 s watchdog | lrdn3284 | **hung** at epoch 2 batch 1; the 180 s gloo timeout fired on ranks 1–3 (§4.2) | `slurm_logs/59846210_B_hang_gloo_timeout.*` |
| 59847084 | B | 320/6/768 | DETAIL, 600 s timeout, **180 s watchdog** | lrdn0456 | **hung** at epoch 2 batch 1; the watchdog dumped all 44 processes' stacks (§4.3) | `slurm_logs/59847084_B_hang_stack_dumps.*`, `stack_dumps/` |
| 59848085 | hang test | 768/12/2048 | DETAIL, 600 s timeout, 300 s watchdog | lrdn2015 | COMPLETED, 16 min 17 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59848085_hang_anchor_completed.*` |
| 59848086 | hang test | 1408/40/5632, GC 2 | DETAIL, 600 s timeout, 300 s watchdog | lrdn2282 | COMPLETED, 24 min 09 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59848086_hang_fullDims_completed.*` |
| 59850614 | old_tiny_1, **commit 656c00c** | 320/6/768 | DETAIL, 300 s watchdog (no timeout knob at that commit: 30 min) | lrdn3120 | COMPLETED, 12 min 16 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59850614_old_tiny_1_at_656c00c_completed.*` |
| 59850615 | old_tiny_2, **commit 656c00c** | 320/6/768 | as above | lrdn3016 | COMPLETED, 12 min 27 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59850615_old_tiny_2_at_656c00c_completed.*` |
| 59852095 | new_tiny_1, current code (control) | 320/6/768 | as old_tiny, `WALRUS_PG_TIMEOUT_SEC` unset | lrdn0901 | COMPLETED, 11 min 47 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59852095_new_tiny_1_current_code_completed.*` |
| 59852101 | new_tiny_2, current code (control) | 320/6/768 | as above | lrdn1955 | COMPLETED, 11 min 48 s: 6 epochs, 5 exposed boundaries, no hang | `slurm_logs/59852101_new_tiny_2_current_code_completed.*` |
| 59853799 / 59853800 | old_B_1 / old_B_2, **commit 656c00c**, B's exact script | 320/6/768 | DETAIL, 300 s watchdog | lrdn0668 / lrdn1040 | **FAILED** after ~10 min: first launch passed epoch 2 (no hang); the resume launch crashed loading `step_1` (§7d) | `slurm_logs/59853799_*`, `59853800_*` |
| 59854572 / 59854574 | old_B_nd_1 / _2, **commit 656c00c**, B's script, DETAIL off | 320/6/768 | 300 s watchdog | lrdn0173 / lrdn1152 | **COMPLETED**, 14 min each: first launch and resume both fine (§7e) | `slurm_logs/59854572_*`, `59854574_*` |
| 59856184 / 59856185 | cur_B_nd_1 / _2, **current code**, B's script, DETAIL off | 320/6/768 | 300 s watchdog | — | **hung** at epoch 2 batch 1; 40 / 44 stack dumps; FAILED after 9 min (§7e) | `slurm_logs/59856184_*`, `59856185_*`, `stack_dumps/cur_B_nd_summary.txt` |
| *Round 2, 2026-10-11, current code `dd08774`, all artifacts cleaned first (§7f):* | | | | | | |
| 60036829 | B | 320/6/768 | DETAIL, 600 s timeout, 300 s watchdog | lrdn2995 | **first launch completed** (both epochs, `step_2` saved); the resume launch crashed loading `step_1` with the §7d `DETAIL` error; FAILED after 11 min | `slurm_logs/60036829_B_round2_*` |
| 60036831 | hang_anchor | 768/12/2048 | as B | lrdn1458 | COMPLETED, 17 min: 5 exposed boundaries, no hang | `slurm_logs/60036831_*` |
| 60037693 | cur_B_nd_1 | 320/6/768 | DETAIL off, 300 s watchdog | lrdn0558 | **hung** at epoch 2 batch 1 of launch 1; 44 stack dumps; FAILED after 10 min | `slurm_logs/60037693_*` |
| 60038181 | hang_fullDims | 1408/40/5632, GC 2 | as B | lrdn1385 | COMPLETED, 24 min: 5 exposed boundaries, no hang | `slurm_logs/60038181_*` |
| 60038527 | A | 320/6/768 | as B | lrdn0050 | COMPLETED, 8 min | `slurm_logs/60038527_*` |
| 60039144 / 60040787 | new_tiny_1 / _2 | 320/6/768 | DETAIL, 300 s watchdog | lrdn0920 / lrdn0146 | COMPLETED, 13 min each: 5 exposed boundaries each, no hang | `slurm_logs/60039144_*`, `60040787_*` |
| 60040489 | cur_B_nd_2 | 320/6/768 | DETAIL off, 300 s watchdog | lrdn3196 | **hung** at epoch 2 batch 1 of launch 1; 44 stack dumps; FAILED after 10 min | `slurm_logs/60040489_*` |

The hang tests use `max_samples=100` (anchor) and `30` (fullDims), with `checkpoint_frequency=0`
(one rolling `last` checkpoint). A and B use `max_samples=100`, 2 epochs and
`checkpoint_frequency=1`. Round 1 used six different nodes and round 2 eight more, so a faulty
node is ruled out.

## 3. Diagnostics used

All set in the generated job scripts (`job_scripts/`, from `gen_pt9_determinism_runs.py`):

- `TORCH_DISTRIBUTED_DEBUG=DETAIL`: before every collective, all ranks are checked for issuing
  the same operation (through a `monitoredBarrier` on a gloo helper group).
- `WALRUS_PG_TIMEOUT_SEC`: an opt-in knob in `walrus/utils/distribution_utils.py` that creates the
  default process group with a short timeout for both gloo and NCCL. When unset, behaviour is
  unchanged.
- NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`). It
  never produced a trace: the timeouts surfaced through gloo, or the watchdog fired first.
- `PYTHONFAULTHANDLER=1` with a shell watchdog: if the SLURM log stops growing for N seconds, it
  sends SIGABRT to every `train.py` worker process (not `torchrun`), so each one prints every
  thread's Python stack to `.err`. For the dump to happen, the watchdog must fire before the
  collective timeout (job 59847084: 180 s < 600 s).

## 4. Evidence

### 4.1 First hang (job 59843238; logs deleted, recorded during the investigation)

- Launch 1 trained epoch 1 normally and saved `step_1` (complete; same file sizes as A's) at
  23:04:35. Epoch 2 started at 23:04:35.9, and the rank-0 save callbacks ("Link checkpoint
  best/last") ran at 23:04:37, i.e. during epoch 2.
- All 4 ranks logged epoch 2, batch 1 at 23:04:40. Nothing after that on any rank, and nothing in
  `.err` for 24 minutes. The job was killed at the 30 min limit (CANCELLED).
- A (job 59843236), identical, logged batch 11 two seconds after batch 1 and completed.

### 4.2 Collective timeout (job 59846210)

`slurm_logs/59846210_B_hang_gloo_timeout.err`:

```
[rank1]: RuntimeError: Rank 1 successfully reached monitoredBarrier, but received errors while
         waiting for send/recv from rank 0. Please check rank 0 logs for faulty rank.
[rank1]:  Original exception: [.../gloo/transport/tcp/unbound_buffer.cc:133] Timed out waiting
         180000ms for send operation to complete
```

- Ranks 1–3 were in `FullyShardedDataParallel.forward` → `_all_gather_flat_param` →
  `all_gather_into_tensor`, i.e. epoch 2's second micro-batch.
- Rank 0 never joined, and printed nothing: `torchrun` sent it SIGTERM first.
- `grad_acc_steps=4`, so between batch 1 and batch 2 there is no optimizer sync, and rank 0 only
  calls `next(data_iter)`.

### 4.3 Stack dumps (job 59847084)

`stack_dumps/59847084_summary.txt` groups the 44 dumps (4 ranks + 40 workers) by main-thread stack:

| processes | where | meaning |
|---|---|---|
| 3 | `distributed_c10d.all_gather_into_tensor` ← FSDP `_pre_forward` ← `training.py:514 rollout_model` ← `training.py:1115` | ranks waiting for the missing rank |
| 1 | `dataloader._try_get_data` → `queue.get` ← `__next__` ← `training.py:1092` | the missing rank: waiting for a batch |
| 39 | `worker._worker_loop` → `queue.get` (multiprocessing) | idle workers (normal) |
| **1** | **`torch/utils/data/_utils/worker.py:259`, `torch.set_num_threads(1)`**, ← `process.start` ← `DataLoader.__iter__` ← `training.py:1067` | **a worker forked at epoch start, frozen in its start-up code** |

Full dumps of one process of each kind: `stuck_worker_full_dump.txt`,
`rank_waiting_for_data_full_dump.txt`, `rank_in_all_gather_full_dump.txt`.

### 4.4 Same data in A and B; the save overlaps the fork in every run

- A and B draw identical data. Rank 0's epoch 1, batch 1 loss is 5.8148 in both, and the epoch-1
  losses agree to float noise (`slurm_logs/59846208_*` vs `59846210_*`). The hang therefore isn't
  data- or seed-dependent.
- `save_vs_epoch_timing.txt`: in every surviving run, at every epoch boundary, the save starts
  1–3 s before the next epoch and its completion callbacks run 1–8 s *after* it started, i.e. after
  the fork. In the old 200 M roofline run (job 59146807, not copied) the save also finished 2.2 s
  into epoch 2.

## 5. Mechanism, in more detail

- `async_save` (torch 2.5.1, `torch/distributed/checkpoint/state_dict_saver.py:215-250`) stages the
  state dict to CPU in the calling thread, then submits `save(...)` to a one-thread
  `ThreadPoolExecutor`. That thread plans, serialises and writes the CPU tensors, and the
  completion callbacks (`save_metadata`, `link_checkpoint`; rank 0 only) run in it too.
- `train()` waits for that future only right before the *next* save (`training.py:1334-1338`), so
  it runs during the next epoch's start, when `iter(dataloader)` forks the workers. The DataLoader
  is not persistent and uses the default `fork` start method.
- `worker.py` runs `torch.set_num_threads(1)` right after the fork, before seeding. With GNU OpenMP,
  forking while another thread of the parent is inside a parallel region can leave the child's
  runtime locked. This is the general fork-safety limitation, not a WALRUS-specific bug.

## 6. Why it never showed up in earlier runs

- **Earlier runs were exposed just as much.** Every fsdp run saves `last` asynchronously every
  epoch. Earlier fsdp training runs crossed roughly 145 such boundaries:
  - smallest7 N1/N4/N8, reaching epochs 34, 34 and 31 over three 24 h jobs each;
  - fullDims_6DS, N4 and N8, reaching epochs 16, 16 and 8;
  - the 200 M roofline (2 boundaries), fc_H256 (5) and torchprofile (1).

  Every TIMEOUT job was still logging within its normal cadence when killed (last log line 3–9
  min before the end). The one FAILED job (fullDims_6DS_N8, 52966529) died of an NCCL remote
  error 27 s after its last log line: a crashed remote process, not this hang.
- **The NCCL timeout is 30 min, not 10.**
  `slurm_logs/52966529_fullDims_6DS_N8_excerpt_nccl_timeout_30min.txt` shows
  `Timeout(ms)=1800000`. That's why the first hang raised nothing before the 30 min job limit.
  In a 24 h production job, a hang would surface as an NCCL timeout error about 30 min after it
  starts.
- **Hypothesis for the different rates (not tested):** the deadlock needs a fork to land at the
  instant a parent OpenMP thread holds the runtime's lock.
  - With a 320/6/768 model the whole save is a ~1–2 s burst of CPU tensor work, and all 10 forks
    happen inside it.
  - With large models the save is dominated by disk writes (GBs), so a fork rarely catches an
    OpenMP region in progress.
  - The determinism changes (per-epoch re-seeding) only seed generators and don't touch threads or
    checkpointing. Even so, the tiny model at `656c00c` did not hang in 10 boundaries (§7b), so a
    contribution from them is not ruled out.

## 7. Larger-model test

Jobs 59848085 (768/12/2048, ≈200 M) and 59848086 (1408/40/5632, ≈1.2 B, gradient checkpointing
every 2 blocks): 6 epochs each, so 5 epoch boundaries each, all with the save still in flight at
the fork (`save_vs_epoch_timing.txt`). **No hang in any of the 10.** Both completed.

| | runs / boundaries exposed | hangs |
|---|---|---|
| 320/6/768 (A and B) | 5 runs, 1 boundary each | 3 (all B) |
| 768/12/2048 | 5 (+ 5 in round 2, §7f) | 0 |
| 1408/40/5632 | 5 (+ 5 in round 2, §7f) | 0 |
| earlier production runs | ~145 | 0 |

## 7b. Same tiny model on the code before the determinism changes (commit `656c00c`)

**Setup.**
- Commit `656c00c` checked out as a git worktree (`/leonardo/home/userexternal/sshamsi0/MHPC_Thesis_656c00c`)
  and run from there with `PYTHONPATH` pointing at it.
- Each job checked the commit and that `import walrus` resolved to the worktree before launching.
  The logs confirm it: warnings come from the worktree's files, and Hydra wrote its outputs there.
- The only code differences from the current tree are the 7 files of the determinism change.
- Config: B's tiny setup (320/6/768, `max_samples=100`, `checkpoint_frequency=1`,
  `data_workers=10`), but `max_epoch=6`. Two jobs, 59850614 and 59850615.

**Result.** Both completed. All 10 boundaries were exposed: the save was still running 0.9–2.3 s
after the next epoch started (`save_vs_epoch_timing.txt`, last section). **0 hangs.**

| | exposed boundaries | hangs |
|---|---|---|
| current code, 320/6/768 (A, B; `max_epoch=2`) | 5, all epoch 1→2 | 3 |
| `656c00c`, 320/6/768 (`max_epoch=6`) | 10: 2 at epoch 1→2, 8 later | 0 |

**Reading.**
- If both groups had the same per-boundary rate, the chance that all 3 hangs fall in the
  current-code group is C(5,3)/C(15,3) = 10/455 ≈ 2.2 % (Fisher's exact test, one-sided).
- That leans toward the current code raising the rate. Two confounders keep it from being a
  conclusion:
  - every current-code boundary was the *first* (epoch 1→2), while only 2 old-commit boundaries
    were; restricted to 1→2 boundaries it's 3 of 5 against 0 of 2, which says almost nothing;
  - the current-code runs used `max_epoch=2` and the old ones 6.
- No mechanism has been found: the changes touch neither threads, OpenMP nor checkpointing, and
  the save/fork timing is the same in both groups.

**Settling it:** done in §7c.

## 7c. Control: the same config on the current code

Jobs 59852095 and 59852101 use the old_tiny jobs' exact config on the current code. The generated
scripts differ from the old_tiny ones only in:
- the description line;
- the worktree redirect and its checks;
- the `WALRUS_PG_TIMEOUT_SEC` export, left out here because `656c00c` ignores it, so both groups
  build the process group the same way.

Both completed. All 10 boundaries were exposed (the save was still running 1.0–1.6 s after the next
epoch started). **0 hangs.**

All tiny-model (320/6/768) runs:

| runs | code | exposed boundaries | hangs |
|---|---|---|---|
| B: 59843238, 59846210, 59847084 | current | 3 (epoch 1→2) | **3** |
| A: 59843236, 59846208 | current | 2 (epoch 1→2) | 0 |
| new_tiny: 59852095, 59852101 | current | 10 (2 at 1→2) | 0 |
| old_tiny: 59850614, 59850615 | `656c00c` | 10 (2 at 1→2) | 0 |

**Reading.**
- **The code version doesn't matter** in this config: 0 of 10 against 0 of 10. (Superseded by §7e:
  B's own config does separate the two code versions.)
- **B is the outlier.** If all 25 tiny boundaries were alike, all 3 hangs would land on B's 3
  boundaries with probability 1/C(25,3) ≈ 1/2300.
- **Nothing obvious explains it.** Before training, B's script is identical to A's except for the
  job/run name and folder: the checkpoint shuffle and resume launch come after the first launch,
  which the hung runs never finished. B and A also share the model, `max_epoch=2`, the timing and
  the filesystem. B was always the second of a pair submitted and started together with A.
  Neither the name nor the order has an obvious way to affect a worker fork, and with 3 events,
  chance can't be excluded: the 1/2300 assumes the boundaries are otherwise exchangeable.

## 7d. B's exact script on the old code (commit `656c00c`)

Jobs 59853799 and 59853800 (`old_B_1`, `old_B_2`): B's script unchanged except for the worktree
redirect, the job name, and the ignored timeout export. 2 epochs, `checkpoint_frequency=1`, then
`step_2` set aside and a second launch to resume.

- **First launch: no hang in either run.** Both trained epochs 1 and 2 on all 4 ranks and saved
  `step_2`, which the script then set aside as `step_2_first`. No watchdog trigger. B hung in this
  same step in 3 of 3 runs on the current code.
- **Resume launch: crashed in both**, while loading `step_1`:
  `RuntimeError: Backend nccl does not support allgather_into_tensor_coalesced`, raised from the
  optimizer-state gather in `torch/distributed/fsdp/_optim_utils.py` ← `_functional_collectives.all_gather_tensor`,
  wrapped by `checkpoints.py:253` ("Failed to load checkpoint").
  - This is a separate problem from the hang. `checkpoints.py` is identical at `656c00c` and now.
  - **Probable cause, not confirmed:** every one of these diagnostic jobs set
    `TORCH_DISTRIBUTED_DEBUG=DETAIL`, which wraps process groups in `_ProcessGroupWrapper`
    (`distributed_c10d.py:4256`). Production runs did not set it, and they resumed fine
    (e.g. smallest7 N4, "Resume from epoch 11"). Untested: a rerun without `DETAIL`.
  - **Consequence:** the A/B determinism *comparison* (`compare.py`) has never produced a result
    on GPU. Current-code B always hung before its resume, and old-code B crashed at it.

**Reading.** On the one configuration that hung, the old code passed 2 of 2 first launches, against
0 of 3 passes for the current code. That is a 5-run contrast (Fisher's exact, one-sided, 1/C(5,3) =
0.1), and the §7c controls showed 0 of 10 on both code versions for a different config. The
earlier B hangs remain unexplained, and a code-version effect is again not ruled out.

## 7e. Reruns without `TORCH_DISTRIBUTED_DEBUG`: the code version separates B's outcome

Four jobs, all B's exact script (2 epochs, a checkpoint every epoch, then `step_2` set aside and a
second launch to resume), `TORCH_DISTRIBUTED_DEBUG` unset, no timeout knob, 300 s stack-dump
watchdog. Two on the `656c00c` worktree and two on the current code.

| job | code | outcome |
|---|---|---|
| 59854572 `old_B_nd_1` | `656c00c` | **COMPLETED**, 14 min. Both launches finished; the resume loaded `step_1` and wrote `step_2`. |
| 59854574 `old_B_nd_2` | `656c00c` | **COMPLETED**, 14 min, same. |
| 59856184 `cur_B_nd_1` | current | **hung** at epoch 2, batch 1; the watchdog dumped 40 processes; FAILED after 9 min. |
| 59856185 `cur_B_nd_2` | current | **hung** at epoch 2, batch 1; 44 processes dumped; FAILED after 9 min. |

Stack dumps (`stack_dumps/cur_B_nd_summary.txt`): in each hung run, 1 worker is frozen in
`worker.py:259`, `torch.set_num_threads(1)`, and 1 rank waits in `DataLoader.__next__`. That's the
same signature as §4.3, so the hang does not depend on `DETAIL`.

**Resume crash explained.** The old-code resume now works without `DETAIL`, which supports the §7d
suspicion that the `DETAIL` wrapper broke the checkpoint load.

**Tally, B's exact script** (2 epochs):

| code | runs | hung | notes |
|---|---|---|---|
| current | 5 | **5** | 3 earlier (no diagnostics / `DETAIL`) + 2 here (`DETAIL` off) |
| `656c00c` | 4 | **0** | 2 with `DETAIL` (first launch ok, resume crashed, §7d) + 2 without (complete) |

If both versions were equally likely to hang, all 5 hangs falling on the current code has
probability 1/C(9,5) = 1/126 ≈ 0.8 % (Fisher's exact, one-sided).

**What it does not explain.** On the current code, other tiny-model runs did not hang: A (0 of 2) and
the 6-epoch controls (0 of 10). A's first launch is script-identical to B's up to the point where B
hung. So the current code is necessary for this hang, not sufficient. The determinism changes touch
seeding only, and no link to thread or lock state has been found. Candidates (untested):
- a bisect over the 7 changed files, starting with the per-epoch `seed_all` in `Trainer.train`
  (the only change that runs between the checkpoint save and the worker fork);
- more A-type runs on the current code, to see whether A's 0 of 2 was chance.

**compare.py.** No current-code B run reached its resume, so there's still no GPU determinism
result for the fix. As a by-product, on the old code the resumed weights differ from the first
attempt by 6.7–6.8e-4 (relative), against a float-noise floor of 4.7e-4 between two uninterrupted
same-seed runs. At this tiny size GPU noise is too large for that to separate "resume replays other
data" from float noise (ratio 1.4), so it's inconclusive either way.

## 7f. Round 2: all artifacts cleaned, the 8 current-code jobs rerun (2026-10-11)

**Setup.**
- All 14 earlier determinism jobs' SLURM logs, offline W&B runs, run folders (49 GB of
  checkpoints) and Hydra outputs were deleted first. Their logs had been copied here.
- The 8 current-code scripts were resubmitted unchanged on `flopcount` (`dd08774`), 2 at a time,
  because the debug QOS allows only 2 submitted jobs per user. Each pair put a tiny job next to a
  large one.
- The 6 `old_*` scripts could not run: they need the `656c00c` worktree, which was removed.
- Before submitting, the CPU checks passed on the same code: `tests/test_flops.py` (24),
  `tests/test_seed.py` (3), `tests/test_determinism.py` (2), and notebook 5 with every assert.

**Results** (§2, round-2 rows):

| run | shape | boundaries exposed | hangs |
|---|---|---|---|
| hang_anchor | 768/12/2048 | 5 | 0 |
| hang_fullDims | 1408/40/5632 | 5 | 0 |
| new_tiny_1, new_tiny_2 | 320/6/768, 6 epochs | 10 | 0 |
| A | 320/6/768, B's first launch | 1 | 0 |
| B (`DETAIL` on) | 320/6/768 | 1 | **0**: the first B run on the current code to pass its first launch |
| cur_B_nd_1, cur_B_nd_2 (`DETAIL` off) | 320/6/768 | 1 each | **2** |

- **Same hang as before.** Each hung run has 44 stack dumps (`stack_dumps/cur_B_nd_summary.txt`,
  round-2 lines):
  - 1 worker frozen in `worker.py:259`, `torch.set_num_threads(1)`;
  - 1 rank waiting in `DataLoader.__next__`;
  - the 3 other ranks in the next micro-batch's forward (`isotropic_model.py:235`), blocked behind
    the collective the missing rank never joins;
  - the 39 remaining workers idle.
- **The save overlapped the fork at every boundary,** in hung and completed runs alike:
  0.6–1.3 s before the next epoch at 320/6/768, with completion 1.0–2.3 s after it started
  (`save_vs_epoch_timing.txt`, round-2 block). The overlap is the same in A, B and the hung runs, so
  timing alone doesn't separate them.
- **B's resume crashed exactly as old-code B did in §7d:**
  `Backend nccl does not support allgather_into_tensor_coalesced` while loading `step_1`, with
  identical frames (`fsdp/_optim_utils.py:396, :488`, `checkpoints.py:253`). This confirms on the
  current code that `TORCH_DISTRIBUTED_DEBUG=DETAIL` breaks the FSDP checkpoint load. The resume
  bug is the diagnostic, not the code.

**Tallies over both rounds.**

| group | code | runs / boundaries | hangs |
|---|---|---|---|
| B's exact script (B, cur_B_nd) | current | 8 runs | **7** |
| B's exact script (old_B, old_B_nd) | `656c00c` | 4 runs | 0 |
| A (B's first launch under another name) | current | 3 runs | 0 |
| new_tiny (6 epochs) | current | 20 boundaries | 0 |
| old_tiny (6 epochs) | `656c00c` | 10 boundaries | 0 |
| 200 M and 1.2 B | current | 20 boundaries | 0 |

**Reading.**
- **Small model only.** At 200 M and 1.2 B, 20 of 20 boundaries passed with the save in flight,
  plus about 145 production boundaries. Every observed hang is at 320/6/768.
- **Code version.** B's script: 7 of 8 on the current code against 0 of 4 on `656c00c`. Under
  equal rates, that split happens with probability C(8,7)/C(12,7) = 8/792 ≈ 1.0 %. That still
  points at the determinism changes, still with no mechanism.
- **B against A is still unexplained.** Among the 11 current-code first launches of this exact
  config (8 B-script, 3 A), all 7 hangs fall on B-script runs with probability
  C(8,7)/C(11,7) = 8/330 ≈ 2.4 %. B passing once makes "something about B's name or folder" less
  likely, but it can't be excluded.
- **Still no GPU resume result for `compare.py`.** The one B run that reached its resume had
  `DETAIL` on, and that crashes the load. A clean resume result needs a B-type run with `DETAIL`
  off that doesn't hang: on this evidence, about 1 in 8 attempts at the tiny size, or a larger
  model.

## 8. Risk and fix options

- **Risk.** With roughly 165 clean large-model boundaries (about 145 in production, 20 in §7 and
  §7f), the per-boundary probability at production size is bounded at about ≤ 1.8 % (rule of
  three). That doesn't make it negligible. A
  100-epoch run has 99 boundaries, and 1 % per boundary would mean about a 63 % chance of at least
  one hang per run, each costing about 30 min (the NCCL timeout) before the job errors out. With
  auto-resuming chains, the job would then restart from the last checkpoint.
- **Option 0, preferred (§8a):** fork the next epoch's workers *before* the save starts. It
  removes the overlap at no cost.
- **Option 1:** wait for the save before the next epoch forks workers. Move the existing
  `checkpoint_future.result()` (`training.py:1334-1338`) to the top of the epoch loop, before
  `train_one_epoch`. It costs the duration of the background write, from `save_vs_epoch_timing.txt`:
  about 2 s per epoch at 320/6/768, about 9–10 s at 1.2 B (2–3 s staging + 6–8 s writing). This also removes the overlap with the validation loaders'
  forks.
- **Option 2:** start DataLoader workers with `multiprocessing_context="forkserver"` or `"spawn"`, so
  they don't inherit the parent's OpenMP state. That costs a torch import in every worker on
  every epoch, for both train and validation loaders.
- **Option 3:** `persistent_workers=True`, so workers are forked once, before any save. But the
  workers' RNG streams then run on across epochs, so a resume would no longer reproduce
  `dt`/rotation exactly. That undoes part of the per-epoch seeding.
- **Not a fix:** giving `async_save` its own gloo process group. The hang is not in a collective.

## 8a. Proposed fix: fork before saving (`fork_before_save.patch`)

**Idea.** Reorder `Trainer.train` (`walrus/trainer/training.py`) so that no DataLoader worker is
ever forked while the async checkpoint save's background thread is running. The workers are
forked first, and then the save starts. Nothing new is waited for, so the per-epoch cost is zero.

**Current order at the epoch E → E+1 boundary, and after the last epoch:**

```
end of epoch E:   validation(E)                 ← forks validation workers
                  wait for save(E-1)
                  start async save(E)           ← background thread starts writing
top of E+1:       seed_all(E+1); set_epoch(E+1)
                  iter(train_dataloader)        ← forks 10 workers ~1 s later, save still running → hang
after last epoch: test validation               ← forks workers; the last save is never waited for
```

**New order:**

```
end of epoch E:   wait for save(E-1)            ← moved before validation; finished long ago
                  validation(E)                 ← forks with no save running
                  seed_all(E+1); set_epoch(E+1)
                  iter(train_dataloader)        ← forks the E+1 workers now, before any save starts
                  start async save(E)           ← the save overlaps E+1's prefetch and training
top of E+1:       train on the iterator built above
after last epoch: wait for the last save
                  test validation               ← forks with no save running
```

**The changes** (one file, +31 / −14; the patch applies cleanly to `flopcount` `3ee04cd`):

1. **New helper `Trainer._start_epoch_iter(train_dataloader, epoch)`.** It holds the per-epoch
   setup moved from the top of the loop: `seed_all(seed_from(SEED_GLOBAL, seed, rank, epoch))`,
   `sampler.set_epoch(epoch)`, then `return iter(train_dataloader)`, which forks the workers and
   draws their base seed. The first epoch of a launch and every later epoch use the same code.
2. **`train_one_epoch(epoch, dataloader, data_iter=None)`.** It uses an iterator built in
   advance, or calls `iter(dataloader)` itself as before.
3. **Top of the epoch loop.** It builds the iterator only for the first epoch of a launch (a fresh
   run or a resume, when no save is in flight), passes it to `train_one_epoch`, then sets
   `data_iter = None` so that epoch's workers shut down as before.
4. **The wait for the previous save moves before validation.** The validation loaders fork
   workers too. That save started a full epoch earlier, so in practice the wait returns
   immediately: it was already waited for every epoch, just later.
5. **The core change.** After validation and before `save_model_if_necessary`:
   `if epoch < max_epoch: data_iter = self._start_epoch_iter(train_dataloader, epoch + 1)`.
6. **After the loop, the last save is waited for before test validation.** Today nothing waits
   for it, so the test loaders' forks can overlap it at the end of every run. That is a second
   hang window the original diagnosis missed.

**Determinism is unchanged.** Every epoch still runs `seed_all(E)` → `set_epoch(E)` → `iter()`
before its first batch. Only what runs between `iter()` and training changes: the save,
`empty_cache` and `gc`, none of which draws random numbers.
`tests/test_determinism.py::test_resume_equals_uninterrupted` covers exactly this. Its
uninterrupted run builds the epoch-2 iterator early, its resumed run builds it at the top of the
loop, and the final weights are bit-identical. `test_determinism` (2), `test_trainer` (4),
`test_checkpoint` (4 + 1 skipped) and `test_seed` (3) all pass with the patch.

**Cost.**
- **Per epoch:** no new waits. The next epoch's workers even start prefetching while the save is
  written.
- **Per run:** one wait for the final save before test validation (a few seconds).
- **No checkpointing** (`skip_checkpointing`, or a local run without a future): nothing waits,
  and workers are forked slightly earlier.

**Limits.**
- It removes the condition the deadlock needs, a fork while the save thread runs, rather than the
  deadlock inside torch. It rests on §4–5: the frozen worker in `set_num_threads`, and the save
  overlapping every boundary that hung.
- **Untested on GPU.** The test is to run B's script several times with the patch, against
  unfixed runs from the same day.
- Other parent threads that might fork-poison workers (W&B, NCCL or gloo internals) are not
  covered. None has been implicated.

**Apply it:** `git apply mydocs/dataloader_fork_hang/fork_before_save.patch`.

## 8b. Native-backtrace attempt, and a day without hangs (2026-10-12)

- **The tool.** Attaching gdb or py-spy is blocked on Leonardo (`kernel.yama.ptrace_scope = 2`),
  and core dumps are off (hard limit 0). The generator's `diagnostics(native=True)` therefore
  preloads glibc's `libSegFault.so` (`LD_PRELOAD`, `SEGFAULT_SIGNALS=abrt`). When the watchdog
  sends SIGABRT, each process prints a native backtrace and its memory map after faulthandler's
  Python dump. Checked on the login node: the backtrace reaches frames inside `libtorch_cpu.so`.

**Runs.** All use B's exact script with `DETAIL` off, on unfixed `flopcount` `3ee04cd`. Its code
outside `mydocs` is identical to `dd08774`, the commit of the hung runs.

| runs | preload | result |
|---|---|---|
| `B_native_1`–`_6` (60050544, 60050549, 60051878, 60051888, 60053835, 60054992) | `libSegFault.so` | 6 of 6 completed both launches, no hang |
| `B_plain_1`, `B_plain_2` (60053834, 60054991) | none: byte-identical to the `cur_B_nd` scripts apart from the name | 2 of 2 completed, no hang |

- **No hang, so no native backtrace.** The preload is not what suppressed the hang: the plain
  runs passed too.
- **The overlap was there.** The save started 0.6–1.1 s before the next epoch and finished
  0.9–1.2 s after it began (`save_vs_epoch_timing.txt`, last block). That's the same as in the
  runs that hung.

**B's exact script, first launches, by date and code** (`fulldet`: that branch's README §7g):

| date | current code (`dd08774` = `3ee04cd`) | `656c00c` | `fulldet` |
|---|---|---|---|
| 2026-10-09 | 3 of 3 hung | — | — |
| 2026-10-10 | 2 of 2 hung | 0 of 4 | — |
| 2026-10-11 | 2 of 3 hung | — | — |
| 2026-10-12 | **0 of 8** | — | 0 of 3 |

**Reading.**
- **The rate depends on something outside the code and the save/fork timing we log.** It was
  high on 2026-10-09/11, and zero across 11 runs on 2026-10-12. Candidates, none tested: system
  state on the nodes (other jobs, page cache, the `$FAST` filesystem's latency during the write),
  or the node pool the debug QOS handed out.
- **What still stands:**
  - the mechanism signature: a worker frozen in `set_num_threads(1)` while the save runs (§4–5);
  - 0 hangs at 200 M / 1.2 B;
  - the one same-day code contrast (2026-10-10).
- **What no longer stands:** the cross-day contrasts, §7f's included, and the `fulldet` branch's
  "it doesn't hang".
- **The fix can't be tested today.** With a baseline of 0 of 8, fixed runs passing would show
  nothing. A test needs fixed and unfixed runs submitted together, on a day when the unfixed ones
  hang. Since the fix costs nothing and removes the condition the mechanism needs, it can be
  merged on that argument alone.

**CPU reproduction.** Two attempts on the login node failed: forking 10 DataLoader workers while a
background thread did save-like CPU work or OpenMP tensor ops. Both timeouts turned out to be
stalls in the test harness: every worker was idle at `worker.py:317`, none frozen at `:259`.

## 9. Open questions

- Which OpenMP region the parent's save thread was in at fork time. The dumps show only Python
  frames; a native stack (e.g. `py-spy dump --native` or `gdb`) of the stuck worker would show
  the lock.
- Why B's script hit it 7 of 8 times and A 0 of 3 with identical first launches, given that the
  timings are essentially the same (`save_vs_epoch_timing.txt`). §7f puts chance at about 2.4 %.
  The first B pass in round 2 makes a B-specific factor less likely, but doesn't settle it.
- Whether, and how, the determinism changes cause the hang: yes by §7e's evidence, mechanism unknown.
  The next steps are a bisect of the 7 changed files, and more A-type runs on the current code.
- Why only B hung (3 of 3, against 0 of 22 other tiny boundaries; §7c). Ways to tell a
  B-specific factor from chance:
  - rerun the A/B pair as is;
  - run B's script under another name, or swap the submission order.

  Until then, the hazard should be treated as real but rare (§8).

## 10. Files

```
README.md                         this document
save_vs_epoch_timing.txt          save start / completion vs next-epoch start, every run and boundary
slurm_logs/
  59846208_A_completed.{out,err}
  59846210_B_hang_gloo_timeout.{out,err}       collective-timeout evidence (§4.2)
  59847084_B_hang_stack_dumps.{out,err}        faulthandler dumps of all 44 processes (§4.3)
  59848085_hang_anchor_completed.{out,err}     768/12/2048, 5 boundaries, no hang (§7)
  59848086_hang_fullDims_completed.{out,err}   1408/40/5632, 5 boundaries, no hang (§7)
  59850614_old_tiny_1_at_656c00c_completed.{out,err}  commit 656c00c, 320/6/768, 5 boundaries, no hang (§7b)
  59850615_old_tiny_2_at_656c00c_completed.{out,err}  commit 656c00c, 320/6/768, 5 boundaries, no hang (§7b)
  59852095_new_tiny_1_current_code_completed.{out,err} current code, same config, 5 boundaries, no hang (§7c)
  59852101_new_tiny_2_current_code_completed.{out,err} current code, same config, 5 boundaries, no hang (§7c)
  59853799_old_B_1_..._first_launch_ok_resume_failed.{out,err}   B's script at 656c00c: no hang, resume crashed (§7d)
  59853800_old_B_2_..._first_launch_ok_resume_failed.{out,err}   same (§7d)
  59854572/59854574_old_B_nd_{1,2}_at_656c00c_DETAILoff_completed.{out,err}   old code, B's script, DETAIL off: complete incl. resume (§7e)
  59856184/59856185_cur_B_nd_{1,2}_current_code_DETAILoff_HUNG.{out,err}      current code, B's script, DETAIL off: hung again (§7e)
  60036829_B_round2_DETAIL_launch1_ok_resume_crashed.{out,err}   round 2: B passed launch 1, resume hit the DETAIL crash (§7f)
  60037693/60040489_cur_B_nd_{1,2}_round2_HUNG.{out,err}         round 2: hung at the epoch 1->2 fork (§7f)
  60038527_A_round2_completed.{out,err}                          round 2 (§7f)
  60039144/60040787_new_tiny_{1,2}_round2_completed.{out,err}    round 2, 5 boundaries each (§7f)
  60036831_hang_anchor_round2_completed.{out,err}                round 2, 768/12/2048, 5 boundaries (§7f)
  60038181_hang_fullDims_round2_completed.{out,err}              round 2, 1408/40/5632, 5 boundaries (§7f)
  52966529_fullDims_6DS_N8_excerpt_nccl_timeout_30min.txt   NCCL Timeout(ms)=1800000 (§6)
fork_before_save.patch            the proposed fix (§8a), against flopcount 3ee04cd
slurm_logs/600*_B_{native,plain}_*_flopcount_completed.{out,err}   2026-10-12, unfixed flopcount: 8 of 8 passed (§8b)
stack_dumps/
  59847084_summary.txt                         44 dumps grouped by main-thread stack
  stuck_worker_full_dump.txt                   the worker frozen in set_num_threads(1)
  rank_waiting_for_data_full_dump.txt          the rank blocked in DataLoader.__next__
  rank_in_all_gather_full_dump.txt             a rank blocked in FSDP all_gather
  cur_B_nd_summary.txt                         dumps of the DETAIL-off current-code hangs (§7e, and round 2 §7f)
job_scripts/                                   snapshot of the generator, compare.py and the generated sbatch scripts
```

`job_scripts/` holds the scripts as they were last generated: 600 s timeout, 300 s watchdog.
Earlier jobs ran with the settings in the §2 table.

The originals on `$FAST` (`/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/`) were deleted after
copying, for both rounds: the SLURM logs, the run folders under `determinism/` (with checkpoints),
the W&B offline runs and the Hydra outputs. Only the copies listed above remain.
