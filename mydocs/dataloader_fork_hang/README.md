# DataLoader worker hang after an async checkpoint save (investigation, Oct 2026)

**Status: diagnosed, not fixed.** The fix options are in §8. Nothing in `walrus/` has been changed
to address it.

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
DataLoader workers. It is **timing-dependent**. It struck 3 of 5 tiny-model (320/6/768) runs on the
current code, but 0 of 10 exposed epoch boundaries at 200 M and 1.2 B parameters (§7), and 0 of
roughly 145 boundaries in earlier production runs (§6).

**The code before the determinism changes does not hang, so those changes are implicated (§7e).**
B's exact script hung in **5 of 5** runs on the current code, with and without
`TORCH_DISTRIBUTED_DEBUG=DETAIL`, and the stack dumps show the same frozen worker every time. The
same script on commit `656c00c` completed **4 of 4**, and the two runs without `DETAIL` also resumed
successfully. If both versions hung equally often, all 5 hangs falling on the current code would
happen by chance with probability 1/C(9,5) ≈ 0.8 % (§7e).

That is not the whole picture: on the current code, other tiny-model runs did *not* hang
(A: 0 of 2; the 6-epoch controls: 0 of 10), and A's first launch is script-identical to B's. So
"the current code" alone isn't the trigger, and the mechanism linking the changes to the hang is
unknown. Earlier sections (§7b, §7c) concluded the changes were not the cause; §7e supersedes that.

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

The hang tests use `max_samples=100` (anchor) and `30` (fullDims), with `checkpoint_frequency=0`
(one rolling `last` checkpoint). A and B use `max_samples=100`, 2 epochs and
`checkpoint_frequency=1`. Six different nodes were involved, so a faulty node is ruled out.

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
| 768/12/2048 | 5 | 0 |
| 1408/40/5632 | 5 | 0 |
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

## 8. Risk and fix options (none implemented)

- **Risk.** With roughly 155 clean large-model boundaries, the per-boundary probability at
  production size is bounded at about ≤ 2 % (rule of three). That doesn't make it negligible. A
  100-epoch run has 99 boundaries, and 1 % per boundary would mean about a 63 % chance of at least
  one hang per run, each costing about 30 min (the NCCL timeout) before the job errors out. With
  auto-resuming chains, the job would then restart from the last checkpoint.
- **Option 1, recommended:** wait for the save before the next epoch forks workers. Move the existing
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

## 9. Open questions

- Which OpenMP region the parent's save thread was in at fork time. The dumps show only Python
  frames; a native stack (e.g. `py-spy dump --native` or `gdb`) of the stuck worker would show
  the lock.
- Why B hit it 3 of 3 times and A 0 of 2 with identical configs, given that the timings are
  essentially the same (`save_vs_epoch_timing.txt`). That's most likely chance at a high per-run
  rate (3 of 5 overall), but unexplained.
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
  52966529_fullDims_6DS_N8_excerpt_nccl_timeout_30min.txt   NCCL Timeout(ms)=1800000 (§6)
stack_dumps/
  59847084_summary.txt                         44 dumps grouped by main-thread stack
  stuck_worker_full_dump.txt                   the worker frozen in set_num_threads(1)
  rank_waiting_for_data_full_dump.txt          the rank blocked in DataLoader.__next__
  rank_in_all_gather_full_dump.txt             a rank blocked in FSDP all_gather
  cur_B_nd_summary.txt                         dumps of the two DETAIL-off current-code hangs (§7e)
job_scripts/                                   snapshot of the generator, compare.py and the generated sbatch scripts
```

`job_scripts/` holds the scripts as they were last generated: 600 s timeout, 300 s watchdog.
Earlier jobs ran with the settings in the §2 table.

Not copied, still on `$FAST` (`/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/`):
- the run folders `determinism/{A,B,hang_anchor,hang_fullDims}`, with checkpoints:
  1.1 G, 526 M, 1.7 G and 27 G;
- the matching W&B offline runs, and the SLURM log originals;
- `determinism_diag/`, the earlier copy of job 59846210's logs.
