# Code changes: deterministic, resume-exact training runs

This is a record of every change made after commit `656c00c` ("Commit before changing
determinism of the runs") to make a run's random draws a function of `(cfg.seed, rank, epoch)`,
plus one resume bug found along the way. Roll back with `git checkout 656c00c -- <files>` or by
reverting the commit that contains these changes.

The background and the full inventory of non-determinism sources are in
[walrus_sampling_and_repetition.md](walrus_sampling_and_repetition.md) §2 and §8.

---

## 1. Summary

| | before (`656c00c`) | after |
|---|---|---|
| Global RNGs (torch CPU+CUDA, numpy, `random`) | seeded once at start-up: `seed`, then `seed + 1_000_003·rank` | additionally re-seeded at the top of **every epoch** from `(seed, rank, epoch)` |
| Window permutation of each dataset | seeded from the global CPU RNG at each pass's first use, so it depended on every earlier forward pass, validation, `data_workers` | own generator per epoch, keyed on `(seed, global rank, epoch, dataset)` |
| Dataset choice per step | `100·epoch + 10·0 + sampling_rank`: ignored `cfg.seed`, and could collide | hash of `(seed, epoch, sampling_rank)` |
| `distribution=local` | `set_epoch` never called, so every epoch used the same dataset order | `set_epoch` always called |
| Resume with a recursive LR scheduler | lr inflated by one extra scheduler step on every resume | loaded lr restored |
| Resumed epoch vs. uninterrupted | different draws, plus a partial replay of epoch 1 | identical (same `data_workers` and GPU count) |

**Behaviour change for all future runs.** Every random stream differs from runs made at or before
`656c00c`, and the dataset order now depends on `cfg.seed`. The PT9 noise floor
(`SCALING_STRATEGY_2D.md` §5.4) therefore now includes dataset-order noise.

---

## 2. Changes by file

### 2.1 `walrus/utils/experiment_utils.py`, new seed helper

```python
SEED_GLOBAL, SEED_DATASET_CHOICE, SEED_WINDOWS = 1, 2, 3

def seed_from(*keys: int) -> int:
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])
```

- One helper for every stream. `SeedSequence` hashes the keys, so two different key tuples
  never give the same seed, unlike the old `100·epoch + 10·seed + rank`.
- The leading tag keeps different streams apart even when their other keys coincide.
- Keys must be non-negative integers, a `SeedSequence` requirement; `cfg.seed` must therefore
  be ≥ 0.

### 2.2 `walrus/trainer/training.py`, per-epoch re-seed

- New constructor argument `seed: int = 0`, stored as `self.seed`, and documented.
- In `Trainer.train`, at the top of each epoch and before the data iterator is created:
  ```python
  seed_all(seed_from(SEED_GLOBAL, self.seed, self.rank, epoch))
  train_dataloader.sampler.set_epoch(epoch)   # was: only `if self.is_distributed`
  ```
  - `self.rank` is the global rank.
  - Everything that draws from the global RNGs during the epoch is now keyed on the epoch: the
    DataLoader's `_base_seed` (and therefore the worker seeds, `dt` and rotation), patch jitter,
    the periodic roll, DropPath, dropout, and the validation that follows the epoch.
  - No RNG state needs to be checkpointed.
- `set_epoch` is now called for `distribution=local` too, so single-GPU runs vary the dataset
  order per epoch instead of replaying it.

### 2.3 `walrus/data/mixed_dset_sampler.py`, sampler seeding

`BatchedMultisetSampler`:
- New constructor argument `global_rank=None`. It defaults to `rank` for callers that don't pass
  it, which suits single-process use.
  - `rank` is the **sampling** rank, which picks the dataset stream (0 on every GPU under fsdp).
  - `global_rank` keys the window permutations.
  - Keying windows on the sampling rank would make every GPU under fsdp draw identical windows.
- At the start of `__iter__`, every sub-sampler is re-keyed for the epoch:
  - a `RandomSampler` gets
    `generator = torch.Generator().manual_seed(seed_from(SEED_WINDOWS, seed, global_rank, epoch, d))`.
    A recycle within the epoch continues the same generator, so the next pass is a new but
    deterministic permutation;
  - a `DistributedSampler` (the `allow_sharding_in_train=True` path) gets `seed = self.seed`, so its
    shared permutation follows `cfg.seed` instead of the constant 0.
- The dataset-choice generator is seeded `seed_from(SEED_DATASET_CHOICE, seed, epoch, rank)`.
  The sampler's `seed` argument is now actually passed: it is `cfg.seed`.

`MultisetSampler` (legacy, unused by training): the same dataset-choice seed change, one line.

### 2.4 `walrus/data/multidatamodule.py`, plumbing

- New constructor argument `seed: int = 0`, stored as `self.seed`, and documented.
- `train_dataloader` passes `seed=self.seed` and `global_rank=self.rank` to
  `BatchedMultisetSampler`, replacing the commented-out `# seed=seed`.
- Validation and test loaders are unchanged: they keep their fixed `seed=0` / `manual_seed(0)`
  samplers, so the same validation windows are scored every epoch and for every seed.

### 2.5 `walrus/train.py`, seed plumbing and the LR resume fix

- Passes `seed=cfg.seed` to the datamodule and to the `Trainer`.
- **LR scheduler on resume:** a real bug, for recursive schedulers only. Verified 2026-10-12; details
  below.

#### The change

```python
resumed_lrs = [group["lr"] for group in optimizer.param_groups]
lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer, max_epochs=cfg.trainer.max_epoch,
                           step_mult_factor=step_mult_factor, last_epoch=max(-1, last_epoch - 1))
if last_epoch > 0:
    for group, lr in zip(optimizer.param_groups, resumed_lrs):
        group["lr"] = lr
    lr_scheduler._last_lr = resumed_lrs
```

It records the lr loaded from the checkpoint just before building the scheduler. On a resume
(`last_epoch > 0`) it puts that lr back afterwards, including the scheduler's `_last_lr`, which
is what `get_last_lr()` and the W&B `lr` log read. Finetuning loads don't set `last_epoch`, so
they're unaffected.

#### Why it is a bug

The same structure is in upstream PolymathicAI `main`: `train.py:264`, `last_epoch=max(-1,
last_epoch - 1)`, the `lr_scheduler.step()` before the save in `training.py`, and the default
`inv_sqrt_w_linear_ramps` in `config.yaml`. Every link below was checked in the code (torch 2.5.1):

1. **The saved lr is already the next epoch's.** `train_one_epoch` ends with
   `lr_scheduler.step()` (`training.py:1217`), and the checkpoint is saved after that in the epoch
   loop. So the optimizer state on disk holds the lr for epoch `E + 1`.
2. **The resume restores it.** `optimizer.load_state_dict` copies each saved param group,
   including `lr` and `initial_lr`. The FSDP path ends in the same call:
   `CheckPointer.load_state_dict` → `set_state_dict` → `_load_optim_state_dict`, which calls
   `optim.load_state_dict` (`torch/distributed/checkpoint/state_dict.py:960`).
3. **The constructor steps once more.** The scheduler is rebuilt with `last_epoch = E − 1`, and
   torch's `LRScheduler.__init__` always calls `_initial_step()` → `step()`
   (`torch/optim/lr_scheduler.py:148-151`). That moves it to `last_epoch = E`, the right index
   for epoch `E + 1`.
4. **A recursive `get_lr` then applies epoch `E`'s factor twice.** `InverseSqrtLinearRamps.get_lr`
   returns `group["lr"] × factor(last_epoch)`, the current lr times this epoch's factor. Here the
   current lr already includes that factor.
5. **The closed-form scheduler is immune.** `InverseSqrtLinearWarmupSqrtCooldown.get_lr` computes
   the lr from `base_lrs` and `last_epoch` alone, ignoring the current lr. It gives the same value
   whatever was loaded.

#### Measured

`mydocs/lr_resume_check.py` replays this logic in plain torch: base lr 2e-4, 100 epochs, warmup
and cooldown 10, warmup factor 0.1. For each resume point it reports the resumed lr divided by the
uninterrupted run's lr.

| Resume after epoch | `InverseSqrtLinearRamps`, no fix: next epoch / epoch 100 | with fix | `InverseSqrtLinearWarmupSqrtCooldown`, no fix |
|---|---|---|---|
| 1 | **×1.900** / ×1.900 | exact | exact |
| 5 | ×1.196 / ×1.196 | exact | exact |
| 10 | ×1.099 / ×1.099 | exact | exact |
| 11 | ×0.960 / ×0.960 | exact | exact |
| 50 | ×0.986 / ×0.986 | exact | exact |
| 90 | ×0.992 / ×0.992 | exact | exact |
| 95 | ×0.857 / ×0.857 | exact | exact |

- **The error is permanent.** Every later lr is computed from the wrong one, so the ratio holds to
  the end of the run.
- **The error is largest early in warm-up and early in the cooldown,** where the per-epoch factor
  is furthest from 1.
- **Chains compound it.** Each resume multiplies in its own factor. This follows from the
  recursion and was not measured separately.
- **The fix is exact.** With it, the resumed schedule equals the uninterrupted one at every epoch
  (max |ratio − 1| = 0). For the closed-form scheduler it changes nothing.

**End to end.** On a throwaway copy of `flopcount` (`dd08774`) with only these 8 lines removed,
`tests/test_determinism.py::test_resume_equals_uninterrupted` **fails**: the resumed run ends with
different weights. With the fix it passes. The test uses the repo's default scheduler, which is
the recursive one. Earlier, a control showed that the per-epoch re-seeding and this fix are both
needed for the test to pass.

#### Who is affected

- **Affected:** any training run that keeps the default `lr_scheduler=inv_sqrt_w_linear_ramps`
  and resumes. That includes upstream WALRUS with its default config.
- **Not this thesis's training runs.** All 79 training scripts in `LeonardoRunScripts/` set
  `lr_scheduler=inv_sqrt_w_sqrt_ramps` (73) or `inv_sqrt_w_sqrt_ramps_longer` (6). Both are the
  closed-form `InverseSqrtLinearWarmupSqrtCooldown`. The scripts without an override are
  validation, smoke-test and benchmark scripts.

#### Alternatives considered

- **Switch `get_lr` to the class's existing `_get_closed_form_lr`.** Rejected: it changes the
  default schedule for every run, because that closed form indexes the decay step one off (about
  0.5 % lower in the decay phase; compare epoch 11 above, 1.919e-4 recursive against 1.916e-4
  closed form).
- **Save and load the scheduler's `state_dict`.** Rejected: it changes the checkpoint format, and
  it wouldn't help, because the recursive lr lives in the optimizer's param groups, which are
  already restored. What has to be stopped is the constructor's extra step.
- **Known leftover:** `trainer.lr_scheduler_per_step=True` passes an epoch count where it needs a
  step count (§4). It is unused, and the fix doesn't touch it.

### 2.5b `walrus/utils/distribution_utils.py`, opt-in collective timeout (diagnostic)

This is the only code change made to diagnose the hang; everything else was job-script settings
(below).

```python
timeout = os.environ.get("WALRUS_PG_TIMEOUT_SEC")
if timeout and not torch.distributed.is_initialized():
    torch.distributed.init_process_group(timeout=timedelta(seconds=int(timeout)))
if cfg.distribution.distribution_type.upper() == "HSDP":   # was `elif`
```

`configure_distribution` reads `WALRUS_PG_TIMEOUT_SEC`. When it is set and no process group
exists yet, it calls `torch.distributed.init_process_group(timeout=timedelta(seconds=...))`
before the device mesh is built. `init_device_mesh` would otherwise create the same default group
(`cpu: gloo, cuda: nccl`, `device_mesh.py:465-467`), only with torch's default timeouts. Torch
documents 30 min for gloo and 10 min for NCCL, but the production log 52966529 shows
`Timeout(ms)=1800000`, i.e. 30 min for NCCL as well
(`dataloader_fork_hang/slurm_logs/52966529_*`). When unset, behaviour is unchanged.

- **The `elif` → `if` change.** The first version put the knob inside the existing
  `if LOCAL / elif HSDP / else` chain. That broke the chain: `mesh` was never assigned
  (`UnboundLocalError`), and jobs 59845924 and 59845925 crashed at start-up. The knob now sits
  before the chain as a separate `if`, and every branch (local, hsdp, fsdp, ddp; knob set and
  unset) was checked with mocks.
- **Job-script diagnostics, not code.** `gen_pt9_determinism_runs.py`'s `diagnostics()` sets:
  - `TORCH_DISTRIBUTED_DEBUG=DETAIL`: cross-rank collective checks. It breaks FSDP checkpoint
    loading (`Backend nccl does not support allgather_into_tensor_coalesced`) on both the old and
    the current code, so a resume needs it off;
  - the NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DUMP_ON_TIMEOUT`), which
    never produced a trace;
  - `PYTHONFAULTHANDLER=1` with a shell watchdog. If the log is silent for N s, it sends SIGABRT
    to every `train.py` process, which prints every thread's Python stack. All the stack dumps
    in `dataloader_fork_hang/` come from this.

  The settings each job used, and what they showed, are in `dataloader_fork_hang/README.md` §2–3.

The knob was added because a first GPU run of the determinism check (B, job 59843238) hung in
epoch 2 for 24 minutes with no timeout error. A short timeout turns such a hang into an error
naming the stuck collective.

### 2.6 `walrus/configs/config.yaml`

Only the comment on `seed` changed. It now says the seed controls weight init and, per epoch,
every RNG including the dataset order and the window draws.

### 2.7 `scripts/count_WALRUS_flops.py`, offline sampler replay

- `replay_epoch(seed, epoch, sampling_rank, dset_batch, windows, max_samples, recycle)` now takes
  the run's seed, the epoch and the sampling rank, and seeds with the same `seed_from` key as the
  sampler. It used to take a precomputed `100·epoch + rank`.
- `microbatch_counts(..., recycle, exact, seed=0)`: the `distributed` argument is gone, because
  `set_epoch` is now always called. `main()` passes `seed=int(cfg.seed)`.

### 2.8 `tests/test_determinism.py`, new

Both tests run on CPU with the dummy dataset and the existing `default_conf` fixture:

| test | checks |
|---|---|
| `test_resume_equals_uninterrupted` | run A trains 2 epochs; run B trains 2 epochs, its `step_2` is deleted and `last`/`best` re-pointed to `step_1` (a job killed after epoch 1, same config), then it auto-resumes. B's resumed `step_2` weights must be bit-identical to B's first attempt and to A |
| `test_seed_controls_the_run` | two `seed=0` runs give bit-identical weights; `seed=1` differs |

Local runs save `full_checkpoint.pt` (not a DCP directory); the test reads weights from it.

### 2.9 `LeonardoRunScripts/PT9_2d/Determinism/`, new GPU check (not yet submitted)

- **`gen_pt9_determinism_runs.py`** generates `p9_determinism_A.sh` and `p9_determinism_B.sh`.
  Both use the PT9 substrate with shape `320/6/768`, `max_samples=100`, 2 epochs, seed 0, 1 node,
  fsdp, `data_workers=10`, and a fixed `folder_override` under `<EXPERIMENT_DIR>/determinism/`.
  - They are sized for the **debug QOS** (`--qos=boost_qos_dbg`, `--time=00:30:00`; it allows
    2 jobs per user, so A and B can run at once).
  - `trainer.max_rollout_steps=2` and `trainer.short_validation_length=2` replace the substrate's
    60 and 20. In the 200 M-anchor roofline log, the last-epoch rollout validation and the
    rollout test pass took about 16 min each. Validation runs after training and never touches
    the weights being compared.
  - B (two launches, three epochs) needs an estimated 12 min.
  - **Hang diagnostics.** A first B run (job 59843238) hung at epoch 2, batch 1, of its first
    launch, right after the asynchronous `step_1` save overlapped epoch 2's first steps. It
    produced no error for 24 minutes, until the 30-minute limit killed it; A, identical,
    completed in 7 min 48 s. The scripts therefore now set:
    - `TORCH_DISTRIBUTED_DEBUG=DETAIL`, which checks collectives for consistency across ranks;
    - `WALRUS_PG_TIMEOUT_SEC=180` (§2.5b);
    - the NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
      per-rank traces in `<RUN_ROOT>/<run>/nccl_trace_rank_*`);
    - `PYTHONFAULTHANDLER=1` with a watchdog that sends SIGABRT to the training workers if the
      log is silent for 7 min, so every thread's Python stack is printed to `.err`.
  - A trains 2 epochs straight.
  - B trains 2 epochs, moves `step_2` to `step_2_first`, re-points `last`/`best` to `step_1`, and
    runs again (resume).
  - Both scripts' overrides are checked with `pt9_substrate.check_composes`.
- **`compare.py`** (login node) converts the three DCP checkpoints and prints relative weight
  distances. It passes if `|B_resumed − B_first| ≤ 2·|A − B_first|`: a resume should add no
  difference beyond the float noise between two uninterrupted runs.

```
python gen_pt9_determinism_runs.py
sbatch p9_determinism_A.sh; sbatch p9_determinism_B.sh
python compare.py
```

### 2.10 Notebooks

- **`demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb`** is rewritten for the new scheme:
  - E0 `run()` mirrors `Trainer.train` (per-epoch re-seed, `seed`/`global_rank` to the sampler).
  - The E1 predictor models the new seeding, including `RandomSampler`'s empty trailing
    `randperm` on exhaustion, which advances a persistent generator. It must match for every
    `data_workers` and forward-draw count.
  - E2 asserts that the dataset order depends on the seed.
  - E4 asserts that forward passes no longer change the windows.
  - E6 asserts that `DistributedSampler` follows the seed.
  - E6b asserts that resumed epochs equal uninterrupted ones.
  - E8 uses the new `replay_epoch`.
  - The old findings are kept as *History* notes.
- **`demo_notebooks/walrus_example_3_Datasets.ipynb`:** the `replay_epoch`/`microbatch_counts`
  calls use the new signatures (`seed=0`), and three passages describing the old seeding are
  corrected. The replayed counts moved by about 1 %, and the notebook's asserts still pass.

### 2.11 Documentation

- `mydocs/walrus_sampling_and_repetition.md`:
  - §2, §3.4–§3.6, §5–§7 rewritten for the new behaviour;
  - new §8, an inventory of every non-determinism source and its status;
  - the old behaviour kept as *History* notes.
- `mydocs/walrus_tokens_and_data_pipeline.md`: §4.2 marked as superseded with a pointer, plus
  three passages corrected (the data-object diagram, the ranks table note, the dataset-choice
  seed bullet).
- `LeonardoRunScripts/PT9_2d/SCALING_STRATEGY_2D.md`: the "Seed" and "What the seed controls"
  paragraphs, and §5.4 "What `seed` varies" (now includes dataset order).

---

## 3. Verification

| check | result |
|---|---|
| `tests/test_determinism.py` before the change (at `656c00c`) | `test_resume_equals_uninterrupted` **fails** |
| …with the LR fix but the per-epoch re-seed disabled | **fails**: both fixes are needed |
| …with both | **passes**, as does `test_seed_controls_the_run` |
| full suite, each file in its own process (login node CPU-time cap is 600 s per process) | all 10 files pass (`test_checkpoint` 4+1 skipped, `test_dataset` 9, `test_determinism` 2, `test_finetune` 2, `test_flops` 24, `test_import` 1, `test_jitterer` 9, `test_seed` 3, `test_trainer` 4, `test_trainer_linear_model` 2 skipped) |
| ruff on changed files, against the same files at `HEAD` | no new findings |
| mypy on changed files, against `HEAD` | same 69 pre-existing errors, none new |
| notebooks 3 and 5 executed in place | all asserts pass |
| GPU check | generated and composes; **not yet run** |

To re-run the tests on the login node, keep thread counts low so a process stays under the CPU
cap:

```bash
module purge; module load python; module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_determinism.py::test_resume_equals_uninterrupted
```

---

## 4. Known limitations

- **The GPU check exposed a DataLoader hang.** A worker forked while the async checkpoint save is
  still running can deadlock in `torch.set_num_threads(1)`. It hit 3 of 5 tiny-model runs and 0 of
  10 boundaries at 200 M / 1.2 B. B's exact script hung in 5 of 5 runs on the current code
  (also with `TORCH_DISTRIBUTED_DEBUG` off) and completed in 4 of 4 on `656c00c`, which is before these
  changes (chance probability ≈ 0.8 %). Other tiny runs on the current code did not hang, so the
  changes are implicated but not sufficient, and the mechanism is unknown. Diagnosed but
  **not fixed**: see
  [dataloader_fork_hang/README.md](dataloader_fork_hang/README.md), which has the logs and stack dumps.

- **GPU floating point is not bit-reproducible:** SDPA backward, the backward of indexed weights,
  and collective reduction order. Same draws, slightly different floats. Not addressed; it would
  need `torch.use_deterministic_algorithms(True)` and a cuBLAS workspace setting, at a speed cost.
- **`data_workers` and the GPU count must stay fixed within a chain.**
  - `dt` and the rotation depend on the worker count, because batches are dealt to workers
    round-robin and each worker has its own numpy stream.
  - Every stream is keyed on the rank, so the rank layout matters too.
- **Ranks are not partitioned.** Two GPUs can still draw the same window; this was accepted by
  design (see the sampling doc §3.4 and §5.3 for the size of the overlap).
- **`trainer.lr_scheduler_per_step=True`** still restarts the schedule at the wrong place on
  resume: it passes an epoch count as the scheduler's `last_epoch`, which counts steps. This
  predates the work and is unused (`False` in every trainer config).

---

## Appendix: earlier changes in the same work, already in `656c00c`

- `scripts/build_well_registry.py`, `split_entry`: each split also records a per-file list
  `files: [{name, n_traj, n_steps, bytes}]`. The existing `n_files`, `bytes` and `steps` keys are
  unchanged, so readers of the registry are unaffected. `scripts/well_dataset_registry.json` and
  `demo_notebooks/data/dataset_table_2d.json` were regenerated with it.
- `demo_notebooks/walrus_example_3_Datasets.ipynb` §3, "Counting tokens": the token, repetition,
  coverage and fixed-FLOP-budget tutorial, with `demo_notebooks/data/dataset_table_2d.json` as
  its cached dataset table.
- `demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb` and
  `mydocs/walrus_sampling_and_repetition.md`: first versions, describing the pre-change sampling.
