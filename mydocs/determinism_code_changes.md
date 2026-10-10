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
- **LR scheduler on resume.**
  - **The bug:** `InverseSqrtLinearRamps.get_lr` (the upstream default `lr_scheduler`) is
    recursive: it multiplies the optimizer's current lr. On resume, the checkpoint restores the
    correct lr, and then the scheduler's constructor takes its initial `step()`, which multiplies
    by the resumed epoch's factor a second time. Every later epoch inherits the inflation (1.9× in
    the test, during warm-up).
  - **The fix:** record the loaded lr just before building the scheduler, and when resuming
    (`last_epoch > 0`) put it back afterwards, including `_last_lr`.
  - For closed-form schedulers (e.g. PT9's `InverseSqrtLinearWarmupSqrtCooldown`) this is a
    no-op, because the restored value equals what the scheduler computed. Finetuning loads don't
    set `last_epoch`, so they're unaffected.
  - **Rejected alternatives:**
    - switching `get_lr` to the class's existing `_get_closed_form_lr` would change the default
      schedule for every run, since that closed form indexes the decay step one off (about
      0.5 % lower in the decay phase);
    - saving the scheduler's `state_dict` would change the checkpoint format.

### 2.5b `walrus/utils/distribution_utils.py`, opt-in collective timeout (diagnostic)

`configure_distribution` reads `WALRUS_PG_TIMEOUT_SEC`. When it is set (and no process group
exists yet), it calls `torch.distributed.init_process_group(timeout=timedelta(seconds=...))`
before the device mesh. `init_device_mesh` would otherwise create the same default group
(`cpu: gloo, cuda: nccl`, `device_mesh.py:465-467`), only with torch's default timeouts
(30 min gloo / 10 min NCCL). When unset, behaviour is unchanged.

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
- **Ranks are not partitioned** on `flopcount`. Two GPUs can still draw the same window; this was
  accepted by design (see the sampling doc §3.4 and §5.3 for the size of the overlap). Branch
  `fulldet` removes it (§5).
- **`trainer.lr_scheduler_per_step=True`** still restarts the schedule at the wrong place on
  resume: it passes an epoch count as the scheduler's `last_epoch`, which counts steps. This
  predates the work and is unused (`False` in every trainer config).

---

## 5. Branch `fulldet`: one window plan per epoch, shared by all ranks

Built on `flopcount` (`dd08774`). Goal: within an epoch, no two GPUs or nodes train on the same
window of a dataset until all of its windows have been used. Recycling stays on. Repeats across
epochs are still allowed.

**Why not the built-in `allow_sharding_in_train=True` (`DistributedSampler`).** It gives each
rank a fixed `1/R` slice. Under HSDP or DDP, rows use up a dataset at different rates, so a rank
that empties its slice repeats it while other slices are untouched. A recycle within an epoch
also replays the same order, and `drop_last=False` pads the slices with duplicates.

| | `flopcount` | `fulldet` |
|---|---|---|
| Window streams | one permutation per rank, keyed on `(seed, global rank, epoch, dataset)` | one stream per dataset and epoch, keyed on `(seed, epoch, dataset, pass)`; no rank |
| Which positions a rank takes | all of its own permutation | its share of the shared stream, fixed by `epoch_plan` (step, then row, then rank) |
| Cross-rank overlap within an epoch | `≈ r²/W` per rank pair | 0 until the epoch's demand exceeds `W_d`; then every window `⌊N/W⌋` or `⌈N/W⌉` times |
| Dataset running out mid-batch | partial batch dropped, new dataset drawn | the batch continues into the next permutation; nothing dropped |
| `recycle=False` | each rank stops when its own samplers run out | a dataset that can't fill the next step is dropped for every rank; all ranks stop on the same step |
| Dataset choice | `seed_from(SEED_DATASET_CHOICE, seed, epoch, sampling_rank)` | same key; one generator per row, replayed by every rank |

### 5.1 Changes by file

- **`walrus/data/mixed_dset_sampler.py`.** New module-level
  `epoch_plan(seed, epoch, sampling_ranks, windows, dset_batch, max_samples, recycle)`. It returns
  `plan[r] = [(dataset, stream start), ...]` for every global rank `r`. Every rank runs it, and
  the result is identical everywhere.

  `BatchedMultisetSampler` no longer builds per-dataset sub-samplers:
  - its constructor takes `sampling_ranks` (the sampling rank of every global rank) and
    `global_rank` in place of `rank`; `base_sampler` stays in the signature, unused;
  - `__iter__` maps this rank's stream positions to window indices through lazily built
    `randperm(W_d)`s, cached per `(dataset, pass)`;
  - `__len__` is `max_samples` with recycling, and the plan's length without;
  - `set_epoch` only stores the epoch.

  The legacy `MultisetSampler` (unused) is untouched.
- **`walrus/data/multidatamodule.py`.** `train_dataloader(sampling_ranks=None)` defaults to every
  rank on its own row. The `DistributedSampler` branch is gone, and `allow_sharding_in_train` is
  accepted but ignored (docstring). The default `rank` is now 0: it was 1 with `world_size=1`,
  which the old sampler never used, and the plan indexes by it.
- **`walrus/trainer/training.py`.** New module-level
  `sampling_ranks_for(strategy, world_size, sync_group_size)`, the old gpu/node/world branches as
  a list over all ranks. `Trainer` stores `self.sampling_ranks` (and `self.sampling_rank` as its
  own entry) and passes the list to `train_dataloader`.
- **`scripts/count_WALRUS_flops.py`.** `replay_epoch(seed, epoch, sampling_ranks, ...)` now calls
  `epoch_plan` and returns micro-batches per dataset summed over all ranks. Its own copy of the
  sampler simulation is deleted. `microbatch_counts` keeps its signature, so the CLI is unchanged.
- **`tests/test_fulldet_sampler.py` (new, CPU).** Simulated 8-rank fsdp (1 row), HSDP (2×4) and
  DDP (8 rows) layouts over a fake 3-dataset mixture. It checks that:
  - every window is used once before any is used twice, within an epoch over all ranks, and there
    are no repeats at all while demand ≤ `W_d`;
  - ranks of a row train on the same dataset at every step;
  - the draws follow `seed` and `epoch`;
  - `recycle=False` stops every rank after the same number of steps.

  The same disjointness check on `flopcount`'s sampler finds windows drawn up to 4 times in one
  epoch, with demand well under `W_d`.
- **`LeonardoRunScripts/PT9_2d/Determinism/`.**
  - New `check_disjoint.py`: under torchrun, it builds the real PT9 datamodule, takes
    `sampling_ranks` from `sampling_ranks_for`, iterates the sampler for 2 epochs without reading
    data, gathers every rank's indices and asserts disjointness.
  - The generator now also writes `p9_fulldet_check.sh` (debug QOS, 1 node, 30 min). It runs the
    check under fsdp and under HSDP (2 groups of 2), then a 1-epoch tiny training as a smoke test.
    The script runs the code of the tree that holds the generator.
  - The generator skips the `656c00c` scripts when that worktree is absent, and leaves them as
    last generated.
- **Notebooks.**
  - Notebook 5 has been rewritten for the plan: the E1 predictor for three layouts, E3 (streams
    wrap, nothing dropped), E5 (disjoint under fsdp, HSDP and DDP, against `flopcount`'s `r²/W`),
    E6 (now a note on `DistributedSampler`), E8 (new model, with a `flopcount` column) and the
    finetune text.
  - Notebook 3 has been updated: §2's hierarchy and trace, the §2.8 rank table, §3.6's coverage
    formula, the coverage simulation and the cheat sheet.
  - Both were re-run with every assert passing.
- **Docs.**
  - `walrus_sampling_and_repetition.md`: §1–3, §5–8;
  - `SCALING_STRATEGY_2D.md`: the seed paragraph and §5.4;
  - `walrus_tokens_and_data_pipeline.md`: a pointer note at the top, since its §2 and §4 describe
    the `flopcount` sampler.

### 5.2 Verification

| check | result |
|---|---|
| `tests/test_fulldet_sampler.py` | 10 passed |
| `tests/test_determinism.py` (resume equals the uninterrupted run, seed control) | 2 passed |
| the other test files (dataset, seed, import, flops, jitterer, linear model, checkpoint, finetune, trainer) | all pass (2 + 1 skipped, as before) |
| ruff, mypy on the touched files | no new findings beyond the `List` annotation style already used in the file; 3 fewer mypy errors than `flopcount` |
| `check_disjoint.py` emulated on the login node (4 CPU ranks, gloo, real PT9 data config) | `DISJOINT OK` for both epochs under fsdp and HSDP |
| notebooks 3 and 5 | execute, all asserts pass |
| `p9_fulldet_check.sh` on GPU | generated, not yet submitted |

### 5.3 What it does and doesn't buy

For PT9 pretraining, each epoch draws about 1,780 windows per dataset over 4 GPUs, against
`W_d ≥ 96 192`. Independent ranks rarely collided, so run-level repetition moves by at most 0.2
percentage points (sampling doc §5.1–5.2). The difference is large when one epoch's demand is
comparable to `W_d`. In the single-dataset finetunes, each epoch now uses 87–100 % of the windows
with at most 1–3 uses each, against 63–100 % and up to 4 uses on `flopcount` (sampling doc §5.3).

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
