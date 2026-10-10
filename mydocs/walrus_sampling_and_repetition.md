# How WALRUS samples training windows, and how often they repeat

This document follows one training window from the dataset mixture to the GPU. It shows exactly
which random number decides each choice, lists every way a window can be drawn again, explains
how to estimate repetition and how to measure it exactly, and inventories everything that can
make a run non-deterministic (§8).

Claims are tagged **[verified]** when an `assert` in
[`demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb`](../demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb)
or [`tests/test_determinism.py`](../tests/test_determinism.py) checks them (labels E1–E8 match the
notebook), and **[code-read]** when they rest on reading the code only. Library line numbers refer
to the Leonardo venv: **torch 2.5.1** and its bundled `the_well`.

**Seeding scheme.** This describes the per-epoch seeding introduced after commit `656c00c`:
every random stream is keyed on `(cfg.seed, rank, epoch[, dataset])` through one helper,
`seed_from` (`walrus/utils/experiment_utils.py`). The earlier behaviour is kept in the *History*
notes, because runs made before the change used it. The code changes themselves, file by file,
are listed in [determinism_code_changes.md](determinism_code_changes.md).

Related: [`walrus_tokens_and_data_pipeline.md`](walrus_tokens_and_data_pipeline.md) §2 (data
objects, one micro-step traced) and §4 (seeding history); notebook 3 §3 (token counts, `R_d`,
coverage).

---

## 1. The sampling hierarchy

One micro-batch is chosen in four nested steps. The first two are decided in the main process;
`dt` and the rotation are decided in the data-loader worker that reads the data.

```
step t of epoch e on rank k
│
├─ 1. WHICH DATASET d          BatchedMultisetSampler, own torch.Generator
│                              seeded seed_from(DATASET_CHOICE, seed, e, sampling_rank)
│                              d ~ Uniform{0..K-1}                          mixed_dset_sampler.py
│
├─ 2. WHICH WINDOWS (B of them) RandomSampler of dataset d: next B entries of a randperm(W_d)
│                              from its own generator, seeded at the start of every epoch with
│                              seed_from(WINDOWS, seed, global_rank, e, d)
│                              → B global indices (local + offset_d)
│
├─ 3. WHERE IN THE FILES        MixedWellDataset: dataset from searchsorted(offsets)   multidataset.py:374-385
│                              WellDataset._load_one_sample(index)                     the_well datasets.py:780-792
│                                file  = searchsorted(file_index_offsets, index) − 1
│                                traj  = local // windows_per_traj(file)   windows_per_traj = n_steps − (T_in+T_out) + 1
│                                t0    = local %  windows_per_traj(file)
│
└─ 4. HOW THE WINDOW IS READ    dt ~ randint(min_dt, min(max_dt, (n_steps − t0 − 1)//(T_in+T_out−1)) + 1)
                               np.random in the worker                                 the_well datasets.py:801-813
                               RandomRotation90 (np.random, once per micro-batch); patch jitter and
                               periodic roll in the forward pass
```

All `B` windows of a micro-batch come from the same dataset and may come from different files,
trajectories and start frames. **[verified E1, E7]**

---

## 2. The random number generators and who seeds them

| decision | generator | seeded by | depends on |
|---|---|---|---|
| dataset `d` per step | the sampler's own `torch.Generator` | `seed_from(SEED_DATASET_CHOICE, seed, epoch, sampling_rank)` | seed, epoch, and `sampling_rank` (0 on every GPU under fsdp) |
| window permutation of each dataset | its own `torch.Generator`, set on the `RandomSampler` at the start of every epoch | `seed_from(SEED_WINDOWS, seed, global_rank, epoch, dataset)` | seed, epoch, **global** rank, dataset |
| everything else on the global RNGs (torch CPU + CUDA, numpy, `random`) | re-seeded at the top of every epoch (`Trainer.train`) | `seed_all(seed_from(SEED_GLOBAL, seed, global_rank, epoch))` | seed, epoch, global rank |
| …which then feeds: DataLoader worker seeds → `dt`, rotation | `_base_seed` (`dataloader.py:666`), then worker numpy | — | + `data_workers` |
| …patch jitter (`patch_jitterers.py:460`), periodic roll (`isotropic_model.py:375`), DropPath, field dropout | global torch CPU, numpy, CUDA | — | + the epoch's sequence of forward passes (deterministic) |

`seed_from` hashes its integer keys with `numpy.random.SeedSequence`, so two different key
tuples never give the same seed. Each use has its own leading tag, so different streams can't
coincide either.

### 2.1 What a draw depends on

The notebook's predictor (E1) re-implements dataset and window choice from these seeds alone. It
reproduces the real `BatchedMultisetSampler` + `DataLoader` **draw for draw**: 3 epochs × 3 ranks,
with recycling, `data_workers ∈ {0, 2, 4}`, and 0 or 2 forward draws per step. So:
**[verified E1, E4]**

- **Window choice is a pure function of `(seed, global rank, epoch, dataset)`.** It doesn't
  depend on `data_workers`, the prefetch depth, the forward passes, validation, or anything that
  happened in earlier epochs.
- `dt` and the rotation are a function of `(seed, rank, epoch, data_workers)`. The worker count
  matters because micro-batches are dealt to workers round-robin and each worker has its own
  numpy stream. **[code-read]**
- Jitter, roll and dropout are a function of `(seed, rank, epoch)` and the model's forward
  passes within the epoch, which are themselves deterministic.

One torch detail matters for exact replay: when a pass runs out, `RandomSampler.__iter__`
still evaluates its trailing `randperm(n)[:n % n]` (an empty slice, `sampler.py:198-200`), which
advances the generator once more before the next pass. **[verified E1]**

*History (before the per-epoch seeding).* The global RNG was seeded only at start-up (`seed +
1_000_003·rank`, `train.py:155`), and each `RandomSampler` seeded every pass from it at first
use. Window choice then depended on the DataLoader's `_base_seed` draw, the prefetch depth
(`2·data_workers` batches drawn up front), and every forward pass's jitter draws in all earlier
epochs, including validation. The dataset-choice seed was `100·epoch + 10·0 + sampling_rank`:
`cfg.seed` never reached it, and the linear formula collided (seed 10, epoch 1 = seed 0, epoch 2).

---

## 3. Repetition, case by case

"Repeat" means **the same window index** is drawn again: the same file, trajectory and start
frame. §4 discusses whether that is the same *sample*.

### 3.1 Within one pass over a dataset: never

A pass is one `randperm(W_d)`, consumed in order, so no window repeats until the dataset runs
out. **[verified E3]**

### 3.2 Within one epoch: only if the dataset runs out

The sampler creates a new iterator for every dataset at the start of each epoch. If a dataset
runs out mid-epoch and `recycle=True` (the default), `StopIteration` is caught:

- the **partial batch is discarded**, so `W_d mod B` windows of every pass are never yielded,
  and *which* ones changes from pass to pass;
- the next pass is a **new permutation** from the same per-dataset generator;
- the step counter is decremented and a **new dataset is drawn** for this step.

**[verified E3]** For the PT9 mixture this never happens: each rank draws about
`B·max_samples/K ≈ 444` windows of each dataset per epoch, against `W_d ≥ 96 192`. For the
held-out datasets finetuned alone it also doesn't (4,000 draws per rank-epoch, against
`W_d ≥ 5 422`). It would start at `max_samples > W_d / B`, e.g. 2,712 for VE.

### 3.3 Across epochs: independent, with no memory

Each epoch draws a **fresh** permutation of every dataset, keyed on the epoch, not a continuation
of last epoch's. The next epoch's draws are an independent random subset, and the overlap between
two epochs is the hypergeometric `r²/W` (E3: 23 shared windows observed vs. 14 expected for
`r ≈ 260`, `W = 5000`). **[verified E3]** Nothing steers draws towards unseen windows, so
coverage grows like `1 − e^{−λ}` (§5), not linearly.

### 3.4 Across GPUs in a node: independent, and nothing deduplicates

- **Which dataset.** Under `fsdp`, `sampling_rank = 0` on every GPU: the FSDP mesh is the whole
  world (`distribution_utils.py:43-47`), so `sync_group_rank = rank // world_size = 0`
  (`training.py:354-376`). Every GPU therefore draws the **same dataset at every step**. The
  sequence depends on `(seed, epoch)` and matches `count_WALRUS_flops.replay_epoch` exactly.
  **[verified E2]**
- **Which windows.** Each rank's window generators are keyed on its **global** rank, so each GPU
  draws its own **independent permutation** of the shared dataset. Nothing compares or partitions
  windows across ranks. **[verified E5]** Keying them on the sampling rank instead would give
  every GPU under fsdp identical windows; that's why the sampler takes `global_rank` separately.

Two consequences, both matching their formulas in E5:
- two GPUs share about `r²/W` windows per epoch;
- in the **same step**, the expected number of GPU pairs holding the same window is
  `C(R,2)·B²/W` per step of that dataset.

This overlap is accepted by design. It is negligible for pretraining and large for
single-dataset finetunes (§5.3).

Other distribution types, in one sentence each **[code-read]**: under `ddp`, `sampling_rank =
rank`, so each GPU also draws its own dataset sequence; under `hsdp`, GPUs of one sharding group
share a dataset sequence and groups differ. Window permutations are per global rank in all cases.

### 3.5 Across nodes: exactly as across GPUs

Nodes are just more ranks: the same shared dataset sequence under fsdp, independent window
permutations, and no deduplication. Doubling the node count doubles the draws per epoch from the
same pool. **[code-read, follows from E2/E5]**

The only built-in alternative is `allow_sharding_in_train=True` (default `False`,
`multidatamodule.py:74`), which uses `DistributedSampler` for each dataset. **[verified E6]**
- Ranks are **disjoint**: one shared permutation, with rank `r` taking every `R`-th entry.
- The permutation is seeded `cfg.seed + epoch`.
- A recycle within an epoch **replays the same order**.
- Not used in PT9.

### 3.6 Across a resumed job: identical to the uninterrupted run

A resume continues the epoch counter from the checkpoint (`start_epoch = epoch + 1`,
`train.py:294`). Every stream of an epoch is keyed on `(seed, rank, epoch)`, so a resumed job
draws exactly what the uninterrupted run would have drawn: datasets, windows, `dt`, rotations,
jitter and dropout, provided `data_workers` and the GPU count are the same.
- **[verified E6b]** epochs 4–5 of a job started fresh at epoch 4 equal epochs 4–5 of an
  uninterrupted job (PT9 window counts, `data_workers=10`, ranks 0 and 3).
- **[verified `tests/test_determinism.py`]** on CPU with the dummy dataset, an uninterrupted
  run and a run killed after epoch 1 and resumed end with **bit-identical weights**. It also
  checks that two same-seed runs are identical and that a different seed differs.

The GPU counterpart is
[`LeonardoRunScripts/PT9_2d/Determinism/`](../LeonardoRunScripts/PT9_2d/Determinism/). It compares
the weight difference after a resume against the float-noise floor of two uninterrupted runs.
Generated, not yet submitted.

The test also exposed an unrelated resume bug, now fixed. The upstream default scheduler
`InverseSqrtLinearRamps` is recursive (`get_lr` scales the current lr), and the constructor's
initial `step()` on resume applied the resumed epoch's factor a second time on top of the lr
loaded from the checkpoint. `train.py` now restores the loaded lr after building the scheduler.
PT9's `InverseSqrtLinearWarmupSqrtCooldown` is closed-form and was never affected.

*History.* Checkpoints store no RNG state, and the global RNG used to be seeded only at start-up.
A resumed job therefore restarted every stream from the seed, and its first epoch partly
**replayed epoch 1**: on average 0.9 datasets per resume, and at least one in 61 % of resume
epochs, redrew exactly their epoch-1 windows in the same order. About 0.2 % of the draws of a
3-job chain.

### 3.7 Deliberate reuse

`trainer.reuse_batches=True` (`training.py:1073`) caches and replays `grad_acc_steps` batches.
It defaults to `False` and is not used in PT9. **[code-read]**

---

## 4. What counts as "the same sample"

Three levels, from strictest to loosest:

| level | same when | how often in PT9 N1, 100 epochs (§5) |
|---|---|---|
| **identical sample** | same window **and** same `dt`, rotation, jitter, roll | rare: a repeated window usually gets a different `dt` (up to 5 values early in a trajectory, 1 near the end, E7), rotation and jitter |
| **same window** | same (file, trajectory, `t0`) | 8.7 % (GS) to 54.5 % (PS) of draws repeat an earlier window |
| **overlapping frames** | any of the 6 input frames was an input of an earlier draw | 77 % (GS) to 98 % (PS) of draws |

The frame level is the one that matters for "has the model seen this data before". Windows
within ±5 start frames of each other share input frames when `dt = 1`, and more with a larger
`dt`. So within 100 epochs almost every draw shares frames with an earlier one, even for GS where
only 17 % of windows have been drawn.

The token-based repetition factor `R_d = T_seen,d / T_unique,d` (notebook 3 §3.5) is the
average number of times each *frame-token* has been an input. It tracks the frame level, not the
window level.

---

## 5. Estimating repetition

**Model** [verified E3/E5, simulated E8]. Under fsdp each (rank, epoch) draws, for dataset `d`, a
uniform random subset of `r_{d,e} = B·c_{d,e}` of its `W_d` windows. Here `c_{d,e}` is the
number of steps on which dataset `d` is drawn in epoch `e`, the same on all ranks, and exactly
replayable with `count_WALRUS_flops.replay_epoch(seed, epoch, sampling_rank, ...)` (E2). Subsets
are independent across ranks and epochs. With `N_d = R·Σ_e r_{d,e}` draws and `λ = N_d/W_d`:

| quantity | exact / approximation |
|---|---|
| times a given window is drawn | sum of `R·E` independent Bernoulli(`r/W`) ≈ Poisson(`λ`) |
| distinct windows drawn (coverage) | `W[1 − Π_e (1 − r_{d,e}/W)^R] ≈ W(1 − e^{−λ})` |
| exact-repeat draws (share of all draws) | `1 − distinct/N` |
| windows drawn ≥ 2 times | `≈ 1 − e^{−λ}(1 + λ)` |
| pairwise rank overlap per epoch | `r²/W` |
| same-step shared windows per epoch | `c_{d,e}·C(R,2)·B²/W` |
| frame overlap | simulated (E8), with `dt` drawn by the real rule |

The closed forms agree with the simulation to within 0.2 percentage points on every row below.
`seed = 0`, `B=2`, `T_in=6`, `max_samples=2000`, fsdp, 100 epochs, `c_{d,e}` replayed. A different
seed changes `c_{d,e}` by about ±1 % (notebook 3 §3.5), so these numbers barely move.

### 5.1 PT9 N1 (1 node, 4 GPUs), 9-dataset mixture

| | W | draws/rank/epoch | λ | window coverage | exact-repeat draws | windows ≥ 2× | frame-overlap draws | same-step shared/epoch | pair overlap/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GS | 955,200 | 444 | 0.19 | 17.0 % | 8.7 % | 1.5 % | 76.7 % | 0.006 | 0.21 |
| EMQ-O | 380,000 | 439 | 0.46 | 37.0 % | 19.9 % | 7.9 % | 89.3 % | 0.014 | 0.51 |
| EMQ-P | 380,000 | 448 | 0.47 | 37.6 % | 20.2 % | 8.2 % | 89.6 % | 0.014 | 0.53 |
| AM | 313,600 | 443 | 0.57 | 43.2 % | 23.6 % | 11.1 % | 91.8 % | 0.017 | 0.63 |
| RB | 271,600 | 448 | 0.66 | 48.4 % | 26.8 % | 14.2 % | 93.0 % | 0.020 | 0.74 |
| AI | 268,608 | 442 | 0.66 | 48.2 % | 26.7 % | 14.1 % | 92.5 % | 0.020 | 0.73 |
| SF | 173,824 | 447 | 1.03 | 64.3 % | 37.5 % | 27.4 % | 95.5 % | 0.031 | 1.15 |
| AD | 153,600 | 444 | 1.16 | 68.6 % | 40.7 % | 32.2 % | 95.7 % | 0.035 | 1.28 |
| PS | 96,192 | 446 | 1.85 | 84.4 % | 54.5 % | 55.3 % | 97.6 % | 0.056 | 2.06 |

### 5.2 PT9 N2 (2 nodes, 8 GPUs)

Same per-rank draws, twice the ranks, so `λ` doubles.

| | λ | window coverage | exact-repeat draws | windows ≥ 2× | frame-overlap draws | same-step shared/epoch |
|---|---:|---:|---:|---:|---:|---:|
| GS | 0.37 | 31.0 % | 16.5 % | 5.4 % | 88.1 % | 0.026 |
| EMQ-O | 0.92 | 60.3 % | 34.7 % | 23.6 % | 94.7 % | 0.065 |
| EMQ-P | 0.94 | 61.1 % | 35.2 % | 24.3 % | 94.8 % | 0.066 |
| AM | 1.13 | 67.8 % | 40.1 % | 31.2 % | 95.9 % | 0.079 |
| RB | 1.32 | 73.3 % | 44.5 % | 38.0 % | 96.5 % | 0.092 |
| AI | 1.32 | 73.2 % | 44.4 % | 37.9 % | 96.3 % | 0.092 |
| SF | 2.06 | 87.2 % | 57.6 % | 60.9 % | 97.7 % | 0.144 |
| AD | 2.31 | 90.1 % | 61.0 % | 67.2 % | 97.9 % | 0.162 |
| PS | 3.71 | 97.6 % | 73.7 % | 88.4 % | 98.8 % | 0.259 |

Same-step sharing grows like `C(R,2)`, so 6 pairs at N1 against 28 at N2.

### 5.3 Held-out datasets, each finetuned alone (N1, PT9 knobs)

`K = 1`, so every step draws the one dataset: 4,000 windows per rank per epoch.

| | W | λ | window coverage | exact-repeat draws | same-step shared/epoch | pair overlap/epoch |
|---|---:|---:|---:|---:|---:|---:|
| HS | 18,304 | 87 | 100 % | 98.9 % | 2.6 | 874 |
| ACT | 13,125 | 122 | 100 % | 99.2 % | 3.7 | 1,219 |
| TRL2 | 6,840 | 234 | 100 % | 99.6 % | 7.0 | 2,339 |
| VE | 5,422 | 295 | 100 % | 99.7 % | 8.9 | 2,951 |

Every window is drawn within the first epoch or two. After that, every draw repeats a window, and
the 4 GPUs of one epoch overlap heavily (VE: any two ranks share 2,951 of their 4,000 windows).
Variation then comes only from `dt`, rotation and jitter. These are the cases where
`allow_sharding_in_train=True` (§3.5) or a smaller `max_samples` would change the picture.

---

## 6. Measuring repetition exactly

**Windows: replay offline.** Window choice is a pure function of `(seed, global rank, epoch,
dataset)` and the dataset sizes (§2.1). The notebook's E1 predictor, about 30 lines that touch
neither the model nor the data, therefore reproduces every rank's window sequence of a real run
exactly. Every window-level quantity in §5 can be computed as an exact count rather than an
expectation: repeats, coverage, same-step cross-rank collisions. Pass the run's `seed`,
`max_samples`, `batch_size`, world size and per-dataset window counts.

**`dt` and rotation: log them.** These depend on `data_workers` and on each worker's numpy stream,
so replaying them means emulating torch's worker seeding and the round-robin assignment. Logging
is simpler.

| where | what | why there |
|---|---|---|
| worker: `BatchInflatedWellDataset.__getitem__` (`inflated_dataset.py:308-336`) | `(index, file_idx, sample_idx, time_idx, dt)` per sample, plus the rotation | the method already collects `file_idxes`, `sample_idxes`, `time_idxes`, `dts` per sample (`:313-325`) and throws all but the last away, so it only needs persisting |
| main process: the batch indices in `train_one_epoch` | `(rank, epoch, step, dataset, global index)` | to align worker records with steps, and to cross-check the offline window replay |

Write one file per rank, and join on `(rank, epoch, step)`. The frame level then uses
`frames = t0 + dt·[0..5]`. Size: N1 for 100 epochs is 4 ranks × 2,000 steps × 100 epochs ×
2 windows = 1.6 M rows.

*History:* before the per-epoch seeding, window choice depended on every global-RNG call the
trainer made (forward passes, validation, loader seeds). Offline replay was possible only by
reproducing all of them, so logging was the only practical exact method.

---

## 7. Summary

| question | answer |
|---|---|
| How is a dataset chosen? | Uniformly per step, by a generator keyed on `(seed, epoch, sampling_rank)`: the same on every GPU (fsdp), different for each seed |
| How is a window chosen? | The next entry of a per-dataset `randperm` whose generator is keyed on `(seed, global rank, epoch, dataset)` |
| How are file, trajectory and `t0` chosen? | Not separately: they are arithmetic on the window index |
| Can the same window be drawn again on the same GPU? | Not within a pass. Across epochs: yes, independently (`r²/W` per epoch pair) |
| Can overlapping windows be drawn? | Yes, freely. By 100 epochs, 77–98 % of PT9 draws share frames with an earlier draw |
| Is anything preventing repeats? | Only the permutation within one pass. Nothing across epochs, GPUs or nodes |
| Across GPUs / nodes? | Same dataset at each step (fsdp), independent windows, no deduplication; collisions grow like `C(R,2)` |
| Resumed jobs? | Identical to the uninterrupted run (same `data_workers` and GPU count) |
| Estimate? | Poisson(`λ = N/W`) with `N` from the replayed dataset counts (§5) |
| Exact? | Windows: replay offline with the E1 predictor. `dt`/rotation: log in the worker (§6) |

---

## 8. Inventory: what can make a run non-deterministic

Everything on the PT9 training path that draws a random number or can change results between
two runs with the same config, and how it is handled.

| # | source | where | status |
|---|---|---|---|
| 1 | Global RNGs seeded only at start-up; no RNG state in checkpoints | `train.py:155`, `checkpoints.py` | **fixed**: re-seeded every epoch from `(seed, rank, epoch)`, `Trainer.train` |
| 2 | Window permutations seeded from the global CPU RNG at each pass | `RandomSampler` | **fixed**: own generator per `(seed, global rank, epoch, dataset)` |
| 3 | DataLoader `_base_seed` → worker seeds → `dt`, rotation | `dataloader.py:666`, `_utils/worker.py:260-267`, the_well `np.random` | **fixed** per epoch by #1; still depends on `data_workers` (keep it fixed within a chain) |
| 4 | Patch jitter, in training and validation | `patch_jitterers.py:460` (global torch CPU) | **fixed** by #1 |
| 5 | Periodic roll between blocks | `isotropic_model.py:375` (numpy) | **fixed** by #1 |
| 6 | DropPath (`drop_path=0.05`), field dropout | timm `DropPath`, `F.dropout3d` (CUDA RNG) | **fixed** by #1 (`torch.manual_seed` seeds CUDA) |
| 7 | Validation consumes the RNGs between epochs | validation loaders and forward passes | **fixed** by #1: the next epoch re-seeds |
| 8 | Dataset-choice seed ignored `cfg.seed` and collided (`100e + 10s + r`) | `mixed_dset_sampler.py` | **fixed**: `seed_from(SEED_DATASET_CHOICE, seed, epoch, sampling_rank)` |
| 9 | `distribution=local` never called `set_epoch`, so every epoch got the same dataset order | `training.py` | **fixed**: called unconditionally |
| 10 | Resume re-applied one LR-scheduler step for recursive schedulers | `train.py` scheduler construction, `InverseSqrtLinearRamps.get_lr` | **fixed**: the loaded lr is restored (PT9's closed-form scheduler was never affected) |
| 11 | GPU floating-point non-determinism: SDPA backward, the backward of indexed weights (`proj1.weight[:, field_indices]`), collective reduction order | CUDA kernels, NCCL | **not addressed**: same draws, slightly different floats. Would need `torch.use_deterministic_algorithms(True)` and a cuBLAS workspace setting, at a speed cost |
| 12 | Changing `data_workers` or the GPU count between the jobs of one chain | — | **by design**: changes `dt`/rotation (workers), or every rank's streams (GPU count). Keep both fixed within a chain |

Already deterministic before the change: the validation samplers (fixed `manual_seed(0)` /
`DistributedSampler(seed=0)`), and gradient-checkpoint recompute (`preserve_rng_state=True`
restores the RNG state around it). `reuse_batches` (python `random.shuffle`) is off and is covered
by #1 anyway. `choose_kernel_size_random` (python `random`) is only reached when
`variable_deterministic_ds` is off; it is on in every encoder config, and covered by #1 anyway.
