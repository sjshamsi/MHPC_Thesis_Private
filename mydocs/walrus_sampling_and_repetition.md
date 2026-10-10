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

**Sampling scheme (branch `fulldet`).** Every random stream is keyed on
`(cfg.seed, rank, epoch[, dataset])` through one helper, `seed_from`
(`walrus/utils/experiment_utils.py`), as on branch `flopcount`. In addition, window choice
follows one **epoch plan** that every rank computes identically (`epoch_plan`,
`walrus/data/mixed_dset_sampler.py`). Each dataset has a single window stream per epoch, keyed on
`(seed, epoch, dataset)` with no rank in the key, and each rank takes its own positions in it.
**Within an epoch no two GPUs or nodes get the same window of a dataset until every window of it
has been used.** The earlier behaviours are kept in the *History* notes, because runs made with
them exist:
- `flopcount`: independent per-rank permutations;
- `656c00c`: seeding from the global RNG.

The code changes themselves, file by file, are listed in
[determinism_code_changes.md](determinism_code_changes.md).

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
├─ 1. WHICH DATASET d          epoch_plan: one torch.Generator per row (sampling rank),
│                              seeded seed_from(DATASET_CHOICE, seed, e, row)
│                              d ~ Uniform{0..K-1}; every rank replays every row  mixed_dset_sampler.py
│
├─ 2. WHICH WINDOWS (B of them) the next B positions of dataset d's epoch stream that the plan
│                              gives rank k (step by step, rows in order, ranks of a row in order);
│                              position p → perm_{p // W_d}[p mod W_d], each perm a randperm(W_d)
│                              seeded seed_from(WINDOWS, seed, e, d, pass)   (no rank in the key)
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
| dataset `d` per step | one `torch.Generator` per row in `epoch_plan`, replayed by every rank | `seed_from(SEED_DATASET_CHOICE, seed, epoch, row)` | seed, epoch, and the row, i.e. the sampling rank (0 on every GPU under fsdp) |
| window permutations of each dataset (one per pass of the epoch's stream) | a fresh `torch.Generator` per permutation | `seed_from(SEED_WINDOWS, seed, epoch, dataset, pass)` | seed, epoch, dataset; **not** the rank |
| which positions of the stream a rank takes | none: arithmetic in `epoch_plan` | — | the dataset choices of all rows and the layout (`sampling_ranks`) |
| everything else on the global RNGs (torch CPU + CUDA, numpy, `random`) | re-seeded at the top of every epoch (`Trainer.train`) | `seed_all(seed_from(SEED_GLOBAL, seed, global_rank, epoch))` | seed, epoch, global rank |
| …which then feeds: DataLoader worker seeds → `dt`, rotation | `_base_seed` (`dataloader.py:666`), then worker numpy | — | + `data_workers` |
| …patch jitter (`patch_jitterers.py:460`), periodic roll (`isotropic_model.py:375`), DropPath, field dropout | global torch CPU, numpy, CUDA | — | + the epoch's sequence of forward passes (deterministic) |

`seed_from` hashes its integer keys with `numpy.random.SeedSequence`, so two different key
tuples never give the same seed. Each use has its own leading tag, so different streams can't
coincide either.

### 2.1 What a draw depends on

The notebook's predictor (E1) re-implements dataset and window choice from these seeds alone. It
reproduces the real `BatchedMultisetSampler` + `DataLoader` **draw for draw**:
- fsdp, HSDP (2×2) and DDP layouts;
- 2 epochs, ranks 0 and 3, with streams that wrap;
- `data_workers ∈ {0, 2, 4}`, and 0 or 2 forward draws per step.

So: **[verified E1, E4]**

- **A rank's windows are a pure function of `(seed, epoch, layout, rank)`** and the dataset
  sizes. The layout is `sampling_ranks`, the sampling rank of every global rank. The windows don't
  depend on `data_workers`, the prefetch depth, the forward passes, validation, or anything that
  happened in earlier epochs.
- `dt` and the rotation are a function of `(seed, rank, epoch, data_workers)`. The worker count
  matters because micro-batches are dealt to workers round-robin and each worker has its own
  numpy stream. **[code-read]**
- Jitter, roll and dropout are a function of `(seed, rank, epoch)` and the model's forward
  passes within the epoch, which are themselves deterministic.

**How the plan is built** (`epoch_plan`, `mixed_dset_sampler.py`). Every rank runs the same loop
over the `max_samples` steps. At each step, for each row in order, the row's generator picks a
dataset `d`. Then each rank of that row, in rank order, takes the next `dset_batch[d]` positions
of `d`'s stream. A rank keeps only its own entries. Measured on the login node at PT9 sizes (9
datasets, `max_samples=2000`), the plan takes 27 ms per epoch for fsdp on 4 GPUs, 79 ms for fsdp
on 64, and 0.67 s for DDP on 64 GPUs (64 rows). On top of that comes one `randperm(W_d)` per
dataset and pass (12 ms for GS's 955,200 windows). **[measured]**

*History (`flopcount`).* Each dataset's `RandomSampler` had its own generator, keyed on
`(seed, global rank, epoch, dataset)`: each rank drew its windows on its own, and
ranks were independent (§3.4). On a recycle, `RandomSampler` also evaluated an empty trailing
`randperm(n)[:0]` (`sampler.py:198-200`), which advanced the generator once more.

*History (before the per-epoch seeding, `656c00c`).* The global RNG was seeded only at start-up
(`seed + 1_000_003·rank`, `train.py:155`), and each `RandomSampler` seeded every pass from it at
first use. Window choice then depended on the DataLoader's `_base_seed` draw, the prefetch depth
(`2·data_workers` batches drawn up front), and every forward pass's jitter draws in all earlier
epochs, including validation. The dataset-choice seed was `100·epoch + 10·0 + sampling_rank`:
`cfg.seed` never reached it, and the linear formula collided (seed 10, epoch 1 = seed 0, epoch 2).

---

## 3. Repetition, case by case

"Repeat" means **the same window index** is drawn again: the same file, trajectory and start
frame. §4 discusses whether that is the same *sample*.

### 3.1 Within one pass over a dataset's stream: never

A pass is one `randperm(W_d)`, and all ranks share it, each taking its own positions. No window
repeats, on any rank, until all `W_d` windows have been used. **[verified E3, E5]**

### 3.2 Within one epoch: only once the dataset is used up

If an epoch's total demand for a dataset, over all ranks, exceeds `W_d`, its stream **wraps**
into the next permutation (`pass + 1`), and the epoch then uses every window `⌊N/W⌋` or `⌈N/W⌉`
times. Nothing is dropped: a micro-batch that crosses the end of a permutation takes the rest
of its windows from the next one. With `recycle=False`, a dataset that can't fill the next step
is removed for every rank at once, and when none is left every rank stops on the same step.
**[verified E3; `tests/test_fulldet_sampler.py`]**

For the PT9 mixture this never happens: the 4 ranks together draw about `1 780` windows of each
dataset per epoch, against `W_d ≥ 96 192`. For the held-out datasets finetuned alone (16,000 draws
per epoch over 4 ranks), it happens for ACT, TRL2 and VE (`W_d ≤ 13 125`), but not HS (18,304).

*History (`flopcount`).* Each rank's own permutation could run out. The partial batch was then
discarded (`W_d mod B` windows of every pass were never yielded), a new permutation started, and
a new dataset was drawn for the step.

### 3.3 Across epochs: independent, with no memory

Each epoch starts **fresh** permutations, keyed on the epoch, rather than continuing the last
epoch's. The next epoch's draws are an independent random subset, and the overlap between two
epochs is the hypergeometric `N_1 N_2 / W` (E3: 202 shared windows observed vs. 203 expected for
`N ≈ 1000`, `W = 5000`). **[verified E3]** Nothing steers draws towards windows unseen in earlier
epochs, so coverage over a run grows like `1 − e^{−λ}` (§5), not linearly.

### 3.4 Across GPUs in a node: disjoint within an epoch

- **Which dataset.** Under `fsdp`, `sampling_rank = 0` on every GPU: the FSDP mesh is the whole
  world (`distribution_utils.py:43-47`), so `sampling_ranks_for` gives `[0] * world_size`
  (`training.py:126`). Every GPU therefore draws the **same dataset at every step**. The
  sequence depends on `(seed, epoch)` and matches `count_WALRUS_flops.replay_epoch` exactly.
  **[verified E2]**
- **Which windows.** All GPUs share each dataset's stream and take disjoint positions in it, so
  two GPUs share a window in an epoch only once the epoch's demand exceeds `W_d`. Even then,
  every window is used before any is used again. **[verified E5]** The GPU check
  `LeonardoRunScripts/PT9_2d/Determinism/check_disjoint.py` asserts the same on the real PT9
  datasets under torchrun: no window drawn twice in 2 epochs, under fsdp and under HSDP.

Under `ddp`, `sampling_rank = rank`, so each GPU draws its own dataset sequence. Under `hsdp`,
the GPUs of one sharding group share a sequence, and groups differ. Rows then use up a stream at
different rates. Every rank replays every row's choices, though, so the positions still never
collide. **[verified E1, E5]**

*History (`flopcount`).* Each GPU drew its own **independent permutation** of the shared dataset.
Two GPUs shared about `r²/W` windows per epoch (`r` = a rank's draws), and in the same step the
expected number of GPU pairs holding the same window was `C(R,2)·B²/W` per step of that dataset.
E5 shows the contrast: for `W = 3000` and 8,000 draws over 4 ranks, the pairwise overlap is 696
against 1,331 for independent ranks, and every window is used 2–3 times instead of 0–4 (measured with `flopcount`'s sampler).

### 3.5 Across nodes: exactly as across GPUs

Nodes are just more ranks: the same plan, the same disjoint positions. Doubling the node count
doubles the draws per epoch from the same pool. Within an epoch, those draws are still distinct
until the pool is used up. **[code-read, follows from E1/E5]**

`allow_sharding_in_train=True`, the built-in `DistributedSampler` partition, is **ignored** on
this branch. As `flopcount`'s E6 showed, it splits each dataset into fixed `1/R` slices. Under
HSDP or DDP, a rank that empties its slice repeats it while other slices still hold unused
windows. A recycle within an epoch also replays the same order, and `drop_last=False` pads the
slices with duplicates. The shared plan avoids all three.

### 3.6 Across a resumed job: identical to the uninterrupted run

A resume continues the epoch counter from the checkpoint (`start_epoch = epoch + 1`,
`train.py:294`). The epoch plan is keyed on `(seed, epoch)` and the layout, and every other
stream on `(seed, rank, epoch)`. So a resumed job draws exactly what the uninterrupted run would
have drawn: datasets, windows, `dt`, rotations, jitter and dropout, provided `data_workers` and the
GPU count are the same.
- **[verified E6b]** epochs 4–5 of a job started fresh at epoch 4 equal epochs 4–5 of an
  uninterrupted job (PT9 window counts, `data_workers=10`, ranks 0 and 3).
- **[verified `tests/test_determinism.py`]** on CPU with the dummy dataset, an uninterrupted
  run and a run killed after epoch 1 and resumed end with **bit-identical weights**. It also
  checks that two same-seed runs are identical and that a different seed differs.

The GPU counterpart is
[`LeonardoRunScripts/PT9_2d/Determinism/`](../LeonardoRunScripts/PT9_2d/Determinism/). It compares
the weight difference after a resume against the float-noise floor of two uninterrupted runs.
Its resume run hit an unrelated DataLoader fork hang; see
[dataloader_fork_hang/README.md](dataloader_fork_hang/README.md).

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
| **same window** | same (file, trajectory, `t0`) | 8.7 % (GS) to 54.3 % (PS) of draws repeat an earlier window |
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

**Model** [verified E3/E5, simulated E8]. Under fsdp, in epoch `e` the `R` ranks together draw
`N_{d,e} = R·B·c_{d,e}` windows of dataset `d`. Here `c_{d,e}` is the number of steps on which
`d` is drawn, the same on all ranks, and exactly replayable with
`count_WALRUS_flops.replay_epoch(seed, epoch, sampling_ranks, ...)` (E2). These draws are the
first `N_{d,e}` positions of the epoch's shared stream:
- a uniform random subset with no repeats when `N_{d,e} ≤ W_d`;
- otherwise, every window `⌊N/W⌋` or `⌈N/W⌉` times.

Epochs are independent. With `N_d = Σ_e N_{d,e}` and `λ = N_d/W_d`:

| quantity | exact / approximation |
|---|---|
| times a given window is drawn | sum of `E` independent Bernoulli(`N_{d,e}/W`) (while `N_{d,e} ≤ W`) ≈ Poisson(`λ`) |
| distinct windows drawn (coverage) | `W[1 − Π_e (1 − min(N_{d,e}, W)/W)] ≈ W(1 − e^{−λ})` |
| exact-repeat draws (share of all draws) | `1 − distinct/N` |
| windows drawn ≥ 2 times | `≈ 1 − e^{−λ}(1 + λ)` |
| windows shared by two ranks in one epoch | 0 while `N_{d,e} ≤ W` |
| frame overlap | simulated (E8), with `dt` drawn by the real rule |

The closed forms agree with the simulation to within 0.2 percentage points on every row below.
`seed = 0`, `B=2`, `T_in=6`, `max_samples=2000`, fsdp, 100 epochs, `c_{d,e}` replayed. A different
seed changes `c_{d,e}` by about ±1 % (notebook 3 §3.5), so these numbers barely move.

**What the shared plan changes, and what it doesn't.** It removes repeats *within* an epoch.
Across a whole run, repetition is dominated by repeats *between* epochs, which both schemes have
in the same amount. When `N_{d,e} ≪ W_d`, as in PT9 pretraining, the independent ranks of
`flopcount` rarely collided anyway, so the run-level numbers barely move (the "flopcount" column:
at most 0.2 percentage points more exact repeats). The difference is large when one epoch's
demand is comparable to `W_d`, as in the single-dataset finetunes (§5.3).

### 5.1 PT9 N1 (1 node, 4 GPUs), 9-dataset mixture

| | W | draws/epoch (4 ranks) | λ | window coverage | exact-repeat draws | ditto, `flopcount` | windows ≥ 2× | frame-overlap draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GS | 955,200 | 1,775 | 0.19 | 17.0 % | 8.7 % | 8.7 % | 1.5 % | 76.6 % |
| EMQ-O | 380,000 | 1,755 | 0.46 | 37.0 % | 19.8 % | 19.9 % | 7.9 % | 89.4 % |
| EMQ-P | 380,000 | 1,792 | 0.47 | 37.7 % | 20.1 % | 20.2 % | 8.2 % | 89.6 % |
| AM | 313,600 | 1,773 | 0.57 | 43.3 % | 23.5 % | 23.6 % | 11.1 % | 91.8 % |
| RB | 271,600 | 1,793 | 0.66 | 48.4 % | 26.6 % | 26.8 % | 14.2 % | 93.0 % |
| AI | 268,608 | 1,767 | 0.66 | 48.3 % | 26.6 % | 26.7 % | 14.1 % | 92.5 % |
| SF | 173,824 | 1,786 | 1.03 | 64.4 % | 37.3 % | 37.5 % | 27.4 % | 95.5 % |
| AD | 153,600 | 1,776 | 1.16 | 68.7 % | 40.5 % | 40.7 % | 32.2 % | 95.8 % |
| PS | 96,192 | 1,782 | 1.85 | 84.6 % | 54.3 % | 54.5 % | 55.3 % | 97.6 % |

### 5.2 PT9 N2 (2 nodes, 8 GPUs)

Same per-rank draws, twice the ranks, so `λ` doubles.

| | λ | window coverage | exact-repeat draws | ditto, `flopcount` | windows ≥ 2× | frame-overlap draws |
|---|---:|---:|---:|---:|---:|---:|
| GS | 0.37 | 31.1 % | 16.3 % | 16.5 % | 5.4 % | 88.1 % |
| EMQ-O | 0.92 | 60.5 % | 34.5 % | 34.7 % | 23.6 % | 94.7 % |
| EMQ-P | 0.94 | 61.2 % | 35.1 % | 35.2 % | 24.3 % | 94.7 % |
| AM | 1.13 | 67.9 % | 39.9 % | 40.1 % | 31.2 % | 95.9 % |
| RB | 1.32 | 73.5 % | 44.3 % | 44.5 % | 38.0 % | 96.5 % |
| AI | 1.32 | 73.4 % | 44.2 % | 44.4 % | 37.9 % | 96.3 % |
| SF | 2.06 | 87.5 % | 57.4 % | 57.6 % | 60.9 % | 97.7 % |
| AD | 2.31 | 90.4 % | 60.9 % | 61.0 % | 67.2 % | 97.9 % |
| PS | 3.71 | 97.7 % | 73.6 % | 73.7 % | 88.4 % | 98.8 % |

### 5.3 Held-out datasets, each finetuned alone (N1, PT9 knobs)

`K = 1`, so every step draws the one dataset: 4,000 windows per rank, 16,000 per epoch over the 4
ranks. Over 100 epochs every window is drawn many times under either scheme (`λ` = 87 to 295;
99 % of draws repeat a window). The shared plan changes what happens **within each epoch**:

| | W | λ (100 epochs) | windows used in one epoch | ditto, `flopcount` | max uses of a window in one epoch | ditto, `flopcount` |
|---|---:|---:|---:|---:|---:|---:|
| HS | 18,304 | 87 | 87.4 % | 62.7 % | 1 | 4 |
| ACT | 13,125 | 122 | 100 % | 76.6 % | 2 | 4 |
| TRL2 | 6,840 | 234 | 100 % | 97.0 % | 3 | 4 |
| VE | 5,422 | 295 | 100 % | 99.5 % | 3 | 4 |

`flopcount`'s values are `1 − (1 − 4000/W)^4` for the share used, and 4 for the maximum (once per
rank). Under the shared plan, each epoch sweeps the dataset evenly: ACT, TRL2 and VE use every
window `⌊N/W⌋` or `⌈N/W⌉` times, and HS uses 16,000 distinct windows. After the first epoch,
variation still comes mostly from `dt`, rotation and jitter. A smaller `max_samples` would change
the picture more than the sampling scheme does.

---

## 6. Measuring repetition exactly

**Windows: replay offline.** Window choice is a pure function of `(seed, epoch, layout, rank)`
and the dataset sizes (§2.1). `BatchedMultisetSampler` itself, built on a stand-in dataset that
only has the window counts (notebook 5 E0), therefore reproduces every rank's window sequence of
a real run exactly, without the model or the data. The E1 predictor does the same in about 30
lines of plain torch. Every window-level quantity in §5 can be computed as an exact count rather
than an expectation: repeats and coverage. Pass the run's `seed`, `max_samples`, `batch_size`,
`sampling_ranks` and per-dataset window counts.

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
| How is a window chosen? | The next positions of the dataset's epoch stream (`randperm`s keyed on `(seed, epoch, dataset, pass)`, no rank) that the shared epoch plan gives this rank |
| How are file, trajectory and `t0` chosen? | Not separately: they are arithmetic on the window index |
| Can the same window be drawn again? | Within an epoch: not on any GPU until every window of the dataset is used. Across epochs: yes, independently (`N_1 N_2/W` per epoch pair) |
| Can overlapping windows be drawn? | Yes, freely. By 100 epochs, 77–98 % of PT9 draws share frames with an earlier draw |
| Is anything preventing repeats? | The shared stream within an epoch, across all GPUs and nodes. Nothing across epochs |
| Across GPUs / nodes? | Same dataset at each step (fsdp); disjoint windows within an epoch (HSDP and DDP too) |
| Resumed jobs? | Identical to the uninterrupted run (same `data_workers` and GPU count) |
| Estimate? | Poisson(`λ = N/W`) with `N` from the replayed dataset counts (§5) |
| Exact? | Windows: replay offline with the sampler on a stand-in dataset (or the E1 predictor). `dt`/rotation: log in the worker (§6) |

---

## 8. Inventory: what can make a run non-deterministic

Everything on the PT9 training path that draws a random number or can change results between
two runs with the same config, and how it is handled.

| # | source | where | status |
|---|---|---|---|
| 1 | Global RNGs seeded only at start-up; no RNG state in checkpoints | `train.py:155`, `checkpoints.py` | **fixed**: re-seeded every epoch from `(seed, rank, epoch)`, `Trainer.train` |
| 2 | Window permutations seeded from the global CPU RNG at each pass | `RandomSampler` | **fixed**: replaced by the shared epoch plan, permutations keyed on `(seed, epoch, dataset, pass)` (`epoch_plan`) |
| 3 | DataLoader `_base_seed` → worker seeds → `dt`, rotation | `dataloader.py:666`, `_utils/worker.py:260-267`, the_well `np.random` | **fixed** per epoch by #1; still depends on `data_workers` (keep it fixed within a chain) |
| 4 | Patch jitter, in training and validation | `patch_jitterers.py:460` (global torch CPU) | **fixed** by #1 |
| 5 | Periodic roll between blocks | `isotropic_model.py:375` (numpy) | **fixed** by #1 |
| 6 | DropPath (`drop_path=0.05`), field dropout | timm `DropPath`, `F.dropout3d` (CUDA RNG) | **fixed** by #1 (`torch.manual_seed` seeds CUDA) |
| 7 | Validation consumes the RNGs between epochs | validation loaders and forward passes | **fixed** by #1: the next epoch re-seeds |
| 8 | Dataset-choice seed ignored `cfg.seed` and collided (`100e + 10s + r`) | `mixed_dset_sampler.py` | **fixed**: `seed_from(SEED_DATASET_CHOICE, seed, epoch, sampling_rank)` |
| 9 | `distribution=local` never called `set_epoch`, so every epoch got the same dataset order | `training.py` | **fixed**: called unconditionally |
| 10 | Resume re-applied one LR-scheduler step for recursive schedulers | `train.py` scheduler construction, `InverseSqrtLinearRamps.get_lr` | **fixed**: the loaded lr is restored (PT9's closed-form scheduler was never affected) |
| 11 | GPU floating-point non-determinism: SDPA backward, the backward of indexed weights (`proj1.weight[:, field_indices]`), collective reduction order | CUDA kernels, NCCL | **not addressed**: same draws, slightly different floats. Would need `torch.use_deterministic_algorithms(True)` and a cuBLAS workspace setting, at a speed cost |
| 12 | Changing `data_workers` or the GPU count between the jobs of one chain | — | **by design**: changes `dt`/rotation (workers), or the epoch plan and every rank's streams (GPU count). Keep both fixed within a chain |
| 13 | Ranks drew windows independently, so two GPUs could train on the same window in one epoch | `BatchedMultisetSampler` (`flopcount`) | **fixed** (`fulldet`): one plan per epoch, disjoint positions of a shared stream per dataset |

Already deterministic before the change: the validation samplers (fixed `manual_seed(0)` /
`DistributedSampler(seed=0)`), and gradient-checkpoint recompute (`preserve_rng_state=True`
restores the RNG state around it). `reuse_batches` (python `random.shuffle`) is off and is covered
by #1 anyway. `choose_kernel_size_random` (python `random`) is only reached when
`variable_deterministic_ds` is off; it is on in every encoder config, and covered by #1 anyway.
