# How WALRUS samples training windows, and how often they repeat

This document follows one training window from the dataset mixture to the GPU. It shows exactly
which random number decides each choice, lists every way a window can be drawn again, and
explains how to estimate repetition and how to measure it exactly.

Claims are tagged **[verified]** when an `assert` in
[`demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb`](../demo_notebooks/walrus_example_5_SamplingAndRepetition.ipynb)
checks them (section labels E1–E8 match the notebook), and **[code-read]** when they rest on
reading the code only. Library line numbers refer to the Leonardo venv: **torch 2.5.1** and its
bundled `the_well`. The RNG details in §2 and §3.6 depend on torch's `sampler.py` and
`dataloader.py`, so recheck them after a torch upgrade by re-running the notebook.

Related: [`walrus_tokens_and_data_pipeline.md`](walrus_tokens_and_data_pipeline.md) §2 (data
objects, one micro-step traced) and §4 (seeding history); notebook 3 §3 (token counts, `R_d`,
coverage).

---

## 1. The sampling hierarchy

One micro-batch is chosen in four nested steps. Only the first two are decided in the main
process; `dt` and the rotation are decided in the data-loader worker that reads the data.

```
step t of epoch e on rank k
│
├─ 1. WHICH DATASET d          BatchedMultisetSampler, own torch.Generator
│                              seeded 100·e + 10·0 + sampling_rank          mixed_dset_sampler.py:172
│                              d ~ Uniform{0..K-1}                          :175-181
│
├─ 2. WHICH WINDOWS (B of them) RandomSampler of dataset d: next B entries of a randperm(W_d)
│                              the permutation is seeded from the GLOBAL CPU torch RNG
│                              on its first next() after (re)creation      torch sampler.py:175-199
│                              → B global indices (local + offset_d)       mixed_dset_sampler.py:182-194
│
├─ 3. WHERE IN THE FILES        MixedWellDataset: dataset from searchsorted(offsets)   multidataset.py:374-385
│                              WellDataset._load_one_sample(index)                     the_well datasets.py:780-792
│                                file  = searchsorted(file_index_offsets, index) − 1
│                                traj  = local // windows_per_traj(file)   windows_per_traj = n_steps − (T_in+T_out) + 1
│                                t0    = local %  windows_per_traj(file)
│
└─ 4. HOW THE WINDOW IS READ    dt ~ randint(min_dt, min(max_dt, (n_steps − t0 − 1)//(T_in+T_out−1)) + 1)
                               np.random in the worker                                 the_well datasets.py:801-813
                               RandomRotation90 (np.random, once per micro-batch), patch jitter and
                               periodic roll in the forward pass
```

All `B` windows of a micro-batch come from the same dataset and may come from different files,
trajectories and start frames. **[verified E1, E7]**

---

## 2. The random number generators and who seeds them

| decision | generator | seeded by | depends on `cfg.seed`? | depends on rank? |
|---|---|---|---|---|
| dataset `d` per step | sampler's own `torch.Generator` | `100·epoch + sampling_rank` (`mixed_dset_sampler.py:172`; the sampler's `seed` is never passed, `multidatamodule.py:419`) | **no** | only via `sampling_rank` (0 on every GPU under fsdp) |
| window permutation of each dataset | fresh `torch.Generator` per pass | one draw from the **global CPU torch RNG** at the pass's first `next()` | yes | yes (`seed + 1_000_003·rank`, `train.py:155`) |
| DataLoader worker seeds | — | `_base_seed`, one global-RNG draw per `iter(loader)` (`dataloader.py:666`); worker `w` seeds torch, numpy and `random` from it (`_utils/worker.py:260-267`) | yes | yes |
| `dt`, rotation | worker's numpy | worker seed | yes | yes |
| patch jitter | **global CPU torch RNG**, one `torch.randint` per non-singleton axis per forward (`FixedPatchJittererBoundaryPad`, `patch_jitterers.py:460`; PT9 sets `use_periodic_fixed_jitter=True`) | — | yes | yes |
| periodic roll between blocks | main-process numpy (`isotropic_model.py:375`) | — | yes | yes |

The important row pair is the second and fifth: **the window permutations and the forward pass
draw from the same global CPU RNG.** The window drawn at any step therefore depends on how many
forward passes ran before its permutation was seeded. **[verified E1, E4]**

### 2.1 Exactly when each draw happens

The notebook's predictor (E1) re-implements the sampler and loader using only these global-RNG
calls, in this order. It reproduces the real `BatchedMultisetSampler` + `DataLoader` **draw for
draw**: 3 epochs × 3 ranks, `data_workers ∈ {0, 2, 4}`, 0 or 2 forward draws per step, with
recycling. **[verified E1]**

1. `iter(loader)` at the start of each epoch (`training.py:1062`): one draw for `_base_seed`.
2. Batches are drawn by the sampler in the main process:
   - `data_workers = 0`: one batch just before each `next()` returns it.
   - `data_workers = w > 0`: `2w` batches up front (`prefetch_factor=2`, `dataloader.py:1227`),
     then one more at each `next()`, *before* the batch is returned (`_process_data`,
     `:1487-1489`). PT9 runs use `w ∈ {8, 10, 16}`, i.e. 16–32 batches prefetched.
3. When a batch is the first of its dataset in this pass, one draw seeds that dataset's
   permutation. `RandomSampler.__iter__` is a Python generator, so the seed is drawn lazily, at
   first use, not when the sampler is built.
4. The forward pass of each batch: 2 jitter draws for 2D data.
5. Between epochs: validation, when it runs (`val_frequency`, `rollout_val_frequency`), draws a
   `_base_seed` for each validation loader iterator and 2 jitter draws per validation forward.
   **[code-read]**

Gradient checkpointing does not change the stream: `torch.utils.checkpoint` saves and restores
the RNG state around the recompute (`preserve_rng_state=True`, the default). **[code-read]**

---

## 3. Repetition, case by case

"Repeat" here means **the same window index** is drawn again: the same file, trajectory and
start frame. §4 discusses whether that is the same *sample*.

### 3.1 Within one pass over a dataset: never

A pass is one `randperm(W_d)`, consumed in order, so no window repeats until the dataset runs
out. **[verified E3]**

### 3.2 Within one epoch: only if the dataset runs out

The sampler creates a new iterator for every dataset at the start of each epoch
(`samplers = [iter(s) for s in self.sub_samplers]`, `mixed_dset_sampler.py:168`). If a dataset
runs out mid-epoch and `recycle=True` (the default), `StopIteration` is caught (`:195-206`):

- the **partial batch is discarded**, so `W_d mod B` windows of every pass are never yielded,
  and *which* ones changes from pass to pass;
- the iterator is recreated, so the next pass is a **new permutation**, seeded at its first use;
- the step counter is decremented and a **new dataset is drawn** for this step.

So a window can be drawn twice in one epoch only after its dataset has run out in that epoch.
**[verified E3]** For the PT9 mixture this never happens: each rank draws about
`B·max_samples/K ≈ 444` windows of each dataset per epoch, against `W_d ≥ 96 192`. For the
held-out datasets finetuned alone it also doesn't (4,000 draws per rank-epoch, against
`W_d ≥ 5 422`). It would start at `max_samples > W_d / B`, e.g. 2,712 for VE.

### 3.3 Across epochs: independent, with no memory

Each epoch draws a **fresh** permutation of every dataset, not a continuation of last epoch's.
So an epoch that draws only 444 of GS's 955,200 windows does not make the next epoch draw
different ones. The next epoch's 444 are an independent random subset, and the overlap between
two epochs is the hypergeometric `r²/W` (E3: 21 shared windows observed vs. 18 expected).
**[verified E3]**

There is no mechanism that steers draws towards unseen windows. Coverage therefore grows like
`1 − e^{−λ}` (§5), not linearly.

### 3.4 Across GPUs in a node: independent, and nothing deduplicates

- **Which dataset.** Under `fsdp`, `sampling_rank = 0` on every GPU (`training.py:370-376`,
  default `sampling_rank_strategy="node"` with the whole world as one sync group). Every GPU
  therefore draws the **same dataset at every step**, and the sequence is the same for every
  `cfg.seed`. It matches `count_WALRUS_flops.replay_epoch` exactly. **[verified E2]**
- **Which windows.** Each rank's global RNG is seeded `seed + 1_000_003·rank` (`train.py:155`),
  so each GPU draws its own **independent permutation** of the shared dataset. Nothing compares or
  partitions windows across ranks. **[verified E5]**

Two consequences, both matching their formulas in E5 (`W = 3 000`: pairwise overlap per epoch
1,391.5 observed vs. 1,384.5 expected):

- two GPUs share about `r²/W` windows per epoch;
- in the **same step**, the expected number of GPU pairs holding the same window is
  `C(R,2)·B²/W` per step of that dataset. The same window can be in two GPUs' micro-batches of
  one optimizer step.

Other distribution types, in one sentence each **[code-read]**: under `ddp`, `sampling_rank =
rank`, so each GPU also draws its own dataset sequence; under `hsdp`, GPUs of one sharding group
share a dataset sequence and groups differ. Window permutations are per rank in all cases.

### 3.5 Across nodes: exactly as across GPUs

Nodes are just more ranks with larger `rank` values: the same shared dataset sequence under fsdp,
independent window permutations, and no deduplication. Doubling the node count doubles the draws
per epoch from the same pool. **[code-read, follows from E2/E5]**

The only built-in alternative is `allow_sharding_in_train=True` (default `False`,
`multidatamodule.py:74`), which uses `DistributedSampler` for each dataset. **[verified E6]**
- Ranks are **disjoint**: one shared permutation, with rank `r` taking every `R`-th entry.
- The permutation is seeded `0 + epoch` and ignores `cfg.seed`.
- A recycle within an epoch **replays the same order**, because the permutation depends only on
  the epoch.
- Not used in PT9.

### 3.6 Across a resumed job: part of epoch 1 is replayed

Checkpoints save no RNG state (`walrus/trainer/checkpoints.py`), and a resumed job runs
`seed_all` again (`train.py:147, 155`), so its global RNG starts **in the same state** as job
1's. At the resumed job's first epoch `k+1`, each dataset's permutation seed is whichever
global draw it reaches first. A dataset reaches the **same** draw as in job 1's epoch 1 iff the
sequence of earlier global calls is identical. In practice that means first appearing at the
same position inside the prefetch window, before any forward pass. That dataset then draws
**exactly the windows it drew in epoch 1, in the same order**, on every rank.

- [verified E6b] A job resumed at epoch 7 (PT9 mixture, real window counts, `data_workers=10`)
  replays all 404 of AD's epoch-1 draws. The other eight datasets share only chance agreement.
- [verified E6b] Which datasets replay is fully determined by the epoch-seeded dataset sequences,
  so it can be computed offline. Over resume epochs 2–100: **0.90 datasets per resume on
  average** (max 4), and at least one in **61 %** of resumes.
- **Size of the effect.** One replayed dataset-epoch is about `R·r ≈ 4·440` draws, i.e. about
  1/9 of one epoch's draws, which become exact repeats of epoch 1. A 3-job chain (2 resumes) over
  100 epochs replays about 1.8 of 900 dataset-epochs, about 0.2 % of all draws. That is
  negligible for totals, but it is ordered, exact duplication.
- **Caveats.** This assumes nothing else consumes the global CPU RNG between `seed_all` and the
  first epoch. `Trainer.train` runs no validation before epoch 1 (`training.py:1285-1300`), but
  that is [code-read] for the full start-up path. Later resumed epochs replay job 1's later
  epochs only if the forward and validation draw counts happen to line up (§2.1 item 5), which
  isn't generally the case. Whether `dt` and the rotation also repeat depends on which worker
  loads which batch, and is not claimed.

### 3.7 Deliberate reuse

`trainer.reuse_batches=True` (`training.py:1073`) caches and replays `grad_acc_steps` batches.
It defaults to `False` and is not used in PT9. **[code-read]**

---

## 4. What counts as "the same sample"

Three levels, from strictest to loosest:

| level | same when | how often in PT9 N1, 100 epochs (§5) |
|---|---|---|
| **identical sample** | same window **and** same `dt`, rotation, jitter, roll | rare: a repeated window usually gets a different `dt` (up to 5 values early in a trajectory, 1 near the end, E7), rotation and jitter |
| **same window** | same (file, trajectory, `t0`) | 8.7 % (GS) to 54 % (PS) of draws repeat an earlier window |
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
replayable with `count_WALRUS_flops.replay_epoch` (E2). Subsets are independent across ranks and
epochs. With `N_d = R·Σ_e r_{d,e}` draws and `λ = N_d/W_d`:

| quantity | exact / approximation |
|---|---|
| times a given window is drawn | sum of `R·E` independent Bernoulli(`r/W`) ≈ Poisson(`λ`) |
| distinct windows drawn (coverage) | `W[1 − Π_e (1 − r_{d,e}/W)^R] ≈ W(1 − e^{−λ})` |
| exact-repeat draws (share of all draws) | `1 − distinct/N` |
| windows drawn ≥ 2 times | `≈ 1 − e^{−λ}(1 + λ)` |
| pairwise rank overlap per epoch | `r²/W` |
| same-step shared windows per epoch | `c_{d,e}·C(R,2)·B²/W` |
| frame overlap | simulated (E8), with `dt` drawn by the real rule |

The closed forms agree with the simulation to within 0.1 percentage points on every row below.
`B=2`, `T_in=6`, `max_samples=2000`, fsdp, 100 epochs, `c_{d,e}` replayed.

### 5.1 PT9 N1 (1 node, 4 GPUs), 9-dataset mixture

| | W | draws/rank/epoch | λ | window coverage | exact-repeat draws | windows ≥ 2× | frame-overlap draws | same-step shared/epoch | pair overlap/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GS | 955,200 | 442 | 0.19 | 16.9 % | 8.7 % | 1.5 % | 76.6 % | 0.006 | 0.20 |
| EMQ-O | 380,000 | 450 | 0.47 | 37.8 % | 20.3 % | 8.2 % | 89.6 % | 0.014 | 0.53 |
| EMQ-P | 380,000 | 442 | 0.47 | 37.2 % | 20.0 % | 8.0 % | 89.4 % | 0.014 | 0.52 |
| AM | 313,600 | 441 | 0.56 | 43.0 % | 23.5 % | 11.0 % | 91.7 % | 0.017 | 0.62 |
| RB | 271,600 | 443 | 0.65 | 48.0 % | 26.5 % | 14.0 % | 92.9 % | 0.020 | 0.72 |
| AI | 268,608 | 447 | 0.67 | 48.6 % | 26.9 % | 14.4 % | 92.6 % | 0.020 | 0.74 |
| SF | 173,824 | 444 | 1.02 | 64.0 % | 37.3 % | 27.2 % | 95.5 % | 0.031 | 1.13 |
| AD | 153,600 | 450 | 1.17 | 69.0 % | 41.0 % | 32.7 % | 95.8 % | 0.035 | 1.32 |
| PS | 96,192 | 441 | 1.83 | 84.1 % | 54.1 % | 54.7 % | 97.6 % | 0.055 | 2.02 |

### 5.2 PT9 N2 (2 nodes, 8 GPUs)

Same per-rank draws, twice the ranks, so `λ` doubles.

| | λ | window coverage | exact-repeat draws | windows ≥ 2× | frame-overlap draws | same-step shared/epoch |
|---|---:|---:|---:|---:|---:|---:|
| GS | 0.37 | 31.0 % | 16.4 % | 5.4 % | 88.0 % | 0.026 |
| EMQ-O | 0.95 | 61.3 % | 35.4 % | 24.5 % | 94.8 % | 0.066 |
| EMQ-P | 0.93 | 60.6 % | 34.9 % | 23.9 % | 94.7 % | 0.065 |
| AM | 1.12 | 67.5 % | 39.9 % | 31.0 % | 95.9 % | 0.079 |
| RB | 1.31 | 72.9 % | 44.1 % | 37.5 % | 96.4 % | 0.091 |
| AI | 1.33 | 73.6 % | 44.7 % | 38.4 % | 96.3 % | 0.093 |
| SF | 2.04 | 87.1 % | 57.4 % | 60.6 % | 97.7 % | 0.143 |
| AD | 2.34 | 90.4 % | 61.4 % | 67.9 % | 97.9 % | 0.164 |
| PS | 3.67 | 97.5 % | 73.4 % | 88.1 % | 98.8 % | 0.257 |

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

**Offline replay is possible in principle but fragile.** E1 shows the draws are a deterministic
function of the seed, rank, `data_workers` and *every* global CPU RNG call the process makes.
Replaying a real run means reproducing all of those calls exactly: every training and
validation forward pass's jitter draws, every validation loader's `_base_seed`, and anything
else that touches the global CPU generator. Any code change that adds or removes one CPU RNG
call silently changes every later window. The dataset sequence alone *can* be replayed exactly,
since it uses its own generator (E2, `replay_epoch`).

**The exact method is to log the draws during the run.** It's cheap: N1 for 100 epochs is
4 ranks × 2,000 steps × 100 epochs × 2 windows = 1.6 M rows.

| where | what | why there |
|---|---|---|
| main process: wrap `BatchedMultisetSampler.__iter__` (or log `batch` indices in `train_one_epoch`) | `(rank, epoch, step, dataset, global index)` | window identity and order, cross-rank alignment by `step` |
| worker: `BatchInflatedWellDataset.__getitem__` (`inflated_dataset.py:308-336`) | `(index, file_idx, sample_idx, time_idx, dt)` per sample, plus the rotation | the method already collects `file_idxes`, `sample_idxes`, `time_idxes`, `dts` per sample (`:313-325`) and throws all but the last away, so it only needs persisting |

Write one file per rank, and join on `(rank, epoch, step)` afterwards. From the joined log, every
quantity in §5 becomes an exact count rather than an expectation: window, sample and frame
repeats; coverage; same-step cross-rank collisions; and resume replays. The frame level uses
`frames = t0 + dt·[0..5]`.

---

## 7. Summary

| question | answer |
|---|---|
| How is a dataset chosen? | Uniformly per step, by a generator seeded `100·epoch + sampling_rank`: the same on every GPU (fsdp) and for every seed |
| How is a window chosen? | The next entry of a per-dataset `randperm`, seeded from the global CPU RNG at the start of each pass |
| How are file, trajectory and `t0` chosen? | Not separately: they are arithmetic on the window index |
| Can the same window be drawn again on the same GPU? | Not within a pass. Across epochs: yes, independently (`r²/W` per epoch pair) |
| Can overlapping windows be drawn? | Yes, freely. By 100 epochs, 77–98 % of PT9 draws share frames with an earlier draw |
| Is anything preventing repeats? | Only the permutation within one pass. Nothing across epochs, GPUs, nodes or resumes |
| Across GPUs / nodes? | Same dataset at each step (fsdp), independent windows, no deduplication; collisions grow like `C(R,2)` |
| Resumed jobs? | Reseeded from scratch; about 0.9 datasets per resume replay their epoch-1 windows exactly |
| Estimate? | Poisson(`λ = N/W`) with `N` from the replayed dataset counts (§5) |
| Exact? | Log `(rank, epoch, step, index, file, traj, t0, dt)` during the run (§6); offline replay is fragile |
