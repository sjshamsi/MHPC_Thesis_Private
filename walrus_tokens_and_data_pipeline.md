# WARUS: tokens and data ingestion
The information in this document will be accompanied by locations in the repo or `the_well`. Parts marked **[verified]**, were checked by running the code rather than only reading it.

---

## 0. TL;DR for the scaling-law question

| Quantity | Walrus answer |
|---|---|
| What is a "token"? | One `(time-frame, spatial patch)` cell of the post-encoder grid. It is *not* a channel/field and *not* a grid point. |
| Tokens per frame | **1024** for 2D data (`32×32`), **4096** for 3D data (`16³`). Fixed by construction, independent of the input resolution. Non-periodic axes get one extra token per axis (`33×33 = 1089` in 2D) from learned boundary padding. |
| Tokens per micro-batch | `B_eff × T_eff × N_spatial`, engineered to be *the same constant* for 2D and 3D batches (`≈24.6k` at the thesis anchor config `batch_size=4, n_steps_input=6`). |
| Datasets per forward pass | Exactly **one**. Batches are never mixed across datasets. |
| Trajectories per forward pass | Up to `B_eff` distinct `(file, trajectory, t0)` draws; two draws can be different time windows of the *same* trajectory. |
| Is data repeated? | **Yes, heavily and by design.** `max_samples` decouples "epoch" from "one pass over the data"; `recycle_datasets: True` resamples exhausted sub-datasets. |
| Same time span for every window in a batch? | **No.** All windows have the same *frame count*, but `dt` is drawn independently per sample (§2.4), so their physical spans differ. The model is never told `dt`. |
| Does one optimizer step see one dataset? | **No.** Each grad-accum micro-step draws afresh, so an update mixes up to `grad_acc_steps × world_size` independently-chosen datasets (§2.6). |
| Do ranks duplicate data? | Under DDP, ranks pick *different datasets* and *different windows* (different RNG). Under HSDP, ranks in a sharding group **must** pick the same dataset (dimensionality-keyed `embed`/`debed` would desynchronise FSDP's per-module `AllGather`) but pick *different windows* (§2.8). Nothing enforces global non-overlap. |

---

## 1. How is a token defined in WALRUS

### 1.1 Adaptive-compute tokenization (Convolutional Stride Modulation)

WALRUS isn't like NLP model with a vocabulary. The encoder is a 2-layer strided-conv hMLP ([walrus/models/encoders/vstride_encoder.py](../walrus/models/encoders/vstride_encoder.py), class `SpaceBagAdaptiveDVstrideEncoder`, which is what [walrus/configs/model/encoder/spacebag_and_vstride_encoder.yaml](../walrus/configs/model/encoder/spacebag_and_vstride_encoder.yaml) selects). Each spatial axis is downsampled by `stride1 * stride2`. The *strides are chosen per sample from the input shape* so that the number of output cells per axis is a constant:

[walrus/models/shared_utils/flexi_utils.py:69-126](../walrus/models/shared_utils/flexi_utils.py#L69-L126)

```python
patch_dict = {0:(1,1), 1:(1,1), 4:(2,2), 8:(4,2), 12:(6,2), 16:(4,4), 24:(6,4), 32:(8,4)}
...
per_axis_tokens = 512 // 16      # == 32, when <= 2 non-singleton spatial dims
...
per_axis_tokens = 256 // 16      # == 16, when all 3 spatial dims are non-singleton
h_patch = H // per_axis_tokens   # per-axis downsampling factor
return (patch_dict[h_patch], ...)   # -> (stride1, stride2) for that axis
```

So the downsampling factor adapts to the resolution and the token count per axis stays fixed. In the WALRUS paper §3.1: "we choose a fixed number of tokens per axis and adjust the downsampling factor accordingly" and App. A.2: "32 per dimension" for 2D, "16" for 3D.

The strides are then applied in [vstride_encoder.py:316-380](../walrus/models/encoders/vstride_encoder.py#L316-L380) (`stride1`/`stride2` come from `kwargs["random_kernel"]`, i.e. `dynamic_ks` computed in [isotropic_model.py:309-319](../walrus/models/isotropic_model.py#L309-L319)).

### 1.2 Token grids for some datasets **[verified by running the model]**

By `IsotropicModel` from a Hydra config and looking at the input to `model.blocks[0]` (the input to the processor block ) we can see the following resulting token grid for various models:

| dataset shape | BCs | token grid entering processor | spatial tokens / frame |
|---|---|---|---|
| `256×256` (active_matter) | periodic | `(32, 32, 1)` | 1024 |
| `512×512` (viscoelastic) | periodic | `(32, 32, 1)` | 1024 |
| `512×128` (rayleigh_benard) | periodic | `(32, 32, 1)` | 1024 |
| `128×384` (TRL 2D) | periodic | `(32, 32, 1)` | 1024 |
| `256×512` (planetswe) | periodic | `(32, 32, 1)` | 1024 |
| `256×256` (acoustic scattering) | open/closed | `(33, 33, 1)` | 1089 |
| `1024×256` (helmholtz_staircase) | open/closed | `(33, 33, 1)` | 1089 |
| `64³` (MHD_64) | periodic | `(16, 16, 16)` | 4096 |
| `128³` (RT instability 3D) | periodic | `(16, 16, 16)` | 4096 |

We can see $33 \times 33$ gridsize for datasets with non-periodic axes. This comes from the learned boundary padding in [patch_jitterers.py:210-258](../walrus/models/shared_utils/patch_jitterers.py#L210-L258) (`get_paddings` adds `(effective_patch_size - effective_stride)//2` plus `effective_stride//2` of jitter room on each non-periodic side). Periodic axes get circular padding that cancels exactly, so they land on exactly 32.

Note the resolution-blindness. `1024×256` and `128×128` produce the same 1024 tokens! WALRUS deliberately trades variable physical information per token for constant compute per sample. This means that token count can be treated as a compute unit!

### 1.3 Time is not compressed

The encoder only strides spatial axes. Time is flattened into the conv batch dimension ([vstride_encoder.py:361](../walrus/models/encoders/vstride_encoder.py#L361), `T B ... -> (T B) ...`) and restored afterwards. The processor therefore operates on a tensor `(T, B, C, H_tok, W_tok, D_tok)`:

- **spatial attention** flattens `(h w d)` into a sequence of length `N_spatial` per `(t, b)` ([full_attention.py:129](../walrus/models/spatial_blocks/full_attention.py#L129));
- **time attention** attends over `T` at each spatial location ([axial_time_attention.py:81-113](../walrus/models/temporal_blocks/axial_time_attention.py#L81-L113)).

So tokens per sample = `T × N_spatial`, where `T = n_steps_input`.

### 1.4 Channels are *not* tokens

Fields (`density`, `velocity_x`, …) are input channels, folded into `hidden_dim` by the first
conv. Every dataset's fields are mapped into a *global* channel index space
(`field_to_index_map`, [multidataset.py:305-373](../walrus/data/multidataset.py#L305-L373)),
and the encoder sub-selects the relevant columns of the projection weight:

```python
weight1 = self.proj1.weight[:, field_indices]   # vstride_encoder.py:349
```

Three extra channels are always appended — boundary/bias flags, created in
[patch_jitterers.py:302-306](../walrus/models/shared_utils/patch_jitterers.py#L302-L306)
(`bc_flag_shape[2] = 3`) with labels `[2, 0, 1]` appended to `state_labels` in
[isotropic_model.py:232-240](../walrus/models/isotropic_model.py#L232-L240).

⚠️ **Reproducibility gotcha for a scaling sweep.** `n_states` is computed in
[train.py:133-135](../walrus/train.py#L133-L135) as `max(field_to_index_map.values()) + 1`,
and with `prefetch_field_names: True` (the default) `_build_subset_dict` loops over **all 24
`WELL_DATASETS`** and silently skips any that fail to load
([multidataset.py:315-338](../walrus/data/multidataset.py#L315-L338)). So *the model's input
channel count depends on which datasets happen to be present on disk*. If you want the
parameter count of the encoder/decoder to be comparable across sweep points, pin
`data/field_index_map_override=full_well_field_index` (66 entries = the paper's 63 state
variables + 3 boundary flags) instead of the default `bc_only_override` (3 entries).

---

## 2. How datasets are ingested

This section answers: *what are all these objects, who does what, and what exactly is in one
forward pass?*

### 2.1 The five objects and their division of labour

```
MixedWellDataModule                             walrus/data/multidatamodule.py:38
│   owns config; builds the datasets; builds the DataLoader on demand
│
├── train_dataset : MixedWellDataset             multidataset.py:53   (built at :245)
│   │   a *concatenation* of per-dataset readers behind one flat index space
│   │
│   ├── sub_dsets : list[BatchInflatedWellDataset]   one per key in `well_dataset_info`
│   │      inflated_dataset.py:246 -> InflatedWellDataset:20 -> the_well.WellDataset
│   │      each knows: its HDF5 files, its own n_steps_input, its own metadata,
│   │      its own field names, its own length in windows
│   │
│   ├── offsets              : [-1, len(d0), len(d0)+len(d1), ...]   multidataset.py:292-302
│   ├── effective_batch_sizes: [1.0, 1.0, ..., 0.5]  (batch_downsample_factor) :301
│   └── field_to_index_map   : global field-name -> channel index    :373
│
├── val_datasets / rollout_val_datasets          multidatamodule.py:273, :303
│
└── train_dataloader(rank_override)              multidatamodule.py:410
     ├── BatchedMultisetSampler                  mixed_dset_sampler.py:126
     │     decides WHICH dataset, and hands out a whole batch of indices at a time
     │     ├── sub_samplers : list[RandomSampler]   one per sub-dataset, :159-166
     │     └── its own torch.Generator, seeded 100*epoch + 10*seed + rank   :172
     └── DataLoader(batch_size=None, collate_fn=None, sampler=sampler)   :429-438
```

Read it as a division of responsibility:

| object | decides | does **not** decide |
|---|---|---|
| `MixedWellDataModule` | which datasets exist, `batch_size`, `max_samples`, which rank it is | anything per-step |
| `MixedWellDataset` | index → which sub-dataset (`np.searchsorted` on `offsets`), attaches `field_indices` + `metadata` | which index |
| `BatchInflatedWellDataset` (a sub-dset) | index → `(file, trajectory, t0)`, draws `dt`, reads HDF5, pads to 3D, augments, collates | which index |
| `BatchedMultisetSampler` | **which dataset** this step, and how many windows to draw from it | which window |
| each `RandomSampler` | **which window** inside that dataset (`randperm`, no repeats within a pass) | which dataset |
| `DataLoader` | prefetching across `data_workers` processes | anything about content — `batch_size=None` means it does *no* batching |

The two random decisions are made by **two different, independent RNGs** (§4.2): dataset choice is
seeded and reproducible; window choice is not.

The `DataLoader` here is unusually inert. `batch_size=None` + `collate_fn=None`
([multidatamodule.py:429-438](../walrus/data/multidatamodule.py#L429-L438)) puts it in
*iterable-style-over-a-batch-sampler* mode: whatever the sampler yields is passed straight to
`dataset[...]` and returned as-is. All batching and collation happens **inside the dataset**.

### 2.2 One training micro-step, traced end to end

Concrete setting — the thesis anchor: 8 datasets (7× 2D + `MHD_64` 3D), `batch_size=4`,
`n_steps_input=6`, `n_steps_output=1`, `min_dt_stride=1`, `max_dt_stride=5`.

1. `train_one_epoch` calls `next(data_iter)`
   ([training.py:1049](../walrus/trainer/training.py#L1049)).
2. The `DataLoader` asks its sampler for the next item. `BatchedMultisetSampler.__iter__`
   ([mixed_dset_sampler.py:167](../walrus/data/mixed_dset_sampler.py#L167)) draws
   `torch.randint(0, 8, generator=g)` → say **`5` = `helmholtz_staircase`**. Uniform over the
   *number* of datasets — dataset size is irrelevant (§4.1).
3. `batch_factor = effective_batch_sizes[5] = 1.0` (it is 2D) →
   `dset_batch = max(1, int(4 * 1.0)) = 4`.
   Had it drawn `MHD_64`, `batch_factor = 0.5` → `dset_batch = 2`.
4. It calls `next(sub_samplers[5])` **4 times**. That sub-sampler is a `RandomSampler` over
   helmholtz's 22 528 windows, currently partway through a `randperm`, so the 4 local indices are
   distinct. Each gets `+ offsets[5]` added → 4 global indices.
5. It `yield`s **one list of 4 ints**. (Note `accumulated_batch = [accumulated_batch]` then
   `yield from` — a one-element list yielded from, i.e. one `yield` of the inner list.)
6. `DataLoader` (in a worker process) calls
   `MixedWellDataset.__getitem__([i0,i1,i2,i3])`
   ([multidataset.py:374](../walrus/data/multidataset.py#L374)). It `searchsorted`s **only
   `indices[0]`** to find the sub-dataset — the code comment says outright *"This will likely fail
   if the list covers multiple datasets, so just assume its all from one."*
7. `BatchInflatedWellDataset.__getitem__` ([inflated_dataset.py:308](../walrus/data/inflated_dataset.py#L308))
   loops the 4 indices, calling `_load_one_sample(i)` on each. **Per index it independently draws
   its own `dt`** (§2.4). Each returns a `(6 + 1) = 7`-frame window.
8. `default_collate(data)` stacks the 4 into leading batch dim → `input_fields (4, 6, H, W, C)`.
9. `_preprocess_data` pads 2D→3D; `transform` = `RandomRotation90` runs **once on the collated
   tensor**, so all 4 windows get the *same* rotation; `_postprocess_data` flattens fields into
   the channel dim; `_construct_sample` splits input/output.
10. Back in `MixedWellDataset.__getitem__`: `field_indices` (one vector for the whole batch) and
    `metadata` (one object for the whole batch) are attached.
11. `rollout_model` → `ChannelsFirstWithTimeFormatter.process_input`
    ([well_to_multi_transformer.py:29-78](../walrus/data/well_to_multi_transformer.py#L29-L78))
    rearranges to `(T, B, C, H, W, D)` and hands the model exactly
    `(x, field_indices, boundary_conditions)`.
12. `IsotropicModel.forward` picks `dim_key = str(n_spatial_dims)`, calls
    `choose_kernel_size_deterministic(x_shape)` → strides → 33×33 tokens/frame for helmholtz
    (non-periodic). Tokens in this micro-batch: `4 × 6 × 1089 = 26 136`.

### 2.3 What is identical vs. what varies **within one micro-batch**

Because step 6 assumes one dataset, a great deal is forced to be constant:

| property | within one micro-batch | why |
|---|---|---|
| dataset / physics | **identical** | sampler draws one `dset_sampled` |
| spatial resolution | **identical** | same dataset |
| dimensionality (2D/3D) | **identical** | same dataset; also picks `embed[dim_key]` |
| field set & `field_indices` | **identical** | one vector attached post-collate, [multidataset.py:405](../walrus/data/multidataset.py#L405) |
| `metadata` object | **identical** | one object, [multidataset.py:406](../walrus/data/multidataset.py#L406) |
| number of frames (`n_steps_input`) | **identical** | per-dataset, fixed at construction ([multidataset.py:259](../walrus/data/multidataset.py#L259)) |
| boundary conditions | **identical** in practice | collated per-sample but the model reads `bcs[0]` ([isotropic_model.py:366](../walrus/models/isotropic_model.py#L366)) |
| `RandomRotation90` orientation | **identical** | applied after `default_collate` |
| patch jitter / periodic roll | **identical** | `torch.roll` on the full tensor incl. batch dim ([isotropic_model.py:371-390](../walrus/models/isotropic_model.py#L371-L390)) |
| trajectory (`sample_idx`) | **varies** — up to `dset_batch` distinct | independent `RandomSampler` draws |
| start frame (`time_idx`) | **varies** | ditto |
| **time stride `dt`** | **varies per sample** | drawn inside `_load_one_sample`, per index |

So: *yes*, all `batch_size` windows in a pass come from the same dataset; *yes* they all have the
same **number** of frames; but *no*, they do **not** all span the same amount of physical time.

### 2.4 The `dt` stride — the one thing that genuinely varies per sample

This is the answer to "the trajectory length is randomised — is the same trajectory sampled again
with a different length?"

```python
# the_well/data/datasets.py:801-813, inside _load_one_sample(index)
dt = self.min_dt_stride
if self.max_dt_stride > self.min_dt_stride:
    effective_max_dt = maximum_stride_for_initial_index(
        time_idx, n_steps_per_trajectory, n_steps_input, n_steps_output)
    effective_max_dt = min(effective_max_dt, self.max_dt_stride)
    if effective_max_dt > self.min_dt_stride:
        dt = np.random.randint(self.min_dt_stride, effective_max_dt + 1)
```

Five consequences worth having straight:

1. **Frame count is fixed, physical span is not.** Every sample returns exactly
   `n_steps_input + n_steps_output` frames. `dt` only changes how far apart they are. That is
   precisely why `default_collate` can stack samples with different `dt` — the tensors are the
   same shape. A `dt=1` and a `dt=5` window sit side by side in the same batch.
2. **The index space is built with `min_dt_stride` only.**
   `windows_per_trajectory = steps - (1 + min_dt*(n_in + n_out - 1)) + 1`
   ([the_well/data/utils.py:88-112](../walrus/data/multidataset.py)). So `max_dt_stride=5` adds
   **no new indices** — it re-interprets existing ones. `len(dataset)` is unchanged.
3. **The same index yields different physics on different draws.** Index `i` → always the same
   `(file, trajectory, t0)`, but a fresh `dt` every time it is loaded. So the same trajectory *is*
   re-sampled with a different span — on the next recycle, or the next epoch. This is real
   augmentation and is *not* counted in the unique-token figures of §4.1 (which is why those are
   a lower bound on distinct samples).
4. **`dt` is position-biased, not uniform.** `maximum_stride_for_initial_index` caps `dt` by how
   much trajectory remains: `(total_steps - time_idx - 1) // (n_in + n_out - 1)`. For
   `helmholtz_staircase` (50 steps, `n_in+n_out=7`): a window starting at `t0=0` may draw
   `dt ∈ {1..5}`, but one starting at `t0=45` is forced to `dt=1`. Late windows are systematically
   less augmented, and the effective dataset is skewed toward short-`dt` samples near trajectory
   ends.
5. ⚠️ **The model is never told what `dt` was.** `process_input` returns only
   `(x, field_indices, boundary_conditions)` — `time_grid` / `output_time_grid` are loaded and
   then dropped. (`walrus/data/utils.py:preprocess_batch` computes a `time_step` but is not on
   the training path.) `TrajectoryMetadata.time_stride`
   ([inflated_dataset.py:333](../walrus/data/inflated_dataset.py#L333)) is set from the **last**
   sample of the loop only, and nothing reads it. So variable `dt` trains the model to be
   *timescale-agnostic*: it must infer the effective timestep from the data. For a scaling study
   this is an implicit task-difficulty knob — sweeping `max_dt_stride` changes the *task*, not
   just the data volume.

### 2.5 One batch = one dataset

`BatchedMultisetSampler`
([mixed_dset_sampler.py:126-234](../walrus/data/mixed_dset_sampler.py#L126-L234)) yields a
**list** of indices, all drawn from a single randomly chosen sub-dataset:

```python
index_sampled = torch.randint(0, len(sampler_choices), (1,), generator=generator)
dset_sampled  = sampler_choices[index_sampled]
batch_factor  = self.dataset.effective_batch_sizes[dset_sampled]
dset_batch    = max(1, int(self.batch_size * batch_factor))
accumulated_batch = [next(samplers[dset_sampled]) + offset for _ in range(dset_batch)]
yield [accumulated_batch]          # one yield == one whole batch
```

The `DataLoader` is created with `batch_size=None` and `collate_fn=None`
([multidatamodule.py:429-438](../walrus/data/multidatamodule.py#L429-L438)), so the list goes
straight into `MixedWellDataset.__getitem__`, which forwards it to
`BatchInflatedWellDataset.__getitem__`
([inflated_dataset.py:308-349](../walrus/data/inflated_dataset.py#L308-L349)), which loads the
samples one by one and `default_collate`s them. (`metadata_aware_collate` at
[multidatamodule.py:27](../walrus/data/multidatamodule.py#L27) is dead code in this path.)

**Consequence:** a forward pass contains **exactly one dataset**, one resolution, one
dimensionality, one field set, one `metadata` object. This is what makes a single `metadata`
per batch legal (`batch["metadata"]`) and what makes the adaptive-stride tokenizer well-defined
per batch. Mixing across datasets happens *between* steps (and across ranks / grad-accum
micro-steps), never *within* a batch.

**Number of trajectories in a batch:** ≤ `dset_batch`. The `RandomSampler` draws a permutation
without replacement, so the `dset_batch` global indices are distinct — but distinct indices can
map to different `time_idx` of the same `sample_idx` (same trajectory), so the number of
*distinct trajectories* is ≤ `dset_batch`.

### 2.6 Gradient accumulation *does* mix datasets

Each accumulation micro-step calls `next(data_iter)` afresh
([training.py:1046-1053](../walrus/trainer/training.py#L1050)) — there is no caching unless
`trainer.reuse_batches=True`. So with `grad_acc_steps=4`, **one optimizer step sees up to 4
different datasets on each rank**, and the gradient that is actually applied is a mixture. The
"one dataset per pass" restriction is a *forward-pass* constraint, not an *update* constraint.

At the 4-node anchor (`world_size=16`, `grad_acc_steps=4`, DDP): one optimizer step aggregates
`16 × 4 = 64` independently-drawn micro-batches over 8 datasets, so in expectation each dataset
contributes ~8 micro-batches per update, and the probability that a given dataset is absent from
an update is `(7/8)^64 ≈ 0.02%`. Dataset balance at the *update* level is therefore very good even
though each forward pass is single-dataset.

### 2.7 Equalising 2D and 3D cost

Two per-dataset knobs in the data config, e.g.
[configs/data/Leonardo_smallest8_2_3d.yaml](../walrus/configs/data/Leonardo_smallest8_2_3d.yaml):

```yaml
MHD_64:
  step_downsample_factor: .5     # halves n_steps_input for this dataset
  batch_downsample_factor: .5    # halves the batch size for this dataset
```

- `step_downsample_factor` → [multidataset.py:259](../walrus/data/multidataset.py#L259)
  `n_steps_input=max(1, int(n_steps_input * step_downsample_factor))`. Applied **at sub-dataset
  construction**, so it is baked into that reader's index space and its returned tensor shape.
- `batch_downsample_factor` → stored in `effective_batch_sizes`
  ([multidataset.py:301](../walrus/data/multidataset.py#L301)) and applied **at sampling time**
  at [mixed_dset_sampler.py:184-185](../walrus/data/mixed_dset_sampler.py#L184-L185).

This is the paper's *differential batch sizes* (App. A.2: "we multiply the 2D batch size and
length of time history by a factor of 2 each"). It makes the token count per micro-batch
**identical** across 2D and 3D:

| | `B_eff` | `T_eff` | `N_spatial` | tokens/micro-batch |
|---|---|---|---|---|
| 2D (`batch_size=4, n_steps_input=6`) | 4 | 6 | 1024 | **24 576** |
| 3D (both factors `0.5`) | 2 | 3 | 4096 | **24 576** |
| 2D, paper Table 3 (`batch_size=2, n_steps_input=6`) | 2 | 6 | 1024 | 12 288 |
| 3D, paper Table 3 | 1 | 3 | 4096 | 12 288 |

⚠️ `dset_batch = max(1, int(batch_size * batch_factor))` **floors**. At `batch_size=4` a 3D dataset
gets 2 — exact. At `batch_size=3` it gets `int(1.5) = 1`, and the token balance silently breaks
(`3×6×1024 = 18 432` vs `1×3×4096 = 12 288`, a 1.5× imbalance). Keep `batch_size` even, and
divisible by 4 if you ever use `0.25` factors.

Note this is *token*-balanced, not FLOP-balanced: spatial attention is `O(N_spatial²)` per
frame, so a 3D batch spends far more of its budget in attention than a 2D one (paper App. A.2:
attention grows from ~5% to ~80% of a block moving 2D→3D). Residual imbalance is handled by
gradient accumulation + topology-aware sampling, not by the tokenizer.

### 2.8 What must match across ranks, and what may differ

This is the FSDP/HSDP question. The rule follows from **one fact**: the encoder and decoder are
`nn.ModuleDict`s keyed by dimensionality
([isotropic_model.py:110, :141](../walrus/models/isotropic_model.py#L110)), selected per forward
by `dim_key = str(n_spatial_dims)` ([isotropic_model.py:194](../walrus/models/isotropic_model.py#L194)).

Under **FSDP**, parameters live sharded and are `AllGather`ed *per submodule, at the moment that
submodule runs*. Every rank in a sharding group must therefore issue **the same sequence of
collectives**. If rank 0 ran a 2D batch (gathering `embed["2"]`) while rank 1 ran a 3D batch
(gathering `embed["3"]`), the two ranks would post mismatched collectives and the group would
hang. Under **DDP** the model is fully replicated and the only collective is the gradient
all-reduce at the end of backward, which is shape-identical regardless of the data — so DDP ranks
are free to disagree.

Hence `sampling_rank_strategy` ([training.py:351-371](../walrus/trainer/training.py#L351-L371)),
whose value becomes the sampler's `rank`, which is what seeds the dataset-choice generator:

| | mesh | `sampling_rank` | dataset choice | window choice | `dt`, jitter, rotation |
|---|---|---|---|---|---|
| **`local`** | — | `0` | — | own | own |
| **`ddp`** (Leonardo scripts) | `("ddp",)` | `rank` — all distinct | **every GPU different** | own | own |
| **`hsdp`** + `"node"` (default) | `("ddp","fsdp")` | `rank // fsdp_size` | **same within a node**, different across nodes | **own, per GPU** | own |
| `"world"` | any | `0` | **same on every GPU** | own | own |

Worked example, HSDP on 4 nodes × 4 GPUs, `sampling_rank_strategy="node"`:

```
                sampling_rank   dataset this step   windows drawn
node 0, gpu 0-3       0            planetswe        4 different windows each -> 16 total
node 1, gpu 0-3       1            gray_scott       16 total
node 2, gpu 0-3       2            MHD_64  (3D)     dset_batch=2 -> 8 total
node 3, gpu 0-3       3            active_matter    16 total
```

- **Same across a sharding (fsdp) group, necessarily:** the dataset, and therefore resolution,
  dimensionality, field set, `n_steps_input`, and which `embed`/`debed` branch runs.
- **Different across a sharding group, freely:** which windows, which trajectories, each sample's
  `dt`, the `RandomRotation90` draw, the patch-jitter rolls. Those are all per-`__getitem__` /
  per-forward randomness with no collective attached.
- **Different across nodes (ddp dim), freely:** everything, including the dataset.

The same DDP-vs-FSDP argument explains why `sampling_rank_strategy="gpu"` is safe under DDP and
would hang under HSDP.

Three further rank facts:

- **Nothing deduplicates across ranks.** `allow_sharding_in_train` (default `False`,
  [multidatamodule.py:74](../walrus/data/multidatamodule.py#L74)) would swap `RandomSampler` for a
  `DistributedSampler` and partition the index space; it is off, and its docstring warns it drops
  a large fraction when a dataset is small relative to the world size. So two ranks *can* draw the
  same window in the same step; over a run they sample from the same pool entirely.
- **The dataset-choice seed is `100*epoch + 10*seed + rank`**. With `seed=0` and `rank ∈ 0..15`,
  the seeds for epoch `e` are `100e … 100e+15` — and epoch `e+1` starts at `100(e+1)`. No overlap
  here, but note that with `world_size > 100` or `seed ≥ 10` the arithmetic collides and different
  (epoch, rank) pairs share a seed. Worth knowing before scaling past ~100 ranks or sweeping
  `seed`.
- **Workers do not multiply sampling.** With `batch_size=None` + a sampler, PyTorch iterates the
  sampler in the **main** process and dispatches index-lists round-robin to workers, so
  `data_workers=8` prefetches but does not change what is drawn. Workers *do* each get their own
  numpy seed (torch's `_worker_loop` seeds numpy from `base_seed`), so per-sample `dt` and
  `RandomRotation90` draws are independent across workers.

### 2.9 What the loaded sample actually is

`n_steps_input` input frames + `n_steps_output` target frames of a single time window
(strided by `dt` — see §2.4), all fields of that dataset, plus constant fields concatenated
along channels by
[`ChannelsFirstWithTimeFormatter.process_input`](../walrus/data/well_to_multi_transformer.py#L29-L78),
padded to 3D by `InflatedWellDataset._pad_to_higher_d`
([inflated_dataset.py:194-224](../walrus/data/inflated_dataset.py#L194-L224)) — this is the
paper's §4.1 "dimension padding" (2D data becomes a plane embedded in 3D, tensor-valued fields
zero-padded), followed by `RandomRotation90`
([configs/data/transform/random_rotation90_train.yaml](../walrus/configs/data/transform/random_rotation90_train.yaml)).

A concrete `batch` dict handed to `rollout_model`, for a `helmholtz_staircase` micro-batch at the
anchor config:

```
input_fields        (4, 6, 1024, 256, 1, C)   4 windows x 6 frames, padded to 3D
output_fields       (4, 1, 1024, 256, 1, C)
constant_fields     (4, 1024, 256, 1, C_const)
boundary_conditions (4, 3, 2)                 per-sample, but model reads [0]
field_indices       (C,)                      ONE vector for the whole batch
padded_field_mask   (C,)
metadata            WellMetadata              ONE object for the whole batch
```

after which the model sees `x = (T=6, B=4, C, 1024, 256, 1)` and emits
`4 × 6 × 1089 = 26 136` tokens into the processor.
---

## 3. What `max_samples` means

`max_samples` is **the number of micro-batches per rank per epoch**. Nothing else.

```python
# mixed_dset_sampler.py:209-220
if count >= self.max_samples:
    break
def __len__(self):
    if self.recycle:
        return self.max_samples          # sampler yields whole batches
```

`DataLoader` has `batch_size=None`, so `len(dataloader) == len(sampler) == max_samples`, and
`train_one_epoch` loops `while i < len(dataloader)`
([training.py:1032](../walrus/trainer/training.py#L1032)). This is exactly the paper's C.1:
*"we instead structure it in terms of logging intervals of 2000 micro-batches which we refer to
as epochs for convenience."*

**So `max_samples: 2000` means: 2000 optimizer *micro*-steps per rank per epoch**, i.e.
`2000 / grad_acc_steps = 500` real optimizer steps per rank per epoch.

Unpacking "how big is a sample":

- **1 sample (in `max_samples` units) = 1 micro-batch = 1 dataset**.
- It contains `B_eff` time windows (`4` for 2D, `2` for 3D at the anchor config).
- It contains `T_eff` input frames each (`6` / `3`).
- It contains all fields of that one dataset (mapped into the global channel space) + 3 BC flags.
- It contains `B_eff × T_eff × N_spatial` = ~24.6k tokens.
- It contains ≤ `B_eff` distinct trajectories, from **one** dataset.

Sanity check against the paper: `200 epochs × 2000 micro-batches × 96 GPUs = 38.4M`
micro-batches, which the paper reports as "38M micro-batches"; with mixed 2D/3D micro-batch
sizes that comes to ~65M trajectory-window samples, matching "≈4M examples from each 2D dataset
and 2M for each 3D".

---

## 4. Repetition, shuffling, and ranks

### 4.1 Data is repeated — a lot

`recycle_datasets: True` (default, [multidatamodule.py:69](../walrus/data/multidatamodule.py#L69))
means an exhausted sub-sampler is **restarted rather than dropped**:

```python
# mixed_dset_sampler.py:195-206
except StopIteration:
    if self.recycle:
        count -= 1                                        # does not count toward max_samples
        samplers[index_sampled] = iter(self.sub_samplers[dset_sampled])
    else:
        sampler_choices.pop(index_sampled)
```

With `recycle=True` an epoch is *defined* by `max_samples`, not by the data; the docstring says
so explicitly ([multidatamodule.py:147-150](../walrus/data/multidatamodule.py#L147-L150)).
With `recycle=False`, an epoch ends when every sub-dataset has been fully consumed once
(or when `max_samples` is hit, whichever comes first) — that is the only setting in which an
"epoch" is close to a true pass.

Scale of the repetition for the thesis dataset mix
(`Leonardo_smallest8_2_3d`, sizes from the paper Table 6, `n_steps_input=6`, `dt_stride=1`,
whole-dataset figures — the train split is ~80% of these):

| dataset | resolution | steps | traj | windows/traj | unique windows | unique model tokens |
|---|---|---|---|---|---|---|
| active_matter | 256×256 | 81 | 360 | 75 | 27 000 | 29.9 M |
| planetswe | 256×512 | 1008 | 120 | 1002 | 120 240 | 123.9 M |
| acoustic_scattering_discontinuous | 256×256 | 102 | 2000 | 96 | 192 000 | 208.9 M |
| gray_scott_reaction_diffusion | 128×128 | 1001 | 1200 | 995 | 1 194 000 | 1230.0 M |
| turbulent_radiative_layer_2D | 128×384 | 101 | 90 | 95 | 8 550 | 9.3 M |
| helmholtz_staircase | 1024×256 | 50 | 512 | 44 | 22 528 | 26.2 M |
| viscoelastic_instability | 512×512 | ~20 | 260 | 14 | 3 640 | 5.3 M |
| MHD_64 (3D) | 64³ | 100 | 100 | 97 | 9 700 | 41.0 M |
| **total** | | | | | **1.58 M windows** | **1.67 G tokens** |

Against that, a 4-node (16-GPU) run at `max_samples=2000`, `max_epoch=21`, `batch_size=4`
consumes `21 × 2000 × 16 × 24576 ≈ 16.5 G` tokens.

**The aggregate repetition factor (16.5 / 1.67 ≈ 10×) is misleading and should not be used.**
Because the sampler picks datasets *uniformly* (`torch.randint` over `sampler_choices`,
[mixed_dset_sampler.py:176-180](../walrus/data/mixed_dset_sampler.py#L176-L180)) rather than in
proportion to size, every dataset receives the *same* number of micro-batches — here
`672 000 / 8 = 84 000` each, i.e. `84 000 × 24 576 = 2.06 G` tokens delivered to each dataset
regardless of whether it holds 5 M or 1 230 M unique tokens. The real repetition factors:

| dataset | unique tokens | tokens delivered | **repetition** |
|---|---:|---:|---:|
| gray_scott_reaction_diffusion | 1230.0 M | 2.06 G | **2×** |
| acoustic_scattering_discontinuous | 208.9 M | 2.06 G | **10×** |
| planetswe | 123.9 M | 2.06 G | **17×** |
| MHD_64 | 41.0 M | 2.06 G | **50×** |
| active_matter | 29.9 M | 2.06 G | **69×** |
| helmholtz_staircase | 26.2 M | 2.06 G | **79×** |
| turbulent_radiative_layer_2D | 9.3 M | 2.06 G | **222×** |
| viscoelastic_instability | 5.3 M | 2.06 G | **388×** |

A 194× spread. The aggregate 10× is a token-weighted mean dominated by `gray_scott`, which alone
holds 73% of the unique tokens in the mix. **Five of the eight datasets are past the ~16-epoch
boundary** beyond which Muennighoff et al. find repeated tokens contribute essentially nothing,
while `gray_scott` has barely been seen twice. `tools/count_tokens.py` prints this column.

Repeats are not bit-identical, though: `RandomRotation90`, the random `dt` stride (§2.4),
patch jittering, and periodic rolls all resample per `__getitem__` / per forward.

### 4.2 Shuffling and seeding

Two independent random processes:

1. **Which dataset** — `BatchedMultisetSampler`'s own `torch.Generator`, seeded deterministically:
   ```python
   # mixed_dset_sampler.py:172
   generator.manual_seed(100 * self.epoch + 10 * self.seed + self.rank)
   ```
   `set_epoch(epoch)` is called every epoch from
   [training.py:1252-1253](../walrus/trainer/training.py#L1252-L1253), so the *dataset-choice
   sequence changes each epoch* and differs across ranks (via `rank`) — but is reproducible.

2. **Which window inside the dataset** — `RandomSampler(dataset)` with **no generator**
   ([multidatamodule.py:414](../walrus/data/multidatamodule.py#L414)). PyTorch then seeds it
   from the process-global RNG at each `iter()`, and there is **no `torch.manual_seed` anywhere
   in the training path** (grep confirms the only seeded generators are the sampler's own and
   the *validation* `RandomSampler(..., manual_seed(0))` at
   [multidatamodule.py:462](../walrus/data/multidatamodule.py#L462)). So:
   - within one pass, `randperm` guarantees **no repeats**;
   - each recycle draws a **fresh permutation**;
   - the order is **not reproducible across runs**, and differs across ranks and epochs.

   ➜ If you need reproducible scaling-law runs, this is the thing to fix (set a global torch
   seed, or pass an explicit generator to the train `RandomSampler`).

The multi-worker case is fine: with `batch_size=None` and a sampler, PyTorch iterates the
sampler in the **main process** and dispatches index-lists to workers, so `data_workers=8` does
not multiply or desynchronise the sampling.

### 4.3 Temporal augmentation

Covered in full in §2.4. In short: `dt` is drawn per sample at load time when
`max_dt_stride > min_dt_stride`; the index space is built with `min_dt_stride` only, so the same
index yields a different physical window on each draw. This is paper §C.1 "variable time-stride",
and it means the unique-token counts in §4.1 are a **lower bound** on the number of distinct
samples the mix can produce.

### 4.4 Ranks and nodes

The full mechanism — including *why* FSDP forces a shared dataset and what is nonetheless free to
differ per GPU — is in §2.8. The repetition-relevant consequences:

- **Nothing deduplicates across ranks.** `allow_sharding_in_train` (default `False`,
  [multidatamodule.py:74](../walrus/data/multidatamodule.py#L74)) would swap the base sampler
  for a `DistributedSampler` and partition indices across ranks — but its docstring warns it
  drops a large fraction when the dataset is small relative to the world size, and it is off by
  default. So **two ranks can and do draw the same window in the same step**; at 1.58 M windows
  and 16 ranks the per-step collision probability is small, but the *long-run overlap is total* —
  every rank samples from the same pool. Adding ranks therefore multiplies `T_seen` without
  adding any unique data.
- **Under HSDP, sharing the dataset within a node does not mean sharing the data.** The model is
  sharded across the node, but each GPU draws its own windows from the shared sub-dataset and
  runs its own micro-batch. Nothing is duplicated by FSDP itself.
- **The only mechanism that deliberately feeds the same batch twice** is
  `trainer.reuse_batches=True` ([training.py:1035-1055](../walrus/trainer/training.py#L1035-L1055)),
  which caches and replays `grad_acc_steps` worth of batches. Default is `False`.

### 4.5 Validation is deliberately *not* randomised

Val/test loaders use `DistributedSampler(seed=0)` or `RandomSampler(manual_seed(0))`
([multidatamodule.py:446-465](../walrus/data/multidatamodule.py#L446-L465)) and are rebuilt each
epoch with `replicas=sync_group_size, rank=rank_in_sync_group`
([training.py:1269-1279](../walrus/trainer/training.py#L1269-L1279)) so the same validation data
is seen every epoch. Only the train sampler gets `set_epoch`.

---

## 5. Quantifying the training data for a scaling law

### 5.1 The compute-side token count (the "D" in `C ≈ 6ND`)

Because token count per micro-batch is constant by construction, the accounting is simple and
exact:

```
tokens_per_microbatch = B_eff × T_eff × N_spatial          # ≈ constant across datasets
                      = batch_size × n_steps_input × 1024  (2D and 3D agree by design)

tokens_per_optimizer_step = tokens_per_microbatch × grad_acc_steps × world_size
tokens_per_epoch          = tokens_per_microbatch × max_samples × world_size
total_tokens              = tokens_per_epoch × max_epoch
optimizer_steps           = max_samples × max_epoch / grad_acc_steps
```

Worked examples:

| config | tokens/µbatch | tokens/opt-step | tokens/epoch | total tokens | opt steps |
|---|---|---|---|---|---|
| thesis anchor, 1 node (`B=4,T=6,ga=4,ws=4,ms=2000,ep=21`) | 24 576 | 393 216 | 0.20 G | **4.1 G** | 10 500 |
| thesis anchor, 4 nodes (`ws=16`) | 24 576 | 1 572 864 | 0.79 G | **16.5 G** | 10 500 |
| paper Walrus (`B=2,T=6,ga=4,ws=96,ms=2000,ep=200`) | 12 288 | 4 718 592 | 2.36 G | **472 G** | 100 000 |

Two things worth flagging before you fit anything:

1. **`max_samples` caps micro-batches *per rank*.** Adding nodes increases tokens/epoch and the
   global batch size; it does **not** shorten an epoch. (Your own
   `anchor_dimension_run_multinode.sh` header already notes this.) So `world_size` is a
   batch-size axis, not a speed axis, in this codebase.
2. **`grad_acc_steps` does not change the token count**, only the number of optimizer updates
   and the effective batch. `global_batch_size` as logged to wandb
   ([train.py:399-401](../walrus/train.py#L399-L401)) is `batch_size × world_size × grad_acc_steps`
   in *windows*, not tokens — multiply by `n_steps_input × 1024` for tokens.

### 5.2 The data-side token count (unique information)

For a Chinchilla-style "compute-optimal `D`", the NLP assumption is that tokens are ~unique. In
Walrus they are emphatically not. I'd recommend tracking three numbers per sweep point and
reporting the ratios:

- `T_seen` — tokens processed (§5.1). This is the compute axis.
- `T_unique_model` — `Σ_datasets (n_traj_train × n_steps × N_spatial(dataset))`, i.e. the number
  of distinct post-tokenizer cells the data can produce. For the 8-dataset mix: **1.67 G** over
  all splits (~1.3 G on the train split).
- `T_unique_raw` — `Σ_datasets (n_traj × n_steps × Π resolution × n_fields)`, the actual
  simulation state on disk (≈62 G *cells* for the same mix, before counting fields). The gap
  between this and `T_unique_model` is exactly what adaptive tokenization throws away, and it
  is very unequal across datasets (`helmholtz_staircase` at 1024×256 loses 8× more per token
  than `gray_scott` at 128×128).

The epoch ratio `T_seen / T_unique_model` is the "repetition factor". At the 4-node anchor it is
≈10; at paper scale over 19 datasets it is in the hundreds. That places these runs firmly in the
*repeated-data* regime, where the Muennighoff-style corrections (repeated tokens decay in value
after ~4 epochs) matter more than the vanilla Chinchilla fit. Sweeping `max_samples` / dataset
count while holding the model fixed will let you measure that decay directly.

### 5.3 Knobs that change tokens vs. knobs that change parameters

For a clean two-axis (`N`, `D`) sweep:

| axis | knobs | effect |
|---|---|---|
| **Parameters `N`** | `model.hidden_dim`, `model.processor_blocks`, `model.processor.space_mixing.mlp_dim`, `model.groups`, `model.intermediate_dim`, `model.projection_dim` | processor is `O(blocks × hidden²)`; encoder/decoder params also scale with `n_states` — pin `field_index_map_override=full_well_field_index` to keep them constant |
| **Tokens `D`** | `data.module_parameters.max_samples`, `trainer.max_epoch`, `data.module_parameters.batch_size`, `n_steps_input`, `world_size` | all multiply `T_seen` linearly |
| **Tokens per sample (context)** | `n_steps_input` (time), `per_axis_tokens` in `flexi_utils.py` (space — currently hard-coded 32/16) | the *only* way to change `N_spatial` is to edit `choose_kernel_size_deterministic`; there is no config knob |
| **Unique data `T_unique`** | number of entries in `well_dataset_info`, `restrict_train_num_trajectories`, `restrict_train_num_samples`, `restriction_seed` ([multidatamodule.py:155-167](../walrus/data/multidatamodule.py#L155-L167)) | `restrict_*` is the clean lever for a data-constrained scaling study — it subsets the index set deterministically |

⚠️ If you sweep `n_steps_input`, remember it changes **both** the token count per sample **and**
the temporal context length (a model capability), so it confounds the `D` axis with an
architectural axis. `max_samples` × `max_epoch` is the clean `D` knob.

⚠️ `per_axis_tokens` is hard-coded and its `patch_dict` only covers downsample factors
`{0,1,4,8,12,16,24,32}`; `choose_kernel_size_deterministic` asserts `H % per_axis_tokens == 0`.
Changing 32 → 16 or 64 will `KeyError`/assert on several Well resolutions unless you extend
`patch_dict` accordingly.

---

## 6. Quick reference: where to look

| Question | File:line |
|---|---|
| How many tokens per axis? | [flexi_utils.py:69-126](../walrus/models/shared_utils/flexi_utils.py#L69-L126) |
| Where strides are chosen per batch | [isotropic_model.py:309-336](../walrus/models/isotropic_model.py#L309-L336) |
| Where strides are applied | [vstride_encoder.py:316-380](../walrus/models/encoders/vstride_encoder.py#L316-L380) |
| Boundary padding / extra token | [patch_jitterers.py:210-258](../walrus/models/shared_utils/patch_jitterers.py#L210-L258) |
| Spatial token sequence | [full_attention.py:129](../walrus/models/spatial_blocks/full_attention.py#L129) |
| One dataset per batch | [mixed_dset_sampler.py:167-210](../walrus/data/mixed_dset_sampler.py#L167-L210) |
| Sampler → dataset → collate hand-off | [multidatamodule.py:429-438](../walrus/data/multidatamodule.py#L429-L438), [multidataset.py:374-406](../walrus/data/multidataset.py#L374-L406), [inflated_dataset.py:308-349](../walrus/data/inflated_dataset.py#L308-L349) |
| Per-sample `dt` draw | `the_well/data/datasets.py:801-813`; cap at `the_well/data/utils.py:114-128` |
| `dt` is discarded before the model | [well_to_multi_transformer.py:71-78](../walrus/data/well_to_multi_transformer.py#L71-L78) |
| Dimensionality-keyed encoder/decoder (the FSDP constraint) | [isotropic_model.py:110,141,194](../walrus/models/isotropic_model.py#L110) |
| Grad-accum draws a fresh batch | [training.py:1046-1053](../walrus/trainer/training.py#L1050) |
| `max_samples` semantics | [mixed_dset_sampler.py:209-220](../walrus/data/mixed_dset_sampler.py#L209-L220) |
| 2D/3D batch & time balancing | [multidataset.py:259,301](../walrus/data/multidataset.py#L259) + [mixed_dset_sampler.py:184-185](../walrus/data/mixed_dset_sampler.py#L184-L185) |
| Recycling / repetition | [mixed_dset_sampler.py:195-206](../walrus/data/mixed_dset_sampler.py#L195-L206) |
| Sampler seeding | [mixed_dset_sampler.py:172](../walrus/data/mixed_dset_sampler.py#L172) |
| Rank → dataset mapping | [training.py:351-371](../walrus/trainer/training.py#L351-L371) |
| Index → (file, traj, t0) | `the_well/data/datasets.py:780-792` |
| Windows per trajectory | `the_well/data/utils.py:88-112` |
| `n_states` derivation | [train.py:133-135](../walrus/train.py#L133-L135), [multidataset.py:305-373](../walrus/data/multidataset.py#L305-L373) |
