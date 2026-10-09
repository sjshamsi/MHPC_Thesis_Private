# FLOP accounting for W&B logging: how it works, and a review

How WALRUS training logs FLOPs, tokens and data repetition to W&B: which code does it, the exact
formulas, what is observed vs computed, and what to trust. Reviewed 2026-10-06.

Related: `mydocs/flop_estimator.md` (pricing a whole job before running it, same formulas),
`mydocs/walrus_tokens_and_data_pipeline.md` (what a token / micro-batch / epoch is).

---

## 1. Pipeline in one line

Encoder hooks observe each micro-batch's geometry → `forward_flops` (closed form) →
`training_step_flops` (×3, plus recompute) → summed per epoch → all-reduced over ranks →
`wandb.log(..., step=epoch)` on rank 0.

**Every FLOP number is analytical.** Nothing is measured with `FlopCounterMode` or a profiler at
run time. What *is* observed at run time is the geometry the formulas are evaluated on.

---

## 2. Code

| file | what it does |
|---|---|
| `walrus/utils/flops.py` | All of it. Per-stage formulas (`conv_stage_flops`, `space_mixing_flops`, `time_mixing_flops`, `channel_mixing_flops`); `ModelShape` (read once from the model); `BatchGeometry` (one micro-batch); `forward_flops`; `training_step_flops`; `unique_model_tokens`; `FlopAccountant` (hooks, accumulation, `epoch_metrics`, `state_dict`). |
| `walrus/trainer/training.py` | Builds the accountant in `Trainer.__init__` (skipped for baseline models that aren't encoder/processor/decoder), restores it on resume. In `train_one_epoch`: `reset_epoch()` at the start, `record_microbatch(dset_name)` after each micro-batch's backward/optimizer step, `epoch_metrics()` at the end, merged into `train_logs`. `train()` logs `train_logs` to W&B. `save_model_if_necessary` writes the counters into the checkpoint. |
| `walrus/trainer/checkpoints.py` | Stores the accountant's run-long counters in each checkpoint's `metadata.pt` under `"flop_accountant"`; exposes them as `loaded_metadata` on load. |
| `tests/test_flops.py` | CPU tests: formulas vs `FlopCounterMode`, accumulation, recompute, granular keys, counter round trip. |
| `tests/test_checkpoint.py::test_resume_continues_flop_counters` | End to end: counters continue across auto-resume. |

### What a run writes

- **W&B:** the keys in §3, once per epoch, at `step=epoch`, in the same log call as `train_loss`.
  On Leonardo that is an offline run under `$WANDB_DIR/wandb/offline-run-*`
  (`/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs`), uploaded later with `wandb sync`.
- **Checkpoints:** `metadata.pt` in every saved checkpoint (`step_N`, `last`, `best`) now holds
  `flop_accountant = {cumulative_flops, cumulative_processor_flops, cumulative_granular_flops,
  cumulative_tokens}`.
- No separate FLOP file is written.

---

## 3. W&B keys

Every key is summed over **all ranks**. "fwd+bwd" means forward × 3 (§5.4).

| key | definition | basis |
|---|---|---|
| `epoch_flops` | Σ over this epoch's micro-batches of fwd+bwd FLOPs, **including** recompute | per epoch |
| `flops_per_microbatch` | `epoch_flops` / micro-batches this epoch: an average over the epoch's dataset mix | per epoch |
| `cumulative_flops` | Σ `epoch_flops` since the run started (continues across resumes) | cumulative |
| `cumulative_processor_flops` | processor-only fwd+bwd, **excluding** recompute | cumulative |
| `epoch_{encoder,decoder}_proj{1,2}_flops` | each encoder/decoder conv stage, fwd+bwd, excluding recompute | per epoch |
| `cumulative_{encoder,decoder}_proj{1,2}_flops` | same, summed | cumulative |
| `enc_dec_flop_fraction` | (encoder + decoder) / total, **forward only** | per epoch |
| `tokens_seen` | Σ tokens processed since the run started (§6) | cumulative |
| `repetition_factor` | `tokens_seen / T_unique_model` (§6) | cumulative |
| `gpu_hours` | training-phase wall time of this epoch × `world_size` (validation and checkpointing excluded) | per epoch |

Identity: `cumulative_flops` = Σ `cumulative_*_proj*_flops` + `cumulative_processor_flops` +
recompute. Encoder + decoder are exactly the four proj stages, and recompute is the only term with
no key of its own (it is 0 without gradient checkpointing).

---

## 4. Observed vs computed

**Observed per micro-batch** by a forward pre-hook and a forward hook on the encoder
(`embed[dim_key]`):

- padded encoder input shape (after the patch jitterer);
- the two strides actually chosen (`random_kernel`, from `choose_kernel_size_deterministic`);
- periodic flag per axis (from `bcs[0]`);
- input channels (`len(field_indices)`, i.e. the dataset's fields + 3 BC flags);
- `T` and `B`;
- the token grid the encoder outputs.

**Fixed per run** (`ModelShape.from_model`): `H`, `M`, `P`, intermediate width `I`, kernels,
`causal_in_time`, `gradient_checkpointing_freq`.

**Computed:** every FLOP number, from these with the formulas in §5. Results are cached per
geometry (and dataset name), so only the first micro-batch of each dataset costs anything.

---

## 5. Formulas

Convention: one multiply-add = 2 FLOPs (as `FlopCounterMode` and "2ND"). `TB = T·B`,
`N` = tokens per frame, `H` = hidden, `M` = MLP width, `I` = intermediate, `P` = blocks.

### 5.1 Encoder (two strided convs, `walrus/models/encoders/vstride_encoder.py`)

```
conv FLOPs = 2 · batch · (output positions) · (kernel volume) · c_in · c_out
proj1: batch = TB, kernel 8³ (collapsed to 1 on a singleton axis), c_in = fields + 3, c_out = I
proj2: batch = TB, kernel 4³,                                     c_in = I,          c_out = H
```

### 5.2 Processor, per block × P

```
space (FullAttention, per frame):
  fused_ff_qkv  2 · TB · N · H · (M + 3H)      one Linear emits [ff | q | k | v]
  attention     4 · TB · N² · H                QKᵀ and AV, all heads
  attn_out      2 · TB · N · H²
  ff_out        2 · TB · N · (M/2) · H         SwiGLU halves M first
time (AxialTimeAttention, per location):
  input_head    2 · TB · N · H · 3H            1×1 conv = fused q/k/v
  attention     4 · (B·N) · T² · H
  output_head   2 · TB · N · H²
channel mixing  0                              Identity in every config
```

### 5.3 Decoder (two transposed convs, `vstride_decoder.py`)

Same conv formula, but a transposed conv costs per **input** position. Strides are the encoder's,
swapped. Periodic axes are circularly padded by `(k − s) // s` per side **before** the conv
(`vstride_decoder.py:210-251`), so they cost more than the token grid implies. Batch is
`T·B` when `causal_in_time` (every step decoded), else `1·B`.

```
proj1: H → I, kernel 4³     proj2: I → (fields), kernel 8³
```

### 5.4 Backward and recompute (`training_step_flops`)

```
useful    = 3 · forward                       backward counted as 2 · forward
recompute = encoder + decoder + processor · ceil(P / freq) / P     if gradient_checkpointing_freq > 0
total     = useful + recompute                → epoch_flops
```

Encoder and decoder are always checkpointed when any checkpointing is on; block `i` is when
`i % freq == 0` (`isotropic_model.py:126-135`).

### 5.5 Not counted

Norms (RMSGroupNorm, LayerNorm on q/k), softmax, SwiGLU gating, RoPE, bias adds, residual adds,
the relative-position bias, drop-path. Same scope as `FlopCounterMode`, which counts matmuls,
convs and SDPA only.

---

## 6. Tokens and repetition

- **Tokens per micro-batch** = `B · T_in · N`, with `N` the *observed* token grid (input frames
  only; the target frame is not a token).
- **`tokens_seen`** = Σ over every micro-batch on every rank, since the run started.
- **`T_unique_model`** (`unique_model_tokens`, computed once at start) = Σ over the training
  datasets of `n_traj × n_steps × N` on the **train split**. `N` is from each dataset's per-axis
  BCs, read from its first HDF5 file.
- **`repetition_factor`** = `tokens_seen / T_unique_model`. Aggregate over the mixture only; for
  per-dataset repetition use `scripts/count_WALRUS_flops.py`.
- **What it does not capture:** `dt` augmentation, rotations and jitter make repeated windows
  differ, so `T_unique_model` is a lower bound on distinct samples, and the factor an upper bound
  on true repetition.

---

## 7. Worked example: one micro-batch

200M anchor (`768/12/2048`, PT9 substrate: `causal_in_time=True`, no checkpointing) on
`euler_multi_quadrants_openBC`: 512², open/open, 6 fields, `B = 2`, `T = 6`.

Observed geometry: encoder input `532×532` (10 cells of learned padding per side), strides
`(4,4)` and `(4,4)`, middle grid `132×132`, tokens `33×33 = 1089`, `c_in = 6 + 3 = 9`, `TB = 12`.

| term | formula | forward |
|---|---|---:|
| encoder proj1 | 2·12·132²·8²·9·352 | 84.79 GF |
| encoder proj2 | 2·12·33²·4²·352·768 | 113.05 GF |
| space, per block | fused 87.36 + attention 43.72 + attn_out 15.42 + ff_out 20.55 | 167.04 GF |
| time, per block | input 46.25 + attention 0.24 + output 15.42 | 61.90 GF |
| processor (×12) | 12 · (167.04 + 61.90) | 2747.36 GF |
| decoder proj1 | 2·12·33²·4²·768·352 (transposed, per input position) | 113.05 GF |
| decoder proj2 | 2·12·132²·8²·352·6 | 56.52 GF |
| **forward total** | | **3114.76 GF** |
| **fwd+bwd** (`useful`, = `epoch_flops` contribution) | × 3, no recompute | **9.344 TF** |
| tokens | 2 · 6 · 1089 | 13,068 |
| `enc_dec_flop_fraction` contribution | (197.8 + 169.6) / 3114.8 | 11.8 % |

Note how small time attention is (0.24 GF, since `T = 6`); spatial attention (`N² = 1.19 M`) is
the quadratic term.

**One epoch, PT9 N1** (`fsdp`, 4 GPUs, `max_samples = 2000`), expected over the 9-dataset mix
(`count_WALRUS_flops.py ... --end-epoch 1`): 8,000 micro-batches, `epoch_flops` ≈ 70.72 PF,
`flops_per_microbatch` ≈ 8.84 TF, `tokens_seen` +101.76 M, `enc_dec_flop_fraction` ≈ 9.7 %. A
real epoch differs by a few tenths of a percent, depending on which datasets the sampler draws.

---

## 8. Evidence it is right

| check | what it proves |
|---|---|
| `tests/test_flops.py` (24 tests, CPU) | encoder + decoder closed form **exactly** equal `FlopCounterMode` on a real forward; token-grid rule; recompute surcharge; per-micro-batch accumulation; granular keys; counter round trip. **Not the processor:** on CPU, `FullAttention`'s SDPA dispatches to a kernel with no registered FLOP formula and counts as 0, so the processor (most of the FLOPs) has no CI check. It rests on the A100 job below. |
| A100 job `57434063` (`LeonardoRunScripts/flop_counter_backend_test.py`) | space and time attention forward **exactly** equal `FlopCounterMode` in fp32 at fullDims (2D and 3D) and Tier-2-small. One bf16-autocast small-shape row differs by 20 % (`AMBIGUOUS`: the flash kernel's counter convention); runs use fp32. |
| `LeonardoRunScripts/PT9_2d/validate_flops_pt9.py` | encoder/decoder exact on a real batch of each of the 9 PT9 datasets. |
| `scripts/build_well_registry.py --verify` | observed geometry on all 18 datasets equals the analytic geometry. |
| `tests/test_checkpoint.py::test_resume_continues_flop_counters` | after resuming for a second epoch, cumulative counters are exactly 2× the first epoch's; fails (1×) with the restore disabled. |

**GPU end-to-end test, 2026-10-06 — ALL PASS (70 checks).** `LeonardoRunScripts/flopcounter/`:
three back-to-back `boost_qos_dbg` jobs (SLURM 59456360, 59457635, 59458355) trained one run of a
256/4/512 model on the full PT9 mixture, `fsdp` on 1 node × 4 GPUs, gradient checkpointing every
2nd block, 40 micro-batches per GPU per epoch, epochs 1–2 / 3–4 / 5–6. `check_flopcounter.py`
compared the offline W&B history, every checkpoint's `metadata.pt`, and
`count_WALRUS_flops.py --exact`:

- one W&B run (`gownn974`) across the three jobs; each epoch logged once; jobs resumed at 3 and 5;
- cumulative counters continuous across both resumes, and equal to each checkpoint's saved values;
- every epoch's `epoch_flops`, tokens, encoder / processor / decoder split, recompute,
  `enc_dec_flop_fraction` and `flops_per_microbatch` equal the analytical replay to 1e-9 relative
  (e.g. epoch 1: 154.5147 TF logged vs 154.5147 TF expected; total 969.79 TF over 6 epochs);
- `repetition_factor = tokens_seen / T_unique_model`.

This covers the FSDP-wrapped hook path, the all-reduce over 4 ranks, the recompute term and the
resume fix on real hardware. The first attempt crashed on finding 9 below.

---

## 9. Review findings (2026-10-06)

| # | finding | severity | status |
|---|---|---|---|
| 1 | Cumulative counters (`cumulative_*`, `tokens_seen`, `repetition_factor`) restarted from 0 on auto-resume while the W&B run continued. Affects every chained run (noise floor: 3 jobs per seed). Per-epoch keys were unaffected. | **bug** | **fixed**: counters saved in `metadata.pt`, restored when `start_epoch > 1`. Checkpoints written before the fix have no counters: resuming from one logs a warning and restarts at 0. Rebuild those curves offline as the cumulative sum of `epoch_flops`. |
| 2 | `unique_model_tokens` inferred per-axis periodicity from the *set* of BC types: fully periodic 2D datasets got 32×33, mixed 3D ones a misplaced wall axis. PT9 `T_unique_model` was 3.309 G instead of 3.260 G, so `repetition_factor` read ~1.5 % low. | bug | fixed earlier today; reads per-axis BCs from the first file. Runs logged before: multiply PT9 `repetition_factor` by 3.309 / 3.260. |
| 3 | Backward is counted as exactly 2 × forward. The memory-efficient/flash attention backward also recomputes the attention forward, so hardware FLOPs are somewhat above the logged ones. | convention | keep; logged numbers are "useful" FLOPs, the usual scaling-law basis. |
| 4 | Elementwise ops are not counted (§5.5). | convention | keep; consistent with `FlopCounterMode`. |
| 5 | `gpu_hours` is training-phase wall time × `world_size`; it excludes validation and checkpoint time. | labelling | keep; use the SLURM accounting for allocation cost. |
| 6 | `flops_per_microbatch` includes recompute and averages over the epoch's dataset mix; it is not one dataset's cost. | labelling | keep. |
| 7 | Hooks also fire on validation forwards. Only training micro-batches call `record_microbatch`, right after their own forward/backward, so the recorded geometry is always the training batch's. (Recompute forwards were *not* harmless: see 9.) | none | — |
| 8 | Encoder `proj1`'s input gradient is counted although the data needs none. It *is* computed: `IsotropicModel` multiplies the input by the `encoder_dummy` parameter, so the input requires grad. | none | — |
| 9 | With gradient checkpointing, backward re-runs the encoder forward and non-reentrant checkpointing can stop that recompute before the encoder returns: the pre-hook fired, overwrote the observed geometry, and the post-hook (which adds the token grid) never ran. The first `record_microbatch` then raised `token_shape unknown`, crashing any run with `gradient_checkpointing_freq > 0` (e.g. the fullDims roofline) on its first micro-batch. Pre-dated the resume fix; untested because no test combined checkpointing, hooks and backward. | **bug** | **fixed 2026-10-06**: the pre-hook writes a pending geometry that only the post-hook commits. Regression test `test_accountant_survives_gradient_checkpointing_backward`; confirmed on GPU by the end-to-end test above. |
| 10 | The encoder pre-hook cost ~200 µs per micro-batch, almost all of it in `_periodic_flags`: per-element tensor indexing of `bcs`, plus a caught `IndexError` on the padded third axis. Pure CPU overhead, ~0.1 % of a step. | perf | **fixed 2026-10-06**: one `bcs.tolist()` (~5 µs); `_dataset_periodic_axes` reuses the same helper. |
| 11 | `ModelShape.from_model` reads channel-mixing width from a `hidden_features` attribute that `shared_utils/mlps.py::MLP` does not have (it has `fc1`/`fc2`). With `channel_mixing=generic_mlp`, channel mixing would be priced at **0 with no warning**. Unreachable today: `generic_mlp.yaml` passes `exp_factor`, which `MLP` rejects, so the model does not build. | latent bug | open; fix (read `fc1.out_features`) when that config is ever repaired. |
| 12 | With `causal_in_time=True` (PT9), time attention is priced as full `T²`, as `FlopCounterMode` also does; the causal mask makes about half of it useful. At `T = 6` the term is ~0.1 % of the forward pass (§7). | convention | keep. |
| 13 | Validation forwards are not counted in `cumulative_flops` (and validation time not in `gpu_hours`, see 5). | labelling | keep; the counters are training compute only. |
| 14 | `repetition_factor`'s denominator uses `analytic_token_shape`, which assumes jitter padding (+1 token per non-periodic axis). With `jitter_patches=False` it would be slightly off. The FLOP counts are unaffected (they use the observed grid). | labelling | keep; every study config jitters. |

---

## 10. Where the accountant is valid, and how it could break (2026-10-06)

### 10.1 Cost

The formulas run once per distinct geometry (dataset × resolution × strides) and are cached. Per
micro-batch: pre-hook + post-hook (shape reads, no GPU sync: `dynamic_ks` is a Python tuple and
`bcs` stays on the CPU) + `record_microbatch` (~5 µs dict lookup and adds). Per epoch: one
all-reduce of a 10-element float64 tensor. Per checkpoint: 7 floats in `metadata.pt`.

### 10.2 Validated envelope (everything the study runs)

- `model=isotropic_model` with `spacebag_and_vstride_encoder` (`learned_pad: true`) and
  `vstride_decoder`;
- `space_mixing=full_spatial_attention`, `time_mixing=axial_time_attention`,
  `channel_mixing=identity`;
- `override_dimensionality=3` (one encoder/decoder pair for all data);
- any `hidden_dim`, `processor_blocks`, `mlp_dim`, `intermediate_dim`, heads,
  `gradient_checkpointing_freq` (read off the live model);
- `causal_in_time` either way; any dataset mix or resolution; deterministic or random strides
  (observed, not predicted); grad accumulation; `reuse_batches` (each reuse is a real fwd/bwd and
  is counted);
- `local` / `ddp` / `fsdp` / `hsdp`;
- AMP on or off: FLOP counts don't depend on precision, only the peak-FLOP/s denominator of any
  MFU figure does.

### 10.3 Outside the envelope

**Fails loudly (safe):**
- `space_mixing=axial_spatial_attention`: its config targets `spatial_blocks/axial_attention.py`,
  which does not exist, so the model won't build.
- Encoders/decoders without `inner_dim` / `base_kernel1` raise in `ModelShape.from_model` at
  `Trainer` construction.
- Baseline wrappers (DPOT, ScOT, MPP): detected by `supports_flop_accounting`, so no keys are
  logged.

**Wrong without any error (the real risks):**
1. `channel_mixing=generic_mlp` once its config is fixed: priced at 0 (finding 11).
2. `hmlp` / `flexivit` encoders, or any encoder called without a `random_kernel` kwarg: the
   pre-hook returns early, `record_microbatch` returns `None`, and **every FLOP key logs 0**. If
   such an encoder pads internally, the `padding=0` assumption in `conv_out_size` also breaks.
3. Separate 2D and 3D encoders (`include_d=[2,3]` without `override_dimensionality`):
   `ModelShape` takes kernels from the *first* encoder/decoder only, so the other dimensionality
   is priced with the wrong kernels. Safe while every plan keeps `override_dimensionality=3`.
4. Multi-step rollout in training (`Trainer.rollout_model` currently raises): the encoder would
   run N times per micro-batch but be counted once, an N× undercount.
5. `torch.compile` (not used): Python hooks that mutate state may run only at trace time or force
   graph breaks, leaving the recorded geometry stale.
6. Changes inside `FullAttention` / `AxialTimeAttention` or the decoder's pad/crop logic
   (`vstride_decoder.py:225` already carries a TODO for non-divisible strides). The processor has
   no CI check (§8) and the decoder geometry is *derived* from the encoder's, not observed.

**Cheapest guards if the configs ever move:** warn when a training epoch ends with zero
micro-batches recorded (catches 2), and re-run `flop_counter_backend_test.py` after any change to
the attention blocks (catches 6).
