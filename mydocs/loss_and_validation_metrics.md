# Training and validation metrics in WALRUS
Reviewed 2026-10-07.

---

## 1. Code map

| Path | Role |
|---|---|
| `walrus/configs/trainer/defaults.yaml` | Pins `loss_fn: MAE`, `validation_suite` (NRMSE, VRMSE, binned spectral MSE, PearsonR), cadences (`val_frequency: 2`, `rollout_val_frequency: 5`), `short_validation_length: 20`, `max_rollout_steps: 60` (effective rollout length is 65, §4.1), `num_time_intervals: 5`, `loss_multiplier: 100`, `batch_aggregation_fns` (mean, median, std) |
| `walrus/trainer/training.py` | `rollout_model` (one-step and autoregressive prediction, (de)normalisation), `train_one_epoch` (training loss, accumulation, clipping), `validation_loop` (metrics, aggregation, the headline scalar, files on disk), `split_up_losses` / `temporal_split_losses` (per-field and per-time-interval keys), `validate_if_necessary` (cadence), `train` (end-of-run test pass) |
| `walrus/data/well_to_multi_transformer.py` | `ChannelsFirstWithTimeFormatter.process_input`: builds the training target (frame-to-frame changes) |
| `walrus/trainer/normalization_strat.py` | `SamplewiseRevNormalization` (= `RMSSamplewiseRevNormalization`), `normalize_target` |
| `walrus/data/inflated_dataset.py` | `get_padded_field_masks`: which channels are 2D→3D padding |
| `walrus/models/temporal_blocks/axial_time_attention.py`, `walrus/models/shared_utils/position_biases.py` | the causal mask in time attention |
| `walrus/data/multidatamodule.py` | train / valid / rollout-valid / test datasets and loaders: frame spacing, full-trajectory mode, samplers, batch sizes |
| `the_well/benchmark/metrics/spatial.py` (installed) | `MAE`, `VRMSE`, `NRMSE` definitions |

---

## 2. Step-by-step forward pass and how losses are calculated

Notation (the paper's, §2):
- $\mathcal{S}$ is a physics-driven spatiotemporal system (e.g. `active_matter`).
- $d$ is the number of spatial dimensions ($x$, $y$, $z$).
- $q$ is the number of observed fields.
- $u^{\mathcal{S}}(x, t): \prod_{i \in [d]} [0, L_i^{\mathcal{S}}] \times [0, \infty) \rightarrow \mathbb{R}^q$ is the continuous state: at every point in space and time it gives one real value per field. In practice the system is discretised in space and time.
- $u^\mathcal{S}_{t} \in \mathbb{R}^{N^\mathcal{S}}$ is the state at the $N^\mathcal{S}$ spatial grid points at time $t$.
- The difference between two consecutive states is
  $\Delta u^\mathcal{S}_{T + \Delta t} = u^\mathcal{S}_{T + \Delta t} - u^\mathcal{S}_{T}$.
- The history window is $U^\mathcal{S}_{t} = [u^\mathcal{S}_{t - \tau \Delta t}, \dots, u^\mathcal{S}_{t}]$ (the paper calls this "τ snapshots"; as written the list has τ+1 entries.).
- We want a model with $u^\mathcal{S}_{t + \Delta t} \approx u^\mathcal{S}_{t} + \mathcal{M}(U^\mathcal{S}_{t})$: it predicts the change to the next frame.

Below, $s$ and $d$ denote the two normalisation scales defined in §2.1 step 3. (This $d$ is a scale, unrelated to the number of spatial dimensions above.)

### 2.1 During training

The full cycle is a call from `train_one_epoch` to `rollout_model(train=True)`. From one dataset we get `B` samples with `C` channels and 6 input frames.

The 6+1 frames of a window are taken every `dt`-th stored frame, with `dt` drawn
per sample between `min_dt_stride=1` and `max_dt_stride=5` (capped near the end of a trajectory). E.g. with `dt = 3` the window is frames 10, 13, 16, 19, 22, 25 → target 28. The model is never told `dt` and it has to infer the timescale from how much the input frames change. Validation and test always use `dt = 1`.

1. The dataset returns `input_fields` $[x_0, \dots, x_5]$ and `output_fields` $[x_6]$.
2. `process_input` (with `causal_in_time=True`, `predict_delta=True`, `train=True`) concatenates to $[x_0, \dots, x_6]$ and takes consecutive differences, giving 6 targets: $[\Delta_1 = x_1 - x_0, \dots, \Delta_6 = x_6 - x_5]$. Constant fields (time-independent fields stored separately in the dataset, `constant_fields`) are appended as extra input channels, and are not predicted. The four combinations of the two flags:

| `causal_in_time` | `predict_delta` | target |
|---|---|---|
| True | True (our runs) | six changes $\Delta_1 \dots \Delta_6$ |
| True | False | the next six frames $x_1 \dots x_6$ |
| False | True | one change $\Delta_6$ |
| False | False | one frame $x_6$ |

3. Two scales are computed from the input window with `RMSSamplewiseRevNormalization.compute_stats`, per sample $b$ and channel $c$, over time and space:

$$
\text{RMS}_{\text{Space}\times\text{Time}}(\mathbf{U})_{b, c} = \sqrt{\frac{1}{T \times H \times W \times D} \sum_{t, h, w, d} \mathbf{U}_{t, b, c, h, w, d}^2}
$$

- input scale $s_{b,c} = \max(\text{RMS}(x_0 \dots x_5), \varepsilon)$, over the 6 frames
- change scale $d_{b,c} = \max(\text{RMS}(\Delta_1 \dots \Delta_5), \varepsilon)$, over the 5 changes *inside* the window (not the target $\Delta_6$);
- $\varepsilon = 10^{-5}$, a floor against division by zero (e.g. an all-zero channel).

The "mean" is fixed at 0, so nothing is subtracted, the data are only rescaled. The paper says this is to avoid drift when the predicted step is de-normalised and added back at every rollout step. A side benefit of this is that since $\text{RMS}^2 = \text{mean}^2 + \text{variance}$, a field with a large background and tiny fluctuations is not blown up the way std-normalisation would.

```python
def compute_stats(self, x: torch.Tensor, metadata, epsilon: float = 1e-5) -> NormalizationStats:
   # x - T B C H [W D] - MPP format
   with torch.autocast(device_type=x.device.type, enabled=False):
      x = x.float()
      dims = self.get_dims_from_metadata(metadata)          # time + spatial axes
      sample_std = x.square().mean(dims, keepdim=True).sqrt()  # s: RMS over time and space
      sample_mean = torch.zeros_like(sample_std)               # offset fixed at 0
      sample_std = torch.maximum(sample_std, torch.tensor(epsilon, device=sample_std.device))
      if x.shape[0] > 1:
         deltas = x[1:] - x[:-1]                            # the 5 changes in the window
         delta_std = deltas.square().mean(dims, keepdim=True).sqrt()   # d
         delta_mean = torch.zeros_like(delta_std)
         delta_std = torch.maximum(delta_std, torch.tensor(epsilon, device=delta_std.device))
      else:                                                    # 1-frame window: never in our runs
         delta_std = torch.ones_like(sample_std)
         delta_mean = torch.zeros_like(sample_mean)
      return NormalizationStats(sample_mean, sample_std, delta_mean, delta_std, epsilon)
   ```

4. The forward pass runs on $x / s$. The network emits $[\hat{y}_1, \dots, \hat{y}_6]$, one prediction per input position, already on the normalised-change scale (units of $d$). Ont thing to note is that frames only interact in the time-attention blocks (the encoder, decoder, spatial attention and MLPs treat each frame separately). With `causal_in_time=True` the time attention gets a relative-position bias with `-inf` on every future key (`position_biases.py:114-116`), so after the softmax position $k$ puts zero weight on frames after $x_{k-1}$. Position $k$ therefore sees frames $x_0 \dots x_{k-1}$. (The scales $s, d$ are computed from the whole window before the model, so they carry the window's overall magnitude to every position.)
5. With `override_dimensionality=3`, 2D data get a size-1 third axis, so vector fields get a third component and rank-2 tensors 9 components. Those padded components (e.g. `momentum_z` in the 2D Euler sets; scalars such as density are never padded) are marked `False` in `padded_field_mask` (`inflated_dataset.py::get_padded_field_masks`). The padded data are zero, so their true change is zero; the prediction is also zeroed there (`masked_fill(~padded_field_mask, 0)`), so those channels give no error and no gradient flows into them. The prediction is then rearranged from the model's `T B C H W [D]` to the metrics' `B T H W [D] C` (`formatter.process_output`), keeping only as many channels as the target.
6. The target is put on the same scale: $y_k = \Delta_k / d$ for every position $k$ (one scale $d$ per sample and channel, shared by all positions. Zero offset, so no shift). Comparing $\hat{y}_k$ with $y_k$ is the same as comparing reconstructed frames, because adding the same previous frame to both cancels:

$$
|\hat{y}_k - \Delta_k/d| = \frac{|(u_{k-1} + \hat{y}_k d) - (u_{k-1} + \Delta_k)|}{d} = \frac{|\hat{u}_k - u_k|}{d}
$$

7. `minimum_context=1` drops position $k = 1$, the "zero-context" prediction: it sees one frame, so it cannot infer the rate of change (and with `dt` varying 1–5 the target scale varies 5× for the same input). Positions $k = 2 \dots 6$ are scored: $k = 2 \dots 5$ predict changes that lie inside the input window, and $k = 6$ predicts the genuinely new change $\Delta_6 = x_6 - x_5$. Position $k$ sees $k$ frames of context. Setting `minimum_context=0` would keep all six. Validation only ever scores the last position ($k = 6$).

8. MAE is a mean over the spatial grid (the L1 norm divided by the number of grid points):

$$
\text{MAE} = \frac{1}{H \times W \times D} \sum_{h=1}^H \sum_{w=1}^W \sum_{d=1}^D \left| \hat{y}_{h, w, d} - y_{h, w, d} \right|
$$

In the code this gives a `B × 5 × C` tensor; the final `.mean()` averages over samples, positions and all channels, padded ones included. So the zeroed channels add no error but still count in the denominator. Each dataset's loss is scaled by (real channels / total channels), e.g. 5/6 for the Euler sets (density, energy, pressure, and momentum padded from 2 to 3 components). Validation drops padded channels instead, so this weighting is training-only.

```python
y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter_dict[dset_name])   # ~line 1110

if y_pred.shape[1] > self.minimum_context:         # drop position 1
   y_ref  = y_ref[:,  self.minimum_context:]
   y_pred = y_pred[:, self.minimum_context:]
   loss = (self.loss_multiplier * self.loss_fn(y_pred, y_ref, current_metadata, eps=self.model_epsilon).mean(() / grad_acc_steps)                                                                
   self.grad_scaler.scale(loss).backward()
```

`y_ref` is divided by $d$ in `rollout_model` (`normalize_target`, ~line 590); `y_pred` is returned undenormalised in training.

$$
\mathcal{L} = \frac{1}{q} \sum_{i=1}^q \frac{\| \mathcal{M}(\mathbf{U}_t)^{(i)} - \Delta \mathbf{u}_{t+1}^{(i)} \|_1}{\text{RMS}_{\text{Space}\times\text{Time}}(\Delta \mathbf{U}_t^{(i)})}
$$

The code's channel average is the $\frac{1}{q}\sum_i$, and $d$ plays the role of the denominator. Differences: the L1 norm is a spatial mean (a constant factor). The code also averages over the batch and the 5 causal positions. Padded channels count in the average and the result is multiplied by `loss_multiplier=100`.

Effect of the denominator: a field that changes a lot over the window has a large $d$, so its errors are down-weighted relative to slowly changing fields, which the paper expects to be more predictable.

9. Backward pass, then on every 4th micro-batch the update:
- gradients accumulate in `.grad` over `grad_acc_steps=4` micro-batches (each loss is divided by 4, so the sum is the mean). The other 3 run inside `model.no_sync()`, so GPUs exchange gradients once per update.
- the accumulated gradient's global L2 norm is clipped to `clip_gradient=10`. Because the loss was multiplied by 100, that is a clip at 0.1 on the unscaled gradient (whether it fires is unknown: `gradient_log_level=0`, so no gradient norms are logged);
- AdamW step, `zero_grad()`. The LR scheduler steps once per epoch (`lr_scheduler_per_step=False`).

On N1 (4 GPUs) one update sees 2 × 4 × 4 = 32 windows; an epoch (`max_samples=2000` micro-batches per GPU) is 500 updates.

The multiplier exists for "paranoia about underflow error" (trainer docstring); in FP32 with Adam it does not change the update (Adam is invariant to a constant gradient scale, apart from its `eps`), but it does change the clipping threshold, as above.

**What is logged from training.**

| key / output | value |
|---|---|
| W&B `train` and `train_loss` | epoch mean of `100 · MAE` (normalised scale), i.e. the multiplier is **included** |
| stdout per-batch line `loss …` | `sqrt(100 · MAE)` (the code prints the square root) |

### 2.2 During validation

`rollout_model(train=False)`. Each step implements $u_{t+1} = u_t + \mathcal{M}(U_t / \text{RMS}(U_t)) \cdot \text{RMS}(\Delta U_t)$:

1. $s$ and $d$ are computed from the current 6-frame window $U_t$, exactly as in §2.1 step 3.
2. $\hat{y} = \mathcal{M}(U_t / s)$ is computed, and only the last position, $\hat{y}_6$, is kept.
3. The next frame is reconstructed in physical units: $\hat{u}_{t+1} = u_t + \hat{y}_6 \cdot d$.
4. Padded channels are zeroed, then dropped before scoring.
5. For a rollout, drop the oldest frame of $U_t$, append $\hat{u}_{t+1}$, and repeat for 65 steps. The scales $s, d$ are recomputed at every step from the current window, which from step 6 onwards contains only predicted frames.

### 2.3 From per-step errors to the numbers in W&B

Code: `validation_loop`, `split_up_losses`, `temporal_split_losses`. The same chain runs for both passes, and only the number of steps `T` differs (`T = 1` for one-step, `T = 65` for a rollout).

*Stage 0 — what scoring receives.* `rollout_model` returns the prediction $\hat{u}$ and the truth $u$, both of shape `B × T × S × C` (samples × steps × space × channels, channel last), in physical units. For a rollout the batch size is 1, so each batch is one trajectory: `1 × 65 × S × C`. For one-step, the single step is the prediction from the full 6-frame window. Padded channels are removed here (`y_pred[..., padded_field_mask]`), so `C` counts real channels only.

*Stage 1 — one error per sample, step and channel.* Each metric reduces space only, giving a `B × T × C` tensor (in that order) per metric:

$$
e^{\text{MAE}}_{b,t,c} = \langle |\hat{u}-u| \rangle_x ,\qquad
e^{\text{VRMSE}}_{b,t,c} = \sqrt{\frac{\langle (\hat{u}-u)^2 \rangle_x}{\operatorname{Var}_x(u) + \varepsilon}} ,\qquad
e^{\text{NRMSE}}_{b,t,c} = \sqrt{\frac{\langle (\hat{u}-u)^2 \rangle_x}{\langle u^2 \rangle_x + \varepsilon}}
$$

with $\langle\cdot\rangle_x$ the mean over grid points, $\operatorname{Var}_x$ = `torch.std(u)²` (unbiased, $N-1$) and $\varepsilon = 10^{-7}$ (see below). PearsonR is computed too. The denominators use the true frame at that step and channel, not the whole trajectory, so a nearly uniform true frame gives a large VRMSE. (What VRMSE means: §3.1.)

*Stage 2 — reduce each sample to scalars* (per metric). For every interval $[a, b)$ of steps:

$$
\bar{e}_{b,c}^{[a,b)} = \frac{1}{b-a}\sum_{t=a}^{b-1} e_{b,t,c}
\quad\text{(one field)},\qquad
\bar{e}_{b}^{[a,b)} = \frac{1}{b-a}\sum_{t=a}^{b-1}\frac{1}{C}\sum_{c} e_{b,t,c}
\quad\text{(`full`: mean over fields)}
$$

- `T=all` is $[0, T)$: all 65 steps for a rollout, the single step for one-step validation.
- The sub-intervals are log-spaced, with boundaries `[0] + [int(exp(z)) for z in linspace(0, ln T, min(T, 5))]`. For `T = 1` this is `[0, 1]`, so the one-step pass gets no sub-intervals, only `T=all`. For `T = 65` it is `[0, 1, 2, 8, 22, 64]` (`int` truncates, e.g. `exp(ln 65) = 64.99…` to 64), giving:

| key | steps averaged (1-indexed) |
|---|---|
| `T=0:1` | step 1 |
| `T=1:2` | step 2 |
| `T=2:8` | steps 3–8 |
| `T=8:22` | steps 9–22 |
| `T=22:64` | steps 23–64 |
| `T=all` | steps 1–65 |

Step 65 appears only in `T=all`. These are exactly the keys in the logged runs.

- Rollout pass only: the per-step curve $\frac{1}{C}\sum_c e_{b,t,c}$ (and one per field) is also kept, as a `B × T` "time log" (`{dataset}/{field|full}_{metric}_rollout`).

Each Stage-2 quantity is a vector of length `B`: one number per scored sample.

*Stage 3 — collect all scored samples.* Those vectors are concatenated across the batches a GPU scores (up to 20, §6), then gathered from every GPU (`all_gather_object`) and concatenated again. Each key now holds `N` values, one per scored window (one-step) or trajectory (rollout): e.g. `N = 80` rollout trajectories per dataset on N1.

*Stage 4 — aggregate over samples.* For each key, `mean`, `median` and `std` over the `N` values (`batch_aggregation_fns`). These are the W&B scalars, `{pass}_{dataset}/{field|full}_{metric}_T={interval}_{agg}`, logged at `step=epoch`. Because the time and field averages happen per sample first, a `median` is the median over samples of each sample's average, not a median over steps.

*Stage 5 — the headline scalar.* `valid` / `rollout_valid` = mean over datasets of the Stage-4 `mean` of `{dataset}/full_MAE_T=all` (§5). `rollout_valid_…_T=0:1` and `valid_…_T=all` are both one-step errors but are not the same number. Tthe rollout's first step is predicted only from the window at the start of each trajectory (one per trajectory), while the one-step pass scores windows from throughout the trajectories.

The $\epsilon$ of VRMSE is 1e-7, not `validation_epsilon`, as `VRMSE.eval` calls `NRMSE.eval(x, y, meta, norm_mode="std")` without forwarding `eps`, so the argument is ignored and the default `1e-7` applies. Checked directly on a tensor whose true variance is ~1e-8: `eps = 1e-7`, `1e-5` and `0.1` all returned the same VRMSE. This only matters where a true frame is almost constant over space.

### 2.4 What is stored on disk

At every validation event (epoch `E`), rank 0 writes into `{experiment_folder}/viz/`.

| file | contents |
|---|---|
| `loss_dicts/{pass}_loss_dict_epoch{E}_rank0.pkl`, `pass` ∈ `valid`, `rollout_valid`, `test`, `rollout_test` | the Stage-4 scalars, i.e. exactly what goes to W&B (2 952 scalars in that run) |
| `loss_dicts/rollout_{valid,test}_time_logs_epoch{E}_rank0.pkl` | per dataset: per-step curves of length 65 for each field / `full` × metric × {mean, median, std} across trajectories |
| `{dataset}/rollout_losses/epoch_{E}_rank0_rollout_{valid,test}/{field\|full}_{metric}_rollout_{agg}.npy` | the same curves as separate files, shape `(65,)` |
| `{dataset}/{field} Histogram/…png` plus `bins` / `xhist` / `yhist` `.npy` | value-distribution plots of prediction vs truth, for the **last scored batch only** |

Metrics in these files: MAE, VRMSE, NRMSE, PearsonR. Padded components are absent (only `momentum_x`, `momentum_y` for Euler).

This happens in **every run that executes a rollout pass**: rollout validation during training
(every `rollout_val_frequency` epochs and the last epoch), the end-of-run test pass, and
validation-only runs (`Trainer.validate()`). It is not gated by `debug_mode` or by full vs capped
validation. The test run above has `rollout_test` files at epochs 2, 4 and 6, one per chained job,
because each job ended at its own `max_epoch` and ran the end-of-run test pass.

The following are not stored:
- the per-sample values of Stages 2–3 (each trajectory's or window's own error).
- the predictions themselves (`dump_prediction_to_disk` is off by default).

A consequence is that a median over trajectories for a different time range (e.g. the paper's steps 1–20) cannot be recomputed from these files. That needs each trajectory's own average, which is gone. Averaging the saved per-step median curve is not the same thing (example in §3.2). Doing it needs a re-evaluation of the checkpoint with a small code change to save per-trajectory values.

### 2.5 How samples from different GPUs, trajectories and datasets are combined

Short version: inside a dataset, every sample is made the same shape first. Each sample is then reduced to plain numbers. Combining is just joining lists of numbers.

**1. Within a dataset, the differences are removed before anything is combined.**

| possible difference | what the code does |
|---|---|
| time step (`dt`) | validation and test always use `dt = 1`; the 1–5 spacing exists only in training |
| trajectory length | rollouts: every trajectory is cut to the length of the dataset's **shortest** trajectory (`lowest_steps` in the Well dataset, full-trajectory mode), then to 65 steps (§4.1). One-step: every sample is exactly 1 step |
| starting point | rollouts all start at frame 0. One-step windows start anywhere, but each window is an independent sample |
| ending step | follows from the above: every rollout in a dataset has the same number of steps |

So step $k$ means the same thing for every trajectory of a dataset, and every sample has the same
`T × C` shape after Stage 1.

**2. Each sample is reduced to plain numbers on its own GPU** (Stage 2, §2.3): e.g. trajectory A's
65 × C VRMSEs become one number for "`full`, `T=22:64`", one for "`full`, `T=all`", and so on.
Nothing about GPUs, batches or order is kept.

**3. The GPUs share their lists of numbers** (`all_gather_object`), the lists are joined, and
`mean`, `median` and `std` are taken (Stage 4). Every scored sample counts once, whichever GPU scored
it (a sample duplicated by the sampler's padding counts twice, §6). Order does not matter.

**4. Datasets are never pooled** in the logged keys: each dataset has its own. The only
cross-dataset number is the headline `valid` / `rollout_valid`, a plain average of the per-dataset
means, so a dataset with 12 scored trajectories counts as much as one with 80.

Toy example, 2 GPUs, one dataset with 4 trajectories:

| GPU | trajectories | each one's average VRMSE over all steps |
|---|---|---|
| 0 | A, C | 0.20, 0.30 |
| 1 | B, D | 0.25, 1.00 |

Joined: [0.20, 0.30, 0.25, 1.00] → `mean` = 0.44, `median` = 0.275. The per-step curves (§2.4) are
combined the same way, step by step: step 1 of A, B, C, D together, then step 2, and so on.

**Caveat for datasets with mixed trajectory lengths.** Because rollouts are cut to the shortest
trajectory, a dataset like `viscoelastic_instability` (20- and 60-frame trajectories) scored as one
dataset would cut **every** rollout to 14 steps (20 − 6), including the 60-frame ones. The strategy
doc's per-regime scoring (`include_filters`) should avoid this, since the shortest length is taken
over the files actually loaded; not yet tested. No PT9 pretraining dataset is affected: all have at
least 101 frames.

---

## 3. The paper's evaluation and how ours compares

### 3.1 VRMSE (paper Appendix E.1.1)

```
VMSE(u, v)  = ⟨|u − v|²⟩ / ( ⟨|u − ū|²⟩ + ε )
VRMSE(u, v) = ⟨|u − v|²⟩^{1/2} / ( ⟨|u − ū|²⟩ + ε )^{1/2}
```

`u` is the **true** field (one step, one channel), `v` the **prediction**, `⟨·⟩` the spatial mean,
`ū = ⟨u⟩` the true field's spatial mean. The numerator is the mean squared error; the denominator is
the true field's spatial variance. So VRMSE is RMS error in units of how much structure the true
field has. It has no units, so fields and datasets can be averaged together, which MAE cannot do.

Predicting a flat field at the true mean, `v = ū`, gives exactly 1:

| VRMSE | meaning |
|---|---|
| 0 | perfect |
| < 1 | better than a flat field at the true mean |
| ≈ 1 | no better than that flat field |
| > 1 | worse than it |
| ≈ 1.41 (√2) | typical long-horizon saturation for chaotic flows: a prediction with the right statistics but decorrelated from the truth has error ≈ 2× the variance (our inference; the paper only calls long-horizon VRMSE "meaningless" for chaotic systems) |

Weak spot: computed per step against that step's own spatial variance, so a nearly uniform true
frame makes it large. In the code the variance uses `torch.std` (N−1), which is negligible on grids
of 128² and above.

### 3.2 How the paper reports its numbers

*"Median VRMSE over trajectories"* (tables' captions: "Losses (Median VRMSE)"). For rollout ranges,
as we read the text:

1. For each trajectory $j$, predict autoregressively from 6 frames of history, starting at the
   paper's **T = 17** (below).
2. Compute VRMSE per step (per field; presumably averaged over fields, not stated).
3. Average each trajectory's per-step VRMSE over the range (e.g. steps 1–20) → one number $s_j$.
4. Report the **median** of the $s_j$ over trajectories.

Toy example, 3 trajectories, a 3-step range:

| trajectory | step 1 | step 2 | step 3 | $s_j$ |
|---|---|---|---|---|
| A | 0.1 | 0.2 | 0.3 | 0.20 |
| B | 0.1 | 0.1 | 0.4 | 0.20 |
| C | 0.2 | 0.9 | 1.6 | 0.90 |

Paper's number: median(0.20, 0.20, 0.90) = **0.20**. The mean would be 0.43: one blow-up dominates
it. Our saved per-step median curve here is (0.1, 0.1, 0.4); its average happens to be 0.20 too, but
in general the median of averages and the average of medians differ (§2.4).

*One-step "over all possible steps"* (tables' captions: "One-step computed over all possible
steps"). In every trajectory, every position with 6 true frames of history is used to predict just
the next frame from true history; each prediction is scored separately. A 101-frame trajectory gives
95 one-step predictions (frames 0–5 → 6, …, 94–99 → 100). The paper does not say whether the median
is over all individual windows or over per-trajectory averages first. Our `valid` pass also uses
every window, and its median is over individual windows; capped runs score only 160 of them per
dataset on N1 (§6).

*Where "T = 17" comes from.* The paper's evaluation section: "all predicted trajectories begin from
T = 17 to allow for equal comparisons with models utilizing context lengths matching that used by
MPP in pretraining (16)"; the Figure 6 caption: "Autoregressive prediction starts at T=17 to account
for context-length differences". So every model is scored from the same point, and MPP has its full
16-frame context. Not stated: whether T is counted from 0 or 1 (a 16-frame context suggests 1-based:
frames 1–16 history, frame 17 first prediction), and which 6 frames Walrus uses as history
(presumably the 6 just before). It is stated for the downstream/comparison evaluation; whether every
table (e.g. the jitter ablation, Table 7) uses it is not clear. The windows reported are
T ∈ [1:10], [11:30] (Fig. 6, Table 8 column headers) and [1:20], [21:60] (Table 1 column headers).

### 3.3 Training setup (paper Appendix B.1)

AdamW (decoupled weight decay), inverse-square-root schedule with linear warmup and square-root
cooldown, base LR 2e-4 (Walrus) / 1e-4 (HalfWalrus), gradient clip 10, FP32 (BF16 degraded
accuracy, FP16 was unstable). An "epoch" is 2000 micro-batches; 200 epochs on 96 H100s = 38 M
micro-batches.

### 3.4 Paper vs this study's runs

| aspect | paper | here (PT9 runs) |
|---|---|---|
| Normalisation | RMS over time×space, zero mean (Eqs. 9–10) | same (`RMSSamplewiseRevNormalization`); per sample and channel |
| Training loss | L1 / `RMS(ΔU)`, averaged over fields (Eq. 8) | `100 ·` spatial-mean MAE / $d$, over positions 2–6 of the causal output, all channels incl. padded |
| History window | 6 frames (2D), 3 (3D) | 6 (all PT9 data is 2D) |
| Frame spacing in training | uniformly spaced samples | drawn 1–5 per sample; validation and test use 1 |
| Dataset sampling | uniform system, then uniform start frame | uniform dataset per micro-batch (`BatchedMultisetSampler`), random window |
| Optimiser / schedule | AdamW; inverse-sqrt, linear warmup, sqrt cooldown; clip 10; FP32 | same; clip 10 (on the ×100 loss), `enable_amp=False`; warmup 10, cooldown 10 epochs |
| Loss multiplier | not mentioned | 100 |
| Base LR | 2e-4 Walrus, 1e-4 HalfWalrus | 2e-4 for **every** shape (§6 of the strategy doc checks this) |
| "Epoch" | 2000 micro-batches | `max_samples=2000` micro-batches per rank per epoch |
| Rollout start | T = 17 (indexing not stated) | frame 6 is the first prediction (history frames 0–5); `start_rollout_valid_output_at_t` is left at −1 |
| Rollout length | ranges up to step 60 | 65 steps (§4.1) |
| Per-trajectory statistic | **median** VRMSE | mean, median and std are all logged; the headline uses the mean; the strategy doc does not say which `f_up` uses |
| Averaging windows | T ∈ [1:10], [11:30] and [1:20], [21:60] | `T=0:1, 1:2, 2:8, 8:22, 22:64` (log-spaced, zero-indexed; step 65 only in `T=all`) plus `T=all` |
| VRMSE ε | unspecified | 1e-7 in effect (§2.3) |
| Headline logged scalar | VRMSE | MAE in physical units |

**Decisions this raises for the study** (not made here): (1) fit `f_up` on the per-trajectory
**mean** or the **median** (as the paper reports); (2) whether to start rollouts at the paper's T = 17
(`start_rollout_valid_output_at_t`) and/or use `trainer.max_rollout_steps=55` for a true 60-step
horizon, for any comparison with the paper's tables (our PT9 datasets are long enough: the shortest
has 101 frames, and 17 + 65 = 82); (3) which normalisation statistics Tier 7 finetuning uses
(per-sample as in pretraining, or global as the paper suggests for small finetuning sets).

---

## 4. Validation passes

Two separate passes, each over every pretraining dataset's `valid` split (one dataloader per
dataset).

| | one-step (`valid`) | rollout (`rollout_valid`) |
|---|---|---|
| Cadence | epochs divisible by `val_frequency=2`, plus the last epoch | epochs divisible by `rollout_val_frequency=5`, plus the last epoch |
| Windows | every 6+1-frame window of every valid trajectory | **one window per trajectory**, starting at frame 0 (`full_trajectory_mode=True`) |
| Frame spacing | 1 (`min_dt_stride` used as both min and max) | 1 |
| Prediction scored | 1 step, from 6 frames of true context | **65** consecutive steps (§4.1) |
| Batch size | 2 (`data.module_parameters.batch_size`) | 1 (hardcoded in `rollout_val_dataloaders`) |
| Headline key | `valid` | `rollout_valid` |

In validation the causal model's six outputs are reduced to the **last** one (`y_pred[-1:]`), so
the one-step metric is the prediction of the 7th frame ($x_6$) from the full 6-frame context.
Predictions are then **denormalised**: the predicted change is multiplied by $d$ and added to the
last input frame. All validation metrics are therefore computed in **physical units** against the
raw ground truth.

### 4.1 How a rollout is computed

1. **The rollout is 65 steps, not 60.** Two caps apply:
   - the dataset returns frames 0–5 as input and up to `min(shortest trajectory − 6, 100)` frames
     as the target (full-trajectory mode; 100 is the `MixedWellDataModule` default for
     `max_rollout_steps`, which the PT9 data config does not set);
   - `rollout_model` then cuts the target to `trainer.max_rollout_steps + (T_in − 1)` when the model
     is causal in time: `60 + (6 − 1) = 65`. The code comment says this is so causal models
     "during training don't include initial context in rollout length", but the line runs in
     validation too, where only the last output is kept, so the extra 5 become 5 extra
     autoregressive steps.

   Every PT9 dataset has at least 95 target frames, so every rollout is exactly **65 steps**,
   frames 6–70. The logged keys confirm it (§2.3). This is upstream behaviour; it is not set
   anywhere in the PT9 scripts.

   **Every rollout starts at the first stored frame of its trajectory**, in every kind of run
   (rollout validation during training, the end-of-run test pass, validation-only runs). In
   full-trajectory mode each trajectory has exactly one window (`n_windows_per_trajectory = 1`), and
   the dataset computes its start as `time_idx = local_idx % windows_per_trajectory`, i.e. always 0.
   The only override, `start_rollout_valid_output_at_t`, stays at −1: no config or PT9 script sets
   it. So the rollout metric only ever measures the **beginning** of each trajectory (frames 0–70):

   | dataset | trajectory length | share covered by frames 0–70 |
   |---|---:|---:|
   | EMQ-O / EMQ-P, AD, AI | 101–102 frames | ~70 % |
   | RB, SF, AM | 200–202 | ~35 % |
   | GS, PS | 1001–1008 | ~7 % |

   For long datasets the rollout scores only the opening transient from the initial condition and
   never the later behaviour. The one-step pass, by contrast, uses windows from throughout the
   trajectories. The paper starts its rollouts at T = 17 partly for this reason ("prediction
   difficulty varies over time", §3.2). Setting `start_rollout_valid_output_at_t` moves the start
   (still one start per trajectory, the same frame for all).
2. For each of the 65 steps, in order:
   - normalisation scales are recomputed from the **current** 6-frame window;
   - one model call (no gradient); the last position's prediction is kept;
   - the prediction is denormalised and added to the last frame;
   - the window slides: the oldest frame is dropped and the prediction appended.
3. After `k` steps the window holds `6 − k` true frames and `k` predicted frames. From step 6
   onwards the model sees only its own output.
4. The 65 predicted frames are compared with the 65 true frames, step by step (§2.3).

So a rollout is fully autoregressive, costs 65 sequential forward passes per trajectory, and
involves 71 frames: 6 true inputs and 65 predicted/true pairs.

---

## 5. Metrics and how they are aggregated (summary of §2.3)

**Metrics computed in validation:** `validation_suite` = NRMSE, VRMSE, binned spectral MSE,
PearsonR, plus the training `loss_fn` (MAE) appended to it. The PT9 substrate sets
`++trainer.skip_spectral_metrics=True`, so the spectral metric is skipped.

| metric | definition (per sample, step, field, over space) |
|---|---|
| MAE | `mean|ŷ − y|`, physical units |
| VRMSE | `sqrt( mean((ŷ − y)²) / var(y) )`, i.e. RMSE divided by the true field's spatial standard deviation; scale-free |
| NRMSE | `sqrt( mean((ŷ − y)²) / mean(y²) )`; scale-free |

**Aggregation, in order:**

1. Each metric gives a `B × T × C` tensor (samples × steps × fields). Padding channels and
   constant fields are removed first.
2. Per field and averaged over fields (`full`), then over time:
   - `T=all`: the **mean of the per-step values** over all steps (65 for rollouts, 1 for one-step).
     A rollout VRMSE is thus the average of 65 per-step VRMSEs, not one VRMSE over the trajectory.
   - Log-spaced intervals (`num_time_intervals=5`): for 65 steps, `T=0:1`, `1:2`, `2:8`, `8:22`,
     `22:64` (the last boundary truncates to 64). Rollout pass only.
3. Across the scored samples: `mean`, `median` and `std` (`batch_aggregation_fns`).

**W&B key pattern:** `{pass}_{dataset}/{field|full}_{metric}_T={interval}_{agg}`, e.g.

- `rollout_valid_gray_scott_reaction_diffusion/full_VRMSE_T=all_mean`
- `rollout_valid_planetswe/height_VRMSE_T=22:64_median`
- `valid_rayleigh_benard/full_NRMSE_T=all_mean`

**The headline scalars.** `valid` and `rollout_valid` are

```
mean over datasets of  mean over samples of  {dataset}/full_MAE_T=all
```

i.e. the **training metric (MAE) in physical units**, equal-weighted across datasets. Because MAE
has units, datasets with large field magnitudes dominate this number, and it cannot be compared
across datasets. **No cross-dataset VRMSE (or NRMSE) aggregate is logged.**

The stdout line `Field-time-averaged VRMSE …` printed per validation batch is that batch's
`full_VRMSE_T=all`, for monitoring only.

---

## 6. How many samples are scored

**Capped validation.** Outside a full pass, each GPU stops after `short_validation_length=20`
batches per dataset. Under `distribution=fsdp` the device mesh has a single `fsdp` dimension, so
all GPUs form one sync group: every dataset is validated by every GPU, each scoring a different
slice chosen by a `DistributedSampler(seed=0, shuffle=True)`.

| run | GPUs | one-step windows per dataset | rollout trajectories per dataset |
|---|---:|---:|---|
| N1 | 4 | 4 × 20 × 2 = 160 | 4 × 20 × 1 = 80 (PS: all 12) |
| N4 | 16 | 640 | up to 320 (all of GS 120, AD/AM 200, RB 175, SF 112; PS 12, padded to 16 with duplicates) |

Rollout coverage at N1, per dataset: GS 80/120, EMQ-O/EMQ-P 80/500, AM 80/200, AI 80/400,
RB 80/175, SF 80/112, AD 80/200, PS 12/12.

**Deterministic, but tied to the GPU count.** The sampler's seed is fixed and `set_epoch` is never
called on validation loaders, so a given run scores the **same** windows and trajectories at
every epoch. But the slice depends on the number of replicas, so **an N1 run and an N4 run score
different subsets** of the same split, and their capped numbers are not strictly comparable.
(Without a process group, a `RandomSampler` with `manual_seed(0)` is used instead.) When the
sampler pads a small dataset to fill every GPU (PS at N4), some trajectories are counted twice.

**Full validation.** `full=True` scores every window and trajectory of the split. It runs at the
last epoch and on the test pass, **unless `debug_mode=True`**, which forces every pass to be
capped. The PT9 substrate sets `++trainer.debug_mode=True`.

---

## 7. What each number is used for

| number | used for |
|---|---|
| training MAE (`train_loss`) | the gradient; nothing else |
| `valid` (one-step MAE) | saved as `val_loss` in checkpoint metadata, and decides which checkpoint the `best` link points to (lowest value); also carried forward on epochs without validation |
| `rollout_valid` (rollout MAE) | logging only |
| per-dataset VRMSE / NRMSE keys | logging only; the study's `f_up` is built from them offline (§8) |
| LR schedule | not driven by validation: `inv_sqrt_w_sqrt_ramps` steps once per epoch on a fixed schedule |

**End-of-run test pass.** After the last epoch, `Trainer.train()` always runs one-step and
rollout validation on the **`test` split** and logs them as `test` / `rollout_test` (plus the
per-dataset keys) at the last epoch's step, and writes the §2.4 files. With `debug_mode=True` this
pass is capped, not skipped.

---

## 8. Consequences for the scaling study

1. **`f_up` is not logged and must be computed offline.** It is the equal-weight mean, over the
   nine pretraining datasets, of `rollout_valid_{dataset}/full_VRMSE_T=all_mean` (or `_median`, §3.4).
   `SCALING_STRATEGY_2D.md` §3.4 describes it as "the `rollout_valid` metric the trainer already
   logs"; that key is rollout MAE in physical units, so that sentence is wrong.
2. **The rollout horizon is 65 steps, not 60** (§4.1). The strategy doc's "60-step rollout VRMSE"
   is in fact a 65-step mean. `trainer.max_rollout_steps=55` would give a true 60.
3. **`f_up` only measures the start of each trajectory** (frames 6–70 predicted from frames 0–5,
   §4.1): about 70 % of an Euler or acoustics trajectory, 35 % of RB/SF/AM, and 7 % of GS/PS. For
   GS and PS it is a score of the opening transient only. State this when reporting `f_up`, or set
   `start_rollout_valid_output_at_t` if later behaviour should count.
4. **`f_up` is logged only every 5 epochs** (`rollout_val_frequency=5`), not every 2. An
   infinite-phase fitting window of epochs 15–50 gives 8 rollout points per run.
5. **The capped `f_up` depends on GPU count** (§6). Compare points only within one node count,
   or re-score chosen checkpoints with full validation (`Trainer.validate()`, `debug_mode=False`).
   The strategy doc's coverage table assumes 40 windows per dataset; the actual figures are in §6.
6. **Per-trajectory values are not saved** (§2.4), so paper-style medians over other time ranges,
   bootstrap CIs over trajectories (needed for `f_down`, strategy doc §3.5), or any re-aggregation
   need a re-evaluation with per-trajectory output.
7. **The `test` split is touched at the end of every completed run**, capped, despite
   `debug_mode=True`. `pt9_substrate.py`'s docstring and the strategy doc say `debug_mode` keeps
   the test split for the final §8 numbers; it only stops the pass from being *full*. Nothing is
   trained on it, but `test` / `rollout_test` keys and §2.4 files exist for every finished run.
   Ignore them until the final comparison, or disable the pass if the split must stay unread.
8. **Training and validation measure different things.** Training: MAE on per-sample-normalised
   changes, teacher-forced, frame spacing 1–5, positions 2–6. Validation: MAE/VRMSE in physical
   units, autoregressive (rollout), frame spacing 1, last position only. A falling training loss
   need not mean a falling rollout VRMSE; this mismatch is worth stating in the thesis.
9. **The `best` checkpoint is the best one-step MAE**, which is dominated by large-magnitude
   datasets and is neither the rollout metric nor VRMSE. Do not treat `best` as the
   `f_up`-optimal checkpoint; select from the logged `f_up` history instead.
10. **`train_loss` includes the ×100 multiplier.** Divide by 100 to get the MAE.
