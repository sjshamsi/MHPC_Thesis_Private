# WALRUS: architecture dimensions and parameter counts

Source: `demo_notebooks/walrus_example_2_ArchitectureAndDimensions.ipynb`. Numbers below were
recomputed from the notebook's `walrus_param_count_analytical` (cell 13), which the notebook
checks exactly against instantiated models. **[verified]** marks figures also confirmed on a built
model (`IsotropicModel` from a Hydra config, 2026-10-04/06).

Related: `mydocs/walrus_tokens_and_data_pipeline.md` (tokens, data, shapes through the model).

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Which knobs change the parameter count? | `hidden_dim` (H), `processor_blocks` (P), `mlp_dim` (M), `intermediate_dim` (I), `n_states` (n_s), and the number of encoder/decoder copies (n_dk). |
| Which knobs do **not**? | `groups`, `projection_dim`, `causal_in_time`, `drop_path`, `jitter_patches`, `gradient_checkpointing_freq`, `input_field_drop`. |
| Where do the parameters live? | In the processor at large H and P (86 % at paper scale); in the encoder/decoder at small H and P (58 % at the 200M anchor with two stems). |
| `override_dimensionality=0` vs `3`? | `0` builds a 2D **and** a 3D encoder/decoder pair (n_dk = 2); `3` builds one pair (n_dk = 1). This alone moves the total by 59 M at the anchor. |
| The paper model? | `1408 / 40 / 5632`, 2 stems: **1286.2 M** **[verified]**. |

---

## 1. The three parts of the model

```
x (T, B, fields, H, W, D) --patch jitterer--> + 3 BC channels
   --encoder--> tokens (T, B, hidden_dim, h_tok, w_tok, d_tok)
   --processor: P x SpaceTimeSplitBlock--> same shape
   --decoder--> (T, B, fields, H, W, D)
```

- **Encoder** (`SpaceBagAdaptiveDVstrideEncoder`): two strided 3D convs. 2D data is padded with a
  singleton axis, so the kernels are 8³ then 4³ even for 2D. `proj1` is stored over the whole
  `n_states` field map, but each batch uses only its own dataset's field columns plus the 3 BC
  channels (`proj1.weight[:, field_indices]`).
- **Processor**: P blocks. Each block is time attention (`AxialTimeAttention`), then space
  attention (`FullAttention`, which also contains a SwiGLU feed-forward in parallel). Channel
  mixing is `Identity` (0 params).
- **Decoder** (`AdaptiveDVstrideDecoder`): the mirror of the encoder. `proj2` is stored over
  `n_states` output channels and sliced by `state_labels` at run time.

⚠️ The default `space_mixing` is `axial_spatial_attention`, whose module
(`walrus/models/spatial_blocks/axial_attention.py`) is missing from this checkout. Every run
overrides it with `model/processor/space_mixing=full_spatial_attention`, and the analytical
function raises `NotImplementedError` for anything else.

---

## 2. Closed-form counts

Symbols: H = `hidden_dim`, P = `processor_blocks`, M = `mlp_dim` (None → 4H), I =
`intermediate_dim`, n_s = `n_states`, n_h = heads, n_dk = encoder/decoder copies, n_b = 32
relative-position buckets.

**Encoder, per copy:**
`E = 512·I·n_s + I + 64·H·I + H`
(`proj1` I×n_s×8³, `norm1` I, `proj2` H×I×4³, `norm2` H; no conv biases)

**Decoder, per copy:**
`D = 512·I·n_s + I + 64·H·I + n_s`
(same, but the only bias is `proj2.bias`, n_s)

**Processor, per block, = space + time:**

- Space (`FullAttention`): `H + H(M+3H) + (M+3H) + (M/2)·H + H + H² + 4·(H/n_h)`
  - `fused_ff_qkv` is one `Linear(H, M+3H)` that emits `[ff | q | k | v]`.
  - `ff_out` is `Linear(M/2, H)`; SwiGLU halves M first.
  - `attn_out` is `Linear(H, H)` without bias. `q_norm` and `k_norm` are small LayerNorms.
  - RoPE frequencies are buffers (0 params).
- Time (`AxialTimeAttention`): `5H + 4H² + 4·(H/n_h) + n_b·n_h`
  - `input_head` is a 1×1 conv H→3H, `output_head` H→H, plus norms and the relative-position
    bias table `Embedding(32, n_h)`.

**Total** = n_dk·(E + D) + P·(space + time) + 1 (`encoder_dummy`, a gradient-checkpointing scalar).

Large-H approximation: per block ≈ `H·(M + 4H)` for space plus `4H²` for time, i.e.
`≈ H·M + 8H²`. At M = 4H that is `12H²`, the standard transformer figure.

### Worked example: the paper model, H=1408, P=40, M=5632

| piece | params |
|---|---:|
| space attention + FF, per block | 19.84 M |
| time attention, per block | 7.94 M |
| **per block** | **27.78 M** |
| processor (×40) | 1111.0 M |
| encoder (per copy) | 43.8 M |
| decoder (per copy) | 43.8 M |
| **total, 2 stems** | **1286.2 M** |

Space attention is 71 % of each block at this size.

---

## 3. Reference models

All use I = 352, n_s = 67, 16 heads (paper-scale settings).

| model | H / P / M | enc | proc | dec | total, **2 stems** (`override=0`) | total, **1 stem** (`override=3`) |
|---|---|---:|---:|---:|---:|---:|
| WALRUS (paper, fullDims) | 1408 / 40 / 5632 | 87.6 | 1111.0 | 87.6 | **1286.2 M** | 1198.6 M |
| HalfWALRUS | 1088 / 30 / 4352 | 73.2 | 497.7 | 73.2 | **644.0 M** | 570.8 M |
| 200M anchor | 768 / 12 / 2048 | 58.8 | 85.1 | 58.8 | **202.6 M** | **143.8 M** |
| roofline small | 464 / 10 / 1120 | 45.1 | 25.1 | 45.1 | 115.2 M | 70.1 M |
| yaml default shape | 768 / 12 / 3072 | 58.8 | 99.2 | 58.8 | 216.7 M | 158.0 M |

Notes:

- **The "200M anchor" is a 2-stem count.** The PT9 substrate uses `override_dimensionality=3`
  (one stem), so its anchor model has **143.8 M** parameters **[verified]**: 85.1 M processor +
  2 × 29.4 M stems. The 2D+3D plan (`override=0`) gives the 202.6 M figure.
- **Processor share of the total**: 86 % (paper), 77 % (HalfWALRUS), 42 % (anchor, 2 stems), 59 %
  (anchor, 1 stem), 22 % (roofline small, 2 stems). The stem is "ballast" at small shapes; for
  FLOPs see `SCALING_STRATEGY_2D.md` §3.1.
- **`n_states` matters for the stem.** At paper scale, `n_states = 3` (the `bc_only_override`) gives
  1240.1 M total; `n_states = 67` (`full_well_field_index`) gives 1286.2 M. Pin the field map
  (`data/field_index_map_override=full_well_field_index`) so counts are comparable across runs.
- The stem has a fixed part that does not shrink with H: `512·I·n_s` = 12.1 M per `proj1` or
  `proj2` at I = 352, n_s = 67.

---

## 4. How the count scales: local elasticity

ε = d ln N / d ln x at the model itself: the percentage change in parameters for a 1 % change in
one dimension, other two held fixed. Computed by central difference.

| model | dimension | encoder | decoder | processor | **total** |
|---|---|---:|---:|---:|---:|
| WALRUS | `hidden_dim` | 0.72 | 0.72 | 1.57 | **1.46** |
| | `processor_blocks` | 0 | 0 | 1.00 | **0.86** |
| | `mlp_dim` | 0 | 0 | 0.43 | **0.37** |
| HalfWALRUS | `hidden_dim` | 0.67 | 0.67 | 1.57 | **1.37** |
| | `processor_blocks` | 0 | 0 | 1.00 | **0.77** |
| | `mlp_dim` | 0 | 0 | 0.43 | **0.33** |
| 200M anchor (2 stems) | `hidden_dim` | 0.59 | 0.59 | 1.67 | **1.04** |
| | `processor_blocks` | 0 | 0 | 1.00 | **0.42** |
| | `mlp_dim` | 0 | 0 | 0.33 | **0.14** |

What the table says:

- **`processor_blocks` is exactly linear in the processor** (ε = 1), but only diluted in the total
  because the stems don't grow with P.
- **`hidden_dim` is between linear and quadratic.** The processor is quadratic in H (attention
  and FF projections), but not fully: at M fixed the FF part is only linear in H. The stems are
  sub-linear (the 4³·H·I term is linear in H, the fixed `512·I·n_s` terms are constant).
- **`mlp_dim` is the weakest lever** at paper scale (ε ≈ 0.4 total), because attention (`4H²`) and
  the time block (`4H²`) are untouched by it.
- **Small models are stem-dominated**, so every elasticity of the total is lower than the
  processor's. A sweep at the 200M anchor therefore changes the total less than the same relative
  step at paper scale.

Practical validity constraints when picking values: H must be divisible by the head counts (16)
and by `groups`; `groups` must also divide I (e.g. `groups=12` fails for I = 352); M must be even
(SwiGLU halves it). The notebook's sweeps use steps of 16 for H, 1 for P, 2 for M.

---

## 5. Default config vs the upstream pretraining script

Condensed from the notebook's comparison tables (script:
`walrus/run_scripts/pretrain_example_distributed_walrus.sh`).

| key | default | upstream | effect on params |
|---|---|---|---|
| `hidden_dim` | 768 | 1408 | yes |
| `processor_blocks` | 12 (yaml; 8 in class) | 40 | yes |
| `space_mixing.mlp_dim` | None → 4H | None → 4H | yes |
| `intermediate_dim` | 192 | 352 | yes (stems) |
| heads (space and time) | 12 | 16 | tiny (norm sizes, 32·n_h bias table) |
| `override_dimensionality` | 3 (yaml) | 0 | yes (n_dk 1 → 2) |
| `groups` | 12 | 16 | no |
| `projection_dim` | 96 | 48 | no (jitterer's `stage_dim`, "Not used") |
| `causal_in_time` | False | True | no (decoder runs on all T steps, same weights) |
| `use_periodic_fixed_jitter` | False | True | no |
| `input_field_drop` | 0.1 | 0.0 | no |
| `gradient_checkpointing_freq` | 0 | 2 | no |
| `drop_path` | 0.05 (yaml) | 0.05 | no |

Data and training differences (`batch_size` 8 → 2, `n_steps_input` 10 → 6, `max_dt_stride` 1 → 5,
`clip_gradient` 5 → 10, validation cadence, scheduler `inv_sqrt_w_linear_ramps` →
`inv_sqrt_w_sqrt_ramps`) do not affect parameter counts. See the notebook's tables for the full
list.

---

## 6. Why some knobs do nothing

- **`groups`**: `RMSGroupNorm` pools statistics over `groups` channel buckets, but its learnable
  scale is one number per channel regardless. It changes the normalization, not the count.
- **`projection_dim`**: passed to the patch jitterer as `stage_dim`. The jitterer classes in use
  (`PatchJittererBoundaryPad` and its fixed subclass) store it and never use it. Only a different
  class (`PatchJitterer`) turns it into learnable padding.
- **`causal_in_time`**: only flips the relative-position bias between one- and two-directional;
  the bias table has the same size either way.
- **FullAttention's `bias_type`**: the constructor hard-codes the RoPE branch
  (`if False and ...` / `elif True or ...`), so the relative-bias branch there is dead code. Space
  position information always comes from RoPE buffers (0 params). Time attention does use the
  learned relative bias (`bias_type="rel"`, 32 × n_h parameters per block).

---

## 7. Finetuning-only additions

Added by `IsotropicModel.add_ft_options` from `finetuning_mods` (the `defaults` config adds
nothing):

- **APE** (`ape_shape`): one parameter of shape `(1, 1, H, *ape_shape)`, i.e. `H · prod(ape_shape)`.
- **`learnable_rope`**: each block's space-RoPE frequency buffer becomes a parameter of length
  `(H / n_h / 4) / 2`, times 3 if `rope_per_axis`. Crashes with time `bias_type="rotary"`.

---

## 8. Using the analytical function

```python
walrus_param_count_analytical(hidden_dim=768, processor_blocks=12, mlp_dim=2048,
                              n_dim_keys=1)   # PT9 substrate: override_dimensionality=3
# -> {'encoder': ..., 'processor': ..., 'decoder': ..., 'other': 1, 'total': ...}
```

- Default `n_dim_keys=None` derives n_dk from `override_dimensionality` (>0 → 1) and `include_d`
  (else `len(include_d)` = 2), with `override_dimensionality=0` as its own default. **For the PT9
  substrate pass `override_dimensionality=3` or `n_dim_keys=1`**, otherwise the result is 59 M too
  high at the anchor.
- Defaults assume `n_states=67`, `intermediate_dim=352`, 16 heads.
- Only the `spacebag_and_vstride_encoder` / `vstride_decoder` pair is covered.
- Building a real model (heavy, a 1.3 B model takes a minute on CPU) is
  `build_model(make_overrides(...))` in cells 6 and 15 of the notebook.

---

## 9. Caveats in the notebook itself

- **Cell 15's consistency check is incomplete.** Its condition is
  `deltaEnc != 0 or deltaDec != 0 or deltaDec != 0` (`deltaProc` is never tested) and its error
  message prints the literal strings `deltaDec`/`deltaProc` instead of values. It is commented out
  anyway; fix it before relying on it.
- **Cell 16's "n_dk = len(include_d)"** is only true for `override_dimensionality = 0`. With a
  positive override the count is 1.
- **Cell 18 ("Prof. Payel's dims", 768/12/2048)** reports 202.6 M. That is the 2-stem figure;
  the one-stem figure the PT9 runs use is 143.8 M (§3).
- **Cell 3 heads**: the table lists 12 vs 16 heads as default vs upstream; with
  `full_spatial_attention` the yaml's `num_heads: 12` is the default and 16 is set on the command
  line.
