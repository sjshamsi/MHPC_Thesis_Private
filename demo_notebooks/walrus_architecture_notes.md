# Walrus `IsotropicModel` architecture & parameter notes

Companion notes to `walrus_example_2_ArchitectureAndDimensions.ipynb`. We talk about the encoder–processor–decoder structure, which hyperparameters size which part, and
step‑by‑step forward + parameter walkthroughs of the encoder, the
spatial‑attention block, and the time‑attention block.

The examples below use the paper‑scale configuration used in `walrus/run_scripts/pretrain_example_distributed_walrus.sh`

| knob | symbol | value |
|---|---|---|
| `hidden_dim` | `H` | 1408 |
| `processor_blocks` | | 40 |
| `intermediate_dim` | `I` | 352 |
| `num_heads` | `nh` | 16 |
| `groups` | | 16 |
| `mlp_dim` | `M` | unset → `4 * hidden_dim` = 5632 |
| `n_states` | | 67 (size of the full Well field‑index map, `full_well_field_index.yaml`) |
| `include_d` | | `[2, 3]` (because the runs set `override_dimensionality=0`) |
| `projection_dim` | | 48 (does **not** affect param count — see §5) |

Source files:

| component | file |
|---|---|
| top‑level assembly | `walrus/models/isotropic_model.py` |
| encoder | `walrus/models/encoders/vstride_encoder.py` → `SpaceBagAdaptiveDVstrideEncoder` |
| decoder | `walrus/models/decoders/vstride_decoder.py` → `AdaptiveDVstrideDecoder` |
| space mixing | `walrus/models/spatial_blocks/full_attention.py` → `FullAttention` |
| time mixing | `walrus/models/temporal_blocks/axial_time_attention.py` → `AxialTimeAttention` |
| norm | `walrus/models/shared_utils/normalization.py` → `RMSGroupNorm` |
| block wrapper | `walrus/models/spatiotemporal_blocks/space_time_split.py` → `SpaceTimeSplitBlock` |

---

## 1. The sandwich: encoder → processor → decoder

`IsotropicModel` wires three Hydra‑instantiated pieces:

- **Encoder** (`self.embed`, an `nn.ModuleDict`): a patch embedding ("tokenizer" /
  "stem"). Takes a raw physics field — a fine grid of numbers — chops it into
  patches, and turns each patch into one `hidden_dim` vector. Output is a small
  grid of tokens. Two strided‑conv stages: `n_states → intermediate_dim → hidden_dim`.
- **Processor** (`self.blocks`, an `nn.ModuleList`): `processor_blocks` repeated
  `SpaceTimeSplitBlock`s. Each does factorized **space** attention + **time**
  attention (+ an identity channel‑mixing slot), all at width `hidden_dim`.
- **Decoder** (`self.debed`, an `nn.ModuleDict`): the mirror image of the encoder —
  transposed convs, `hidden_dim → intermediate_dim → n_states`, upsampling each
  token back to its patch of full‑resolution field values.

Roughly (paper scale):

| component | params | share |
|---|---:|---:|
| encoder (2 copies) | ≈ 87.6 M | 7 % |
| processor (40 blocks) | ≈ 1.11 B | 86 % |
| decoder (2 copies) | ≈ 87.6 M | 7 % |
| **total** | **≈ 1.29 B** | |

The processor dominates, which is why the scaling study sweeps `hidden_dim`,
`processor_blocks`, and the space block's `mlp_dim`.

---

## 2. Which hyperparameter sizes which part

| hyperparameter | affects | how |
|---|---|---|
| `n_states` | encoder input / decoder output channels | fixed by the dataset mixture (the field‑index map), **not** a tuning knob |
| `intermediate_dim` | encoder & decoder only | bottleneck width between their two conv stages (see §4) |
| `hidden_dim` | everything | token width; the width the processor runs at; the encoder's output / decoder's input |
| `processor_blocks` | processor | number of stacked `SpaceTimeSplitBlock`s |
| `mlp_dim` | processor (space block) | feed‑forward width inside `FullAttention` (defaults to `4 * hidden_dim`) |
| `num_heads` | processor | attention head count in the space & time blocks |
| `groups` | none (param‑count‑wise) | only how `RMSGroupNorm` pools channels for its statistic (see §5) |
| `projection_dim` | none | only fed to the patch jitterer as `stage_dim`, which stores it and never makes a parameter from it (see §5) |
| `include_d` / `override_dimensionality` | encoder & decoder | how many dedicated encoder/decoder copies get built (see §3) |

---

## 3. `include_d`, the per‑dimensionality encoders, and `override_dimensionality`

Walrus trains on a mix of **2‑D and 3‑D** datasets. A `Conv2d` and a `Conv3d` have
differently‑shaped weights, so the model keeps a **separate encoder and decoder per
spatial dimensionality**, stored in `nn.ModuleDict`s keyed by the string `"2"`,
`"3"`:

```python
self.embed = nn.ModuleDict({
    str(i): encoder(spatial_dims=3, input_dim=n_states,
                    inner_dim=intermediate_dim, output_dim=hidden_dim, ...)
    for i in range(1, max_d + 1) if i in include_d
})
```

- `include_d = [2, 3]` → build `self.embed["2"]` **and** `self.embed["3"]`. At
  forward time the model computes `dim_key` from the input and routes to
  `self.embed[dim_key]`. `self.debed` is the matching dict of decoders.
- **Both copies are built with `spatial_dims=3`** (hard‑coded in
  `IsotropicModel`), so they are *structurally identical* `Conv3d` stacks. That is
  why the parameter formula says "one identically‑sized copy per `include_d` entry"
  and just multiplies by `n_dim_keys = len(include_d)`.
- `max_d = 3` is the upper bound of the `range(1, max_d + 1)` loop (and a value
  handed to the patch jitterer) — the highest dimensionality the model can handle.
- **`override_dimensionality`** overrides `include_d`:

  ```python
  if override_dimensionality is not None and override_dimensionality > 0:
      include_d = [override_dimensionality]
  ```

  The checked‑in default `isotropic_model.yaml` has `override_dimensionality: 3`,
  which collapses `include_d` to `[3]` (one encoder/decoder, everything forced
  through the 3‑D path — a 2‑D field gets a singleton depth axis). **All the thesis
  run scripts set `model.override_dimensionality=0`**, which keeps `[2, 3]` so 2‑D
  datasets use the real 2‑D routing. `n_dim_keys = 2` for those runs.

Because 2‑D data is padded to 3‑D (`max_d = 3`) and both encoders use `Conv3d`, the
**kernel volumes in the parameter formula are always** `8·8·8 = 512` and
`4·4·4 = 64` (from `base_kernel_size3d = [[8,4],[8,4],[8,4]]` in
`spacebag_and_vstride_encoder.yaml`), regardless of whether the data is 2‑D or 3‑D.

---

## 4. `intermediate_dim` is encoder/decoder‑only

In `IsotropicModel` the value is passed in exactly two places:

```python
# encoder construction
inner_dim = intermediate_dim   # isotropic_model.py, self.embed
# decoder construction
inner_dim = intermediate_dim   # isotropic_model.py, self.debed
```

The `processor(...)` call gets `hidden_dim` only — **no `intermediate_dim`**. The
patch jitterer gets `projection_dim`, not `intermediate_dim`. So `intermediate_dim`
is purely the encoder/decoder bottleneck width (`n_states → intermediate_dim →
hidden_dim` on the way in, mirror on the way out). It has zero effect on processor
size.

---
   
## 5. `groups` and `projection_dim` are not capacity knobs

**`groups`** only feeds the normalization layers. `RMSGroupNorm(num_groups,
num_channels, affine=True)` stores `weight` of shape `(num_channels,)` and **no
bias** (RMS‑norm rescales but never re‑centers). `num_groups` changes only how the
channels are pooled to compute the RMS statistic — the number of parameters is
`num_channels` regardless. That is why every norm term in the parameter formulas is
`+ H` (or `+ I`), not `+ 2H`, and why `groups` is not a formula argument.

**`projection_dim`** is passed to the patch jitterer as `stage_dim`. The jitterer
class the model actually uses (`PatchJittererBoundaryPad` / its `Fixed…` subclass)
does `self.stage_dim = stage_dim` — the code literally comments `# Not used` — and
never builds a parameter from it. The jitterer is pure geometry (boundary padding +
random patch‑grid shift), zero learnable parameters. (A *different* jitterer class,
`PatchJitterer`, does turn `stage_dim` into learnable padding tokens; it is not the
one instantiated here.)

---

## 6. Why the runs override `space_mixing=full_spatial_attention`

`space_time_split.yaml` defaults to `space_mixing: axial_spatial_attention`, whose
`_target_` is `walrus.models.spatial_blocks.axial_attention.AxialAttentionBlock` —
**a module that does not exist in this codebase**. `walrus/models/spatial_blocks/`
contains only `full_attention.py`. So:

1. The config default would fail at model construction; `full_spatial_attention`
   (`FullAttention`) is the only implemented Walrus spatial mixer.
2. The released checkpoint was trained with it — its `extended_config.yaml` has
   `space_mixing._target_: ...full_attention.FullAttention` — so a validation run
   *must* pass `full_spatial_attention` to load `walrus.pt` at all.
3. Upstream's own `pretrain_example_distributed_walrus.sh` also passes it. Every
   script in the repo does. The YAML default is a stale leftover.

Only the **time** axis keeps a real axial block (`axial_time_attention.py`).

Conceptually: an axial spatial block (had it existed) would attend along each
spatial axis separately — cheaper, but can't model arbitrary 2‑token interactions
in one step. `FullAttention` flattens all spatial tokens and does one dense
self‑attention — every patch attends to every other patch. More expressive,
`O((Gh·Gw·Gd)^2)` cost. Walrus's encoder downsamples aggressively (small token
grid), so full spatial attention is affordable.

---

## 7. The token grid: `Gh × Gw × Gd`

`Gh × Gw × Gd` is the **token grid** — the coarse 3‑D grid of vectors that comes
out of the encoder and goes into the processor.

1. **The raw field lives on a fine simulation grid.** Per timestep, a 2‑D Well
   dataset like `active_matter` is `(n_states, 256, 256)` — a few physical fields
   on a 256×256 lattice of *cells* ("pixels"). `MHD_64` is
   `(n_states, 64, 64, 64)`.
2. **The encoder patchifies.** Two strided convolutions whose strides multiply to
   the patch size (~32 per axis). A stride‑`P` conv reads a `P×P(×P)` block, emits
   one `hidden_dim` vector, slides by `P`.
3. **The output grid is `Gh × Gw × Gd`:**
   ```
   Gh ≈ Yres / (total stride in y)
   Gw ≈ Xres / (total stride in x)
   Gd ≈ Zres / (total stride in z)   # = 1 for 2-D data
   ```
   The "v" in vstride = the encoder nudges the strides per input so `Gh·Gw·Gd`
   stays roughly constant across datasets of different native resolution.
   Example: 256×256 → 8×8×1; `MHD_64` 64³ → 8×8×8.
4. **A "spatial token" is one cell of that grid** — a single `hidden_dim` vector
   tagged with a location `(gh, gw, gd)`. The continuum‑physics analogue of a
   word‑piece in an LLM or a patch token in a ViT. There are `N = Gh·Gw·Gd` of
   them per (timestep, batch element).
5. **The processor** flattens `Gh×Gw×Gd` into a length‑`N` sequence; `FullAttention`
   lets every spatial token attend to every other (`N×N` attention). Time is
   handled separately — see §8.4.

---

## 8. Encoder walkthrough (shapes + parameters)

### 8.1 Config → layer inventory

`SpaceBagAdaptiveDVstrideEncoder`, built by `IsotropicModel` with `spatial_dims=3`,
`input_dim=n_states=67`, `inner_dim=I=352`, `output_dim=H=1408`,
`norm_layer=RMSGroupNorm`, kernels from `base_kernel_size3d = [[8,4],[8,4],[8,4]]`:

| attribute | module | shape | bias? |
|---|---|---|---|
| `proj1` | `Conv3d(67, 352, kernel=(8,8,8))` | weight `(352, 67, 8, 8, 8)` | **no** |
| `norm1` | `RMSGroupNorm(16, 352)` | weight `(352,)` | no (RMS‑norm) |
| `act` | `SiLU` / `GELU` | — | — |
| `proj2` | `Conv3d(352, 1408, kernel=(4,4,4))` | weight `(1408, 352, 4, 4, 4)` | **no** |
| `norm2` | `RMSGroupNorm(16, 1408)` | weight `(1408,)` | no |

Forward is literally `proj1 → norm1 → act → proj2 → norm2 → act`.

### 8.2 Step‑by‑step forward (example)

Example batch: 2‑D `active_matter`, native grid 256×256, `T = 6` input timesteps,
`B = 2`, and say **3** physical fields present in this dataset. The axis order in
the code is `x is (T, B, C, H, W, D)`; a size‑1 depth axis is added for 2‑D data,
and `adaptive_conv` forces stride 1 (and sums the kernel taps) on any size‑1 axis,
so this behaves like an 8×8 / 4×4 conv over the 256×256 plane.

| # | operation | input shape | parameter used (shape) | output shape |
|---|---|---|---|---|
| 0 | *(pre‑encoder: field dropout + patch‑jitter / boundary‑pad, in `_encoder_forward`)* | `(6, 2, 3, 256, 256, 1)` | — | `(6, 2, 6, 256, 256, 1)` — 3 fields **+ 3 boundary/bias channels** appended |
| 1 | flatten time `"T B ... -> (T B) ..."` | `(6, 2, 6, 256, 256, 1)` | none (pure reshape) | `(12, 6, 256, 256, 1)` |
| 2 | **proj1** (patchify): `Conv3d`, stride 8 | `(12, 6, 256, 256, 1)` | `proj1.weight (352, 67, 8,8,8)`, **sliced at runtime** to the present columns → `(352, 6, 8,8,8)`; **no bias** | `(12, 352, 32, 32, 1)` |
| 3 | **norm1** (`RMSGroupNorm`, 16 groups) + act | `(12, 352, 32, 32, 1)` | `norm1.weight (352,)`; no bias | `(12, 352, 32, 32, 1)` |
| 4 | **proj2** (dim‑raise): `Conv3d`, stride 4 | `(12, 352, 32, 32, 1)` | `proj2.weight (1408, 352, 4,4,4)`; **no bias** | `(12, 1408, 8, 8, 1)` |
| 5 | **norm2** + act | `(12, 1408, 8, 8, 1)` | `norm2.weight (1408,)`; no bias | `(12, 1408, 8, 8, 1)` |
| 6 | un‑flatten time `"(T B) ... -> T B ...", T=6` | `(12, 1408, 8, 8, 1)` | none | `(6, 2, 1408, 8, 8, 1)` |

**Output = the token grid:** `Gh × Gw × Gd = 8 × 8 × 1 = 64` spatial tokens, each a
1408‑vector, per (timestep, batch). Total spatial downsample = 8 × 4 = **32×** per
axis.

What each layer does:

- **proj1 — "patchify."** Slides an 8×8(×8) window over the raw field, stride 8; a
  full 3‑D convolution over the channel axis produces one 352‑vector per position.
  The channel axis of the *stored* weight is the **entire 67‑slot field map**, so
  the count is `I · n_states · 512`. At runtime `proj1.weight[:, field_indices]`
  keeps only the columns for the fields this batch contains (+ 3 boundary/bias
  channels the jitterer added) — that is the **"space bag,"** and it is what lets
  one encoder serve datasets with different variable sets.
- **norm1 + act.** RMS‑normalize the 352 channels (pooled into 16 groups) per
  spatial position, learned per‑channel gain (352 numbers), then SiLU/GELU.
- **proj2 — "dim‑raise."** Slides a 4×4(×4) window, stride 4; each window of
  352‑vectors → one 1408‑vector. Downsamples another 4× and lifts the width to the
  processor's `hidden_dim`.
- **norm2 + act.** Same as norm1 but on 1408 channels.

### 8.3 Parameter accounting

```
enc_per_key = I * n_states * 512   # proj1.weight  (I, n_states, 8,8,8)   patchify conv
            + I                    # norm1.weight  (I,)
            + H * I * 64           # proj2.weight  (H, I, 4,4,4)           dim-raise conv
            + H                    # norm2.weight  (H,)
```

`512 = 8·8·8` and `64 = 4·4·4` are the kernel *volumes*. Both convs are
`bias=False`, so the only additive/scale learnables are the two `RMSGroupNorm`
weight vectors — hence `+ I` and `+ H`, not `+ 2I` / `+ 2H`.

| term | tensor | shape | count |
|---|---|---:|---:|
| `I * n_states * 512` | `proj1.weight` | `(352, 67, 8, 8, 8)` | 12,075,008 |
| `I` | `norm1.weight` | `(352,)` | 352 |
| `H * I * 64` | `proj2.weight` | `(1408, 352, 4, 4, 4)` | 31,719,424 |
| `H` | `norm2.weight` | `(1408,)` | 1,408 |
| **enc_per_key** | | | **43,796,192** |
| `× n_dim_keys = 2` | `self.embed["2"]`, `self.embed["3"]` (identical) | | **87,592,384** |

**Decoder** is the mirror (`vstride_decoder.yaml`, same `base_kernel_size3d`):
`proj1` is the 4³ dim‑lowering conv `H → I` (`bias=False`), `proj2` is the 8³
"un‑patchify" conv `I → n_states` **with a bias** (`bias=True`, shape
`(n_states,)`):

```
dec_per_key = H * I * 64            # proj1.weight  (H, I, 4,4,4)
            + I                     # norm1.weight  (I,)
            + I * n_states * 512    # proj2.weight  (I, n_states, 8,8,8)
            + n_states              # proj2.bias    (n_states,)
```

= 31,719,424 + 352 + 12,075,008 + 67 = **43,794,851** per key → **87,589,702** for
`n_dim_keys = 2`.

### 8.4 Why flatten/unflatten time (steps 1 & 6) does not mix timesteps

Two things together guarantee it:

**(a) The flatten targets the batch axis, and every op in the encoder is
"batch‑independent."** `rearrange("T B ... -> (T B) ...")` moves `T` into dimension
0 alongside `B`. No layer here has a weight or a reduction that spans dim 0:

| op | acts on | touches dim 0? |
|---|---|---|
| `Conv3d` (proj1, proj2) | slides the kernel over the **last 3 axes** only; dim 0 is the `N` in `(N, C_in, D, H, W)` | no — applied separately to each item, no weight links different items, padding is spatial only |
| `RMSGroupNorm` | `normalized_shape` is a **suffix** of the shape; statistic computed per row of dim 0 | no — each `(T B)` item normalized using only its own values |
| SiLU / GELU | elementwise | no |

So `t=2, b=0` and `t=3, b=0` travel through the encoder as two unrelated batch
entries.

**(b) The reshape is a lossless bijection undone with the same `T`.**
`"T B ... -> (T B) ..."` is row‑major: `[t, b, …] ↦ index n = t·B + b`. Pure view,
nothing combined. The code captures `T = x.shape[0]` *before* flattening and hands
it back to `"(T B) ... -> T B ...", T=T`, which inverts exactly (`t = n // B`,
`b = n % B`). This works because every op preserves the size of dim 0
(`N = T·B`).

**Where time *is* deliberately mixed:** not in the encoder or decoder. In
`SpaceTimeSplitBlock`, the `(T B)` fold is applied *only* around `space_mixing`
(so spatial attention runs per‑timestep), while `time_mixing` (axial attention
along the `T` axis) runs *before* the fold on the full `(T, B, C, H, W, D)`.
Modeling temporal dynamics is exclusively the processor's `time_mixing` job.

---

## 9. Spatial attention (`FullAttention`) walkthrough

### 9.1 What the block computes

A **parallel attention + MLP transformer block** on a grid of spatial tokens
`x` of shape `(B, C=H, Gh, Gw, Gd)` — every token is a length‑`H` vector, "full"
means every token attends to every other (the `Gh·Gw·Gd` grid is flattened to one
length‑`N` sequence). Note: `SpaceTimeSplitBlock` folds the `T` timesteps into the
batch dimension before calling this, so the "`B`" here is really `T·B`.

1. **Pre‑norm:** `xn = norm1(x)`.
2. **One fused projection:** `fused_ff_qkv` maps each token `H → (M + 3H)`, split
   into `ff` (width `M`, feeds the MLP branch) and `q, k, v` (width `H` each).
   Both branches read the *same* `xn`, so their first linear layers are fused into
   one matmul.
3. **Attention branch:** reshape `q,k,v` to `nh` heads of size `H/nh`; `q_norm` /
   `k_norm` each head's q and k (QK‑norm stability trick); rotary position
   embedding (no params); `scaled_dot_product_attention`; merge heads back to `H`;
   `attn_out` (`H → H`, no bias).
4. **MLP branch:** `SwiGLU(ff)` splits the width‑`M` vector in half and gates →
   width `M/2`; `ff_out` maps `M/2 → H`.
5. **Combine:** `x = drop_path(attn_out + ff_out(...)) + x` (residual). Single
   norm, two branches summed — that is why there is one `norm1`, not two.

### 9.2 Step‑by‑step forward (example)

`E = hidden_dim = 1408`, `nh = 16`, per‑head dim `hd = E/nh = 88`, `M = 4E = 5632`
(`M/2 = 2816`). Token grid `Gh × Gw × Gd`, `N = Gh·Gw·Gd`. `B` here = `T·B`.

Every `nn.Linear(in, out)` stores `weight (out, in)` and, unless `bias=False`,
`bias (out,)`, applied to the **last** axis.

| # | operation | input shape | parameter (shape) — what it does | output shape |
|---|---|---|---|---|
| 0 | input; `input = x.clone()` for the residual | `(B, 1408, Gh, Gw, Gd)` | — | `(B, 1408, Gh, Gw, Gd)` |
| 1 | `norm1` (`RMSGroupNorm`, 16 groups) | `(B, 1408, Gh, Gw, Gd)` | `norm1.weight (1408,)`, no bias — RMS‑normalize + per‑channel gain | `(B, 1408, Gh, Gw, Gd)` |
| 2 | rearrange to channels‑last | `(B, 1408, Gh, Gw, Gd)` | none | `(B, Gh, Gw, Gd, 1408)` |
| 3 | `fused_ff_qkv` = `Linear(1408, 9856)` | `(B, Gh, Gw, Gd, 1408)` | `weight (9856, 1408)` + `bias (9856,)` — one affine map producing `[ff \| q \| k \| v]` = `M`,`E`,`E`,`E` | `(B, Gh, Gw, Gd, 9856)` → split → `ff (…,5632)`, `q/k/v (…,1408)` |
| 4 | split q/k/v into heads | `(B, Gh, Gw, Gd, 1408)` each | none | `(B, 16, Gh, Gw, Gd, 88)` each |
| 5 | `q_norm`, `k_norm` = `LayerNorm(88)` | `(B, 16, Gh, Gw, Gd, 88)` (q, k) | `weight (88,)` + `bias (88,)` each — QK‑norm over the per‑head vector | unchanged |
| 6 | rotary position embedding | `(B, 16, Gh, Gw, Gd, 88)` (q, k) | **none** (fixed frequency buffers) — rotates channel pairs by position‑dependent angles | unchanged |
| 7 | flatten grid to a sequence | `(B, 16, Gh, Gw, Gd, 88)` | none | `(B, 16, N, 88)` |
| 8 | `scaled_dot_product_attention` | q,k,v `(B, 16, N, 88)` | **none** — `softmax(q·kᵀ/√88) · v`; the `N×N` matrix is the "full" part and the `O(N²)` cost | `(B, 16, N, 88)` |
| 9 | merge heads | `(B, 16, N, 88)` | none | `(B, Gh, Gw, Gd, 1408)` |
| 10 | `attn_out` = `Linear(1408, 1408, bias=False)` | `(B, Gh, Gw, Gd, 1408)` | `weight (1408, 1408)`, **no bias** — output projection mixing the heads | `(B, Gh, Gw, Gd, 1408)` |
| 11 | `SwiGLU` on `ff` | `(B, Gh, Gw, Gd, 5632)` | **none** — chunk into two `(…,2816)`, return `silu(gate)·value` | `(B, Gh, Gw, Gd, 2816)` |
| 12 | `ff_out` = `Linear(2816, 1408)` | `(B, Gh, Gw, Gd, 2816)` | `weight (1408, 2816)` + `bias (1408,)` — MLP second layer | `(B, Gh, Gw, Gd, 1408)` |
| 13 | `att_out + ff_out(...)`, then `drop_path` | two `(B, Gh, Gw, Gd, 1408)` | none — `drop_path` is identity at eval, stochastic‑depth in training | `(B, Gh, Gw, Gd, 1408)` |
| 14 | rearrange to channels‑first, `+ input` | `(B, Gh, Gw, Gd, 1408)` | none | `(B, 1408, Gh, Gw, Gd)` — **same shape as the input**, so blocks stack |

### 9.3 Parameter accounting

```
space  = H                          # (1)  norm1: RMSGroupNorm affine weight only
space += H*(M + 3*H) + (M + 3*H)     # (3)  fused_ff_qkv: Linear(H, M+3H), weight + bias
space += (M//2)*H + H                # (12) ff_out: Linear(M//2, H), weight + bias
space += H*H                         # (10) attn_out: Linear(H, H, bias=False)
space += 4*(H//nh)                   # (5)  q_norm + k_norm: LayerNorm(H/nh), weight + bias each
```

| line | step | count rule | count @ paper scale |
|---|---|---|---:|
| `H` | 1 | `RMSGroupNorm` weight `(H,)`, no bias | 1,408 |
| `H*(M+3H) + (M+3H)` | 3 | `Linear` weight `(M+3H, H)` + bias `(M+3H,)` | 13,887,104 |
| `(M//2)*H + H` | 12 | `Linear` weight `(H, M/2)` + bias `(H,)` | 3,966,336 |
| `H*H` | 10 | `Linear` weight `(H, H)`, `bias=False` | 1,982,464 |
| `4*(H//nh)` | 5 | 2 × `LayerNorm(H/nh)`, weight + bias each | 352 |
| **per block** | | | **19,837,664** |
| **× `processor_blocks` = 40** | | | **≈ 793.5 M** |

Zero‑parameter steps: rotary embedding (fixed buffers), `scaled_dot_product_attention`,
`SwiGLU`, `drop_path`. Also dead code in `FullAttention.__init__`: the `bias_type`
branches guarded by `if False and …` / `elif True or …` mean the learnable
`RelativePositionBias` list is never built here; `weight_tied_axes` is stored but
unused (full attention is not axis‑factorized).

The other half of each processor block, `time_mixing` = `AxialTimeAttention`, is
walked through in §10. A processor block ≈ 19.84 M (space) + 7.94 M (time) ≈
27.8 M, × 40 ≈ **1.11 B**.

---

## 10. Time attention (`AxialTimeAttention`) walkthrough

### 10.1 What the block computes

The **time** half of `SpaceTimeSplitBlock`. It runs *before* the `(T B)` fold
(§8.4), so it sees the full 6‑D tensor `(T, B, E, Gh, Gw, Gd)` and mixes information
**along the time axis only** — "axial" = attention restricted to one axis, here `T`.

Structurally simpler than `FullAttention`: **no MLP branch, no SwiGLU** — just
pre‑norm → a fused q/k/v projection → attention over `T` → an output projection →
residual. Key differences from the space block:

| | space (`FullAttention`) | time (`AxialTimeAttention`) |
|---|---|---|
| projections | `nn.Linear` | 1×1 `nn.Conv3d` (channel‑axis linear, **with bias**) |
| MLP branch | yes (`fused_ff_qkv` also emits `ff`, then SwiGLU + `ff_out`) | **none** |
| position info | rotary (0 params) | learned **relative‑position bias**: `Embedding(num_buckets, nh)` |
| attention sequence | the `N = Gh·Gw·Gd` spatial tokens (per timestep) | the `T` timesteps (per spatial location) |
| causal? | no | yes when `causal_in_time=True` — the rel‑pos bias sets future offsets to `-inf`, so step `t` attends only to steps `≤ t` |

`causal_in_time=True` in all the run scripts, matching the physical arrow of time
and the autoregressive rollout.

### 10.2 Step‑by‑step forward (example)

`E = hidden_dim = 1408`, `nh = 16` (`model.processor.time_mixing.num_heads=16`),
per‑head dim `hd = E/nh = 88`, `num_buckets = 32`. Input is the encoder's output
from §8.2: `T = 6`, `B = 2`, token grid `Gh × Gw × Gd = 8 × 8 × 1`. (The code names
the spatial grid axes `H, W, D`; below they are `Gh, Gw, Gd` to avoid clashing with
`E`.)

| # | operation | input shape | parameter (shape) — what it does | output shape |
|---|---|---|---|---|
| 0 | input; `input = x.clone()` for the residual | `(6, 2, 1408, 8, 8, 1)` | — | `(6, 2, 1408, 8, 8, 1)` |
| 1 | flatten time `"t b c h w d -> (t b) c h w d"` | `(6, 2, 1408, 8, 8, 1)` | none (pure reshape) | `(12, 1408, 8, 8, 1)` |
| 2 | `norm1` (`RMSGroupNorm`, 16 groups) | `(12, 1408, 8, 8, 1)` | `norm1.weight (1408,)`, no bias | `(12, 1408, 8, 8, 1)` |
| 3 | `input_head` = `Conv3d(1408, 4224, kernel=1)` | `(12, 1408, 8, 8, 1)` | `weight (4224, 1408, 1,1,1)` + `bias (4224,)` — pointwise `1408 → 3·1408`, the fused q/k/v projection | `(12, 4224, 8, 8, 1)` |
| 4 | rearrange to "time as sequence" `"(t b) (he c) h w d -> (b h w d) he t c"` | `(12, 4224, 8, 8, 1)` | none | `(128, 16, 6, 264)` — `b·Gh·Gw·Gd = 128` independent problems, `he = 16` heads, seq len `t = 6`, `c = 3·hd = 264` |
| 5 | `tensor_split(3, dim=-1)` | `(128, 16, 6, 264)` | none | `q, k, v` each `(128, 16, 6, 88)` |
| 6 | `qnorm`, `knorm` = `LayerNorm(88)` | `(128, 16, 6, 88)` (q, k) | `weight (88,)` + `bias (88,)` each — QK‑norm | unchanged |
| 7 | `rel_pos_bias(T, T)` | — | `RelativePositionBias.relative_attention_bias (32, 16)` — one learned scalar per (relative‑time bucket, head); one‑directional (future → `-inf`) because `causal_in_time=True` | bias `(1, 16, 6, 6)` |
| 8 | `scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias)` | q,k,v `(128, 16, 6, 88)` | **none** — `softmax(q·kᵀ/√88 + bias) · v` over the length‑6 time axis; attention matrix `(128, 16, 6, 6)` | `(128, 16, 6, 88)` |
| 9 | rearrange back `"(b h w d) he t c -> (t b) (he c) h w d"` | `(128, 16, 6, 88)` | none | `(12, 1408, 8, 8, 1)` |
| 10 | `output_head` = `Conv3d(1408, 1408, kernel=1)` | `(12, 1408, 8, 8, 1)` | `weight (1408, 1408, 1,1,1)` + `bias (1408,)` — output projection | `(12, 1408, 8, 8, 1)` |
| 11 | un‑flatten time `"(t b) c h w d -> t b c h w d", t=6` | `(12, 1408, 8, 8, 1)` | none | `(6, 2, 1408, 8, 8, 1)` |
| 12 | `drop_path(x) + input` | `(6, 2, 1408, 8, 8, 1)` | none — `drop_path` identity at eval, stochastic depth in training | `(6, 2, 1408, 8, 8, 1)` — **same shape as input** |

The rearrange in step 4 is the whole trick: it makes the **length‑`T` time axis**
the attention sequence and folds `(batch × every spatial location)` into the batch
dim, so each spatial location's `T`‑step history is attended **independently** —
spatial locations never interact here (that is the space block's job). Steps 1 & 11
fold/unfold `(t b)` and are undone with `t = T` — the same lossless‑bijection
argument as §8.4; nothing crosses that axis.

### 10.3 Parameter accounting

```
time  = H                    # (2)  norm1: RMSGroupNorm affine weight only
time += 3*H*H + 3*H          # (3)  input_head:  Conv3d(H, 3H, 1), weight + bias
time += H*H + H              # (10) output_head: Conv3d(H, H, 1),  weight + bias
time += 4*(H//nh)            # (6)  qnorm + knorm: LayerNorm(H/nh), weight + bias each
time += num_buckets * nh     # (7)  rel_pos_bias: Embedding(num_buckets, nh)   [only if bias_type='rel']
```

A 1×1 `Conv3d(a, b)` has `weight (b, a, 1, 1, 1)` and `bias (b,)`, so it counts
exactly like a `Linear(a, b)`: `a·b + b`.

| line | step | count rule | count @ paper scale (H=1408, nh=16, num_buckets=32) |
|---|---|---|---:|
| `H` | 2 | `RMSGroupNorm` weight `(H,)`, no bias | 1,408 |
| `3*H*H + 3*H` | 3 | 1×1 conv weight `(3H, H, 1,1,1)` + bias `(3H,)` | 5,951,616 |
| `H*H + H` | 10 | 1×1 conv weight `(H, H, 1,1,1)` + bias `(H,)` | 1,983,872 |
| `4*(H//nh)` | 6 | 2 × `LayerNorm(H/nh)`, weight + bias each | 352 |
| `num_buckets * nh` | 7 | `Embedding(32, 16)` | 512 |
| **per block** | | | **7,937,760** |
| **× `processor_blocks` = 40** | | | **≈ 317.5 M** |

Zero‑parameter steps: the rearranges, `tensor_split`, `scaled_dot_product_attention`,
`drop_path`. `bias_type='rotary'` would drop the `num_buckets·nh` term and add a
`RotaryEmbedding` (also 0 params unless `make_rope_learnable` is called);
`bias_type='none'` drops it with nothing added.

Space (≈793.5 M) + time (≈317.5 M) = **≈1.11 B** for the 40‑block processor.

---

## 11. The analytical parameter‑count formula

### 11.1 The formula

```python
def walrus_param_count_analytical(
        hidden_dim=768, processor_blocks=12, mlp_dim=None, intermediate_dim=192,
        n_states=67, num_heads=12, n_dim_keys=2, bias_type_time="rel", num_buckets=32):
    H = hidden_dim
    M = mlp_dim if mlp_dim is not None else H * 4
    nh = num_heads
    I = intermediate_dim

    # space_mixing: FullAttention (per block)
    space  = H
    space += H * (M + 3 * H) + (M + 3 * H)
    space += (M // 2) * H + H
    space += H * H
    space += 4 * (H // nh)

    # time_mixing: AxialTimeAttention (per block)
    time  = H
    time += 3 * H * H + 3 * H
    time += H * H + H
    time += 4 * (H // nh)
    if bias_type_time == "rel":
        time += num_buckets * nh

    processor = (space + time) * processor_blocks

    enc_per_key = I * n_states * 512 + I + H * I * 64 + H
    encoder = enc_per_key * n_dim_keys

    dec_per_key = H * I * 64 + I + I * n_states * 512 + n_states
    decoder = dec_per_key * n_dim_keys

    total = encoder + processor + decoder
    return {"encoder": encoder, "processor": processor, "decoder": decoder, "total": total}
```

### 11.2 Why it is welded to these submodule choices

Parameter count is `sum(p.numel() for p in module.parameters())` — the total size
of the concrete weight/bias tensors a module's `__init__` registers. This function
is a hand‑transcription of that sum for **one** stack of `nn.Module`s.
`IsotropicModel` is a Hydra‑composed *family*: `encoder`, `decoder`, `processor`,
`space_mixing`, `time_mixing`, `channel_mixing`, `norm_layer` are all swappable
`_target_`s. Each swap changes the tensor inventory — shapes, presence of biases,
how things scale with the knobs — none of which is visible from the interface.

| swap | what changes |
|---|---|
| **`norm_layer`** | `RMSGroupNorm(affine)` = weight only → every norm term is `+ H`. `LayerNorm` / `GroupNorm(affine)` add a bias → every one becomes `+ 2H`. `affine=False` → `0`. |
| **`space_mixing`** | `FullAttention`'s exact layout: one *fused* `Linear(H, M+3H)` with bias, `ff_out` taking `M//2` (SwiGLU halves the FF), `attn_out` with no bias, `q/k` `LayerNorm(H/nh)`, rotary → 0 pos params. An axial block ties/replicates projections per axis, may split Q/K/V, may carry learned relative‑position tables, may not fold in the FF. |
| **`time_mixing`** | `AxialTimeAttention` uses kernel‑1 `Conv3d` (with bias) for projections + a `RelativePositionBias` embedding. A plain MHA or an SSM/Mamba block scales completely differently (SSM ~ state dim, not `H²`). |
| **`channel_mixing`** | currently `nn.Identity` → 0, omitted entirely. An MLP adds `~2·H·C` per block; MoE multiplies by the expert count. |
| **`encoder` / `decoder`** | this formula hardcodes two conv stages with kernel volumes `8³ = 512` and `4³ = 64`, `bias=False`, `spatial_dims=3`, one copy per `include_d`. `hmlp_encoder` uses fixed stride‑4/4 stages (`4³` both); `flexivit` / plain ViT change the stage count and kernels; some add conv bias. |

Even holding the classes fixed, a few flags outside the arg list move the count:
`weight_tied_axes` in `FullAttention`, and any active finetuning mod
(`add_ft_options` adds an `ape` parameter block, or makes RoPE learnable). The
formula assumes base pretraining with none of those on.
