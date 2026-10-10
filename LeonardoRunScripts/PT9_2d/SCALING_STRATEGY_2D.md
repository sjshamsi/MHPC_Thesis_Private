# Compute-optimal shape scaling for WALRUS with 13 2D datasets

We use SoViT's star/grid shape-optimisation recipe (Alabdulmohsin et al. 2023, *Getting ViT in Shape*) and Choshen et al.'s intermediate-checkpoint economy technique, organised as Tiers 0–7 with decision gates between them. An earlier version of this strategy was run on `Leonardo_smallest7_2d`, the only 2D datasets available to us at the time. Now that more disk space means that we ball, this plan runs
on a mixture chosen by design.

We have 13 2D Well datasets on disk now. We do not want to train on all 13, though, because we can now do a true blind downstream assessment. So we hold out 4 datasets.

We have some basic facts that we would like to keep constant:

| fact | value |
|---|---|
| Compute budget | ~30K hours |
| Pretraining mix | 9 datasets: AD, AI, AM, EMQ-O, EMQ-P, GS, PS, RB, SF (`data=Leonardo_pretrain9_2d`) |
| Downstream mix | 4 datasets: `active_matter`, `helmholtz_staircase`, `turbulent_radiative_layer_2D`, `viscoelastic_instability` |
| Upstream metric `f_up` | 60-step rollout VRMSE on the `valid` split of the pretraining datasets |
| Downstream metric `f_down` | Finetuned VRMSE on the 4 held-out datasets (§3.5, Tier 7) |
| Shape axes `D` | `hidden_dim`, `processor_blocks`, `mlp_dim` |
| HP reduction | LR×WD sensitivity at the extreme shapes only (§6), then frozen; batch size fixed |
| Validating endpoint | Head-to-head against fullDims at **matched FLOPs**, upstream *and* downstream |
| Data repetition | Measured as an explicit sub-study (RQ3), designed for this mixture's low sweep-scale repetition |

Dataset codes used throughout: AD/AI/AM = `acoustic_scattering_{discontinuous,inclusions,maze}`,
EMQ-O/EMQ-P = `euler_multi_quadrants_{openBC,periodicBC}`, GS = `gray_scott_reaction_diffusion`,
PS = `planetswe`, RB = `rayleigh_benard`, SF = `shear_flow`, HS = `helmholtz_staircase`,
ACT = `active_matter`, TRL2 = `turbulent_radiative_layer_2D`, VE = `viscoelastic_instability`.
Family numerals (I–VI) and axis definitions are those of
[well_walrus_dataset_categorisation.md](../../well_walrus_dataset_categorisation.md) §2.2 and §5.

Shapes are written `H/L/M` = `hidden_dim/processor_blocks/mlp_dim`. Named shapes: **fullDims**
`1408/40/5632` (paper scale), **HalfWalrus** `1088/30/4352`, the **200 M anchor** `768/12/2048`,
and **SoWalrus**, the shape this study predicts (§8.1).

---

## 0. What changed relative to the earlier 7-dataset plan

| item | 7-dataset plan | this plan | consequence |
|---|---|---|---|
| Pretraining data | `Leonardo_smallest7_2d` (7 datasets, by availability) | `Leonardo_pretrain9_2d` (9 datasets, by design, §2) | every data-dependent number is redone below |
| Held-out physics | none; transfer to unseen physics explicitly untested | 4 datasets spanning every categorisation axis (§2.3) | new RQ5 and Tier 7 |
| Unique train tokens `T_unique_model` | ≈ 1.3 G | **≈ 3.26 G** (≈ 2.41 G + ≈ 0.85 G Euler, all measured) | 2.5× more distinct data |
| Worst per-dataset repetition at 60 ep, N1 | ≈ 122× (VE) | **6.6×** (PS) | the repetition bound on sweep duration no longer binds (§7.0) |
| Spread in per-dataset repetition | 143× | 10× | RQ3's aggregate `R` is no longer one dataset's number (§9) |
| GATE C star-centre point | existing fullDims runs, "zero marginal cost" | **must be re-run** on the new mixture (§7.2) | ≈ +470 GPU-h |
| `roofline_staged.sh` | stage the mixture (~653 GB) to node-local disk | infeasible: train splits alone are ≈ 5.8 TB with Euler (§5.2) | redesigned as a paired subset test |
| Tier 0 scripts (roofline, noise floor) | built on smallest7, never submitted | **regenerated** in `LeonardoRunScripts/PT9_2d/{Roofline,NoiseFloor}/` (§5.2, §5.4) | no sunk cost: none were run |
| Euler availability | not on disk | **on disk and surveyed** (2026-10-02; 2.69 TB openBC, 2.65 TB periodicBC, all splits) | Tier 0 is unblocked: preflight and FLOP re-validation pass (§5.1) |
| `override_dimensionality` | documented as `=2` | **`=3`** (§3.3) | the documented `=2` fails on the first forward pass; every generator already used `=3` |

**Nothing already executed is invalidated by this change**, because no Tier 0–6 run of the
7-dataset plan was ever submitted (no `roofline_*` / `noise_floor_*` logs exist under
`/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs`). What *is* lost is the reuse of the historical
fullDims / `smallest7_2d_PT_*` runs as free data points: they trained on a different mixture.

---

## 1. Research questions and method

### 1.1 Research questions

**RQ1 (primary).** How should WALRUS's shape dimensions — token width `hidden_dim`, processor
depth `processor_blocks`, and feed-forward width `mlp_dim` — be scaled jointly as pretraining
compute increases, for a cross-domain continuum-dynamics foundation model?

**RQ2 (validating).** Does a shape-optimised WALRUS derived from RQ1 match or beat the paper-scale
configuration (`H=1408, L=40, M=5632`) when both are pretrained with the same FLOPs — measured
upstream *and* on the held-out physics?

**RQ3 (sub-study).** Over what range does the "unlimited data" assumption behind SoViT's Eq. 2 hold
for this mixture, and how does repetition change the compute-optimal answer? At sweep scale the
mixture is only mildly repeated (≤ 6.6× per dataset), so the question matters mainly at
head-to-head scale, where the small arm still reaches ≈ 240× on PS (§2.5, §9).

**RQ4 (methodological).** Does fitting on intermediate checkpoints (Choshen §6) reproduce the
exponents obtained from end-of-run points, at a fraction of the cost — and does WALRUS's WSD-style
inverse-sqrt LR schedule make those intermediate points legitimately comparable?

**RQ5 (upstream vs downstream).** Is the compute-optimal shape, and are its exponents `s_k`, the
same when "loss" means finetuned performance on physics the model never saw during pretraining, as
when it means held-out-trajectory loss on the pretraining physics? SoViT raise this question in
their §2 (upstream metrics may not reflect downstream ranking) and partly answer it in §4.2
(exponents near-identical across classification and image-to-text). Here it is asked for PDE
surrogates, separately for in-family and out-of-family targets (§2.3).

### 1.2 The SoViT recipe

SoViT models a single shape dimension `x_k` jointly with compute `t`:

```
f_k(x_k, t) ≈ α_k·x_k^(−a_k) + (β_k·x_k^(b_k) + ξ_k)·t^(−c) + ε_k        [SoViT Eq. 2]
```

For fixed `t` this is quasiconvex in `x_k`, with optimum

```
x*_k = (α_k·a_k·t^c / (β_k·b_k))^(1/(b_k+a_k)) = O(t^(s_k)),   s_k = c/(b_k+a_k)   [Eq. 4]
```

The exponents `s_k` are estimated by a **star sweep** (§7.2): fix a large star centre `x⁽ᶜ⁾`, vary
one dimension at a time over an exponentially spaced grid of values *much smaller* than the centre,
at several compute durations. A separate cheap **grid sweep** (§7.1) over small, short-trained
models finds the starting point `x⁽⁰⁾` on the small-compute Pareto frontier. Finally, increasing
compute by a factor `τ` scales each dimension by `τ^(s_k/D)`, with `D` the number of shape
dimensions (SoViT's `w_k = 1/D`; here `D = 3`).

> The paper's prose says the increment is "`τ^(w_k)`", but their own Figure 1 confirms the
> intended rule is `x_k ∝ t^(s_k/D)`: at `τ = 100` with `D = 3` and `s_depth = 0.45`, depth grows
> by `100^(0.45/3) = 2.00×`, exactly the "×2.0" annotated on that figure. We use `x_k ∝ t^(s_k/D)`.

### 1.3 Six places WALRUS is not a ViT

These adaptations make or break the study; each has a design response later in this document.

**(a) Parameter count is a poor summary statistic here, worse than in a ViT.** The encoder and
decoder are each a two-stage conv stack, and the `I · n_states · 512` patchify term does not depend
on any shape knob. With one encoder/decoder pair (`override_dimensionality=3`, §3.3), measured with
`scripts/count_WALRUS_params.py`:

| config | encoder | processor | decoder | total | enc+dec share |
|---|---:|---:|---:|---:|---:|
| fullDims `1408/40/5632` | 43.8 M | 1111.0 M | 43.8 M | **1198.6 M** | 7 % |
| HalfWalrus `1088/30/4352` | 36.6 M | 497.7 M | 36.6 M | **570.8 M** | 13 % |
| 200 M anchor `768/12/2048` | 29.4 M | 85.1 M | 29.4 M | **143.8 M** | **41 %** |

Of the encoder+decoder mass, `2 · I · n_states · 512 = 24.2 M` (at `I = 352`, `n_states = 67`) is
*completely fixed* whatever the shape: at the 200 M anchor that is 17 % of the model doing nothing
that scales. The ballast share is worst exactly where the study runs the most points: 84 % of
parameters at the grid sweep's smallest corner (§7.1). The "200 M anchor" name is kept as a label;
its true count is 143.8 M. Historical runs that passed `override_dimensionality=0` built a second,
never-used 3D encoder/decoder pair, so their totals are larger (fullDims 1286.2 M, anchor 202.6 M);
compare against them with care.

*Response:* pin `intermediate_dim = 352`, `data/field_index_map_override=full_well_field_index`
and `override_dimensionality = 3` on every run (§4.1), so the encoder/decoder are a known,
constant, minimal ballast. Every plot reports **processor parameters** alongside total parameters,
and the analysis favours SoViT's compute-parameterised form over a Chinchilla-style `(N, D)` fit.

**(b) `n_states` silently depends on what is on disk.** `train.py` derives `n_states` as
`max(field_to_index_map.values()) + 1`, and with `prefetch_field_names: True` the datamodule loops
over every Well dataset and silently skips any that fail to load. Adding or removing a dataset
directory would change every model's encoder/decoder parameter count.

*Response:* the `full_well_field_index` pin makes it deterministic: `n_states = 67` whatever is on
disk, because every field name any of the 13 datasets produces is already in the 67-entry map
(§2.1). `n_states` is checked in every run's logged `extended_config.yaml` as a post-condition
(§5.3).

**(c) The LR schedule is WSD-shaped, and this is a gift.**
`InverseSqrtLinearWarmupSqrtCooldown` (`walrus/optim/schedulers.py`, config
`lr_scheduler=inv_sqrt_w_sqrt_ramps`) has three phases: linear warmup (`warmup_epochs`), a
**length-independent** inverse-sqrt "infinite" phase, then a cooldown ramp starting at
`max_epoch − cooldown_epochs`. Unlike with a cosine schedule, intermediate checkpoints inside the
infinite phase are mutually comparable, because the LR at epoch `E` does not depend on the run's
total length. This is exactly the condition Choshen §6 worries about ("changes in the learning
rate schedule, which depend on the size of the full dataset... render losses from intermediate
checkpoints uninformative"), and WALRUS satisfies it.

*Response:* fit only on infinite-phase points (§7.3), and separately measure the cooldown gain to
confirm it is shape-independent.

**(d) Repeated data.** SoViT Eq. 2 assumes unlimited data. On this mixture the aggregate
repetition `R = T_seen / T_unique_model` is mild at sweep scale (≈ 1.9 at 60 epochs on N1), but
reaches ≈ 12 for a fullDims head-to-head arm and ≈ 66 for the ≈ 5.4×-longer SoWalrus arm, with
the most repeated dataset (PS) at ≈ 236× (§2.5).

*Response:* the RQ3 sub-study (§9), a repetition column (`R` and max `R_d`) on every results table,
and a constraint on how far `t_target` may be pushed (§8.2).

**(e) Upstream WALRUS has no seed control.** Upstream never calls `torch.manual_seed` (only the
data samplers are seeded), so every run had an uncontrolled weight-init seed. Choshen §4 finds seed
variation alone moves loss by up to 4 %, and §7 warns that a single unlucky seed can poison a whole
scaling fit.

*Response, done:* a top-level `seed:` config key, and `seed_all` (§5.1) called before model
construction. Every claim of a difference is stated against a measured noise floor `ε_noise`
(§5.4).

**(f) The encoder/decoder are a FLOP ballast too, and it is worst where the exponents come from.**
Encoder FLOPs depend on `hidden_dim` only through `proj2`'s output channels, and **not at all** on
`processor_blocks` or `mlp_dim`. So the encoder and decoder add a large, nearly shape-independent
constant to `t` at exactly the points the fit uses. At fullDims they are 1.4–1.7 % of forward
FLOPs; at the grid sweep's smallest corner they are **29–45 %**, depending on the dataset (§3.1).
SoViT's functional form assumes compute is essentially the model; at the small end of this study,
a third to nearly half of it is patchify stem that barely responds to the swept knobs.

*Response:* log the enc+dec FLOP fraction on every point alongside `t`, and fit the exponents
**both** on total FLOPs and on processor-only FLOPs (§7.1). If they diverge materially, that
divergence is itself a WALRUS-vs-ViT structural finding, not a nuisance to smooth over.

### 1.4 The three given model scales map onto SoViT's three roles

| WALRUS config | shape | SoViT role |
|---|---|---|
| fullDims / paper scale | `1408/40/5632` | **star centre** `x⁽ᶜ⁾`: never fit on; the held-out extrapolation check (SoViT's violet point, their Fig. 4 left) |
| HalfWalrus | `1088/30/4352` | optional mid-scale sanity target for the derived trajectory; large-extreme stand-in in Tier 1 |
| 200 M anchor | `768/12/2048` | sits just *above* the grid-sweep region; the fixed shape for the noise floor and the RQ3 sub-study |

SoViT's star sweep requires the centre to be *large*, and every swept value to be *much smaller
than the centre in that dimension*. The existing top-level
[`gen_star_search_runs.py`](../gen_star_search_runs.py) is still centred on the 200 M anchor
(`H ∈ 384…1024` around 768), so its PT9 replacement must be centred on fullDims (§12).

---

## 2. Dataset selection

### 2.1 Inventory of the 2D datasets on disk

Read directly from each file's HDF5 metadata (`n_trajectories`, the `time` and spatial dimension
arrays, `boundary_conditions`, `t{0,1,2}_fields`). Tokens per frame `N_sp` use the rule in
[`flops.analytic_token_shape`](../../walrus/utils/flops.py): 32 per axis, plus one on each
non-periodic axis. "Train tokens" = `Σ n_traj × n_steps × N_sp` over the train split, i.e. this
dataset's contribution to `T_unique_model` (§3.2). "Valid windows" = `Σ n_traj × (n_steps − 6)`
at `n_steps_input=6`, `n_steps_output=1`, `dt=1`. "Ch." = physical input channels after 2D→3D
tensor padding (vectors → 3, rank-2 tensors → 9), the per-forward encoder `c_in` minus the 3 BC
flags.

| code | resolution | BCs (x, y) | steps | train traj | **train tokens** | valid traj / windows | `N_sp` | ch. | train size | role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GS | 128×128 | per, per | 1001 | 960 | **984.0 M** | 120 / 119,400 | 1024 | 2 | 122 G | pretrain |
| EMQ-O | 512×512 | open, open | 101 | 4000 | **440.0 M** | 500 / 47,500 | 1089 | 6 | 2.15 T | pretrain |
| EMQ-P | 512×512 | per, per | 101 | 4000 | **413.7 M** | 500 / 47,500 | 1024 | 6 | 2.12 T | pretrain |
| AM | 256×256 | wall/open, wall/open | 202 | 1600 | **352.0 M** | 200 / 39,200 | 1089 | 6 | 238 G | pretrain |
| AI | 256×256 | wall/open, wall/open | 102 | 2798 | **310.8 M** | 400 / 38,400 | 1089 | 6 | 226 G | pretrain |
| RB | 512×128 | per, wall | 200 | 1400 | **295.7 M** | 175 / 33,950 | 1056 | 5 | 274 G | pretrain |
| SF | 256×512 | per, per | 200 | 896 | **183.5 M** | 112 / 21,728 | 1024 | 5 | 438 G | pretrain |
| AD | 256×256 | wall/open, wall/open | 102 | 1600 | **177.7 M** | 200 / 19,200 | 1089 | 6 | 121 G | pretrain |
| PS | 256×512 | open (θ), per (φ) | 1008 | 96 | **102.2 M** | 12 / 12,024 | 1056 | 3 | 142 G | pretrain |
| HS | 1024×256 | open/wall, open/wall | 50 | 416 | **22.7 M** | 48 / 2,112 | 1089 | 3 | 41 G | **held out** |
| ACT | 256×256 | per, per | 81 | 175 | **14.5 M** | 24 / 1,800 | 1024 | 22 | 39 G | **held out** |
| TRL2 | 128×384 | per, open | 101 | 72 | **7.7 M** | 9 / 855 | 1056 | 5 | 5.4 G | **held out** |
| VE | 512×512 | per, wall | 20 / 60 | 213 | **7.1 M** | 22 / 548 | 1056 | 14 | 53 G | **held out** |
| **pretrain total** | | | | | **≈ 3.26 G** | | | | ≈ 5.8 T | |
| **held-out total** | | | | | **52.0 M (1.6 %)** | | | | 138 G | |

PS's channel count was listed as 4 until 2026-10-06. It is **3** (`height`, `velocity_theta`,
`velocity_phi`): PS is on an equiangular grid, so its vector field is not padded to three
components. Checked against the real pipeline by `scripts/build_well_registry.py --verify`, which
also confirms every other "ch." value above. §3.1's PS column is corrected to match.

**Euler figures are measured** (re-surveyed 2026-10-02, after the download completed; every
train/valid/test file of both variants opens). Each variant has 10 γ-files per split, 400 / 50 / 50
trajectories per file, 512², and stored fields `density`, `energy`, `pressure` and `momentum` (2-vector)
in float32, i.e. 5 stored channels, with train files of ≈ 212–215 GB. The earlier estimate's only error was
the trajectory length: **101 steps, not 100**, which raises the train tokens by 1 % (from ~435.6 / ~409.6 M)
and the valid windows from ~47,000 to 47,500. No conclusion in this document changes.

Upstream's own [`all_2d.yaml`](../../walrus/configs/data/all_2d.yaml) is exactly the first twelve rows
minus HS — i.e. these thirteen are all of The Well's 2D data. Every field name any of them produces
is already in the 67-entry
[`full_well_field_index`](../../walrus/configs/data/field_index_map_override/full_well_field_index.yaml)
map, so pinning it keeps `n_states = 67` whatever subset is loaded (§1.3b).

### 2.2 Selection criteria

In priority order:

1. **The held-out set must span every categorisation axis**, with at least one target on each side
   of every binary axis in [well_walrus_dataset_categorisation.md](../../well_walrus_dataset_categorisation.md)
   §5: equation family, linear/nonlinear, diffusive/non-diffusive, sharp/smooth, chaotic/regular,
   and boundary topology. Otherwise RQ5 can only say "transfers to *this kind* of physics".
2. **It must contain both in-family and out-of-family targets.** A target whose equation family is
   represented in pretraining measures near transfer; one whose family is absent measures far
   transfer. RQ5 is asked separately for each.
3. **Holding out should reduce, not worsen, repetition in pretraining.** Sampling is uniform over
   datasets: `BatchedMultisetSampler` picks the dataset for each micro-batch uniformly, regardless
   of its size ([walrus_tokens_and_data_pipeline.md](../../mydocs/walrus_tokens_and_data_pipeline.md)
   §4.1). So dataset `d` receives `T_seen / K` tokens regardless of its size, and its repetition is
   `R_d = T_seen / (K · T_unique,d)`. `R_d` is set by the *small* datasets. Removing a small dataset
   removes the worst `R_d` from the mixture; removing a large one just raises everyone else's `R_d`
   by `K/(K−1)`.
4. **The pretraining mixture must still cover every axis**, so every in-family target has a
   pretrained counterpart and the mixture stays "cross-domain".
5. **Held-out targets should be cheap and data-scarce to finetune on.** Pretraining helps most
   when finetuning data is scarce ([WALRUS] Fig. 8), and Tier 7 finetunes many checkpoints.

Criteria 1, 3 and 5 turn out to point at the same four datasets.

### 2.3 The split and its coverage

**Held out: HS, ACT, TRL2, VE** — the four smallest datasets on disk.

| | family | linear | explicit diffusion | sharp features | regime (proxy T21–60) | BC topology | in pretraining? |
|---|---|---|---|---|---|---|---|
| AD / AI / AM | I waves | ✓ | none | ✓ interfaces | regular (0.04–0.13) | wall/open both axes | pretrain |
| EMQ-O | II compressible | — | none | ✓ shocks | intermediate (0.28) | open | pretrain |
| EMQ-P | II compressible | — | none | ✓ shocks | intermediate (0.28) | periodic | pretrain |
| PS | II (SWE analogue), sphere | — | hyperviscosity only | — | slowly chaotic, predictable at 60 steps (0.02) | periodic + open | pretrain |
| RB | IV incompressible + field | — | ✓ | — | **chaotic** (0.64) | periodic + wall | pretrain |
| SF | IV incompressible + field | — | ✓ | — | chaotic in principle, predictable at 60 steps (0.08) | periodic | pretrain |
| GS | VI reaction–diffusion | — | ✓ | — | regular (0.13) | periodic | pretrain |
| **HS** | **I** waves | **✓** | none | — smooth | **regular** (0.003) | open + wall | **held out** |
| **TRL2** | **II** compressible | — | none | **✓ interfaces** | **chaotic** (0.84) | periodic + open | **held out** |
| **VE** | **IV** incompressible + tensor | — | **✓** | borderline (sharp in C) | **mixed, by file** | periodic + wall | **held out** |
| **ACT** | **V** Stokes + kinetic | — | **✓** | — smooth | **chaotic** (1.25) | periodic | **held out** |

Axis attributes are those of the categorisation doc §5.2 and §5.4.3–5.4.5. Read down the held-out
rows:

| axis | held-out targets on side A | held-out targets on side B |
|---|---|---|
| equation family | I (HS), II (TRL2), IV (VE) — in-family | V (ACT) — out-of-family |
| linear vs nonlinear | HS | TRL2, VE, ACT |
| diffusive vs non-diffusive | VE, ACT | HS, TRL2 |
| sharp vs smooth | TRL2 (+ VE's conformation tensor) | HS, ACT |
| chaotic vs regular | TRL2, ACT, VE-CAR/EIT | HS, VE-AH |
| boundary topology | fully periodic: ACT · periodic+open: TRL2 | periodic+wall: VE · open+wall: HS |

Every axis is covered on both sides, and all four boundary topologies appear. The pretraining
mixture also covers both sides of every axis — linear (AD/AI/AM) and nonlinear, diffusive (RB, SF,
GS) and not (acoustics, Euler), sharp (acoustics, Euler) and smooth (RB, SF, GS, PS), chaotic (RB,
plus Euler's intermediate regime) and regular — and every held-out BC topology also occurs in
pretraining. So no held-out target is out-of-distribution *by representation alone*, except where
that is the point (next table).

**What each held-out target tests.** The four sit at different distances from the pretraining data,
along two separate ladders — equations and field names:

| target | nearest pretraining relative | equation distance | field-name overlap with pretraining | what a good transfer result would mean |
|---|---|---|---|---|
| TRL2 | EMQ-O/P — the "equation twin" (categorisation §2.4) | same compressible Euler + one cooling sink | **full** (density, pressure, velocity) | equation-level transfer works when the fields line up; the classic near-transfer case |
| VE | SF → RB (categorisation §5.4.2 ladder) | one more step on no-feedback → scalar-feedback → **tensor**-feedback | partial (pressure, velocity; C, c_zz new) | transfer survives an increase in the rank of the coupled field |
| HS | AD/AI/AM | same wave equation; constant medium, complex boundary, time-harmonic | **none** (`pressure_re`, `pressure_im`, `mask` all new) | transfer runs through the processor, not the field embeddings |
| ACT | none (family V absent); equation-level cousin of VE (also held out) | out-of-family: Stokes flow, no inertia, kinetic closure | partial (velocity; concentration, D, E new) | far transfer: pretraining helps even without a related family |

**Why these four also serve the repetition criterion.** They are the four smallest datasets on disk
(7.1–22.7 M train tokens each, 52 M together = 1.6 % of all unique train tokens). In the
7-dataset mixture they were the most repeated datasets — 40–122× at 60 epochs — and were already
suspected of contributing a flat, saturated floor rather than scaling signal to `f`. Removing them
costs pretraining almost no distinct data, and removes the worst repetition (§2.5).

**Why they are good downstream targets.** Finetuning on them is cheap (one single-dataset N1 epoch covers
TRL2's whole train split ≈ 13 times), they are genuinely data-scarce (72–416 train trajectories), and
the paper's own downstream results show pretraining matters most in that regime. The price is small
evaluation sets — see §3.5.

### 2.4 Alternatives considered

| variant | what it adds | why not the default |
|---|---|---|
| also hold out **PS** | a geometry probe (sphere, curvilinear, rotation, forcing); settles the categorisation doc's open question (§5.4.2) of whether PS behaves like family II or IV; drops pretraining max `R` to 4.4× | PS is the only curvilinear, rotating, externally forced dataset; removing it narrows pretraining more than any other single removal |
| also hold out **GS** | a second out-of-family target with a 119k-window valid set | GS is the only non-fluid physics in pretraining, and it is the *least* repeated dataset, so removing it does nothing for repetition |
| also hold out **AI** | a near-transfer control (same equations as AD/AM); rebalances acoustics from 3/9 to 2/8 of micro-batches | AI's transfer is barely "blind" — it largely re-measures the upstream valid-split metric; a fifth target raises Tier 7's cost by 25 % |

Each remains a cheap later extension: a single-dataset config exists or is trivial to add, and none
of them requires re-running pretraining *if decided before Tier 0 is submitted*.

### 2.5 Repetition in the new mixture

Per-dataset tokens delivered per epoch on the §4.1 N1 substrate (`max_samples=2000`,
`batch_size=2`, `n_steps_input=6`, `world_size=4`, `K=9`): `96 000 × N_sp / 9` ≈ 10.9–11.6 M.

| dataset | train tokens | epochs per full pass | `R` at 60 ep, N1 | `R` at 100 ep, N1 | fullDims arm (100 ep, N4) | SoWalrus arm (≈ 5.4× longer) |
|---|---:|---:|---:|---:|---:|---:|
| GS | 984.0 M | 90.1 | 0.7× | 1.1× | 4.5× | ≈ 24× |
| EMQ-O | 440.0 M | 37.9 | 1.6× | 2.6× | 10.6× | ≈ 57× |
| EMQ-P | 413.7 M | 37.9 | 1.6× | 2.6× | 10.6× | ≈ 57× |
| AM | 352.0 M | 30.3 | 2.0× | 3.3× | 13.2× | ≈ 71× |
| AI | 310.8 M | 26.8 | 2.2× | 3.7× | 15.0× | ≈ 80× |
| RB | 295.7 M | 26.2 | 2.3× | 3.8× | 15.2× | ≈ 82× |
| SF | 183.5 M | 16.8 | 3.6× | 6.0× | 23.8× | ≈ 128× |
| AD | 177.7 M | 15.3 | 3.9× | 6.5× | 26.1× | ≈ 140× |
| PS | 102.2 M | 9.1 | **6.6×** | **11.0×** | **44.1×** | **≈ 236×** |
| aggregate `T_seen / T_unique_model` | ≈ 3.26 G | — | 1.9× | 3.1× | ≈ 12× | ≈ 66× |

For comparison, the 7-dataset mixture at 60 epochs ran from 0.9× (GS) to ≈ 122× (VE) on the
train split, and its head-to-head aggregate was `R ≈ 30` / `≈ 160`.

Three things follow:

- **At sweep scale (≤ 100 epochs, N1) every dataset stays within or near Muennighoff et al.'s
  "negligible degradation" band (≤ 4×) or their "half-life" (~16×).** No dataset crosses 16× before
  warmup ends, as VE and TRL2 did in the 7-dataset mixture.
- **At head-to-head scale, repetition is still real.** PS reaches ≈ 44× in the fullDims arm and
  ≈ 236× in the SoWalrus arm. RQ3 (§9) is still needed, and needs to probe `R` in the tens to
  hundreds per dataset.
- **The aggregate is no longer one dataset's number.** GS is 30 % of `T_unique_model` (it was
  ≈ 75 % of the 7-dataset mixture), so an aggregate-`R` fit in §9 describes the mixture rather
  than Gray–Scott.

**Mixture balance, for the record.** Uniform-over-dataset sampling gives each dataset 1/9 of the
micro-batches, so by family: acoustics (I) 3/9, compressible (II) 3/9 (EMQ ×2 + PS), incompressible
(IV) 2/9, reaction–diffusion (VI) 1/9. This imbalance is incidental — the same kind WALRUS's own
mixture has (categorisation doc §4.1) — and is reported, not corrected; changing the sampler would
be a departure from upstream WALRUS that confounds every comparison with the paper.

### 2.6 Blindness: how "never seen" is guaranteed

- **Data.** The four targets are absent from
  [`Leonardo_pretrain9_2d.yaml`](../../walrus/configs/data/Leonardo_pretrain9_2d.yaml). Training *and*
  validation datasets are both built from that config, so no training step and no mid-training
  validation ever reads them. The run scripts must never pass them via overrides; the §5.3
  post-condition check (resolved dataset list identical across runs) enforces this.
- **Field embeddings.** `n_states = 67` includes rows for the held-out-only fields (`C`, `c_zz`,
  `concentration`, `D`, `E`, `pressure_re`, `pressure_im`, `mask`). The encoder only ever uses
  `proj1.weight[:, field_indices]` ([vstride_encoder.py:349](../../walrus/models/encoders/vstride_encoder.py#L349))
  and the decoder slices `proj2` by `state_labels` likewise, so those columns receive **exactly zero
  gradient** throughout pretraining. Their Adam moments stay zero, so the Adam update is zero. The
  only thing that touches them is AdamW's decoupled weight decay, which scales the whole tensor by
  `(1 − lr·wd)` per step: at the pinned `lr ≤ 2e-4`, `wd = 1e-4`
  ([adam.yaml](../../walrus/configs/optimizer/adam.yaml), no parameter-group exclusions) that is
  `(1 − 2e-8)` per step, ≤ 0.06 % shrinkage over 30 000 optimizer steps. The held-out fields' rows
  are therefore still their random initialisation, to within 0.06 %, when finetuning starts.
- **Normalisation.** `SamplewiseRevNormalization` normalises per sample, so no per-dataset
  statistics are computed or stored anywhere that could carry information from the targets.
- **Intended transfer channels.** Shared field *names* — `velocity` (every target except HS),
  `pressure` (TRL2, VE), `density` (TRL2, via EMQ; the acoustic density is zeroed and renamed
  `zeros_like_density`) — map to embedding rows that *were* trained. That is a legitimate part of
  what a foundation model transfers and is **not** removed. It is, however, a confound between
  "the physics transfers" and "the field embedding transfers", which the field-overlap column of
  §2.3 makes explicit. It is separated by the embedding-reinitialisation ablation in Tier 7 (§10).

---

## 3. Definitions: `t`, `D`, `N`, `f_up`, `f_down`

### 3.1 Compute `t`

SoViT uses GFLOPs, arguing it is hardware-agnostic and correlates ~0.99 with core-hours. We adopt
FLOPs as the scaling variable, but **verify the correlation rather than assume it** (the roofline,
§5.2), because WALRUS breaks the ViT analogy in two ways:

- **Encoder cost is a step function of resolution, not a smooth scaling.** Conv FLOPs are
  `output positions × kernel volume × c_in × c_out`. The kernels are *fixed* (8, then 4) and the
  token grid is *fixed* at 32 per axis by construction, so only the **stride** varies with
  resolution. Resolution therefore enters only through the intermediate grid between the two conv
  stages, `mid = 31·s₂ + 4`, and `patch_dict` admits just two values of `s₂`: **any axis ≤ 384 →
  mid = 66; any axis ≥ 512 → mid = 128**. Datasets fall into two encoder cost tiers, not a
  continuum: GS (128²) and AD (256²), 4× apart in pixels, have identical encoder FLOPs.
- **Per-micro-batch FLOPs are not constant across the mixture, although tokens are.** A forward
  pass contains exactly one dataset, and datasets differ in encoder tier, channel count and
  boundary conditions. Across the nine pretraining datasets, per-micro-batch FLOPs span 1.07× at
  fullDims and up to 1.38× at the grid's smallest corner (table below). `cumulative_flops` must
  therefore be accumulated per micro-batch using the dataset actually drawn, never as
  `shape × steps`; `FlopAccountant` does exactly that (§5.1).

**The analytical formula is the primary instrument, and `FlopCounterMode` is only the
cross-check.** [`walrus/utils/flops.py`](../../walrus/utils/flops.py) holds a closed-form FLOP
count for every submodule, validated exactly — to the FLOP, not to 5 % — against
`torch.utils.flop_counter.FlopCounterMode` (§5.1). It is exact, backend-independent, free to
evaluate, and defined for shapes that have never been run. `FlopCounterMode`, by contrast, counts
only ops with a registered formula for the kernel actually dispatched: on a CPU build
`F.scaled_dot_product_attention` dispatches to a kernel with no formula, so spatial attention —
the `O(N_sp²)` term — is silently counted as **zero**. On GPU it is counted exactly under this
study's fp32 configuration (§5.1). `FlopCounterMode` never runs inside a training run.

Backward is charged at 2× forward. The gradient-checkpointing recompute surcharge is computed
explicitly (encoder and decoder are always recomputed when any checkpointing is on; processor
block `i` when `i % freq == 0`) and reported as its own term. It changes *measured* FLOPs without
changing *useful* FLOPs, and the two head-to-head arms may not use the same setting.

Closed form, for planning only (processor terms per block; the analysis uses the logged values):

```
MACs/token/block = [ H(M+3H) + H² + (M/2)H + 2·N_sp·H ]   (space)
                 + [ 3H² + H² + 2·T·H ]                   (time)
FLOPs_fwd = 2 · MACs/token/block · (B·T·N_sp) · L
t         = 3 · FLOPs_fwd · max_samples · world_size · epochs
```

**Mixture-dependent numbers**, recomputed with `flops.py`'s own functions
(`ModelShape`/`BatchGeometry` built by hand from each dataset's resolution, `patch_dict` strides,
BCs and channel count, then `forward_flops`; `scripts/count_WALRUS_flops.py` /
`well_registry.dataset_geometry`). The same code reproduces the corresponding table of the
7-dataset plan exactly, which validates the hand-built geometry.

**Encoder cost tiers.** Five of the nine pretraining datasets — EMQ-O, EMQ-P (512²), PS, SF
(256×512) and RB (512×128) — have an axis ≥ 512, so their encoder intermediate grid is 128 per
axis. AD, AI, AM (256²) and GS (128²) have mid = 66.

**Enc+dec share of forward FLOPs, per dataset:**

| shape | AD/AI/AM | EMQ-O | EMQ-P | GS | PS | RB | SF | per-µbatch FLOP spread (pretrain9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fullDims `1408/40/5632` | 1.4 % | 1.7 % | 1.7 % | 1.4 % | 1.4 % | 1.5 % | 1.5 % | 1.07× |
| HalfWalrus `1088/30/4352` | 2.4 % | 3.0 % | 3.0 % | 2.4 % | 2.4 % | 2.5 % | 2.6 % | 1.08× |
| 200 M anchor `768/12/2048` | 8.8 % | 11.8 % | 12.0 % | 8.7 % | 9.0 % | 9.6 % | 9.9 % | 1.11× |
| SoWalrus predicted `720/24/3616` | 4.0 % | 5.5 % | 5.6 % | 3.9 % | 4.1 % | 4.4 % | 4.5 % | 1.09× |
| roofline small `464/10/1120` | 15.9 % | 23.2 % | 23.6 % | 15.4 % | 16.5 % | 18.0 % | 18.5 % | 1.19× |
| Tier 2 grid, largest `560/12/1344` | 11.7 % | 16.7 % | 17.0 % | 11.4 % | 12.1 % | 13.1 % | 13.5 % | 1.15× |
| **Tier 2 grid, smallest `320/6/768`** | **30.8 %** | **44.3 %** | **45.0 %** | **29.1 %** | **32.0 %** | **35.0 %** | **35.8 %** | **1.38×** |

The PS column was recomputed on 2026-10-06 with PS's corrected channel count (3, §2.1); every
other column and the spread reproduce unchanged.

**Ballast warning.** The worst case is Euler (`EMQ-P`, 45 % at the grid's smallest corner). At the
grid sweep's small end, a third to nearly half of every micro-batch's compute is the patchify stem,
which barely responds to the swept knobs (§1.3f). At fullDims the encoder really is a small
fraction, so "the processor dominates" is true there and **false across the grid sweep**, which is
where `x⁽⁰⁾` and hence the exponents come from. The dual fit (total vs processor-only FLOPs, §7.1)
is therefore mandatory.

⚠️ The 7-dataset plan quoted its per-micro-batch FLOP spread as "1.03× at fullDims, 1.95× at
`320/10/1280`". The current `flops.py` gives **1.07× and 1.46×** for that same old mixture (and
1.07× / 1.27× for the new one at `320/10/1280`). The quoted figures probably predate a correction
in `flops.py`; only numbers from the current code are used here.

### 3.2 Data `D`

Three numbers, always reported together (definitions as in
[walrus_tokens_and_data_pipeline.md](../../mydocs/walrus_tokens_and_data_pipeline.md) §5):

- **`T_seen`** = `batch_size × n_steps_input × N_sp × max_samples × world_size × epochs`, summed
  per micro-batch with the drawn dataset's `N_sp`. This is the token axis; it is exact.
- **`T_unique_model`** = `Σ_datasets n_traj × n_steps × N_sp` over the train split, the number of
  distinct post-tokenizer cells the mixture can produce: **≈ 3.26 G** for this mixture (≈ 2.41 G
  plus ≈ 0.85 G Euler, all measured, §2.1).
- **Repetition factor** `R = T_seen / T_unique_model`, reported for every run, alongside the
  per-dataset `R_d = T_seen / (K · T_unique,d)` (§2.5).

`T_seen` and `R` are logged per epoch by `FlopAccountant` (`tokens_seen`, `repetition_factor`).
The denominator (`flops.unique_model_tokens`) is computed from the loaded datamodule, reading each
dataset's per-axis boundary conditions from its first HDF5 file, so it follows the mixture
automatically.

`max_samples` caps micro-batches *per rank*, so `world_size` is a batch-size axis, not a speed
axis: adding nodes multiplies `T_seen` and the global batch, and does not shorten an epoch
(historical fullDims timings: 1 node ≈ 4 nodes per epoch, §5.2).

### 3.3 Parameters `N`

Reported two ways on every plot: **total** (from the analytical formula, cross-checked against
`sum(p.numel())`, `scripts/count_WALRUS_params.py`) and **processor-only**, since that is what the
shape knobs actually control (§1.3a). With `override_dimensionality=3` there is one
encoder/decoder pair (`n_dim_keys = 1`), and under the `full_well_field_index` pin parameter counts
do not depend on the mixture.

**Why the pin is `=3`, not `=2`.** `IsotropicModel.forward` always sets
`metadata.n_spatial_dims = max_d` (3) and looks up `self.embed["3"]`
([isotropic_model.py:285-313](../../walrus/models/isotropic_model.py#L285-L313)). Under
`override_dimensionality > 0` the encoder/decoder routing also reads that overwritten
`n_spatial_dims`. So `=2` builds only a `"2"` encoder and fails with `KeyError: '3'` on the first
forward pass (reproduced on a real `gray_scott` batch). `=3` builds exactly one pair, `"3"`, and
routes 2D data through it: 2D data is already padded to a singleton third axis, and `adaptive_conv`
sums the kernel over that axis. This is also the `isotropic_model.yaml` default ("Force it to treat
all data as 3D"). `=0` is not an alternative: `include_d` then stays at its default `[2, 3]` and
builds a second, never-used encoder/decoder pair (§1.3a). Every generator in `LeonardoRunScripts/`
passes `=3`, and `shape_cost_table.py` counts parameters under `include_d = [3]`.

### 3.4 Upstream metric `f_up`

**Rollout VRMSE at the 60-step horizon on the `valid` split, averaged over the 9 pretraining
datasets with equal weight** (the `rollout_valid` metric the trainer already logs). Every
pretraining dataset's trajectories are long enough for a 60-step rollout after a 6-frame history
(the shortest are EMQ at 101 steps and AD/AI at 102).

**Secondary metrics**, for the exponent-robustness check SoViT make in their Fig. 5: one-step
`valid` VRMSE, one-step `valid` NRMSE, and rollout VRMSE at a shorter horizon. The `test` split is
touched **only** for the final §8 numbers, never during fitting.

Because acoustics and compressible flow are each 3/9 of the mixture (§2.5), also report the
**family-balanced mean** (average within family, then across the four families) as a secondary
`f_up`. If `s_k` differs between the dataset-balanced and family-balanced `f_up`, that is itself a
finding about how mixture composition leaks into scaling exponents.

**The 60-step horizon and cost.** `max_rollout_steps=60` matches the historical WALRUS pretraining
runs' convention. Rollout-validation cost scales with it, so it costs 3× a 20-step horizon. Every
GPU-h estimate in this document that was first derived at 20 steps is marked provisional until the
roofline re-measures (§5.2, §11).

**How validation is run during training.**

- `short_validation_length=20` caps validation micro-batches during training (the trainer's own
  default). Full validation at the final epoch is switched off by `++trainer.debug_mode=True`
  (§4.1); full-split scores come from re-evaluating chosen checkpoints instead (§5.3b).
- Validation is deliberately *not* re-randomised across epochs: the validation sampler is a
  `RandomSampler` with a fixed `torch.Generator().manual_seed(0)`
  ([multidatamodule.py:462](../../walrus/data/multidatamodule.py#L462), upstream code). So the same
  held-out windows are scored at every logged checkpoint of every run in the study. Epoch-to-epoch
  variation is therefore the model, not sampling, and the capped subset adds no seed-to-seed noise
  to `ε_noise` (§5.4).

**Coverage of the capped mid-training validation** (`short_validation_length=20`, `batch_size=2` →
40 windows per dataset per check):

| dataset | valid windows | coverage at 40 windows |
|---|---:|---:|
| GS | 119,400 | 0.034 % |
| EMQ-O / EMQ-P | 47,500 each | 0.084 % |
| AM | 39,200 | 0.10 % |
| AI | 38,400 | 0.10 % |
| RB | 33,950 | 0.12 % |
| SF | 21,728 | 0.18 % |
| AD | 19,200 | 0.21 % |
| PS | 12,024 | 0.33 % |

The spread across datasets is ≈ 10× (it was 216× in the 7-dataset mixture, whose tiny valid sets
VE and TRL2 are now held out). But every dataset is sampled at a fraction of a percent, and the
capped reading alone cannot tell whether its one fixed slice is representative of the dataset's
true valid-split difficulty.

🔲 **Full-validation spot-check (planned, not yet run).** Score a handful of saved checkpoints with
`full=True` over the entire valid split via `Trainer.validate()`'s standalone path (no new
training), prioritising GS and the two Euler sets, and compare with the capped reading logged at
the same epoch. Close agreement means the capped numbers can be trusted as they are; divergence
means raising `short_validation_length` for the worst-covered datasets, or flagging the affected
points' precision in the fit (§12).

### 3.5 Downstream metric `f_down`

`f_down` is measured after finetuning a pretrained checkpoint on one held-out target under a fixed
protocol (Tier 7, §10). Scoring rules are set per target because the four differ sharply in
trajectory length, regime and evaluation-set size:

| target | steps/traj | max rollout after 6-frame history | regime | primary `f_down` | valid set |
|---|---:|---:|---|---|---|
| HS | 50 | 44 | regular | rollout VRMSE, T1–44 | 48 traj / 2,112 windows |
| TRL2 | 101 | 95 | chaotic | one-step VRMSE, T1–10 VRMSE, binned spectral error | **9 traj** / 855 windows |
| ACT | 81 | 75 | chaotic | one-step VRMSE, T1–10 VRMSE, binned spectral error | 24 traj / 1,800 windows |
| VE-CAR, VE-EIT | 60 | 54 | chaotic | one-step VRMSE, T1–10 VRMSE, spectral | 6 traj |
| VE-AH | 20 | 14 | regular | rollout VRMSE, T1–14 | **2 traj** |
| VE edge states (`transto*`) | 20 | 14 | sensitive by construction | one-step VRMSE only | 14 traj |

Rules:

- **Chaotic targets are never scored on long-horizon VRMSE**, which saturates once any model
  decorrelates (categorisation doc §5.4.5). The binned spectral metrics are already implemented in
  `the_well.benchmark.metrics`, but are switched off by the substrate's
  `++trainer.skip_spectral_metrics=True`; Tier 7 finetuning runs turn them back on.
- **VE is scored per regime**, not pooled, using `include_filters` on the file names
  (`_AH.`, `_CAR.`/`_EIT.`, `transto`). Pooling would average a steady travelling wave with
  elasto-inertial turbulence.
- **Full validation, not capped.** All four valid splits are small, so `full=True` over the whole
  valid split is cheap and removes the coverage problem of §3.4 entirely.
- **Uncertainty by trajectory bootstrap.** TRL2 (9 valid trajectories) and VE-AH (2) are too small
  for a single number to be trusted. Report every `f_down` with a bootstrap CI resampling
  *trajectories*, not windows (windows of one trajectory are strongly correlated). A shape
  difference in `f_down` counts only if it exceeds both this CI and `ε_noise` (§5.4).
- **The `test` split is touched only for the final §8 numbers**, exactly as for `f_up`.

---

## 4. Fixed experimental substrate

### 4.1 Held constant on every run in every tier

The single source of truth is [`pt9_substrate.py`](pt9_substrate.py) (`SUBSTRATE`); every PT9
generator builds on it, and this listing mirrors it:

```
distribution=fsdp
data=Leonardo_pretrain9_2d
data/field_index_map_override=full_well_field_index      # n_states = 67 -- §1.3b
model=isotropic_model
model.intermediate_dim=352      model.projection_dim=48
model.groups=16                 model.override_dimensionality=3        # n_dim_keys = 1 -- §3.3
model.processor.space_mixing.num_heads=16
model.processor.time_mixing.num_heads=16
model/processor/space_mixing=full_spatial_attention
model.causal_in_time=True       model.jitter_patches=True
model.drop_path=0.05
data.module_parameters.batch_size=2      n_steps_input=6      n_steps_output=1
data.module_parameters.min_dt_stride=1   max_dt_stride=5      max_samples=2000
trainer.grad_acc_steps=4        trainer.enable_amp=False     trainer.clip_gradient=10
trainer.prediction_type=delta
# optimizer: not overridden -- adam.yaml defaults, lr=2e-4, weight_decay=1e-4
lr_scheduler=inv_sqrt_w_sqrt_ramps   (warmup_epochs=10, cooldown_epochs=10)
trainer.short_validation_length=20   trainer.val_frequency=2   trainer.rollout_val_frequency=5
trainer.max_rollout_steps=60         trainer.video_validation=False
++trainer.skip_spectral_metrics=True
checkpoint=defaults
seed=<per run>
++model.use_periodic_fixed_jitter=True   ++model.input_field_drop=0.0   # upstream pins
++trainer.debug_mode=True                                               # capped validation only
```

- **Heads and drop-path.** `num_heads` is pinned at 16 throughout, as SoViT pinned heads at 16 and
  patch size at 14. All swept widths are divisible by 16 (for `num_heads` and `groups`).
  `drop_path` is pinned at one value across all depths. That is a mild confound (stochastic depth
  is depth-dependent regularisation), noted as a limitation rather than made a variable; SoViT made
  the same simplification.
- **The two upstream pins.** Upstream's canonical pretraining script
  ([pretrain_example_distributed_walrus.sh](../../walrus/run_scripts/pretrain_example_distributed_walrus.sh))
  and every historical fullDims/halfDims/`smallest7_2d_PT_*` run set them, but the 7-dataset plan's
  generators did not. Without them the model trains with the class defaults (`False`, `0.1`), i.e.
  with random input-field dropout and a different jitter.
- **`++trainer.debug_mode=True`** is an evaluation pin, not a training one. Its only effect
  ([training.py:657, 1296–1355](../../walrus/trainer/training.py#L1296-L1355)) is to keep every
  validation capped at `short_validation_length`. Without it, the epoch that reaches `max_epoch`
  runs *full* one-step and full-trajectory rollout validation over all nine valid splits (≈ 378 k
  windows, ≈ 2 200 trajectories) and then the same over the `test` split. That costs hours at fullDims,
  distorts the 3-epoch roofline timings, and touches the `test` split on every run, contrary to
  §3.5. Full-split scores come from re-evaluating chosen checkpoints instead (§3.4 spot-check, §5.3b).

**Allowed to vary, for engineering reasons only and never numerics:** `distribution`
(`fsdp`/`hsdp`), `model.gradient_checkpointing_freq`, `data_workers`, `trainer.log_interval`,
`checkpoint.checkpoint_frequency`, and node count when it is *not* being used as a `T_seen` knob.
Each is recorded per run so it can be ruled out as an explanation. Avoid `ddp` (§5.1: under it,
unseeded ranks would not share weights; seeding fixes it, but `fsdp` never had the problem).

[`Leonardo_pretrain9_2d.yaml`](../../walrus/configs/data/Leonardo_pretrain9_2d.yaml) lists the nine
datasets, with `density: torch.zeros_like` on AD/AI/AM as in upstream
[`all_2_3d.yaml`](../../walrus/configs/data/all_2_3d.yaml) and [WALRUS] App. C.1, and no transforms on
Euler (upstream applies none). Hydra composition with the pins above is verified: 9 datasets, the
three density transforms, `n_states = 67`. The full datamodule also instantiates and yields a real
batch from every dataset (`validate_flops_pt9.py`, §5.1, 2026-10-02).

**Naming discipline.** `get_experiment_name`
([experiment_utils.py](../../walrus/utils/experiment_utils.py)) encodes only
`{name}-{data}-{prediction_type}-{model}[...]-{optimizer}-{lr}`. **It does not encode
`hidden_dim`, `processor_blocks` or `mlp_dim`**, so two sweep points that differ only in shape
resolve to the same experiment folder and silently resume into each other's checkpoint. Every
generated script therefore carries the full shape, and the data config, in `name=` (e.g.
`p9_star_H0448_L40_M5632`), so that shapes, and runs on different mixtures, can never collide.
The generators enforce this.

### 4.2 Storage

- **$WORK (datasets):** Euler added 5.34 TB (both variants, all splits). `cinQuota` on 2026-10-02,
  with every download complete: **9.58 TB of the 20 TB** quota on `/leonardo_work/ICT26_MHPC_0`.
  Re-check before any further download.
- **$FAST (logs, checkpoints):** 321 GB of 1 TB on `/leonardo_scratch/fast/ICT26_MHPC`
  (2026-10-02). This is the binding storage constraint. A sharded AdamW checkpoint costs roughly
  `12–16 bytes × params`: ≈ 15 GB for fullDims, ≈ 4 GB for a 400 M star point. Eighteen star runs
  at three retained checkpoints each would be ≈ 220 GB.

**Checkpoint economy: the scaling-law fit needs the W&B metric history, not the weights.** So:

- sparse weights, dense metrics on every sweep run: `checkpoint.checkpoint_frequency` 20–25, but
  `trainer.val_frequency=2`. Metrics are what we fit;
- dense weight checkpoints only on the few runs used for the cooldown-branching check (§7.3) and
  on the noise floor, whose `checkpoint_frequency=10` adds ≈ 18 GB per seed (§5.4);
- after each tier completes and its metrics are synced, delete `step_*` directories with
  `LeonardoRunScripts/cleanup_run.sh`;
- Tier 7 adds finetuned checkpoints: keep only their final weights, and delete them after `f_down`
  is extracted.

### 4.3 W&B

Compute nodes have no outbound internet, so runs use `WANDB_MODE=offline`, with periodic
`wandb sync` from the login node, as documented in
[LeonardoRunScripts/README.md](../README.md). `Trainer.train()` passes `step=epoch` to every
`wandb.log()`, and the W&B run id is persisted in the experiment folder (`wandb_run_id.txt`), so a
resumed job joins the same W&B run and continues its history rather than clobbering it. This is
what makes chains of 24 h-limited jobs usable as continuous scaling curves.

---

## 5. Tier 0 — instrumentation, roofline, noise floor

**This tier gates everything else.** Nothing in Tiers 2–5 is submitted until §5.2 and §5.4 report.

### 5.1 FLOP counter and seed — done, mixture-independent

**FLOP counter.** [`walrus/utils/flops.py`](../../walrus/utils/flops.py) holds the closed form
(§3.1), and `Trainer` owns a `FlopAccountant` that logs per-epoch compute to W&B alongside
`train_loss`, with no config change. How it works:

1. `supports_flop_accounting(model)` checks that the (possibly FSDP-wrapped) model is an
   encoder/processor/decoder `IsotropicModel`. Baseline wrappers (DPOT/ScOT/MPP) go unaccounted
   rather than mis-accounted.
2. `ModelShape.from_model` reads every shape knob off the live model (`hidden_dim`, `mlp_dim`,
   `processor_blocks`, the encoder/decoder kernels, `causal_in_time`,
   `gradient_checkpointing_freq`), so it cannot drift from what the model actually is.
3. **Geometry is observed, not re-derived.** A forward pre-hook on the encoder records the padded
   input shape, the strides actually chosen, the boundary conditions and the real per-dataset
   channel count (`len(field_indices)`, not the global `n_states`); a forward hook records the
   token grid. Every FLOP bug found in this study came from re-deriving geometry the model already
   knew.
4. Each micro-batch is priced from that geometry with the cached closed form (the first
   micro-batch of each `(dataset, resolution, shape)` evaluates it; every later one is a dict
   lookup).
5. At epoch end the totals are all-reduced across ranks (exact, not `rank 0 × world_size`) and
   merged into the logged metrics. The run-long counters are saved in each checkpoint's
   `metadata.pt` and restored on auto-resume.

Logged per epoch: `epoch_flops`, `flops_per_microbatch`, `cumulative_flops`,
`cumulative_processor_flops` (for the processor-only fit, §7.1),
`cumulative_{encoder,decoder}_proj{1,2}_flops`, `enc_dec_flop_fraction`, `tokens_seen`,
`repetition_factor` and `gpu_hours`. Full key definitions, formulas, a worked example and the
validity envelope are in
[mydocs/flop_accounting_wandb.md](../../mydocs/flop_accounting_wandb.md).

**Evidence it is right:**

| check | what it proves |
|---|---|
| `tests/test_flops.py` (24 tests, CPU) | encoder + decoder closed form **exactly** equal to `FlopCounterMode` on a real forward; token-grid rule; recompute surcharge; per-micro-batch accumulation; counter round trip. Not the processor: CPU SDPA counts as zero |
| A100 job `57434063` ([`flop_counter_backend_test.py`](../flop_counter_backend_test.py)) | processor attention terms exact on GPU (table below) |
| [`validate_flops_pt9.py`](validate_flops_pt9.py) | encoder/decoder exact on a real batch of each of the nine PT9 datasets (below) |
| GPU end-to-end test, 2026-10-06 (`LeonardoRunScripts/flopcounter/`) | 3 chained `fsdp` jobs, 1 node × 4 GPUs, gradient checkpointing on: every logged epoch's FLOPs, tokens and splits equal the analytical replay to 1e-9 relative, and the counters continue exactly across both resumes; all 70 checks pass |

**Attention-backend check on GPU** (A100, 3 shapes × 2 dtypes × 2 modules): 11 of 12 exact.

| shape | module | dtype | backend dispatched | residual/attn | verdict |
|---|---|---|---|---|---|
| fullDims 1408/5632, 2D | space | fp32 | `efficient_attention` | 0.000 | exact |
| fullDims 1408/5632, 2D | space | bf16 | `flash_attention` | 0.000 | exact |
| Tier-2-small 320/1280, 2D | space | fp32 | `efficient_attention` | 0.000 | exact |
| **Tier-2-small 320/1280, 2D** | **space** | **bf16** | **`flash_attention`** | **+0.200** | **AMBIGUOUS** |
| fullDims 1408/5632, 3D | space | fp32 / bf16 | `efficient_attention` / `flash_attention` | 0.000 | exact |
| all shapes | time | fp32 / bf16 | none (math decomposition) | 0.000 | exact |

The one AMBIGUOUS row is exactly +20 % of that case's attention term. It isolates to
`head_dim = 320/16 = 20`, which is not a multiple of 8: flash attention's tensor-core kernels pad
it to 24, and the registered formula counts the padded width (`24/20 = 1.2`). This is the kernel
counting real, if wasted, padding compute, and only occurs in bf16. Under the pinned
`enable_amp=False`, every shape dispatched to `efficient_attention` and counted exactly, so
**`FlopCounterMode` is a valid GPU cross-check for this study.** Time attention passes an additive
float `attn_mask` (relative-position bias), so SDPA decomposes it into plain matmul + softmax +
matmul in every case: unconditionally exact, but without fused-kernel memory savings (a candidate
MFU optimisation, orthogonal to this study).

**Encoder/decoder findings that shaped the counter**, all now exact and pinned by tests: the
encoder convolves only over the current dataset's fields (`proj1.weight[:, field_indices]`); a
uniform-`n_states` assumption would overestimate encoder cost by 1.4×–3.1×. The decoder swaps the
encoder's `(stride1, stride2)` pair per axis, slices `proj2`'s output by `state_labels`
(`field_indices` minus the 3 BC flags), and on periodic axes circularly pads by `(k − s) // s`
before the transposed conv and crops `k − s` after
([vstride_decoder.py:213-238](../../walrus/models/decoders/vstride_decoder.py#L213-L238)), so it runs on
more positions than the token grid implies. On a 384-pixel axis (stride 6, e.g. TRL2) the decoder
returns 394 where the encoder consumed 398 (the case `vstride_decoder.py:225` flags with a TODO):
the FLOP count is still exact; it is the reconstruction that is lossy.

> **Branch note (`flopsonly`):** this paragraph and the next describe the seeding and determinism work on the `flopcount` branch. `flopsonly` has none of it: no `cfg.seed`, no `seed_all`, no per-epoch re-seeding, and the sampler keeps upstream's fixed `100·epoch + rank` dataset-choice seed, so every run starts from OS-entropy randomness. Read it as a record of `flopcount` (`git show flopcount:LeonardoRunScripts/PT9_2d/SCALING_STRATEGY_2D.md`), not as a description of this branch.

**Seed.** `seed: 0` is a top-level key in `config.yaml`, and `train.py::train` calls
`seed_all(cfg.seed)` immediately before model construction, identically on every rank.
[`seed_all`](../../walrus/utils/experiment_utils.py) seeds torch, numpy **and** python's `random`,
all three of which the training path draws from (torch for weight init, dropout and the jitterer's
rolls; numpy for the periodic roll in `isotropic_model.py`; `random` for
`choose_kernel_size_random`). Two `seed=0` runs give a bit-identical train loss; `seed=1` differs.
After construction the seed is re-derived per rank (`cfg.seed + 1_000_003 · rank`), and at the
top of **every epoch** all RNGs are re-seeded from `(cfg.seed, rank, epoch)` (`Trainer.train`, via
`seed_from` in `experiment_utils.py`). Every random stream of an epoch is therefore a function of
the seed, the rank and the epoch, so a resumed epoch draws exactly what the uninterrupted run
would (`tests/test_determinism.py`; full inventory in
[walrus_sampling_and_repetition.md](../../mydocs/walrus_sampling_and_repetition.md) §8).

Seeding identically on every rank also closes a cross-rank hole. Each process's default torch
generator is otherwise seeded from OS entropy, and `distribute_model` builds FSDP without
`sync_module_states`, so nothing broadcasts rank 0's weights. Under `ddp` (`NO_SHARD`) that is a
correctness bug: every rank keeps a different full replica forever, and the averaged gradient is
not the gradient of any single model. Under `fsdp`/`hsdp` it was only a reproducibility gap (one
coherent but unreproducible random init). This study uses `fsdp`.

**What the seed controls in the data.** Both levels of sampling now follow `cfg.seed`:
- which dataset fires at each step is `BatchedMultisetSampler`'s own generator, keyed on
  `(seed, epoch, sampling_rank)`;
- which window fires *within* it is a per-dataset generator keyed on `(seed, global rank,
  epoch, dataset)`.

The sampling rank is 0 on every GPU under fsdp, so all GPUs share the dataset sequence; the
global rank keeps their windows independent. Before this change (commit `656c00c` and earlier) the
dataset-choice seed was `100·epoch + 10·0 + rank`, so the **dataset-selection pattern was
identical across seeds**, and the window permutations drew their seeds from the global generator,
so they depended on every earlier forward pass.

**PT9 re-validation, done:**

- ✅ **Closed-form FLOPs re-validated on the nine pretraining datasets** (2026-10-02) by
  [`validate_flops_pt9.py`](validate_flops_pt9.py), the PT9
  counterpart of [`validate_flops_real_data.py`](../validate_flops_real_data.py),
  which still loops over smallest7. Rather than per-dataset configs, it builds the datamodule from the
  §4.1 substrate itself (`override_dimensionality=3`, `n_steps_input=6`, the acoustic density
  transforms), so each dataset is checked with the geometry, field list and encoder routing that
  training uses. Results: encoder **and** decoder FLOPs exact against `FlopCounterMode` on all
  nine; observed token grids equal `analytic_token_shape` (33×33 for open/wall 2D, 32×32 fully
  periodic, 33×32 for PS and RB); `n_states = 67`; one encoder key, `"3"`. This covers EMQ-O/EMQ-P
  (512², open and fully periodic), RB (512×128, periodic + wall), SF (256×512) and AI/AM, which had
  never been checked on real data, including the decoder's periodic-padding path. Processor terms
  are covered by the GPU backend test above (CPU SDPA has no FLOP formula).
  The run takes ≈ 4 min on a login node, most of it datamodule construction.

### 5.2 Roofline: is cost FLOP-proportional or dataloader-bound?

**The single most important unknown.** SoViT's GFLOPs ↔ core-hours proxy holds only if cost is
FLOP-proportional. If small models instead sit on a dataloader floor, the grid sweep becomes
unaffordable and FLOP-based scaling recipes mis-price small models on this system.

**Reference timings** (fullDims, measured on older mixtures, so only estimates for this one):

| run | nodes | median s/epoch |
|---|---|---|
| `smallest7_2d_PT_BS2000_E100_N1` (fullDims, 7×2D) | 1 | 6981 |
| `smallest7_2d_PT_BS2000_E100_N4` (fullDims, 7×2D) | 4 | 7503 |
| `fullDims_6DS` | 1 | 7488 |
| `fullDims_6DS_N4` | 4 | 7633 |

≈ 1.95 h/epoch ⇒ **≈ 7.8 GPU-h per epoch at 1 node**; 100 epochs ≈ 780 GPU-h (N1) / 3 120 GPU-h
(N4). N1 ≈ N4 confirms `max_samples` is per-rank. ≈ 26 TFLOP/s per GPU of processor FLOPs is
plausible for TF32 with `enable_amp=False` and gradient checkpointing, i.e. *probably*
compute-bound. But ≈ 3.5 s per micro-batch is also slow enough to be an I/O floor, and **no
small-model run has ever completed**, so the slope is unmeasured.

**Experiment.** Three shapes × 3 epochs at N1, a `data_workers` variation at the smallest shape,
and a staging test. Fit `sec/epoch = a + b·FLOPs` and report `a` (the floor) and the implied MFU.
The scripts are **generated and ready to submit** in [`Roofline/`](Roofline/) (generator
`gen_pt9_roofline_runs.py`). Every point runs the §4.1 substrate for 3 epochs at N1, `seed=0`;
only shape and `data_workers` vary:

| script | shape | `data_workers` | time cap | role |
|---|---|---:|---|---|
| `p9_roofline_run_0.sh` | fullDims `1408/40/5632` (GC 2) | 10 | 10 h | main family, `sec/epoch = a + b·FLOPs` fit |
| `p9_roofline_run_1.sh` | 200 M anchor `768/12/2048` | 10 | 5 h | main family |
| `p9_roofline_run_2.sh` | small `464/10/1120` | 10 | 5 h | main family |
| `p9_roofline_run_3.sh` / `_4.sh` | small | 8 / 16 | 5 h | dataloader-worker sensitivity |
| `p9_roofline_pair_unstaged.sh` | small, GS + PS only, read from $WORK | 10 | 6 h | paired staging test |
| `p9_roofline_pair_staged.sh` | identical, GS + PS first copied to `/scratch_local` | 10 | 6 h | paired staging test |

Submit with `run_p9_roofline_all.sh` (the main five) and `run_p9_roofline_pair.sh` (the pair,
together, but not in the same burst as the main five, since shared-filesystem contention would bias
the unstaged arm). Both wrappers run `preflight_pt9.py` first and refuse to submit if any HDF5
file fails to open or the config does not resolve to the expected datasets. The generator also
composes every script's full override list before writing it.

Design points specific to this mixture:

- **The dataloader-floor risk is high.** The mixture's largest per-window read is Euler's:
  7 frames × 512² × 5 stored channels × 4 B ≈ 37 MB per window, drawn 2/9 of the time from ≈ 212–215 GB
  HDF5 files. SF (≈ 15 MB) and PS (≈ 11 MB) are also mid-sized. Random access into very large files
  may behave differently from anything the storage benchmarks
  (`storage_bench_walrus.sh`) measured. The roofline must report sec/epoch **and** per-dataset
  data-wait time, not just the aggregate.
- **Per-dataset data-wait comes from the stdout logs, not W&B.** W&B receives only the aggregate
  `avg_data_loading_time`. The per-dataset split exists only in the per-microbatch lines the trainer
  prints every `log_interval` microbatches (dataset name plus data/fwd/bw seconds), so the roofline
  runs use `trainer.log_interval=10` (the upstream default) instead of 50. That is a 1-in-10 sample,
  unbiased because `BatchedMultisetSampler` draws the dataset independently of the batch index.
  [`Roofline/parse_roofline_logs.py`](Roofline/parse_roofline_logs.py)
  produces the GATE A table: train-phase seconds per epoch (validation and checkpointing excluded;
  epoch 1 reported separately) and, per dataset, the median and p90 data-wait. fwd/bw are not
  CUDA-synchronised, so only data-wait and epoch wall time are trustworthy. FLOPs for the fit come
  from W&B's `epoch_flops` after `wandb sync`. As a check, the parser reads the old smallest7
  fullDims N1 logs as 7 598 s per training epoch (8.4 GPU-h, epochs 2+), with a 1.6–1.9 ms median
  data-wait for every dataset, i.e. compute-bound on that mixture.
- **End-of-run full validation is switched off** (`++trainer.debug_mode=True`, §4.1). Otherwise the
  third epoch would run full valid and test passes over all nine datasets, which would overrun the
  time caps and pollute the timing.
- **Staging the whole mixture is infeasible.** The 7-dataset plan copied its whole mixture
  (~653 GB) to node-local `/scratch_local` first. This mixture's train splits total ≈ 1.56 TB
  without Euler and ≈ 5.8 TB with it. Staging is instead tested with a **paired staged-vs-unstaged
  test on a two-dataset sub-mixture**: GS (cheap encoder tier, 128²) + PS (≥512 tier, 256×512).
  That is 264 GB of train data and 330 GB copied, since the staging copies whole dataset
  directories (GS + SF would copy 699 GB). The pair runs the exact PT9 config minus seven entries
  (Hydra `~` deletions), at the roofline-small shape. The staging effect is the difference between
  the paired arms; neither arm's sec/epoch is comparable with the main family's. Booster nodes do
  not advertise node-local capacity (SLURM `TmpDisk=0`), so the staged arm checks free space
  itself and aborts cleanly if the pair does not fit with 10 % headroom; it logs
  `STAGING_SECONDS` separately, since that cost is paid once and not per epoch.
- Time caps are set generously above the ≈ 7.8 GPU-h/epoch fullDims estimate (10 h for 3 epochs);
  auto-resume covers an overflow.

**Decision gate (GATE A):**

| outcome | consequence |
|---|---|
| `a` small (cost ≈ FLOP-proportional) | Proceed with the full plan as budgeted in §11. SoViT's GFLOPs↔core-hours proxy holds; show it with our own version of their Fig. 4 (right). |
| `a` large (dataloader floor dominates for small models) | The 64-point grid sweep becomes unaffordable (64 × 40 × 7.8 ≈ 20 k GPU-h). Contingency: shrink the grid to 3×3×3 = 27, cut grid epochs to 25, stage hot data to node-local disk, and **report the FLOPs↔hours decoupling as a finding**: published FLOP-based recipes mis-price small models on this system. |

Also record whether `enable_amp=True` (bf16) is a large win. It is not adopted mid-study (it must
be identical across all points), but it is a legitimate lever if decided *before* Tier 1. If it is
adopted, see §7.1 on which grid widths a bf16 `FlopCounterMode` reading over-counts.

### 5.3 Reproducibility post-conditions

An automated check over every run's `extended_config.yaml`: `n_states` identical (67), the §4.1
pinned values identical, `seed` recorded (a point whose seed is unknown cannot be re-run, and
re-running is the only way to tell an outlier from a real effect), and the **resolved dataset list
equal to the nine pretraining datasets exactly**, on every pretraining run. Any mismatch
invalidates that point. The dataset-list check is what keeps §2.6's blindness guarantee true.

It is enforced before submission at two points: `preflight_pt9.py` (run by every submit wrapper)
checks the substrate, and each PT9 generator composes every generated script's full override list
and asserts its dataset list, shape, seed and duration before writing the script.

### 5.3b Re-evaluating a checkpoint: what repeats can and cannot tell you

This applies to both `f_up` and `f_down`. **A forward pass is stochastic even in `eval()`**:
`PatchJitterer` is not gated on training mode, and its random roll
(`patch_jitterers.py:148`) is the sole source of the randomness. Measured on real data, with the
model in `eval()`:

| what was done | result |
|---|---|
| same checkpoint, same input, two forward passes, no reseeding | **differ**, max element delta ≈ 0.20–0.30 |
| same checkpoint, both passes reseeded to the same value | **bit-identical** |
| same checkpoint, passes reseeded to different values | differ, same magnitude as above |
| same checkpoint, `model.jitter_patches=False` | identical under *any* seed |

**The trap.** A repeat evaluation that inherits the same `seed` is **bit-identical**, so its
spread is exactly zero. Reporting that zero as "the error on this checkpoint" would be badly
wrong: it measures the determinism of the code, not the uncertainty of the number.

**Protocol.** Decide which of three quantities is wanted; they are not interchangeable, and only
the first is obtainable from one checkpoint:

1. **Eval-path noise** (how much the metric moves under the jitterer's draw). Re-run the same
   checkpoint with *explicitly different* `seed` values (`seed=0,1,2,…`) and take the spread.
   Hold `world_size` fixed while doing it: the per-rank offset `seed + 1_000_003·rank` and the data
   sharding both change with node count, so "same seed, different node count" is not a controlled
   repeat.
2. **Sample uncertainty** (how much the metric depends on *which* validation samples were drawn).
   Seeds cannot give this, because the validation loader uses a fixed generator (§3.4). Bootstrap
   over the per-sample metrics instead (over trajectories, for `f_down`, §3.5).
3. **Run-to-run noise `ε_noise`** (§5.4). Not recoverable from a checkpoint at all: it needs
   separate *training* runs. Eval-path noise is a **lower bound** on it, so it can never justify a
   claim that two shapes differ.

If the goal is a *stable* number rather than a spread, raise
`validation_one_step_ensemble_size` / `validation_full_trajectory_ensemble_size`: those average over
stochastic passes, reducing quantity 1 rather than measuring it. `jitter_patches=False` also
removes it, but it changes model behaviour, so it belongs in a diagnostic, never in a run whose
number is compared with others.

### 5.4 Noise floor `ε_noise`

Three seeds (0, 1, 2) of the 200 M anchor `768/12/2048` on the §4.1 substrate, identical except
for `seed`. The scripts are **generated and ready to submit** in [`NoiseFloor/`](NoiseFloor/)
(`p9_noise_floor_run_{0,1,2}.sh`, generator `gen_pt9_noise_floor_runs.py`, submit with
`run_p9_noise_floor_all.sh`). Report the spread in field-time-averaged VRMSE (and every secondary
metric of §3.4) at each epoch.

> **Branch note (`flopsonly`):** this branch has no `cfg.seed`. Its noise-floor runs are `p9_noise_floor_..._run0..2`, identical apart from their names, so they differ only by unseeded OS-entropy randomness and are not reproducible. The paragraph below describes the seeded design on `flopcount`.

**What `seed` varies.** Weight init, dropout, augmentation, the within-dataset sample order **and
the dataset-selection order** (§5.1). This is Choshen's combined run-to-run figure, including data
order, not an init-only measurement. Since the per-epoch seeding change, a chain's resumes no
longer add noise of their own: a resumed epoch is identical to the uninterrupted one, provided
`data_workers` and the GPU count stay the same across the chain's links. Sweep runs that share a
seed also share their dataset order, so comparisons between them are not affected by dataset-order
noise; comparisons across seeds are.

**Sequencing: all three seeds run to `max_epoch=100` and are read at the same epoch `E ≤ 90`.**
The noise floor must be measured at the duration the sweeps will use, which is not yet known
(§7.0), and seed 0's curve is also the small-model saturation pilot that helps fix it. Running
seed 0 to 100 epochs, fixing `E` from it, and then capping seeds 1–2 at `max_epoch = E` would be
wrong: with `cooldown_epochs=10` the schedule differs from epoch `E − 10` on, so seed 0's
*un-cooled* epoch `E` would be compared with seeds 1–2's *cooled* epoch `E`, and the "noise" would
include a schedule difference. `inv_sqrt_w_sqrt_ramps` does not depend on run length until the
cooldown starts at `max_epoch − 10 = 90`, so an uncapped run's epoch-`E` metrics and checkpoint
are exactly what a run capped at a later `max_epoch` would have produced. Seed 0 continues to 100
as the saturation pilot; seeds 1–2 are cancelled once they pass `E`. They can all be submitted at
once, since the results are identical either way.

- `checkpoint.checkpoint_frequency=10` (default 20). The `f_down` noise floor below finetunes each
  seed's epoch-`E` checkpoint and `E` is not yet known, so any multiple of 10 is usable (≈ 18 GB per
  seed).
- **24 h wall clock.** There is no 200 M-anchor timing yet. At ≈ 0.2–0.5 h per epoch, depending on the
  data floor, 100 epochs is 20–50 h. The submit wrapper queues each seed as a chain of **3 jobs with
  `--dependency=singleton`**: SLURM runs at most one job per name at a time, so the links run back
  to back and each auto-resumes from the last checkpoint. A surplus link resumes past `max_epoch`,
  skips the epoch loop ([training.py:1317](../../walrus/trainer/training.py#L1317)) and exits within
  minutes. To stop a seed early, cancel the chain by name (`scancel -n <job_name>`).
- End-of-run validation is capped like every other (`debug_mode`, §4.1), so all three seeds' metric
  at `E` is the same logged quantity.

**`ε_noise` for `f_down` too.** Finetune each of the three seed checkpoints on each held-out
target under the Tier 7 protocol (12 short runs at the anchor shape). Without it, no RQ5 shape
comparison can be judged against noise.

**Why this is non-negotiable.** Every tier past Tier 0 makes an "is this actually different"
claim:

- **The grid sweep picks `x⁽⁰⁾` as the argmin over 64 points** (§7.1). If seed noise is comparable
  to the gaps between grid points, the "minimum" is as likely to be the luckiest seed as the
  compute-optimal shape. A bad `x⁽⁰⁾` does not look wrong; it silently biases every exponent fit
  downstream.
- **GATE C and the validity checks of §12 use "acceptable ARE"/"agreement" as pass conditions**,
  which are undefined until there is a number to compare against. `ε_noise` is that number.
- **The deviation checks (§8.3) are explicitly gated on it**: they are only worth ≈ 3 100 GPU-h if
  the expected differences exceed `ε_noise`.
- **The head-to-head runs both arms at one shared seed**, so its single number is only informative
  relative to how much it would have moved on a different seed. Without `ε_noise`, "SoWalrus
  matches fullDims at less compute" and "SoWalrus got a lucky draw" are indistinguishable.

Choshen §4 finds seed variation alone moves loss by up to 4 %, while many published pretraining
interventions claim gains of only 4–50 %. Choshen §7 also recommends spending marginal budget on
*more distinct shapes* rather than more seeds per shape, so this is deliberately the **only**
place in the study where seeds are replicated. It cannot be obtained more cheaply from a checkpoint
already in hand: re-evaluating one checkpoint under different seeds only re-samples the eval path
(§5.3b), a lower bound on `ε_noise`.

---

## 6. Tier 1 — hyperparameter reduction

**Goal:** confirm that a single frozen `(lr, weight_decay) = (2e-4, 1e-4)` is safe across the
entire shape range Tiers 2–3 explore, so that shape differences are shape differences and not an
artefact of one end of the range being under- or over-tuned.

### 6.0 The premise: this is not "SoViT's recipe"

SoViT's per-shape LR/WD sweep (their App. C: LR ∈ {1e−3, 3e−4, 1e−4}, WD ∈ {3e−4, 1e−4, 3e−5}
at the largest and smallest architectures) belongs to their multitask-decoder experiment (§4.2,
captioning/VQA), not to shape scaling. Their actual star/grid sweeps (Table 5, App. B.1) used **one
fixed** `(lr = 8e-4, wd = 0.03 × 8e-4)` across the entire shape range, 320-wide to 1968-wide,
with no per-shape tuning.

So there is no SoViT precedent for spending compute on this. The reason to do it anyway is our own:
`lr=2e-4`/`weight_decay=1e-4` ([adam.yaml](../../walrus/configs/optimizer/adam.yaml)) has only ever
been exercised at or near fullDims scale, never at the grid sweep's `hidden_dim=320` floor, and
without µP the optimal LR is known to shift with width.

### 6.1 What is actually a candidate for being size-dependent

| knob | risk | disposition |
|---|---|---|
| `optimizer.lr` | known to shift with width without µP | **checked below** |
| `weight_decay` | couples with LR via effective regularisation strength (`lr × wd`, even decoupled) | **checked below** |
| `trainer.clip_gradient=10` | raw grad norms scale with parameter count | watched for free from the same runs (§6.2) |
| `lr_scheduler.warmup_epochs=10` | fixed across a 6–40 block range; deeper nets typically want more warmup | watched for free from the same runs' early epochs (below) |
| `drop_path=0.05` | depth-dependent regulariser, fixed across depth | not re-opened: a stated confound (§4.1), quantified by the sensitivity ablation |
| `batch_size=2` / `grad_acc_steps=4` | none identified | fixed by design, as SoViT pin one batch size across their whole range |
| `model.gradient_checkpointing_freq` | none | recompute/memory trade-off, not numerics (§4.1) |
| optimizer `eps`/betas | none identified in the literature | not varied |

**Warmup is folded into the LR/WD runs rather than given its own axis.** The runs below are long
enough to clear `warmup_epochs=10` and observe steady-state behaviour, so an early loss or
grad-norm spike as warmup ends would show in the same trace. This catches warmup *insufficiency*,
not a needlessly long warmup; it is a deliberately cheaper substitute for a warmup sweep, not an
equivalent of one.

*Why not µP.* Maximal update parametrisation would let LR transfer across width by construction,
which is better science. But WALRUS's fused `ff/qkv` projection, `RMSGroupNorm`, QK-norm and conv
encoders each need their own µP treatment, and getting one wrong silently breaks equivalence with
the released checkpoint, without any error. That is too much implementation risk for a thesis
timeline. The checks below are the deliberate, cheaper substitute.

### 6.2 Step 1 — check what is already free

Completed fullDims-scale runs exist (`fullDims_6DS`, `smallest7_2d_PT_BS2000_E100_N1` and others),
all at `lr=2e-4`. Pull their train-loss curves first: no divergence, no NaN, no sustained spikes.

This is **partial evidence** for two reasons. First, they trained on `smallest7_2d` / `6DS`, not
this mixture: they still show that `lr=2e-4` is stable at fullDims scale on WALRUS data, but not on
this data. Second, `trainer.gradient_log_level` defaults to 0 and no historical run overrode it, so
no gradient-norm trace exists for any of them. If the loss curves are unambiguous, they close the
large-extreme question at zero cost; if they are anything less, run the large-extreme leg of §6.3
with `gradient_log_level=1`.

### 6.3 Step 2 — cheap dedicated checks at both extremes

**Design: local sensitivity around the pin, not a search for a new optimum.** One factor at a time,
sharing the centre point: 5 runs per extreme instead of a 9-run grid. Decoupled AdamW's WD is
designed to weaken the LR×WD interaction that would justify the missing 4 combinations.

- LR ∈ {1e-4, 2e-4, 4e-4} at `wd=1e-4` (half / pin / double).
- WD ∈ {3e-5, 3e-4} at `lr=2e-4` (≈ 3.3× either side of the pin).

**Small extreme: `320/6/768`**, the grid sweep's actual smallest corner (all three axes at their
floor), not a width-only proxy. It is the cheapest shape in the study, so it gets the check
outright: 5 runs × 20 epochs (10 warmup + 10 steady state: long enough for the warmup read, short
enough that divergence, which shows early, is still caught). `gradient_log_level=1`.

**Large extreme: only if §6.2 is inconclusive, using HalfWalrus `1088/30/4352` as a stand-in for
the star centre.** fullDims costs ≈ 7.8 GPU-h per epoch however few epochs are run, so even a
10-epoch fullDims check across 5 runs would cost ≈ 390 GPU-h. HalfWalrus is the lever that makes
this cheap; a shorter run is a second, smaller lever. Same 5-run design, 20 epochs,
`gradient_log_level=1`.

**Selection criterion**, adapted from SoViT: at both extremes, `(2e-4, 1e-4)`'s loss curve and
gradient-norm trace must be smooth and *not visibly worse* than its neighbours. This validates a
pin; it does not search for a replacement. If an extreme looks unstable, report the
LR-sensitivity ablation (loss difference between adjacent LR values, compared with `ε_noise`)
rather than silently re-picking a global LR, which would invalidate runs already in flight.

*Known residual confound, whatever the outcome.* A single frozen LR across the width range is not
scale-invariant; the widest points may be slightly under-tuned even if nothing looks unstable.
SoViT accepted the same limitation. The sensitivity ablation quantifies it rather than hides it.

**Cost.** Small-extreme leg: 5 runs × 20 epochs at the cheapest shape. Large-extreme leg: 0 GPU-h
if §6.2 is conclusive, otherwise 5 runs × 20 epochs at HalfWalrus. Exact figures await the
roofline's small-shape timings (§5.2); §11 budgets the pessimistic case.

---

## 7. Tiers 2–3 — the two sweeps

### 7.0 Choosing the sweep duration (applies to both tiers)

Neither sweep's `max_epoch` can be borrowed from SoViT: their 500 K / 1 M / 2 M steps translate
*their* FLOP targets into *their* step counts on an effectively unlimited corpus. Duration is
derived here from WALRUS's own structure.

**Why a fixed compute budget is required at all.** `x⁽⁰⁾ = argmin_x f(x, t)` is only well-posed
relative to a stated `t`: narrower/shallower shapes win at small `t`, wider/deeper ones at large
`t`. Both sweeps need target compute levels `t₁ < … < tₖ` fixed before shapes are compared.

**Getting several `tᵢ` without training to each separately.** Run each shape once, long enough to
pass through several `tᵢ`, and read the loss at each from that one trajectory. SoViT do this
informally;
[Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*](https://arxiv.org/abs/2405.18392)
(NeurIPS 2024) formalise it for WSD-style schedules: one long stable-phase run, with cooldowns read
or branched from intermediate checkpoints at each `tᵢ`. That is §7.3's protocol.

**Lower bound on `t₁` — LR warmup.** Only infinite-phase checkpoints are mutually comparable
across eventual run lengths (§1.3c). With `warmup_epochs = 10`, the earliest usable `tᵢ` needs real
margin past epoch 10: **epoch 15–18**.

**Upper bound 1 — cooldown fraction.** Hägele et al. find the cooldown should be **~10–20 % of the
total run** (too short loses the cooldown gain; too long wastes stable-phase compute). With
`cooldown_epochs = 10` fixed:

| `max_epoch` | cooldown fraction | in the 10–20 % band? |
|---|---|---|
| 40 | 25 % | no, too decay-heavy |
| 50 | 20 % | edge, ok |
| 60 | 16.7 % | yes, comfortably |
| 100 | 10 % | edge, ok |

So `max_epoch ∈ [50, 100]`. The same paper also implies that a raw infinite-phase checkpoint at
`tᵢ` **understates** what that compute level can achieve, because cooldown recovers additional
gains. §7.3 step 2 (cooldown branches, checking the gain is shape-independent) is the control for
exactly this, which is why it is required rather than optional.

**Upper bound 2 — small-model capacity saturation.** At fixed model size, loss approaches a
capacity-limited floor as compute grows (`L(N, D) = E + A/N^α + B/D^β`; Kaplan/Hoffmann, reaffirmed
by [*Scaling Laws in the Tiny Regime*](https://arxiv.org/pdf/2603.07365) and
[*Reusing Overtrained Language Models Saturates Scaling*](https://arxiv.org/pdf/2510.06548)).
Smaller grid shapes hit that floor first. If `t_max` is so large that the shapes competitive for
`x⁽⁰⁾` are already flat, the argmin reads noise off a plateau. (Seeing the *smallest* shapes
saturate first is signal, not a bug: it is where the frontier curls away from them.) There is no
WALRUS-specific `α`/`β` yet, so this bound is empirical: **noise-floor seed 0** (§5.4) runs to
100 epochs, comfortably past the whole [50, 100] window, and its loss curve doubles as the
saturation pilot. The 200 M anchor sits just above the grid region (§1.4), so it is a reasonable
but imperfect stand-in for where that scale saturates. The roofline's 3-epoch runs are too short to
show saturation.

**Upper bound 3 — data repetition ceiling: no longer binding.** Muennighoff et al.
([*Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264), NeurIPS 2023)
give landmarks for repeated data: **≤ 4×** negligible degradation, **~16×** a "half-life" of
diminishing returns, **~40×** added compute close to worthless. In the 7-dataset mixture two
datasets crossed 16× before warmup even finished, so no duration could both clear warmup and keep
every dataset informative. In this mixture the worst dataset (PS) is at 6.6× at epoch 60 and 11×
at epoch 100 (§2.5): every dataset stays below the half-life across the entire `[50, 100]` window.
**`max_epoch` is therefore set by the cooldown fraction and the saturation pilot alone.** Every
shape runs the identical schedule, so whatever repetition there is would be a shared bias across
sweep points, not point-to-point noise.

**Synthesis — current status.** `t₁` ≈ epoch 15–18. `t_max` is bounded by the cooldown fraction
(`max_epoch ∈ [50, 100]`) and by wherever the saturation pilot shows saturation, whichever is
tighter. Use 2–4 `tᵢ`, log-spaced between the bounds, each read via a §7.3 cooldown branch.
🔲 **Pending:** submitting the pilot and reading its curve, then fixing **one** derived
`max_epoch` for both sweeps. The current numbers are placeholders and mutually inconsistent: the
grid generator uses 40, this plan's star-sweep cost uses 60, and the old
`gen_star_search_runs.py` hardcodes 101. The noise floor is read at the same epoch (§5.4).

### 7.1 Tier 2: grid sweep → `x⁽⁰⁾`

**Purpose** (SoViT §3): find a *small* architecture on the Pareto frontier at *small* compute. It
absorbs the leading coefficients in `x*_k = O(t^(s_k))`, so the star sweep only has to estimate the
exponents. SoViT warn that a bad `x⁽⁰⁾` "can significantly skew results".

Grid (4 × 4 × 4 = 64 points, ≈ 20 % increments as SoViT prescribe, all widths divisible by 16):

```
hidden_dim       ∈ {320, 384, 464, 560}
processor_blocks ∈ {6, 8, 10, 12}
mlp_dim          ∈ {768, 928, 1120, 1344}
```

**Sizes and costs across the grid** (corners plus one interior point; all 64 rows from
[`shape_cost_table.py`](../shape_cost_table.py), parameters with one encoder/decoder pair, disk at
≈ 12.4 bytes × params). Parameter counts are mixture-independent. The GPU-h columns use a FLOP
proxy calibrated on the 7-dataset mixture, at the 40-epoch placeholder: they are **provisional**
pending GATE A and §7.0, and move only slightly with this mixture's enc+dec mix (§3.1).

| shape `H/L/M` | total params | processor params | enc+dec share | ckpt size | GPU-h/epoch | GPU-h @ 40 ep | vs anchor (params) | vs anchor (FLOPs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 320/6/768 | 46 M | 7 M | 84 % | 0.6 GB | 0.05 | 2 | 0.32× | 0.09× |
| 320/6/1344 | 47 M | 9 M | 81 % | 0.6 GB | 0.07 | 3 | 0.33× | 0.11× |
| 320/12/768 | 53 M | 14 M | 73 % | 0.7 GB | 0.10 | 4 | 0.37× | 0.17× |
| 320/12/1344 | 56 M | 18 M | 69 % | 0.7 GB | 0.13 | 5 | 0.39× | 0.22× |
| 384/8/928 | 55 M | 14 M | 75 % | 0.7 GB | 0.10 | 4 | 0.38× | 0.17× |
| 560/6/768 | 68 M | 19 M | 72 % | 0.8 GB | 0.13 | 5 | 0.48× | 0.22× |
| 560/6/1344 | 71 M | 22 M | 69 % | 0.9 GB | 0.16 | 6 | 0.50× | 0.26× |
| 560/12/768 | 87 M | 38 M | 57 % | 1.1 GB | 0.26 | 10 | 0.61× | 0.44× |
| 560/12/1344 | 93 M | 44 M | 53 % | 1.2 GB | 0.30 | 12 | 0.65× | 0.52× |

Sweep totals at 40 epochs: **≈ 365 GPU-h** if FLOP-proportional, **≈ 20 000 GPU-h** under the
dataloader floor (every shape at fullDims's 7.8 GPU-h/epoch, §5.2's GATE A contingency case). This
is the tier most exposed to the floor risk. Disk at 3 retained checkpoints per run: ≈ 152 GB.

**Ballast, on both axes.** 53–84 % of *parameters* across the grid are the fixed encoder/decoder
(the processor is 7 M of a 46 M model at the smallest corner), and 11–45 % of *forward FLOPs*
(29–45 % at the smallest corner, depending on dataset; §3.1). Two consequences: (i) `x⁽⁰⁾` is
chosen by comparing points whose `t` is dominated by a nearly constant term, which compresses the
dynamic range the fit sees; (ii) SoViT's assumption that compute tracks the model is weakest here.
**Fit `x⁽⁰⁾` and the exponents twice, once on total FLOPs and once on processor-only FLOPs, and
report both.** Agreement is a robustness result; disagreement is a finding about conv-stem
foundation models that ViT-derived recipes do not anticipate. If they disagree, the processor-only
fit answers RQ1 (how should the *shape* scale) and the total-FLOP fit answers RQ2 (what should we
*spend*).

**If AMP is ever adopted** (§5.2), only `hidden_dim = 384` of the four grid widths gives an exact
bf16 `FlopCounterMode` reading: bf16 flash attention over-counts space attention by
`(next multiple of 8)/head_dim` when `head_dim = hidden_dim/16` is not 8-aligned (`320→20`,
`464→29`, `560→35`). Under the pinned `enable_amp=False` this does not apply. The analytical
counter that the study actually uses is unaffected either way.

**SoViT's four cautions, applied:**

1. **The optimum must lie strictly in the interior.** If the argmin lands on a boundary, we cannot
   tell whether it is optimal or merely the edge of the box. Extend the grid in that direction and
   re-run those points: one extension round is budgeted (≈ 16 points, ≈ 120 GPU-h).
2. **It must be optimal over a compute *range*, not a point.** Read the argmin at several
   cumulative-FLOP levels along the training curves (free, from per-epoch logging) and pick the
   shape that wins across a band. Where a range applies, take `t⁽⁰⁾` as the **least** compute it is
   optimal for, erring towards larger models as we scale, as SoViT do.
3. **The grid must be tightly packed** (≈ 20 % steps): satisfied above.
4. **Compute must be large enough for power laws to have taken effect.** SoViT insisted on
   > 100 M examples in their grid sweep. At 40 epochs each grid run sees
   `40 × 2000 × 4 × 2 × 6 × N_sp` ≈ 4.1 G tokens, ≈ 1.25× `T_unique_model`: less than two
   aggregate passes, so comfortably met without leaning on repeated data. Re-check once §7.0
   fixes `max_epoch`, and confirm the loss curves are in power-law decay, neither still in the
   early transient nor already flat.

### 7.1b Alternative candidate: the WALRUS authors' own grid

[`ExampleRunScripts/gen_grid_search.py`](../../ExampleRunScripts/gen_grid_search.py) (and its
`_gpuxl` twin) is the WALRUS authors' own shape grid search (`wandb_project_name="MPPX_Scaling_Grid"`).
It emits exactly 64 scripts, a full 4×4×4 factorial:

```
hidden_dim       ∈ {768, 928, 1088, 1360}
processor_blocks ∈ {6, 8, 10, 12}
mlp_dim          ∈ {416, 512, 608, 768}
```

It shares §7.1's `processor_blocks` set exactly, and its `mlp_dim` ceiling (768) is §7.1's floor.
Where they differ is scale (same provenance and caveats as the §7.1 table):

| shape `H/L/M` | total params | processor params | enc+dec share | ckpt size | GPU-h/epoch | GPU-h @ 40 ep | vs anchor (params) | vs anchor (FLOPs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 768/6/416 | 90 M | 31 M | 65 % | 1.1 GB | 0.21 | 8 | 0.63× | 0.35× |
| 768/6/768 | 92 M | 34 M | 64 % | 1.1 GB | 0.23 | 9 | 0.64× | 0.38× |
| 768/12/416 | 121 M | 62 M | 48 % | 1.5 GB | 0.41 | 16 | 0.84× | 0.69× |
| 768/12/768 | 126 M | 67 M | 47 % | 1.6 GB | 0.45 | 18 | 0.88× | 0.76× |
| 928/8/512 | 127 M | 61 M | 52 % | 1.6 GB | 0.40 | 16 | 0.88× | 0.67× |
| 1360/6/416 | 179 M | 94 M | 48 % | 2.2 GB | 0.60 | 24 | 1.25× | 1.02× |
| 1360/6/768 | 184 M | 98 M | 47 % | 2.3 GB | 0.64 | 25 | 1.28× | 1.08× |
| 1360/12/416 | 273 M | 188 M | 31 % | 3.4 GB | 1.19 | 48 | 1.90× | 2.02× |
| 1360/12/768 | 282 M | 197 M | 30 % | 3.5 GB | 1.26 | 50 | 1.96× | 2.14× |

| | §7.1 grid | authors' grid |
|---|---|---|
| params range | 46 → 93 M | 90 → 282 M |
| vs. the 200 M anchor | 0.32×–0.65×, entirely **below** it | 0.63×–1.96×, **brackets** it |
| enc+dec parameter ballast | 53–84 % | 30–65 % |
| GPU-h, FLOP-proportional | ≈ 365 | ≈ 1 473 (**4×**) |
| GPU-h, dataloader floor | ≈ 20 000 | ≈ 20 000 (**identical**) |
| disk @ 3 ckpts/run | ≈ 152 GB | ≈ 380 GB |

Four things decide between them:

1. **The cost difference only exists in one GATE A outcome.** FLOP-proportional: the authors'
   grid is 4× dearer. Dataloader floor: both cost the same, because cost stops depending on shape.
   So the choice cannot be made before GATE A.
2. **Disk may bind before GPU-h.** 380 GB is a large share of $FAST, which must also hold 18 star
   runs (§4.2).
3. **SoViT want `x⁽⁰⁾` at *small* compute, which favours §7.1's grid.** `t⁽⁰⁾` sets the
   extrapolation base; moving `x⁽⁰⁾` into the authors' range raises `t⁽⁰⁾` by roughly the FLOP
   ratio and cuts `τ` (§8.1) from ≈ 355 to order 100, a weaker extrapolation claim.
4. **But the authors' grid has better-conditioned shape signal.** At 30–65 % ballast versus
   53–84 %, more of each point's compute is the processor being varied. It also probes `mlp/hidden`
   ratios of 0.31×–1.0×, far narrower than any WALRUS reference config (fullDims and HalfWalrus
   4.0×, the anchor 2.67×) and narrower than §7.1's 1.4×–4.2×. That matters because SoViT found
   `mlp_dim` carries the *largest* exponent (0.60), so small-compute-optimal shapes are expected to
   have relatively narrow MLPs.

🔲 **Not decided.** Revisit at GATE A with the roofline's cost model in hand. If §7.1's argmin lands
on its upper boundary (caution 1), the authors' grid is the natural extension region, since it
starts where §7.1's `mlp_dim` ends and shares its `processor_blocks` set.

### 7.2 Tier 3: star sweep → `s_k`

**Star centre** `x⁽ᶜ⁾ = (1408, 40, 5632)` = fullDims. As in SoViT, the centre is **never used to
fit**: it is the held-out extrapolation check (their violet point).

Grids at the relative offsets SoViT used (0.2–0.85 of the centre, exponentially spaced, ratio
≈ 1.22), with the other two dimensions **pinned at the centre value** so they cannot bottleneck
(parameters with one encoder/decoder pair; GPU-h provisional, from the 7-dataset FLOP proxy):

| axis | grid | others pinned at | params range | GPU-h @ 60 ep, N1 |
|---|---|---|---|---|
| `hidden_dim` | 448, 544, 672, 816, 992, 1184 | `L=40, M=5632` | 260 M → 927 M | 97 → 361 |
| `processor_blocks` | 8, 10, 12, 15, 19, 24 | `H=1408, M=5632` | 310 M → 754 M | 94 → 281 |
| `mlp_dim` | 1024, 1280, 1600, 2000, 2432, 2816 | `H=1408, L=40` | 809 M → 961 M | 319 → 377 |

18 runs × `max_epoch` (60 is the current placeholder, §7.0) at N1 ≈ **4 343 GPU-h**. These are large
models by design; that is the expensive part of SoViT's method, since two of the three dimensions
are always at full scale. Note how flat the `mlp_dim` row is: the fused `ff/qkv` projection is
only one of several `O(H²)` terms, so shrinking `M` from 5632 to 1024 removes only ~32 % of the
FLOPs. That compresses the dynamic range on that axis and risks the precision of `s_mlp` (§12).

**Durations.** Many compute levels are read from each single long run's per-epoch metric history
(§7.3): SoViT's *structure* (several compute levels per point) with the Choshen economy, not their
specific step counts. At a 60-epoch `max_epoch` the usable infinite-phase band is epochs ~15–50, a
≈ 3.3× range in `t`, similar to SoViT's 4×; treat both band and cost as provisional until §7.0
resolves.

**GATE C's held-out centre point is not free.** The existing fullDims runs (`fullDims_6DS*`,
`smallest7_2d_PT_*`) trained on other mixtures and cannot be compared with star points trained on
this one. **A fullDims run on `Leonardo_pretrain9_2d` to the star sweep's `max_epoch` is
required**: ≈ 7.8 GPU-h/epoch × 60 ≈ **470 GPU-h at N1** (old-mixture timing, provisional). It
doubles as Arm A's early trajectory if the head-to-head is run at N1, and as the fullDims point for
Tier 7.

### 7.3 Intermediate checkpoints (Choshen §6, and RQ4)

Choshen's headline recommendation is *"Estimate scaling laws from intermediate checkpoints, not
just fully trained models"*, with two provisos: **drop the earliest part of training** (their
first-10B-tokens cut moved ARE from > 15 % to 4–10 %), and beware that LR-schedule effects are
unmodelled. Protocol:

1. **Fit only on infinite-phase points:** `epoch ∈ [warmup + 5, max_epoch − cooldown]`, i.e.
   `[15, 50]` at a 60-epoch `max_epoch` (recompute once §7.0 settles it). This drops the warmup
   ramp, Choshen's early transient *and* the cooldown, so every fitted point comes from the same
   length-independent LR trajectory (§1.3c). That property is what makes the
   intermediate-checkpoint shortcut defensible here rather than merely convenient.
2. **Measure the cooldown gain separately.** Fork a 10-epoch cooldown from two infinite-phase
   checkpoints (e.g. epochs 25 and 40 at a 60-epoch `max_epoch`) of 3 star runs spanning the shape
   range, via `experiment.checkpoint_override` pointed at `checkpoints/<name>/step_N`, with
   `trainer.max_epoch` set so the run starts immediately in the off-ramp. Those runs need
   `checkpoint_frequency` dense enough to have the fork points (§4.2). If the gain
   `Δ = L_cooled(E) − L_infinite(E)` is approximately shape-independent, the *argmin over shape*,
   which is all `x*_k` depends on, is unaffected. ≈ 12 short runs, ≈ 250 GPU-h. Hägele et al. show
   raw infinite-phase checkpoints systematically understate what a cooldown would reach, so this
   check is what licenses comparing shapes on uncooled checkpoints at all. If `Δ` turns out to be
   shape-*dependent*, fall back to end-of-run-only fitting: less precise, still valid.
   > Functional check first: verify on the debug config that resuming from an arbitrary `step_N`
   > with a modified `max_epoch` restores the scheduler state and enters the off-ramp.
3. **RQ4.** Compare exponents fitted from (a) infinite-phase intermediate points only, (b)
   end-of-run points only, (c) both, and report ARE against the held-out star centre for each. This
   replicates Choshen's Fig. 4 on a new modality: a publishable secondary result in its own right.

---

## 8. Tiers 4–5 — prediction and the head-to-head

### 8.1 Deriving the shape

With `s_k` from §7.2 and `x⁽⁰⁾, t⁽⁰⁾` from §7.1:

```
τ = t_target / t⁽⁰⁾ ;    x_k(t_target) = x⁽⁰⁾_k · τ^(s_k / 3)
```

**`t_target` = the FLOPs consumed by fullDims trained for 100 epochs on 4 nodes**
(`world_size = 16`, ≈ 3 120 GPU-h), the largest paper-faithful run the budget supports. Matching
is on **total** FLOPs, not processor FLOPs: fullDims carries 1.4–1.7 % enc+dec, the predicted
SoWalrus shape 3.9–5.6 % (§3.1). Matching on processor FLOPs alone would hand the SoWalrus arm a few
percent more real compute at nominal parity, a bias in favour of the arm we hope wins, which is
the one direction RQ2 cannot afford. `t_target` is stated on measured total FLOPs (`cumulative_flops`)
and τ solved accordingly.

**Worked example**, using SoViT's *own* exponents (0.22 / 0.45 / 0.60) as placeholders to show
what the machinery produces and check feasibility. Shapes depend only on the exponents and on
`τ`; parameters are with one encoder/decoder pair:

| `t_target` | τ | predicted shape | params | FLOPs/µb vs fullDims | epochs at matched FLOPs |
|---|---|---|---|---|---|
| fullDims 100 ep N1 | 89 | (640, 20, 2752) | 172 M | 0.12× | 847 |
| **fullDims 100 ep N4** | **355** | **(720, 24, 3616)** | **250 M** | **0.19×** | **536** |
| fullDims 200 ep N4 | 709 | (752, 27, 4160) | 307 M | 0.24× | 844 |

So the recipe's prior expectation is a **~4× smaller model trained ~5× longer**, structurally the
same finding as SoViT-400m/14 vs ViT-g/14. `τ ≈ 355` is an extrapolation of comparable audacity to
SoViT's (their SoViT-400m sits ≈ 750× above their `x⁽⁰⁾`). Our measured `s_k` will differ: WALRUS's
factorised space/time processor and conv encoder/decoder have no ViT counterpart.

### 8.2 The repetition confound

At matched FLOPs the small arm runs ~5× more epochs, so it sees ~5× more *repeated* data:
aggregate `R` ≈ 12 for the fullDims arm vs ≈ 66 for a ≈ 5.4×-longer SoWalrus arm, and the most
repeated dataset, PS, goes from ≈ 44× to ≈ 236× (§2.5). If SoWalrus loses, we cannot immediately
tell whether the shape recipe is wrong or the extra repetition stopped paying. SoViT never face
this, because JFT-3B is effectively inexhaustible.

Mitigations, in order of preference:

1. **Use the RQ3 curve (§9) to bound it.** If tokens still carry near-full value out to the
   SoWalrus arm's repetition, the confound is quantified away. If not, report the compute-optimal
   shape *subject to the measured data constraint*, a more honest and arguably more interesting
   result than SoViT's.
2. **Cap `t_target`.** If RQ3 shows decay biting first, fall back to the `fullDims 100 ep N1` target
   (τ = 89, 847 epochs at N1 for the small arm: aggregate `R` ≈ 26, PS ≈ 93×) and accept the
   shorter extrapolation.
3. **Report `R` and `R_d` on every point**, so a reader can re-derive the comparison under their own
   assumptions.

The ordering is deliberate: RQ3 must report *before* `t_target` is fixed (GATE B).

### 8.3 The experiment

- **Arm A:** fullDims `(1408, 40, 5632)`, 100 epochs, N4. ≈ 3 120 GPU-h.
- **Arm B:** SoWalrus `(H*, L*, M*)` from §8.1, epochs set so total FLOPs match Arm A exactly (not
  approximately: computed from the §5.1 counter). ≈ 3 120 GPU-h.
- Both from the same seed; both evaluated on the `test` split with full validation, reporting
  per-dataset one-step and rollout VRMSE/NRMSE, plus GPU-hours and inference cost per sample
  (SoViT's Fig. 2 also sells the inference win).
- **Plus the downstream arm:** finetune both final checkpoints on all four held-out targets under
  the Tier 7 protocol (§10), in both data regimes, and report `f_down` per target with bootstrap
  CIs. RQ2 is answered twice. "SoWalrus wins upstream but loses on held-out physics" is a
  publishable, not a failed, outcome.

**A cheaper mid-scale sanity check, run first.** Before committing 6 240 GPU-h, evaluate the
fitted law at HalfWalrus-equivalent compute and check that its prediction is consistent with the
star centre's held-out performance. If the law cannot predict the star centre to within a few
percent ARE, do not proceed: refit or extend the sweeps instead.

**Deviation check (SoViT Fig. 6; optional, budget permitting, ≈ 3 100 GPU-h).** From the derived
SoWalrus shape, perturb one dimension at a time by −33 % / +33 % / +200 %, holding total FLOPs fixed
by adjusting training duration, and retrain. SoViT found width most sensitive (it has the smallest
exponent). Only worth running if the expected differences exceed `ε_noise` (§5.4); check that
*before* spending.

---

## 9. Tier 6 — RQ3: the repeated-data sub-study

**Question:** over what range does `T_seen` behave like unique data?

**Design.** Fix the shape at the 200 M anchor `768/12/2048`, and vary unique data independently of
compute by restricting the training set, which subsets deterministically given
`restriction_seed`. At full data the anchor sees aggregate `R` ≈ 1.9 at 60 epochs and per-dataset
`R_d` ≤ 6.6. Restricting by fractions 1 → 1/8 would only reach `R_d` ≈ 50 on PS, below the
head-to-head's ≈ 236×, which is the regime the answer is needed for. The restriction has to go
deeper.

**Granularity floor.** `restrict_train_num_trajectories` applies **per sub-dataset**
([multidatamodule.py:155-158](../../walrus/data/multidatamodule.py#L155-L158) → each
`BatchInflatedWellDataset`). PS has only 96 train trajectories: 1/16 leaves 6, 1/64 leaves 1–2,
which is no longer "the same dataset, less of it". For the deepest level use
`restrict_train_num_samples` (window-level) instead. It keeps trajectory diversity and thins
windows, which is a *different* kind of restriction. Before relying on it, confirm from
`multidataset.py` how the samples are chosen (random windows vs contiguous blocks).

| level | knob | fraction | aggregate `R` at 60 ep | PS `R_d` | GS `R_d` |
|---|---|---|---:|---:|---:|
| 0 | — | 1 | ≈ 1.9 | 6.6 | 0.7 |
| 1 | trajectories | 1/4 | ≈ 7.4 | 26 | 2.7 |
| 2 | trajectories | 1/16 | ≈ 30 | 106 | 11 |
| 3 | samples | 1/64 | ≈ 119 | 423 | 44 |

Each level is run at 30 and 60 epochs, giving 8 runs at the anchor shape and a
`(T_seen, T_unique)` grid at fixed `N` (≈ 320 GPU-h, provisional).

**Analysis.** Fit a Muennighoff-style repeated-token model (effective tokens decaying with repeat
count) and report the repetition count at which a token's marginal value has halved. That number
is what §8.2 consumes, and it is the answer to RQ3. Also break it down per dataset, with no new
runs: the trainer already logs every metric split by dataset (`{dataset}/full_VRMSE_T=all`, further
split by field and rollout-time interval), so the per-dataset curves across the four levels are a
W&B history query against the same 8 runs. Per-dataset `R_d` spans a 10× range at each level, so
every dataset contributes points across the whole curve. A dataset whose curve is already flat at
level 0 is saturated throughout, and its contribution to the sweeps' aggregate `f` should be
flagged as a flat floor rather than scaling signal (§12).

*Ordering constraint:* RQ3 reports before `t_target` is fixed (GATE B). It is cheap, so run it
concurrently with the star sweep.

---

## 10. Tier 7 — RQ5: blind downstream finetuning

### 10.1 Protocol

Reuse the existing finetuning pipeline — `experiment=finetune_leonardo`,
`checkpoint=finetune_leonardo`, and the existing single-dataset configs
[`Leonardo_active_matter`](../../walrus/configs/data/Leonardo_active_matter.yaml),
[`Leonardo_helmholtz_staircase`](../../walrus/configs/data/Leonardo_helmholtz_staircase.yaml),
[`Leonardo_turbulent_radiative_layer_2D`](../../walrus/configs/data/Leonardo_turbulent_radiative_layer_2D.yaml),
[`Leonardo_viscoelastic_instability`](../../walrus/configs/data/Leonardo_viscoelastic_instability.yaml)
— as already exercised by [`fullDims_FT_ActiveMatter.sh`](../fullDims_FT_ActiveMatter.sh).
Pinned identically across every finetune in the study:

- the §4.1 substrate, except `model.drop_path=0.0` and `optimizer.lr=1e-4` (the choices made and
  documented in `fullDims_FT_ActiveMatter.sh`), with `data/field_index_map_override=full_well_field_index`
- **a fixed finetuning budget in micro-batches**, the same for every shape: default 20 epochs at
  `max_samples=500` = 10 000 micro-batches ≈ 5 standard epochs
- `++trainer.skip_spectral_metrics=False`, full validation at the end (§3.5)
- **two data regimes per target**:
  - *scarce*: `restrict_train_num_trajectories` = 16 trajectories per target (≈ 22 % of TRL2's
    72, ≈ 4 % of HS's 416). This is the regime where pretraining effects are largest and where
    shapes are most likely to separate. For VE, restrict per regime file rather than over the
    whole dataset: a random 16 of its 213 trajectories would often contain no AH trajectory at all
    (AH has 16 train trajectories).
  - *full*: the whole train split.
- **field embeddings kept as pretrained** (primary; this is how WALRUS is finetuned), with one
  ablation that reinitialises the rows of fields shared with pretraining. The ablation runs at the
  star centre and at SoWalrus only, on TRL2 (full name overlap) and VE (partial). It separates the
  "physics transfers" and "embedding transfers" channels of §2.6.

### 10.2 Which checkpoints

| set | checkpoints | regimes | purpose |
|---|---|---|---|
| 7a — core | 6 grid-sweep points on the Pareto frontier around `x⁽⁰⁾`; per star axis the smallest, middle and largest point (9) + the centre; both head-to-head arms | scarce | is the upstream-optimal shape also downstream-optimal? (rank agreement) |
| 7a — core | star centre + both head-to-head arms | full | RQ2's downstream arm |
| 7a — core | 2 scratch baselines (the anchor shape and SoWalrus, no pretraining) | scarce + full | transfer gain `g = 1 − VRMSE(ft) / VRMSE(scratch)` |
| 7b — optional | the remaining 9 star points, at end of run and at one intermediate `tᵢ` | scarce | enough points to **fit `s_k` on `f_down`** and compare with `f_up`'s |

7a answers RQ5 as a *ranking* question: does the shape order by `f_down` agree with the order by
`f_up`, within CIs and `ε_noise`? 7b answers it as an *exponent* question, as SoViT's Fig. 5 does,
and costs roughly as much again. The choice between them is made at GATE B, once `f_up`'s exponents
and their CIs are known.

### 10.3 Analysis

- Report `f_down` per target, and aggregated over the three in-family targets (HS, TRL2, VE)
  separately from the out-of-family one (ACT). The two may disagree, and the disagreement is the
  result.
- For each target, test whether the upstream argmin shape is within noise of the downstream argmin
  at matched pretraining FLOPs.
- For 7b, fit SoViT Eq. 2 with `f_down` as `f` and compare `s_k` with the upstream fit's bootstrap
  CIs.
- Relate the results to the categorisation axes only descriptively: four targets are enough to
  *illustrate* axis effects, not to estimate them. The crossover designs in the categorisation doc
  §5 are the separate study for that.

---

## 11. Budget and sequencing

Costs assume the FLOP-proportional regime; the right column is the pessimistic dataloader-floor
regime (every run pinned at ≈ 7.8 GPU-h/epoch whatever its shape), which §5.2 will discriminate.
All GPU-h figures are provisional: several were first derived at a 20-step validation rollout (now
60, §3.4) or on the 7-dataset mixture, and the roofline re-measures them.

| tier | content | runs | GPU-h (FLOP-prop.) | GPU-h (floor) |
|---|---|---:|---:|---:|
| 0 | FLOP re-validation, roofline (incl. paired staging test) | ~12 short | 400 | 500 |
| 0 | Noise floor, 3 seeds × 60 ep, + 12 `f_down` finetunes | 3 + 12 | 170 | 1 450 |
| 1 | LR × WD sensitivity | 5–10 | 630 | 2 340 |
| 2 | Grid sweep 4×4×4 @ 40 ep (+1 extension round) | 64 (+16) | 580 | 20 000 ✗ |
| 3 | Star sweep 18 × 60 ep | 18 | 4 343 | 8 424 |
| 3 | **Star-centre re-run on the new mixture** (GATE C) | 1 | **470** | 470 |
| 3 | Cooldown-branch validation (RQ4) | 12 short | 250 | 900 |
| 6 | Repetition sub-study (RQ3), 4 levels × 2 durations | 8 | 320 | 3 744 |
| 5 | Head-to-head, 2 arms at matched FLOPs | 2 | 6 240 | 6 240 |
| 7a | **Downstream core** (≈ 20 checkpoints × 4 targets, scarce; 3 × 4 full; scratch) | ≈ 110 short | **≈ 1 500** | ≈ 3 000 |
| — | **Core total** | | **≈ 14 900** | **≈ 47 000** |
| 7b | Downstream exponent fit (optional, GATE B) | ≈ 72 short | ≈ 1 300 | ≈ 2 800 |
| 5 | Deviation checks (optional) | 4 | 3 120 | 3 120 |
| — | Contingency / reruns (~25 %) | | 3 700 | — |
| — | **Planned total** | | **≈ 23 000** | — |

The Tier 7a estimate uses the star points' per-epoch costs (≈ 1.6–6.3 GPU-h/epoch at N1),
5 epoch-equivalents per finetune, the grid points at ≈ 0.1–0.3 GPU-h/epoch, and fullDims at 7.8.
It is dominated by the fullDims-scale checkpoints (star centre, Arm A).

≈ 23 k GPU-h sits inside the ~30 k budget with room for failures. Under the floor regime the plan
does **not** fit, and the pre-committed contingency is: grid sweep → 3×3×3 = 27 points at 25
epochs, noise floor → 30 epochs, RQ3 → 4 runs, deviation checks and Tier 7b dropped.

### Sequencing and gates

```
Euler download complete ──► re-survey (§2.1) ──► FLOP re-validation on 9 sets     ✅ 2026-10-02
        │
        ▼
T0 roofline ──[GATE A: cost model]──► T1 HP sensitivity
                                              │
                           ┌──────────────────┴──────────────────┐
                           ▼                                     ▼
                 T2 grid sweep → x⁽⁰⁾                   T6 repetition (RQ3)
                           │                                     │
                           ▼                                     │
                 T3 star sweep → s_k  +  star-centre re-run      │
                 T3 cooldown branches (RQ4)                      │
                           │                                     │
                           └────────────► [GATE B] ◄─────────────┘
                                             │   (also: choose 7a only vs 7a + 7b)
                                             ▼
                         predict SoWalrus + verify against star centre
                                             │
                                  [GATE C: ARE vs centre]
                                             ▼
                              T5 head-to-head  +  T7 downstream (RQ5, RQ2 downstream arm)
```

The noise floor (§5.4) runs alongside T0/T1. Tier 7a's grid- and star-point finetunes can start as
soon as their pretraining runs finish; they do not wait for GATE C.

- **Prerequisite (before GATE A) — met 2026-10-02:** Euler fully downloaded and every file
  readable, the re-survey replacing every estimated figure (§2.1), and `validate_flops_pt9.py`
  exact on all nine datasets (§5.1). The roofline and noise floor can be submitted.
- **GATE A** — cost model known; sweep sizes and contingency chosen (§5.2).
- **GATE B** — `s_k` fitted, `x⁽⁰⁾` interior-verified, repetition decay measured; `t_target` fixed
  (§8.2); Tier 7b decided (§10.2).
- **GATE C** — the fitted law predicts the *held-out star centre's* performance to acceptable ARE,
  judged against `ε_noise`. Do not spend 6 240 GPU-h on a law that cannot predict a point we
  already have. SoViT's Fig. 4 (left) is exactly this check.

**Practical scheduling.** `boost_usr_prod` caps jobs at 24 h. Long runs are chains of jobs with
`--dependency=singleton` (one job per name at a time), relying on `auto_resume=True` and
`wandb_run_id.txt` to join the history (§4.3); the 60-epoch star runs need 2–6 links each. The
≈ 536-epoch SoWalrus arm is ≈ 8 days of N4 wall clock (~9 chained links): confirm the
resume-and-continue path end to end (`test_resume_wandb.sh`) before starting it.

---

## 12. Analysis protocol and deliverables

**Fitting.** For each axis `k`, fit SoViT Eq. 2 by minimising *relative* error (as SoViT do, not
absolute), over all `(x_k, t, f)` triples from the infinite-phase intermediate checkpoints of that
axis's six star runs. Report `a_k, b_k, c, ξ_k, ε_k` and the derived `s_k = c/(b_k + a_k)`, each
with bootstrap confidence intervals over runs. In Tier 7b the same applies to `f_down`, with the
trajectory-bootstrap CIs of §3.5 in place of window-level ones.

**Fit hygiene: screen for saturation before fitting.** Some checkpoints, especially the smallest
grid shapes at the largest `tᵢ`, may already be flat against their own capacity floor (§7.0).
[*(Mis)Fitting: A Survey of Scaling Laws*](https://proceedings.iclr.cc/paper_files/paper/2025/file/7f77492bb8070a5c825a87c0c5181da2-Paper-Conference.pdf)
(ICLR 2025) documents this as a routine failure mode: the best fit over *all* observed points is
often not the best fit for the region being extrapolated to. Before fitting, plot each shape's loss
against cumulative FLOPs and residual-check for flattening; exclude or flag saturated points rather
than including them by default. Use a robust loss (Huber, not plain least squares). Datasets
flagged as already saturated by RQ3 (§9) are noted alongside the fits.

**Three pre-registered validity checks**, mirroring the source papers:

1. **Metric robustness** (SoViT Fig. 5). Refit `s_k` under each secondary metric of §3.4, and under
   the family-balanced `f_up`. SoViT found `a_k, b_k, c` move but `s_k` — a ratio — stays put. If
   ours does not, that is a finding, and it undermines the single-metric choice.
2. **`c` should be shape-independent** (SoViT §4.1: `c = Θ(1/d)` for intrinsic data-manifold
   dimension `d`). SoViT got `c ≈ 0.65 ± 0.06` across all three axes. Agreement across our axes is
   strong evidence the functional form suits this modality; disagreement is evidence it does not.
   For a physics mixture with a plausibly low-dimensional attractor, the value of `c` is
   independently interesting.
3. **Held-out extrapolation to the star centre** (SoViT Fig. 4 left), using the re-run fullDims
   point of §7.2. This is GATE C.

**Known precision risk.** The `mlp_dim` axis spans only ≈ 0.68–0.81× of the centre's FLOPs (§7.2),
because `M` is one of several `O(H²)` terms, versus ≈ 0.21–0.77× for `hidden_dim`. A narrow
compute range makes `b_mlp`, and hence `s_mlp`, the least well-determined exponent. If its bootstrap
CI is too wide to be useful, extend that grid downward (`M ∈ {512, 640, 800}`) rather than
over-claim.

**Reporting.** Every table carries: shape, total and processor params, `T_seen`, aggregate `R`
and max `R_d`, measured FLOPs (total and processor-only), GPU-hours, `f_up` with its `ε_noise`
band, and — where measured — `f_down` per target with its CI. Every claim of a difference is
stated relative to `ε_noise`.

**Deliverables — status in this pass:**

- ✅ `walrus/utils/flops.py` + `FlopAccountant` + `seed_all` (§5.1), with `tests/test_flops.py`,
  `tests/test_seed.py`, resume of the counters across checkpoints, and the 2026-10-06 GPU
  end-to-end test.
- ✅ `walrus/configs/data/Leonardo_pretrain9_2d.yaml` — the pretraining mixture, composition
  verified.
- ✅ this document.
- ✅ Re-survey of the Euler datasets after download, and the resulting table updates (§2.1, §2.5,
  §3.2, §3.4), 2026-10-02.
- ✅ FLOP re-validation on the nine pretraining datasets: `validate_flops_pt9.py`, all exact (§5.1).
- ✅ `LeonardoRunScripts/PT9_2d/`: shared `pt9_substrate.py` (the §4.1 pins, once), `preflight_pt9.py`;
  `Roofline/` (5 main scripts, the paired staging test, `parse_roofline_logs.py`, §5.2) and
  `NoiseFloor/` (3 seeds with singleton resubmission chains, §5.4). Generated, compose-checked,
  not yet submitted.
- 🔲 Grid and star generators on `pt9_substrate.py`. The existing top-level
  `gen_grid_sweep_runs.py` (40-epoch placeholder) and `gen_star_search_runs.py` still target
  `Leonardo_smallest7_2d`. The star one needs a full pass, not just a duration edit: re-centre on
  fullDims (§1.4, §7.2), take the §4.1 pins (it lacks `seed=`, the field-index pin and `fsdp`), and
  replace its hardcoded `max_epoch=101` with the single §7.0 value.
- 🔲 `gen_downstream_runs.py` — Tier 7 finetune generator (§10), parameterised by checkpoint path,
  target, regime and embedding policy.
- 🔲 `gen_repetition_runs.py` — Tier 6 with the levels of §9.
- 🔲 `analysis/` — W&B history export, Eq. 2 fitting, bootstrap CIs, `f_down` extraction with
  trajectory bootstrap, and the figure set.

**Figures** (each an intentional counterpart to a source figure):

1. IsoFLOP curves per axis — SoViT Fig. 4 (middle).
2. Predicted vs actual, with the held-out star centre marked — SoViT Fig. 4 (left).
3. Measured FLOPs vs GPU-hours — SoViT Fig. 4 (right); it may or may not show their 0.99
   correlation, and either way it is a result.
4. Exponents `a, b, c, s` per axis across metrics — SoViT Fig. 5.
5. The efficiency frontier: width / depth / MLP / params vs compute for WALRUS — SoViT Fig. 1.
6. ARE vs fraction of training used and number of models — Choshen Fig. 4, replicated on a PDE
   foundation model.
7. Repetition decay: loss vs `R` at fixed `N`, aggregate and per dataset — the answer to RQ3.
8. Head-to-head at matched FLOPs, with per-dataset breakdown — SoViT Fig. 2 / Fig. 6.
9. Upstream vs downstream shape ranking at matched FLOPs, per held-out target (RQ5, Tier 7a).
10. `s_k` from `f_up` vs `s_k` from `f_down` with CIs, in-family vs out-of-family (RQ5, Tier 7b).
11. Head-to-head per held-out target, both data regimes (RQ2's downstream arm).

---

## 13. Open risks

| risk | mitigation |
|---|---|
| Cost is dataloader-bound, not FLOP-bound | GATE A; pre-committed contingency (§11); staging tested (§5.2); reported as a finding |
| Euler's large per-window reads from ≈ 212–215 GB files raise the dataloader floor | Roofline reports per-dataset data-wait via `parse_roofline_logs.py` (§5.2); if Euler alone is the floor, report it, and consider staging only Euler's hot files as a targeted fix |
| Seed variance swamps shape effects | `ε_noise` measured first, for `f_up` and `f_down`; every claim stated against it (§5.4). Seeds are explicit and reproducible (§5.1), so a suspicious point can be re-run exactly |
| Zero-spread error bars from same-seed repeats | Evaluation is stochastic (the jitterer runs in `eval()`), but a same-seed repeat is bit-identical. §5.3b's protocol: vary `seed` explicitly, hold `world_size` fixed, never quote eval-path spread as run-to-run noise |
| Frozen LR under-tunes the widest shapes | LR-sensitivity ablation quantifies it (§6); acknowledged limitation |
| Grid-sweep optimum lands on a boundary | Budgeted extension round (§7.1); the authors' grid as the natural extension region (§7.1b) |
| `s_mlp` poorly determined by a narrow compute range | Extend the `mlp_dim` grid downward (§12) |
| Cooldown gain turns out shape-*dependent* | Fall back to end-of-run-only fitting; costs precision, not validity (§7.3) |
| Sweep `max_epoch` not yet derived (placeholders 40 / 60 / 101 disagree) | §7.0 derives it from the warmup floor, the cooldown fraction and the saturation pilot; one value for both sweeps |
| Enc/dec FLOP ballast biases the grid-sweep exponents (29–45 % of forward FLOPs at the smallest corner) | Dual fit on total vs processor-only FLOPs, both logged per epoch (§1.3f, §3.1, §7.1) |
| Capped mid-training validation covers 0.03–0.3 % of each valid split | Deterministic subset keeps `ε_noise` clean; full-validation spot-check on saved checkpoints, prioritising GS and Euler (§3.4) |
| GPU-h estimates predate the 60-step validation horizon and this mixture | Treated as provisional until the roofline re-measures (§3.4, §11) |
| `n_states` changes with what is on disk | `full_well_field_index` pin, `n_states = 67` checked as a post-condition (§1.3b, §5.3) |
| End-of-run full validation/test inflates cost and touches `test` | `++trainer.debug_mode=True` on every run; full-split scores only via checkpoint re-evaluation (§4.1) |
| 24 h wall-clock limit vs an 8-day run | Chained `--dependency=singleton` submissions; resume path verified before starting (§11) |
| $FAST (1 TB) fills with checkpoints | Sparse weights, dense metrics; delete `step_*` after each tier (§4.2) |
| Historical fullDims runs no longer usable as GATE C's centre | Budgeted re-run (§7.2) |
| Tiny held-out evaluation sets (TRL2 9 valid trajectories, VE-AH 2) | Full validation, trajectory bootstrap, per-regime VE scoring; claim a shape effect only above both CI and `ε_noise` (§3.5) |
| Shared field names give a transfer channel that is not "physics" | Stated explicitly (§2.6); embedding-reinit ablation (§10.1) |
| Mixture overweights acoustics and compressible flow (3/9 each) | Family-balanced `f_up` reported alongside the dataset-balanced one (§3.4) |
| Deep RQ3 restriction collapses PS to 1–2 trajectories | Window-level restriction for the deepest level, with its different semantics stated (§9) |
| A held-out dataset leaks into pretraining via an override | §5.3 post-condition: resolved dataset list must equal the nine exactly |
| Four targets cannot estimate per-axis transfer effects | Stated as illustrative only; the categorisation doc's crossover study is the instrument for that (§10.3) |
| ~~Euler figures are estimates~~ | **Closed 2026-10-02:** re-surveyed; only the trajectory length was off (101, not 100 steps; +1 % tokens) (§2.1) |
| ~~Closed-form FLOPs never checked on the new geometries~~ | **Closed 2026-10-02:** encoder and decoder exact on all nine (§5.1) |
| ~~`FlopCounterMode` silently drops spatial attention~~ | **Resolved:** CPU-only; exact on GPU under the pinned `enable_amp=False`. A bf16, non-8-aligned-`head_dim` over-count remains only if AMP is adopted (§5.1, §7.1) |
