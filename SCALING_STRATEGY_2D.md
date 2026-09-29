# Compute-optimal shape scaling for WALRUS — 13-dataset 2D substrate with a blind held-out set

This document supersedes [SCALING_STRATEGY.md](SCALING_STRATEGY.md) as the live plan. The method —
SoViT's star/grid shape-optimisation recipe (Alabdulmohsin et al. 2023), Choshen's
intermediate-checkpoint economy, the Tier 0–6 structure and its gates — is unchanged. What changed
is the data. The old plan pretrained on `Leonardo_smallest7_2d`, i.e. whichever seven 2D datasets
had happened to be downloaded. Thirteen 2D Well datasets are now on disk, so this version chooses
the pretraining mixture deliberately and **holds four datasets out entirely** for blind downstream
finetuning.

Anything below that does not depend on the data is carried over by pointer rather than re-pasted:
the old document remains the record of *how* each piece was built and validated. Sections are
numbered to match the old document, so `§7.0` here and `§7.0` there are about the same thing.

| decision | value |
|---|---|
| Compute budget | ~30K hours on Boost |
| Pretraining mixture | 9 datasets: AD, AI, AM, EMQ-O, EMQ-P, GS, PS, RB, SF (`data=Leonardo_pretrain9_2d`) |
| Held out (never seen in pretraining) | 4 datasets: `active_matter`, `helmholtz_staircase`, `turbulent_radiative_layer_2D`, `viscoelastic_instability` |
| Upstream metric `f_up` | Held-out-trajectory 60-step rollout VRMSE on the `valid` split of the 9 pretraining datasets |
| Downstream metric `f_down` | Finetuned VRMSE on the 4 held-out datasets (§3.5, Tier 7) |
| Shape axes `D` | `hidden_dim`, `processor_blocks`, `mlp_dim` |
| HP reduction | LR×WD sensitivity at the extreme shapes only (§6 of the old doc), then frozen |
| Validating endpoint | Head-to-head against fullDims at **matched FLOPs**, upstream *and* downstream |
| Data repetition | Measured as an explicit sub-study (RQ3), redesigned for the new, much lower repetition regime |

Dataset codes used throughout: AD/AI/AM = `acoustic_scattering_{discontinuous,inclusions,maze}`,
EMQ-O/EMQ-P = `euler_multi_quadrants_{openBC,periodicBC}`, GS = `gray_scott_reaction_diffusion`,
PS = `planetswe`, RB = `rayleigh_benard`, SF = `shear_flow`, HS = `helmholtz_staircase`,
ACT = `active_matter`, TRL2 = `turbulent_radiative_layer_2D`, VE = `viscoelastic_instability`.
Family numerals (I–VI) and axis definitions are those of
[well_walrus_dataset_categorisation.md](well_walrus_dataset_categorisation.md) §2.2 and §5.

---

## 0. What changed relative to SCALING_STRATEGY.md

| item | old plan | this plan | consequence |
|---|---|---|---|
| Pretraining data | `Leonardo_smallest7_2d` (7 datasets, by availability) | `Leonardo_pretrain9_2d` (9 datasets, by design, §2) | every data-dependent number is redone below |
| Held-out physics | none; transfer to unseen physics explicitly untested (old §3.4) | 4 datasets spanning every categorisation axis (§2.3) | new RQ5 and Tier 7 |
| Unique train tokens `T_unique_model` | ≈ 1.3 G | **≈ 3.25 G** (≈ 2.4 G measured + ≈ 0.85 G Euler, estimated) | 2.5× more distinct data |
| Worst per-dataset repetition at 60 ep, N1 | ≈ 122× (VE) | **6.6×** (PS) | old §7.0 "upper bound 3" no longer binds (§7.0) |
| Spread in per-dataset repetition | 143× | 10× | RQ3's aggregate `R` is no longer one dataset's number (§9) |
| GATE C star-centre point | existing fullDims runs, "zero marginal cost" | **must be re-run** on the new mixture (§7.2) | ≈ +470 GPU-h |
| `roofline_staged.sh` | stage the mixture (~653 GB) to node-local disk | infeasible: train splits alone are ≈ 5.8 TB with Euler (§5.2) | redesigned as a paired subset test |
| Tier 0 scripts (roofline, noise floor) | built on smallest7, not submitted | must be regenerated with the new data config | no sunk cost: none were run |
| Euler availability | not on disk | downloading (both variants, ≈ 2.65 TB each) | **blocks every Tier 0 submission** until the download completes and is re-surveyed |

**Nothing already executed is invalidated by this change**, because no Tier 0–6 run of the old plan
was ever submitted (no `roofline_*` / `noise_floor_*` logs exist under
`/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs`). What *is* lost is the reuse of the historical
fullDims / `smallest7_2d_PT_*` runs as free data points: they trained on a different mixture.

---

## 1. Research questions

**RQ1 (primary).** How should WALRUS's shape dimensions — `hidden_dim`, `processor_blocks`,
`mlp_dim` — be scaled jointly as pretraining compute increases, for a cross-domain
continuum-dynamics foundation model? *(Unchanged.)*

**RQ2 (validating).** Does a shape-optimised WALRUS derived from RQ1 match or beat the paper-scale
configuration (`H=1408, L=40, M=5632`) when both are pretrained with the same FLOPs — measured
upstream *and* on the held-out physics? *(Extended with the downstream arm.)*

**RQ3 (sub-study).** Over what range does the "unlimited data" assumption behind SoViT's Eq. 2 hold
for this mixture, and how does repetition change the compute-optimal answer? *(Reframed: at sweep
scale the new mixture is only mildly repeated, ≤ 6.6× per dataset, so the question now matters
mainly at head-to-head scale, where the small arm still reaches ≈ 240× on PS — §2.5, §9.)*

**RQ4 (methodological).** Do intermediate checkpoints of WALRUS's WSD-shaped schedule reproduce the
exponents from end-of-run points (Choshen §6)? *(Unchanged.)*

**RQ5 (new — upstream vs downstream).** Is the compute-optimal shape, and are its exponents `s_k`,
the same when "loss" means finetuned performance on physics the model never saw during pretraining,
as when it means held-out-trajectory loss on the pretraining physics? This is the question SoViT
raise in their §2 (upstream metrics may not reflect downstream ranking) and partly answer in §4.2
(exponents near-identical across classification and image-to-text). Here it is asked for PDE
surrogates, and is asked separately for in-family and out-of-family targets (§2.3).

---

## 2. Dataset selection

### 2.1 Inventory of the 2D datasets on disk

Read directly from each file's HDF5 metadata (`n_trajectories`, the `time` and spatial dimension
arrays, `boundary_conditions`, `t{0,1,2}_fields`). Tokens per frame `N_sp` use the rule in
[`flops.analytic_token_shape`](walrus/utils/flops.py): 32 per axis, plus one on each non-periodic
axis. "Train tokens" = `Σ n_traj × n_steps × N_sp` over the train split (the `T_unique_model`
definition of old §3.2). "Valid windows" = `Σ n_traj × (n_steps − 6)` at `n_steps_input=6`,
`n_steps_output=1`, `dt=1`. "Ch." = physical input channels after 2D→3D tensor padding (vectors
→ 3, rank-2 tensors → 9), the per-forward encoder `c_in` minus the 3 BC flags.

| code | resolution | BCs (x, y) | steps | train traj | **train tokens** | valid traj / windows | `N_sp` | ch. | train size | role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GS | 128×128 | per, per | 1001 | 960 | **984.0 M** | 120 / 119,400 | 1024 | 2 | 122 G | pretrain |
| EMQ-O\* | 512×512 | open, open | 100 | ~4000 | **~435.6 M** | ~500 / ~47,000 | 1089 | 6 | ~2.1 T | pretrain |
| EMQ-P\* | 512×512 | per, per | 100 | ~4000 | **~409.6 M** | ~500 / ~47,000 | 1024 | 6 | ~2.1 T | pretrain |
| AM | 256×256 | wall/open, wall/open | 202 | 1600 | **352.0 M** | 200 / 39,200 | 1089 | 6 | 238 G | pretrain |
| AI | 256×256 | wall/open, wall/open | 102 | 2798 | **310.8 M** | 400 / 38,400 | 1089 | 6 | 226 G | pretrain |
| RB | 512×128 | per, wall | 200 | 1400 | **295.7 M** | 175 / 33,950 | 1056 | 5 | 274 G | pretrain |
| SF | 256×512 | per, per | 200 | 896 | **183.5 M** | 112 / 21,728 | 1024 | 5 | 438 G | pretrain |
| AD | 256×256 | wall/open, wall/open | 102 | 1600 | **177.7 M** | 200 / 19,200 | 1089 | 6 | 121 G | pretrain |
| PS | 256×512 | open (θ), per (φ) | 1008 | 96 | **102.2 M** | 12 / 12,024 | 1056 | 4 | 142 G | pretrain |
| HS | 1024×256 | open/wall, open/wall | 50 | 416 | **22.7 M** | 48 / 2,112 | 1089 | 3 | 41 G | **held out** |
| ACT | 256×256 | per, per | 81 | 175 | **14.5 M** | 24 / 1,800 | 1024 | 22 | 39 G | **held out** |
| TRL2 | 128×384 | per, open | 101 | 72 | **7.7 M** | 9 / 855 | 1056 | 5 | 5.4 G | **held out** |
| VE | 512×512 | per, wall | 20 / 60 | 213 | **7.1 M** | 22 / 548 | 1056 | 14 | 53 G | **held out** |
| **pretrain total** | | | | | **≈ 3.25 G** | | | | ≈ 5.8 T | |
| **held-out total** | | | | | **52.0 M (1.6 %)** | | | | 138 G | |

\* **Euler figures are estimates.** Both variants are mid-download (one truncated ~212 GB file each
at the time of writing, unreadable by `h5py`). The estimate assumes The Well's published geometry —
512², 100 steps, 10 γ-files per boundary condition, 500 trajectories per file split 400/50/50 —
which is consistent with the observed ≈ 212 GB train-file size (400 traj × 100 steps × 512² × 5
stored channels × 4 B ≈ 210 GB) and The Well's ≈ 5.2 TB total. **Re-survey after the download
completes (§11 dependency) and replace every `~` in this document.**

Upstream's own [`all_2d.yaml`](walrus/configs/data/all_2d.yaml) is exactly the first twelve rows
minus HS — i.e. these thirteen are all of The Well's 2D data. Every field name any of them produces
is already in the 67-entry
[`full_well_field_index`](walrus/configs/data/field_index_map_override/full_well_field_index.yaml)
map, so pinning it keeps `n_states = 67` whatever subset is loaded (old §2.2b's risk stays closed).

### 2.2 Selection criteria

In priority order:

1. **The held-out set must span every categorisation axis**, with at least one target on each side
   of every binary axis in [well_walrus_dataset_categorisation.md](well_walrus_dataset_categorisation.md)
   §5: equation family, linear/nonlinear, diffusive/non-diffusive, sharp/smooth, chaotic/regular,
   and boundary topology. Otherwise RQ5 can only say "transfers to *this kind* of physics".
2. **It must contain both in-family and out-of-family targets.** A target whose equation family is
   represented in pretraining measures near transfer; one whose family is absent measures far
   transfer. RQ5 is asked separately for each.
3. **Holding out should reduce, not worsen, repetition in pretraining.** Sampling is uniform over
   datasets (old §2.2d; [walrus_tokens_and_data_pipeline.md](walrus_tokens_and_data_pipeline.md)
   §4.1), so dataset `d` receives `T_seen / K` tokens regardless of its size, and its repetition is
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
(7.1–22.7 M train tokens each, 52 M together = 1.6 % of all unique train tokens). They are exactly
the datasets that were most repeated under the old plan — 40–122× at 60 epochs — and that
old §9 already suspected contributed a flat, saturated floor rather than scaling signal to `f`.
Removing them costs pretraining almost no distinct data, and removes the worst repetition (§2.5).

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
| EMQ-O\* | ~435.6 M | ~37.5 | ~1.6× | ~2.7× | ~10.7× | ≈ 57× |
| EMQ-P\* | ~409.6 M | ~37.5 | ~1.6× | ~2.7× | ~10.7× | ≈ 57× |
| AM | 352.0 M | 30.3 | 2.0× | 3.3× | 13.2× | ≈ 71× |
| AI | 310.8 M | 26.8 | 2.2× | 3.7× | 15.0× | ≈ 80× |
| RB | 295.7 M | 26.2 | 2.3× | 3.8× | 15.2× | ≈ 82× |
| SF | 183.5 M | 16.8 | 3.6× | 6.0× | 23.8× | ≈ 128× |
| AD | 177.7 M | 15.3 | 3.9× | 6.5× | 26.1× | ≈ 140× |
| PS | 102.2 M | 9.1 | **6.6×** | **11.0×** | **44.1×** | **≈ 236×** |
| aggregate `T_seen / T_unique_model` | ≈ 3.25 G | — | 1.9× | 3.1× | ≈ 12× | ≈ 66× |

For comparison, the old mixture at 60 epochs ran from 0.9× (GS) to ≈ 122× (VE) on the train split
(old §7.0's whole-dataset figures differ slightly but tell the same story), and the old head-to-head
aggregate was `R ≈ 30` / `≈ 160`.

Three things follow:

- **At sweep scale (≤ 100 epochs, N1) every dataset stays within or near Muennighoff et al.'s
  "negligible degradation" band (≤ 4×) or their "half-life" (~16×).** No dataset crosses 16× before
  warmup ends, as VE and TRL2 did in the old plan.
- **At head-to-head scale, repetition is still real.** PS reaches ≈ 44× in the fullDims arm and
  ≈ 236× in the SoWalrus arm. RQ3 (§9) is still needed, but now needs to probe `R` in the tens to
  hundreds per dataset rather than already starting there.
- **The aggregate is no longer one dataset's number.** GS is 30 % of `T_unique_model`, down from
  ≈ 75 %, so an aggregate-`R` fit in §9 describes the mixture rather than Gray–Scott.

**Mixture balance, for the record.** Uniform-over-dataset sampling gives each dataset 1/9 of the
micro-batches, so by family: acoustics (I) 3/9, compressible (II) 3/9 (EMQ ×2 + PS), incompressible
(IV) 2/9, reaction–diffusion (VI) 1/9. This imbalance is incidental — the same kind WALRUS's own
mixture has (categorisation doc §4.1) — and is reported, not corrected; changing the sampler would
be a departure from upstream WALRUS that confounds every comparison with the paper.

### 2.6 Blindness: how "never seen" is guaranteed

- **Data.** The four targets are absent from
  [`Leonardo_pretrain9_2d.yaml`](walrus/configs/data/Leonardo_pretrain9_2d.yaml). Training *and*
  validation datasets are both built from that config, so no training step and no mid-training
  validation ever reads them. The run scripts must never pass them via overrides; the §5.3
  post-condition check (resolved dataset list identical across runs) enforces this.
- **Field embeddings.** `n_states = 67` includes rows for the held-out-only fields (`C`, `c_zz`,
  `concentration`, `D`, `E`, `pressure_re`, `pressure_im`, `mask`). The encoder only ever uses
  `proj1.weight[:, field_indices]` ([vstride_encoder.py:349](walrus/models/encoders/vstride_encoder.py#L349))
  and the decoder slices `proj2` by `state_labels` likewise, so those columns receive **exactly zero
  gradient** throughout pretraining. Their Adam moments stay zero, so the Adam update is zero. The
  only thing that touches them is AdamW's decoupled weight decay, which scales the whole tensor by
  `(1 − lr·wd)` per step: at the pinned `lr ≤ 2e-4`, `wd = 1e-4`
  ([adam.yaml](walrus/configs/optimizer/adam.yaml), no parameter-group exclusions) that is
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

Unchanged in method (old §3.1). `walrus/utils/flops.py` is the primary instrument and does not
depend on the mixture. Two numbers are mixture-dependent and are recomputed here with
`flops.py`'s own functions (`ModelShape`/`BatchGeometry` built by hand from each dataset's
resolution, `patch_dict` strides, BCs and channel count, then `forward_flops`). The same code
reproduces every figure in old §2.2(f)'s table exactly, which validates the hand-built geometry.

**Encoder cost tiers.** Five of the nine pretraining datasets — EMQ-O, EMQ-P (512²), PS, SF
(256×512) and RB (512×128) — have an axis ≥ 512, so their encoder intermediate grid is 128 per
axis. AD, AI, AM (256²) and GS (128²) have mid = 66. (Old §3.1 explains why this is a two-tier step
function, not a continuum.)

**Enc+dec share of forward FLOPs (§2.2f equivalent), per dataset:**

| shape | AD/AI/AM | EMQ-O | EMQ-P | GS | PS | RB | SF | per-µbatch FLOP spread (pretrain9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fullDims `1408/40/5632` | 1.4 % | 1.7 % | 1.7 % | 1.4 % | 1.4 % | 1.5 % | 1.5 % | 1.07× |
| HalfWalrus `1088/30/4352` | 2.4 % | 3.0 % | 3.0 % | 2.4 % | 2.5 % | 2.5 % | 2.6 % | 1.08× |
| 200 M anchor `768/12/2048` | 8.8 % | 11.8 % | 12.0 % | 8.7 % | 9.3 % | 9.6 % | 9.9 % | 1.11× |
| SoWalrus predicted `720/24/3616` | 4.0 % | 5.5 % | 5.6 % | 3.9 % | 4.2 % | 4.4 % | 4.5 % | 1.09× |
| roofline small `464/10/1120` | 15.9 % | 23.2 % | 23.6 % | 15.4 % | 17.3 % | 18.0 % | 18.5 % | 1.19× |
| Tier 2 grid, largest `560/12/1344` | 11.7 % | 16.7 % | 17.0 % | 11.4 % | 12.6 % | 13.1 % | 13.5 % | 1.15× |
| **Tier 2 grid, smallest `320/6/768`** | **30.8 %** | **44.3 %** | **45.0 %** | **29.1 %** | **33.6 %** | **35.0 %** | **35.8 %** | **1.38×** |

The worst case is now Euler (`EMQ-P`, 45 % at the grid's smallest corner). In the old mixture it
was VE (57 % at the same corner), which is now held out. The ballast warning of old §2.2(f) stands
unchanged in substance: at the grid sweep's small end, a third to nearly half of every
micro-batch's compute is the patchify stem, which barely responds to the swept knobs. The dual fit
(total vs processor-only FLOPs) remains mandatory.

⚠️ **Old per-µbatch spread figures not reproduced.** Old §3.2 quotes the per-micro-batch FLOP
spread across the old mixture as "1.03× at fullDims, 1.95× at `320/10/1280`". The current
`flops.py` gives **1.07× and 1.46×** for that same old mixture (and 1.07× / 1.27× for the new one).
The quoted figures probably predate a correction in `flops.py`; either way, only numbers from the
current code are used here. The conclusion — per-micro-batch FLOPs vary across datasets, so
`cumulative_flops` must be accumulated per micro-batch, never as `shape × steps` — is unaffected
and already implemented.

### 3.2 Data `D`

As old §3.2, with new values: `T_unique_model` ≈ **3.25 G** train tokens (≈ 2.40 G measured plus
≈ 0.85 G Euler, estimated). `T_seen` and `R = T_seen / T_unique_model` are logged per epoch by
`FlopAccountant` (`repetition_factor`), whose denominator (`flops.unique_model_tokens`) is computed
from the loaded datamodule, so it picks up the new mixture automatically. Report per-dataset `R_d`
alongside the aggregate on every results table (§2.5).

### 3.3 Parameters `N`

Unchanged (old §3.3, §2.2a): total and processor-only, with `override_dimensionality=3` so
`n_dim_keys = 1`. Parameter counts do not depend on the mixture under the `full_well_field_index`
pin.

⚠️ **The pin is `=3`, not `=2` as old §2.2a/§4.1 state.** `IsotropicModel.forward` always sets
`metadata.n_spatial_dims = max_d` (3) and looks up `self.embed["3"]`
([isotropic_model.py:285-313](walrus/models/isotropic_model.py#L285-L313)). Under
`override_dimensionality > 0` the encoder/decoder routing also reads that overwritten
`n_spatial_dims`. So `=2` builds only a `"2"` encoder and fails with `KeyError: '3'` on the first
forward pass (reproduced on a real `gray_scott` batch). `=3` builds exactly one pair, `"3"`, and
routes 2D data through it: 2D data is already padded to a singleton third axis, and `adaptive_conv`
sums the kernel over that axis. This is also the `isotropic_model.yaml` default ("Force it to treat
all data as 3D"). Every generator in `LeonardoRunScripts/` already passes `=3`, and
`shape_cost_table.py` already counts parameters under `include_d = [3]`, so only the documents
were wrong, not any script.

### 3.4 Upstream metric `f_up`

**Rollout VRMSE at the 60-step horizon on the `valid` split, averaged over the 9 pretraining
datasets with equal weight** — the same logged quantity as before, over a different mixture. Every
pretraining dataset's trajectories are long enough for a 60-step rollout after a 6-frame history
(the shortest is EMQ and AD/AI at 100–102 steps). Secondary metrics for the exponent-robustness
check are unchanged (one-step VRMSE/NRMSE, shorter-horizon rollout).

Because acoustics and compressible flow are each 3/9 of the mixture (§2.5), also report the
**family-balanced mean** (average within family, then across the four families) as a secondary
`f_up`. If `s_k` differs between the dataset-balanced and family-balanced `f_up`, that is itself a
finding about how mixture composition leaks into scaling exponents.

**Coverage of the capped mid-training validation** (`short_validation_length=20`, `batch_size=2` →
40 windows per dataset per check):

| dataset | valid windows | coverage at 40 windows |
|---|---:|---:|
| GS | 119,400 | 0.034 % |
| EMQ-O\* / EMQ-P\* | ~47,000 each | ~0.085 % |
| AM | 39,200 | 0.10 % |
| AI | 38,400 | 0.10 % |
| RB | 33,950 | 0.12 % |
| SF | 21,728 | 0.18 % |
| AD | 19,200 | 0.21 % |
| PS | 12,024 | 0.33 % |

The spread is now ≈ 10× rather than the old 216×, because the tiny valid sets (VE 548, TRL2 855)
are now held out. Every dataset is sampled at a fraction of a percent, though, so the planned
full-validation spot-check of old §3.4 — score a few saved checkpoints with `full=True` via
`Trainer.validate()` and compare with the capped reading — is **still required**, now prioritising
GS and the two Euler sets. The validation sampler's fixed `manual_seed(0)` still makes the capped
subset deterministic, so it adds no seed-to-seed noise (old §3.4 explains the mechanism).

### 3.5 Downstream metric `f_down` (new)

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

Identical to old §4.1 except for the data config, and the `override_dimensionality` value, which
old §4.1 misstates as 2 (§3.3):

```
data=Leonardo_pretrain9_2d                                # was Leonardo_smallest7_2d
data/field_index_map_override=full_well_field_index      # n_states = 67, unchanged
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
lr_scheduler=inv_sqrt_w_sqrt_ramps   (warmup_epochs=10, cooldown_epochs=10)
trainer.short_validation_length=20   trainer.val_frequency=2   trainer.rollout_val_frequency=5
trainer.max_rollout_steps=60         trainer.video_validation=False
++trainer.skip_spectral_metrics=True
seed=<per run>
```

[`Leonardo_pretrain9_2d.yaml`](walrus/configs/data/Leonardo_pretrain9_2d.yaml) lists the nine
datasets, with `density: torch.zeros_like` on AD/AI/AM as in upstream
[`all_2_3d.yaml`](walrus/configs/data/all_2_3d.yaml) and [WALRUS] App. C.1, and no transforms on
Euler (upstream applies none). Hydra composition with the pins above is verified: 9 datasets, the
three density transforms, `n_states = 67`. Instantiating the datamodule is not possible until the
Euler files are complete.

Naming discipline (old §4.2) and W&B offline sync (old §4.4) are unchanged. Add the data config to
every generated `name=` prefix (e.g. `p9_star_H0448_L40_M5632`) so that runs on the two mixtures
can never resolve to the same experiment folder and silently resume into each other.

### 4.2 Storage

- **$WORK (datasets):** Euler adds ≈ 5.3 TB (both variants, all splits), bringing the dataset tree
  to roughly 10 TB of the 20 TB quota (22 Sep snapshot, before the six new downloads). Re-check
  `cinQuota` before any further download.
- **$FAST (logs, checkpoints):** unchanged by this plan. The checkpoint economy of old §4.3 (sparse
  weights, dense metrics) still applies. Tier 7 adds finetuned checkpoints; keep only their final
  weights and delete them after `f_down` is extracted.

---

## 5. Tier 0 — instrumentation, roofline, noise floor

### 5.1 FLOP counter and seed — done, mixture-independent

`walrus/utils/flops.py`, `FlopAccountant`, `seed_all` and their tests (old §5.1, §2.2e, §12's
change manifest) need no change. One follow-up is mixture-dependent:

- 🔲 **Re-run [`validate_flops_real_data.py`](LeonardoRunScripts/validate_flops_real_data.py) on
  the nine pretraining datasets** once Euler is complete. It currently loops over the smallest7
  list. EMQ-O/EMQ-P (512², open and fully periodic), RB (512×128, periodic + wall), SF (256×512,
  fully periodic), AI and AM are geometries the closed form has never been checked against on real
  data. The decoder's periodic-padding path in particular was only validated on smallest7 geometries.
  This is a login-node, ~1 min check; it must pass before any Tier 0 job is submitted.

### 5.2 Roofline

The question and decision gate (GATE A: is cost FLOP-proportional or dataloader-bound?) are
unchanged from old §5.2. The six scripts must be **regenerated** with `data=Leonardo_pretrain9_2d`
before submission. None has run, so nothing is lost.

What changes:

- **The dataloader-floor risk is higher.** The new mixture's largest per-window read is Euler's:
  7 frames × 512² × 5 stored channels × 4 B ≈ 37 MB per window, drawn 2/9 of the time from ≈ 212 GB
  HDF5 files. The old mixture's largest (VE, ≈ 59 MB) was drawn only 1/7 of the time from a 53 GB
  dataset. SF (≈ 15 MB) and PS (≈ 11 MB) are also mid-sized. Random access into very large files
  may behave differently from anything the storage benchmarks
  (`storage_bench_walrus.sh`) measured. The roofline must report sec/epoch **and** per-dataset
  data-wait time, not just the aggregate.
- **`roofline_staged.sh` is infeasible as designed.** It copied the whole mixture (~653 GB) to
  node-local `/scratch_local` first. The new train splits total ≈ 1.56 TB without Euler and
  ≈ 5.8 TB with it. Replace it with a **paired staged-vs-unstaged test on a two-dataset
  sub-mixture**: one cheap-tier and one ≥512-tier dataset, e.g. GS + SF (≈ 560 GB of train data).
  Run both arms at the roofline-small shape. The effect of staging is then the difference between
  the paired arms, not an absolute number. Check node-local capacity (`df -h /scratch_local` on a
  booster node) before building it.
- The fullDims reference timing (≈ 7.8 GPU-h/epoch at N1) was measured on the old mixture and is
  now only an estimate for the new one.

### 5.3 Reproducibility post-conditions

As old §5.3, plus: the **resolved dataset list must equal the nine pretraining datasets exactly**,
on every pretraining run. This is the automated check that keeps §2.6's blindness guarantee true.

### 5.3b Re-evaluating a checkpoint

Unchanged; old §5.3b's protocol applies to both `f_up` and `f_down` (the patch jitterer runs in
`eval()` too, so same-seed repeats are bit-identical and their spread is not an error bar).

### 5.4 Noise floor `ε_noise`

Design unchanged (three seeds of the 200 M anchor `768/12/2048`, seed 0 first as the §7.0
saturation pilot, seeds 1–2 read at the settled `max_epoch`). Regenerate
`LeonardoRunScripts/NoiseFloor/` with the new data config. One addition: measure `ε_noise` for
`f_down` too, by finetuning each of the three seed checkpoints on each held-out target under the
Tier 7 protocol (12 short runs at the anchor shape). Without it, no RQ5 shape comparison can be
judged against noise.

---

## 6. Tier 1 — hyperparameter reduction

Design unchanged (old §6): local LR/WD sensitivity around the `(2e-4, 1e-4)` pin at the grid's
smallest corner, and at HalfWalrus only if needed. One caveat on old §6.2's "free" evidence: the
historical fullDims runs whose loss curves were to settle the large-extreme question trained on
`smallest7_2d` / `6DS`, not this mixture. They still show that `lr=2e-4` is stable at fullDims
scale on WALRUS data, but that is **partial** evidence here. If the loss curves are anything less
than unambiguous, run the HalfWalrus leg rather than lean on them.

---

## 7. Tiers 2–3 — the two sweeps

### 7.0 Choosing the sweep duration

Old §7.0's four bounds carry over: warmup floor (`t₁` ≳ epoch 15–18); cooldown fraction 10–20 %
(`max_epoch ∈ [50, 100]` for `cooldown_epochs=10`); small-model saturation (empirical, from the
noise-floor seed-0 pilot); and the repetition ceiling.

**What changes: upper bound 3 (repetition) no longer binds before the other two.** In the old
mixture two datasets crossed Muennighoff's 16× "half-life" before warmup even finished, and there
was no duration that cleared warmup while keeping every dataset informative. In the new mixture the
worst dataset (PS) is at 6.6× at epoch 60 and 11× at epoch 100 (§2.5). Every dataset stays below
the half-life across the entire `[50, 100]` window that bound 1 allows. `max_epoch` is therefore
set by the cooldown fraction and the saturation pilot alone. The old plan could not claim this.

The three inconsistent placeholders flagged in old §7.0 — grid sweep 40, star sweep 60 in the doc
but 101 in `gen_star_search_runs.py` — still need reconciling into one derived number when the
generators are switched to this mixture.

### 7.1 Tier 2: grid sweep → `x⁽⁰⁾`

Grid unchanged (`H ∈ {320, 384, 464, 560}`, `L ∈ {6, 8, 10, 12}`, `M ∈ {768, 928, 1120, 1344}`,
64 points). Parameter counts are mixture-independent. FLOP-proportional GPU-h move only slightly with
the enc+dec mix (§3.1), and all costs remain provisional pending GATE A. The ballast share across
the grid is now 29–45 % of forward FLOPs at the smallest corner, depending on dataset (was 29–57 %),
and 11–17 % at the largest. SoViT's four cautions (old §7.1) apply unchanged. Old §7.1b (the
authors' own grid as an alternative) also stands unchanged.

SoViT's caution 4 (enough data for power laws to take hold) is comfortably met: at 40 epochs each
grid run sees ≈ 3.9 G tokens, ≈ 1.2× `T_unique_model` — now *less* than one aggregate pass, whereas
the old plan's ≈ 3× already meant repeated data.

### 7.2 Tier 3: star sweep → `s_k`

Star centre `(1408, 40, 5632)` and the three six-point axes are unchanged (old §7.2).

**GATE C's held-out centre point is no longer free.** Old §11 relied on "the fullDims runs we
already have" for SoViT's violet point. Those runs (`fullDims_6DS*`, `smallest7_2d_PT_*`) trained on
other mixtures and cannot be compared with star points trained on this one. **A fullDims run on
`Leonardo_pretrain9_2d` to the star sweep's `max_epoch` is required**: ≈ 7.8 GPU-h/epoch × 60 ≈
**470 GPU-h at N1** (old-mixture timing, provisional). It doubles as Arm A's early trajectory if the
head-to-head is run at N1, and as the fullDims point for Tier 7.

### 7.3 Intermediate checkpoints

Unchanged (old §7.3): fit on infinite-phase points only, measure the cooldown gain separately on
three star runs, and answer RQ4 by comparing intermediate-only, end-only and combined fits.

---

## 8. Tiers 4–5 — prediction and the head-to-head

### 8.1 Deriving the shape

Unchanged (old §8.1): `x_k(t_target) = x⁽⁰⁾_k · τ^(s_k/3)`, with `t_target` = total FLOPs of
fullDims at 100 epochs on N4, matched on **total** FLOPs, not processor FLOPs. The worked example
with SoViT's exponents is unchanged, since it depends only on shapes.

### 8.2 The repetition confound

Much smaller than before, but not gone. Aggregate `R` ≈ 12 for the fullDims arm vs ≈ 66 for a
≈ 5.4×-longer SoWalrus arm (old plan: 30 vs 160). The most repeated dataset, PS, goes from ≈ 44×
to ≈ 236× (§2.5). The mitigation order of old §8.2 stands: RQ3 bounds the effect (and must report
before `t_target` is fixed); fall back to a shorter `t_target` if decay bites; report `R` and `R_d`
on every point.

### 8.3 The experiment

As old §8.3 — Arm A fullDims, Arm B SoWalrus at exactly matched total FLOPs, same seed, final
numbers on the `test` split — **plus the downstream arm**: finetune both final checkpoints on all
four held-out targets under the Tier 7 protocol (§10), in both data regimes, and report `f_down`
per target with bootstrap CIs. RQ2 is answered twice. "SoWalrus wins upstream but loses on held-out
physics" is a publishable, not a failed, outcome.

The mid-scale sanity check (old §8.3, HalfWalrus-equivalent) and the optional deviation checks are
unchanged.

---

## 9. Tier 6 — RQ3: the repeated-data sub-study (redesigned)

**Why it has to change.** At full data the anchor shape sees aggregate `R` ≈ 1.9 at 60 epochs and
per-dataset `R_d` ≤ 6.6. The old design (restriction fractions 1 → 1/8) would only reach
`R_d` ≈ 50 on PS — below the head-to-head's ≈ 236×, the regime the answer is needed for. The
restriction has to go deeper.

**Granularity floor.** `restrict_train_num_trajectories` applies **per sub-dataset**
([multidatamodule.py:155-158](walrus/data/multidatamodule.py#L155-L158) → each
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
`(T_seen, T_unique)` grid at fixed `N`, as before (≈ 320 GPU-h, provisional). The analysis is
unchanged: a Muennighoff-style effective-token fit, the repetition count at which a token's marginal
value halves, and a per-dataset breakdown from `{dataset}/full_VRMSE_T=all`. That breakdown is now
far more informative than before: per-dataset `R_d` spans a 10× range at each level instead of 143×,
so every dataset contributes points across the whole curve.

*Ordering constraint unchanged:* RQ3 reports before `t_target` is fixed (GATE B).

---

## 10. Tier 7 — RQ5: blind downstream finetuning (new)

### 10.1 Protocol

Reuse the existing finetuning pipeline — `experiment=finetune_leonardo`,
`checkpoint=finetune_leonardo`, and the existing single-dataset configs
[`Leonardo_active_matter`](walrus/configs/data/Leonardo_active_matter.yaml),
[`Leonardo_helmholtz_staircase`](walrus/configs/data/Leonardo_helmholtz_staircase.yaml),
[`Leonardo_turbulent_radiative_layer_2D`](walrus/configs/data/Leonardo_turbulent_radiative_layer_2D.yaml),
[`Leonardo_viscoelastic_instability`](walrus/configs/data/Leonardo_viscoelastic_instability.yaml)
— as already exercised by [`fullDims_FT_ActiveMatter.sh`](LeonardoRunScripts/fullDims_FT_ActiveMatter.sh).
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
regime (every run pinned at ≈ 7.8 GPU-h/epoch). All GPU-h figures inherit old §3.4's warning:
they predate `max_rollout_steps` 20 → 60 and are provisional until the roofline re-measures.

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

The Tier 7a estimate uses the star points' old per-epoch costs (≈ 1.6–6.3 GPU-h/epoch at N1),
5 epoch-equivalents per finetune, the grid points at ≈ 0.1–0.3 GPU-h/epoch, and fullDims at 7.8.
It is dominated by the fullDims-scale checkpoints (star centre, Arm A). Under the floor regime the contingency of
old §10 still applies (grid → 27 points × 25 epochs, noise floor → 30 epochs, RQ3 → 4 runs,
deviation checks dropped), and 7b is dropped.

### Sequencing and gates

```
Euler download complete ──► re-survey (§2.1), fix every "~" ──► validate_flops_real_data on 9 sets
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

The noise floor (§5.4) runs alongside T0/T1 as in the old plan. Tier 7a's grid- and star-point
finetunes can start as soon as their pretraining runs finish; they do not wait for GATE C.

- **New prerequisite (before GATE A):** Euler fully downloaded, the re-survey replacing every
  estimated figure, and `validate_flops_real_data.py` passing on all nine datasets.
- **GATE A, B, C:** as old §10. GATE B additionally decides Tier 7b.

---

## 12. Analysis protocol and deliverables

The fitting protocol, fit hygiene and three validity checks of old §11 apply unchanged to `f_up`.
They also apply to `f_down` in Tier 7b, with the trajectory-bootstrap CIs of §3.5 in place of the
window-level ones. Every table carries: shape, total and processor params, `T_seen`, aggregate `R`
and max `R_d`, measured FLOPs, GPU-hours, `f_up` with its `ε_noise` band, and — where measured —
`f_down` per target with its CI.

**Deliverables — status in this pass:**

- ✅ `walrus/configs/data/Leonardo_pretrain9_2d.yaml` — the pretraining mixture, composition
  verified.
- ✅ this document.
- 🔲 Re-survey of the Euler datasets after download, and the resulting table updates (§2.1, §2.5,
  §3.4).
- 🔲 `validate_flops_real_data.py` extended to the nine pretraining datasets and re-run (§5.1).
- 🔲 Switch the generators to `data=Leonardo_pretrain9_2d` and regenerate: `gen_roofline_runs.py`
  (plus a redesigned paired staging script, §5.2), `gen_noise_floor_runs.py`,
  `gen_grid_sweep_runs.py`, `gen_star_search_runs.py`. The last one still needs the full pass old
  §12 describes (re-centre on fullDims, add the §4.1 pins, reconcile `max_epoch`).
- 🔲 `gen_downstream_runs.py` — Tier 7 finetune generator (§10), parameterised by checkpoint path,
  target, regime and embedding policy.
- 🔲 `gen_repetition_runs.py` — Tier 6 with the redesigned levels (§9).
- 🔲 `analysis/` — as old §12, plus `f_down` extraction with trajectory bootstrap.

**Figures:** as old §12, plus
- (9) upstream vs downstream shape ranking at matched FLOPs, per held-out target (RQ5, Tier 7a);
- (10) `s_k` from `f_up` vs `s_k` from `f_down` with CIs, in-family vs out-of-family (RQ5, Tier 7b);
- (11) head-to-head per held-out target, both data regimes (RQ2's downstream arm).

---

## 13. Open risks

Every risk in old §13 that does not depend on the data still applies (dataloader floor, frozen LR,
seed variance, boundary optimum, `s_mlp` precision, 24 h wall-clock limit, cooldown-gain shape
dependence, enc/dec ballast, capped-validation coverage). New or changed:

| risk | mitigation |
|---|---|
| Euler figures are estimates; download not finished | Every Euler number is marked `~`; re-survey is a hard prerequisite for Tier 0 (§11) |
| Euler's large per-window reads from ≈ 212 GB files raise the dataloader floor | Roofline reports per-dataset data-wait (§5.2); if Euler alone is the floor, report it, and consider staging only Euler's hot files as a targeted fix |
| Closed-form FLOPs never checked on the new geometries | `validate_flops_real_data.py` on all nine sets before any submission (§5.1) |
| Historical fullDims runs no longer usable as GATE C's centre | Budgeted re-run (§7.2) |
| Tiny held-out evaluation sets (TRL2 9 valid trajectories, VE-AH 2) | Full validation, trajectory bootstrap, per-regime VE scoring; claim a shape effect only above both CI and `ε_noise` (§3.5) |
| Shared field names give a transfer channel that is not "physics" | Stated explicitly (§2.6); embedding-reinit ablation (§10.1) |
| Mixture overweights acoustics and compressible flow (3/9 each) | Family-balanced `f_up` reported alongside the dataset-balanced one (§3.4) |
| Deep RQ3 restriction collapses PS to 1–2 trajectories | Window-level restriction for the deepest level, with its different semantics stated (§9) |
| A held-out dataset leaks into pretraining via an override | §5.3 post-condition: resolved dataset list must equal the nine exactly |
| Four targets cannot estimate per-axis transfer effects | Stated as illustrative only; the categorisation doc's crossover study is the instrument for that (§10.3) |
