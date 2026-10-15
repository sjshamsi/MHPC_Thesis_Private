# Compute-optimal shape scaling for WALRUS — 2D + 3D substrate with a blind held-out set

This is the third version of the scaling plan. It keeps everything in
[SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) that does not depend on the data, and redoes the
parts that change once 3D data enters pretraining. The method is unchanged: SoViT's star/grid
shape recipe, Choshen's intermediate-checkpoint economy, the tier structure and its gates. The
older [SCALING_STRATEGY.md](SCALING_STRATEGY.md) remains the record of how each instrument was
built and validated. Sections are numbered to match the 2D document.

All 18 of The Well's datasets used in WALRUS pretraining are now on disk and surveyed (Euler
completed and re-surveyed 2026-10-02). Only FBHarmonics, an external FlowBench dataset, is missing from the paper's 19.
This plan pretrains on 12 of the 18 and holds 6 out for blind downstream finetuning.

| decision | value |
|---|---|
| Compute budget | ~30K hours on Boost |
| Pretraining mixture | 12 datasets — 2D: AD, AI, AM, EMQ-O, EMQ-P, GS, PS, RB, SF · 3D: SN, TGC, TRL3D (`data=Leonardo_pretrain12_2d3d`) |
| Held out (never seen in pretraining) | 6 datasets — 2D: ACT, HS, TRL2, VE · 3D: MHD, RT |
| Encoder/decoder stems | **separate 2D and 3D stems**, `model.override_dimensionality=0` (§3.3) |
| Upstream metric `f_up` | 60-step rollout VRMSE on the `valid` split of the 12 pretraining datasets, reported as 2D, 3D and overall means (§3.4) |
| Downstream metric `f_down` | finetuned VRMSE on the 6 held-out datasets (§3.5, Tier 7) |
| Shape axes | `hidden_dim`, `processor_blocks`, `mlp_dim` |
| Validating endpoint | head-to-head against fullDims at matched FLOPs, upstream and downstream |
| Data repetition | measured sub-study (RQ3); TRL3D now provides a high-repetition dataset without any restriction (§9) |

Dataset codes: the 2D codes are as in the 2D document. SN = `supernova_explosion_128`,
TGC = `turbulence_gravity_cooling`, TRL3D = `turbulent_radiative_layer_3D`, MHD = `MHD_64`,
RT = `rayleigh_taylor_instability`. Family numerals (I–VI) and axis definitions are those of
[well_walrus_dataset_categorisation.md](well_walrus_dataset_categorisation.md) §2.2 and §5.

---

## 0. What changed

| item | SCALING_STRATEGY_2D.md | this plan | consequence |
|---|---|---|---|
| Pretraining data | 9 × 2D | 9 × 2D + 3 × 3D (K = 12) | T_unique 3.26 G → **3.88 G** train tokens |
| Held out | ACT, HS, TRL2, VE | the same four, **plus MHD and RT** | adds a far 3D target (MHD) and two transfer-across-dimension tests (§2.4) |
| Stems | `override_dimensionality=3`: one shared stem, correct for 2D-only | **`=0`: separate 2D and 3D stems, both used** | parameters match the paper exactly (fullDims = 1 286.2 M); encoder/decoder parameters double (§3.3) |
| Where compute goes | ~uniform across datasets | 3D takes 3/12 of micro-batches but **34 % (fullDims) to 61 % (grid corner) of FLOPs** | the stem-ballast problem of old §2.2(f) is far worse at small shapes (§3.1) |
| Cost per micro-batch vs 2D-only mix | 1× | 1.14× (fullDims) … 1.90× (grid corner) | every FLOP-proportional budget line rises (§11) |
| Worst repetition, 60 ep, N1 | 6.6× (PS) | **16.5× (TRL3D)** | §7.0's repetition bound partly binds again (§2.6) |
| Per-sample I/O | ≤ 37 MB (Euler) | up to **≈ 335 MB** (TRL3D) | GATE A's dataloader-floor risk is the largest yet (§5.2) |
| Upstream pins | — | adds `++model.use_periodic_fixed_jitter=True` and `++model.input_field_drop=0.0` to §4.1 | every script already sets them, but neither earlier doc listed them |
| Fallback | — | if 3D I/O makes the study unaffordable, fall back to `SCALING_STRATEGY_2D.md` unchanged (§11) | the two plans share their 2D split, so no work is lost |

---

## 1. Research questions

RQ1–RQ5 are as in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §1, with two changes.

- **RQ5 gains a dimensionality axis.** Blind downstream targets now include transfer *across
  dimensionality* in both directions (TRL3D → TRL2 and RB → RT), plus an out-of-family 3D target
  (MHD). The question "is the upstream-optimal shape also downstream-optimal?" is asked for 2D
  and 3D targets separately.
- **Exponents are measured at a fixed number of pretraining systems, K = 12, and are stated as
  such.** §2.2 explains why K is probably not neutral for the shape optimum. Varying K is a
  separate study this budget cannot afford.

A further comparison comes free, because the two plans share their 2D split. If both are run,
**do the exponents change when 3D data is added?** 3D spatial attention is 33–52 % of processor
FLOPs, against 9–18 % in 2D (§3.1), and `mlp_dim` does not touch attention. So 3D data plausibly
shifts the optimal `mlp_dim/hidden_dim` balance, and a difference between the two plans' `s_k`
would be a finding in itself.

---

## 2. Dataset selection

### 2.1 Inventory

The 2D rows are as measured in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §2.1. The 3D rows
are read from HDF5 metadata in the same way. 3D token rule: 16 tokens per axis, plus one on each
non-periodic axis. With upstream's `step_downsample_factor=0.5` and `batch_downsample_factor=0.5`,
a 3D micro-batch carries `T_in = 3` frames of `B = 1` window at `batch_size=2`, against 6 × 2 for
2D. "Ch." counts physical channels after 2D→3D padding; the encoder adds 3 boundary channels.

| code | dim | resolution | BCs | steps | train traj | **train tokens** | valid traj | `N_sp` | ch. | train size | role |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GS | 2D | 128² | per, per | 1001 | 960 | **984.0 M** | 120 | 1024 | 2 | 122 G | pretrain |
| TGC | 3D | 64³ | open ×3 | 50 | 2160 | **530.6 M** | 270 | 4913 | 6 | 633 G | pretrain |
| EMQ-O | 2D | 512² | open, open | 101 | 4000 | **440.0 M** | 500 | 1089 | 6 | 2.15 T | pretrain |
| EMQ-P | 2D | 512² | per, per | 101 | 4000 | **413.7 M** | 500 | 1024 | 6 | 2.12 T | pretrain |
| AM | 2D | 256² | wall/open ×2 | 202 | 1600 | **352.0 M** | 200 | 1089 | 6 | 238 G | pretrain |
| AI | 2D | 256² | wall/open ×2 | 102 | 2798 | **310.8 M** | 400 | 1089 | 6 | 226 G | pretrain |
| RB | 2D | 512×128 | per, wall | 200 | 1400 | **295.7 M** | 175 | 1056 | 5 | 274 G | pretrain |
| SF | 2D | 256×512 | per, per | 200 | 896 | **183.5 M** | 112 | 1024 | 5 | 438 G | pretrain |
| AD | 2D | 256² | wall/open ×2 | 102 | 1600 | **177.7 M** | 200 | 1089 | 6 | 121 G | pretrain |
| PS | 2D | 256×512 | open (θ), per (φ) | 1008 | 96 | **102.2 M** | 12 | 1056 | 4 | 142 G | pretrain |
| SN | 3D | 128³ | open ×3 | 59 | 208 | **60.3 M** | 26 | 4913 | 6 | 576 G | pretrain |
| TRL3D | 3D | 128×128×256 | per, per, open | 101 | 72 | **31.6 M** | 9 | 4352 | 5 | 569 G | pretrain |
| MHD | 3D | 64³ | per ×3 | 100 | 77 | **31.5 M** | 10 | 4096 | 7 | 57 G | **held out** |
| RT | 3D | 128³ | per, per, wall | 119 | 45 | **23.3 M** | 10 | 4352 | 4 | 168 G | **held out** |
| HS | 2D | 1024×256 | open/wall ×2 | 50 | 416 | **22.7 M** | 48 | 1089 | 3 | 41 G | **held out** |
| ACT | 2D | 256² | per, per | 81 | 175 | **14.5 M** | 24 | 1024 | 22 | 39 G | **held out** |
| TRL2 | 2D | 128×384 | per, open | 101 | 72 | **7.7 M** | 9 | 1056 | 5 | 5.4 G | **held out** |
| VE | 2D | 512² | per, wall | 20 / 60 | 213 | **7.1 M** | 22 | 1056 | 14 | 53 G | **held out** |
| **pretrain total** | | | | | | **≈ 3.88 G** | | | | ≈ 7.6 T | |
| **held-out total** | | | | | | **106.8 M (2.7 %)** | | | | 363 G | |

Euler figures are measured (re-surveyed 2026-10-02; details in
[SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §2.1). The earlier estimate's only error was the
trajectory length, 101 steps rather than 100, i.e. +1 % train tokens; no conclusion here changes.

Every field name these datasets produce, including the log10-transformed `log10_density` and
`log10_temperature`, and `magnetic_field_{x,y,z}`, is already in the 67-entry
`full_well_field_index` map. `n_states = 67` holds; this was checked by instantiating the
datamodule.

### 2.2 Should a compute-limited study pretrain on more datasets or fewer?

The WALRUS paper pretrained on 19 datasets and evaluated on 8 unseen ones. We have a small
fraction of its compute. Does that argue for less variety (so each system gets more training) or
more? Seven considerations:

**1. We are compute-limited, not data-limited, and per-system exposure is far below the paper's.**

| run | total tokens | per dataset (K = 12) | per dataset (K = 18) |
|---|---:|---:|---:|
| canonical WALRUS (96 GPUs, 200 ep, K = 19) | 472 G | 24.8 G (K = 19) | — |
| our grid point (40 ep, N1) | 4.2 G | 0.35 G | 0.23 G |
| our star point (60 ep, N1) | 6.3 G | 0.52 G | 0.35 G |
| our fullDims arm (100 ep, N4) | 42 G | 3.5 G | 2.3 G |

Each system sees 7–70× less than in the paper. But at sweep scale, 9 of the 12 pretraining
datasets are seen fewer than 4 times (§2.6), which is the "effectively unique data" regime. Adding
a dataset does not starve anyone of *data*. It divides a fixed budget of *optimizer steps* more
thinly.

**2. More datasets lower everyone else's repetition.** `R_d = T_seen / (K · T_d)`, so raising K
lowers every existing dataset's `R_d`. Only a newcomer that is itself tiny ends up highly repeated.
The repetition argument therefore says: add large datasets, exclude tiny ones. The paper's own
regime was the extreme opposite, with its smallest datasets repeated roughly 10³ times (train
split: VE ≈ 3 500×, TRL2 ≈ 3 200×, RT ≈ 1 100×, MHD ≈ 800×). That is tolerable when the goal is a
strong model, but it confounds a scaling law (RQ3).

**3. Breadth gives the scaling fit headroom.** A power-law fit needs loss that is still falling
across the compute range. With few, heavily repeated datasets, small models flatten early and the
argmin reads noise off a plateau ([SCALING_STRATEGY.md](SCALING_STRATEGY.md) §7.0, upper bound 2).

**4. The counterweight is capacity dilution, and it interacts with the swept variable.** Our grid
models have 84–142 M parameters under `=0`, of which only 7–44 M are processor (§3.3). In
multilingual NLP, the "curse of multilinguality" (Conneau et al. 2020, XLM-R) shows that at fixed
capacity, adding languages helps up to a point and then hurts all of them, and the turning point
moves with model size. By analogy, K plausibly shifts the compute-optimal shape toward larger
models. K is therefore fixed at 12 and stated as a condition of every exponent (§1).

**5. 3D is the expensive kind of variety.** A 3D micro-batch costs 1.3–1.6× a 2D one at fullDims,
but 3.5–5× at the grid's small end (§3.1). 60–72 % of that is the 3D encoder/decoder (8³ and 4³
kernels), which the swept knobs barely change. Three 3D datasets out of twelve take 34 % of FLOPs
at fullDims and 61 % at the grid's smallest corner. Per-sample reads reach ≈ 335 MB (§5.2). Each
3D dataset in pretraining costs as much as two to five 2D ones.

**6. RQ2 fidelity pulls the other way.** The paper's fullDims model, which the head-to-head is
against, was pretrained on 2D and 3D data with separate stems. A mixture that includes 3D makes
that comparison much more like-for-like than a 2D-only mixture.

**7. Downstream evaluation caps how many can be held out.** Each held-out target multiplies Tier
7's cost (checkpoints × targets × data regimes). The paper's 19 : 8 is about 70 : 30. This plan's
12 : 6 is 67 : 33. The paper filled most of its downstream slots with external data (PDEBench,
PDEGym and others), which we do not have, so every held-out slot here comes out of pretraining.

**Conclusion.** Pretrain on every *large* dataset: breadth is nearly free in repetition and helps
the fit. Hold out *small* datasets, which would be the most repeated anyway and are cheap,
data-scarce downstream targets, and choose them to span the categories. Keep 3D in pretraining for
RQ2 fidelity, but keep the number of 3D sets small because of their FLOP and I/O cost.

### 2.3 Selection criteria

As in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §2.2 — axis coverage by the held-out set;
both in-family and out-of-family targets; holding out small datasets to cut repetition; the
pretraining mixture still covering every axis; cheap, data-scarce targets — plus two for 3D:

6. **The 3D stem must see more than one kind of 3D physics.** With separate stems, the 3D stem
   learns only from the 3D pretraining datasets. They should span more than one BC topology and
   more than one regime.
7. **Every 3D pretraining dataset has to earn its FLOP and I/O cost.** It should serve a specific
   purpose: a transfer source for a held-out target, or coverage the 2D sets cannot give.

### 2.4 The split and its coverage

**Held out: ACT, HS, TRL2, VE (2D) and MHD, RT (3D).** The 2D four are unchanged from the 2D plan,
so the two plans stay comparable. Coverage matrix for all 18:

| | dim | family | linear | explicit diffusion | sharp | regime (proxy T21–60) | BCs | role |
|---|---|---|---|---|---|---|---|---|
| AD / AI / AM | 2D | I | ✓ | none | ✓ interfaces | regular (0.04–0.13) | wall/open ×2 | pretrain |
| EMQ-O / EMQ-P | 2D | II | — | none | ✓ shocks | intermediate (0.28) | open / periodic | pretrain |
| PS | 2D | II (SWE), sphere | — | hyperviscosity | — | slowly chaotic (0.02) | per + open | pretrain |
| RB | 2D | IV | — | ✓ | — | chaotic (0.64) | per + wall | pretrain |
| SF | 2D | IV | — | ✓ | — | predictable over 60 steps (0.08) | periodic | pretrain |
| GS | 2D | VI | — | ✓ | — | regular (0.13) | periodic | pretrain |
| SN | 3D | II (+ gravity, cooling; SPH) | — | artificial viscosity only | ✓ blast shells | saturates (0.97) | open ×3 | pretrain |
| TGC | 3D | II (same equations as SN) | — | artificial viscosity only | ✓ filaments | chaotic (0.62) | open ×3 | pretrain |
| TRL3D | 3D | II (+ cooling) | — | none | ✓ interfaces | chaotic (0.78) | per, per, open | pretrain |
| **HS** | 2D | **I** | **✓** | none | — | **regular** (0.003) | open + wall | **held out** |
| **TRL2** | 2D | **II** | — | none | **✓** | **chaotic** (0.84) | per + open | **held out** |
| **VE** | 2D | **IV** | — | **✓** | borderline | **mixed, by file** | per + wall | **held out** |
| **ACT** | 2D | **V** | — | **✓** | — | **chaotic** (1.25) | periodic | **held out** |
| **MHD** | 3D | **III** | — | none (ideal) | **✓** at high Mach | **chaotic + stochastically forced** (1.23) | periodic ×3 | **held out** |
| **RT** | 3D | **IV** (variable density) | — | **✓** | — (diffuse interface) | **chaotic onset** (0.48) | per, per, wall | **held out** |

Held-out coverage now includes every family available on disk: I (HS), II (TRL2), III (MHD),
IV (VE, RT) and V (ACT). Both sides of every binary axis are covered in both 2D and 3D, except
that **no regular 3D dataset exists to hold out**: every 3D dataset on disk is chaotic or
saturating. The 3D pretraining set spans two BC topologies (fully open for SN/TGC, periodic + open
for TRL3D) and three regimes (saturating, chaotic, chaotic with sharp interfaces), which satisfies
criterion 6. What it shares is one equation family, II. That matches the categorisation doc's §5.6
3D design ("pretrain on the compressible family and test on MHD and RT").

**Transfer ladder, now two-dimensional** (equation distance × dimensionality):

| target | nearest pretraining relatives | equation distance | dimensionality | field-name overlap | what it tests |
|---|---|---|---|---|---|
| TRL2 | **TRL3D** (identical equations); EMQ (equation twin) | none (from TRL3D) | **3D → 2D** | partial: `pressure`, `velocity` shared; `density` shared with EMQ only, because TRL3D's density is log-transformed into `log10_density` | same physics across dimension — The Well's App. D "transfer across dimensionality" |
| RT | **RB** (Boussinesq buoyancy); SF | same family (IV); Boussinesq → variable density | **2D → 3D** | partial: `velocity`; `density` (from EMQ, a different quantity) | in-family physics across dimension, from a 2D source into a stem trained only on 3D compressible data |
| VE | SF → RB ladder | one step: tensor feedback | 2D → 2D | partial | as in the 2D plan |
| HS | AD/AI/AM | same equation; new geometry and regime | 2D → 2D | none | transfer through the processor alone |
| ACT | none (family V absent) | out of family | 2D → 2D | partial | far transfer, 2D |
| MHD | TGC / SN / TRL3D (MHD = Euler + induction) | adds a field and an equation (induction) | 3D → 3D | partial: `density`, `velocity`; **`magnetic_field_*` never trained** | far transfer, 3D: a new field type entirely |

Note the subtlety in the TRL2 row. Upstream log-transforms density in the three 3D compressible
sets but not in TRL2, so "the same physics" reaches the model under a different field name. The
TRL3D → TRL2 test is therefore a physics-and-processor transfer. It cannot ride on a shared
density embedding. That makes it cleaner, not weaker.

### 2.5 Alternatives considered

| variant | what changes | why not chosen |
|---|---|---|
| **B:** also hold out TRL3D (11 : 7) | 3D pretraining is only SN + TGC; worst `R` 10.7×; 3D FLOP share 27–49 % | SN and TGC share equations, solver (SPH) and BCs (fully open), so the 3D stem would see one narrow slice; loses the 3D → 2D test |
| **C:** hold out only MHD (13 : 5) | adds RT to 3D pretraining | worst `R` 20.7× (RT); 3D FLOP share 40–66 %; only one 3D target, and the hardest to score |
| **all 18, nothing held out** | maximal breadth | brings back 44–48× repetition on the tiny 2D sets and removes RQ5 |
| **2D pretraining only, 3D all held out** | = SCALING_STRATEGY_2D.md, plus 3D downstream targets | a legitimate cheaper plan (it is the fallback, §11), but it gives up RQ2's like-for-like comparison with paper WALRUS |

### 2.6 Repetition in the new mixture

Tokens per dataset per epoch on the §4.1 N1 substrate (`max_samples=2000`, `world_size=4`,
K = 12): 8.2–9.8 M. A 3D micro-batch carries slightly more tokens (1 × 3 × 4 913 = 14.7 k for
SN/TGC) than a 2D one (≈ 12.3–13.1 k).

| dataset | train tokens | epochs per pass | `R` 60 ep | `R` 100 ep | fullDims arm (100 ep, N4) | SoWalrus arm (≈ 5.4×) |
|---|---:|---:|---:|---:|---:|---:|
| GS | 984.0 M | 120 | 0.5× | 0.8× | 3.3× | ≈ 18× |
| TGC | 530.6 M | 54 | 1.1× | 1.9× | 7.4× | ≈ 40× |
| EMQ-O / EMQ-P | 440.0 / 413.7 M | 50 | 1.2× | 2.0× | 7.9× | ≈ 43× |
| AM | 352.0 M | 40 | 1.5× | 2.5× | 9.9× | ≈ 53× |
| AI | 310.8 M | 36 | 1.7× | 2.8× | 11.2× | ≈ 61× |
| RB | 295.7 M | 35 | 1.7× | 2.9× | 11.4× | ≈ 62× |
| SF | 183.5 M | 22 | 2.7× | 4.5× | 17.9× | ≈ 96× |
| AD | 177.7 M | 20 | 2.9× | 4.9× | 19.6× | ≈ 106× |
| PS | 102.2 M | 12 | 5.0× | 8.3× | 33.1× | ≈ 179× |
| SN | 60.3 M | 6.1 | **9.8×** | **16.3×** | 65× | ≈ 352× |
| TRL3D | 31.6 M | 3.6 | **16.5×** | **27.5×** | 110× | ≈ 595× |
| aggregate | 3.88 G | — | 1.6× | 2.7× | ≈ 11× | ≈ 58× |

Compared with the 2D plan, the 2D datasets are all *less* repeated, because K rose from 9 to 12.
The two small 3D pretraining sets become the most repeated. TRL3D crosses Muennighoff's 16×
"half-life" around epoch 58 and SN around epoch 98.

**Why keep TRL3D despite that.** It is the source for the TRL3D → TRL2 test and the only 3D
pretraining set with periodic axes (criterion 6). Its high `R` also gives RQ3 a heavily repeated
dataset with no restriction applied (§9). The cost is a partly binding repetition bound in §7.0.

GS is now 25 % of the unique tokens (75 % in the original plan, 30 % in the 2D plan), so the
aggregate `R` describes the mixture rather than one dataset.

### 2.7 Blindness

As in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §2.6, with these additions:

- `Leonardo_pretrain12_2d3d.yaml` contains none of the six held-out datasets. Its header comment
  lists them. The §5.3 post-condition (resolved dataset list = exactly these twelve) enforces it.
- **`magnetic_field_x/y/z` (indices 39–41) is never trained.** MHD is the only dataset with a
  magnetic field, so those encoder and decoder columns get zero gradient throughout pretraining.
  AdamW's decoupled decay shrinks them by ≤ 0.06 %, as for the other held-out-only fields. MHD's
  finetuning therefore starts from untrained embeddings for its defining field. That is the
  intended "far" condition.
- Held-out 3D targets are finetuned through the 3D stem, which has seen only SN, TGC and TRL3D.

---

## 3. Definitions

### 3.1 Compute `t`

The closed form in [`walrus/utils/flops.py`](walrus/utils/flops.py) handles 3D geometry unchanged.
The numbers below come from its functions with hand-built geometry: 3D strides are
`patch_dict[res // 16]`, and batches are B = 1, T = 3. The same method reproduces every figure in
the 2D documents exactly. The closed form has **not yet been validated against `FlopCounterMode`
on real 3D data** (§5.1).

**Per-micro-batch forward FLOPs, relative to the mean 2D pretraining micro-batch:**

| shape | 2D sets | SN / TGC | TRL3D | MHD\* | RT\* | enc+dec share, 2D | enc+dec share, 3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| fullDims `1408/40/5632` | 0.96–1.03× | 1.63× | 1.41× | 1.30× | 1.39× | 1–2 % | 5–7 % |
| 200 M anchor `768/12/2048` | 0.95–1.06× | 2.36× | 2.20× | 2.00× | 1.99× | 9–12 % | 26–34 % |
| grid largest `560/12/1344` | 0.94–1.08× | 2.78× | 2.69× | 2.40× | 2.31× | 11–17 % | 32–42 % |
| **grid smallest `320/6/768`** | 0.87–1.20× | **4.40×** | **4.98×** | 4.29× | 3.54× | 29–45 % | **60–72 %** |

\* Held out: costs apply to Tier 7 finetuning, not to pretraining.

**Share of all pretraining FLOPs taken by the three 3D datasets** (3/12 of micro-batches under
uniform sampling):

| shape | 3D share of FLOPs | 12-mix cost vs 2D-only mix | processor share of 12-mix FLOPs |
|---|---:|---:|---:|
| fullDims | 34 % | 1.14× | 97 % |
| HalfWalrus | — | 1.18× | 95 % |
| SoWalrus predicted `720/24/3616` | — | 1.24× | 91 % |
| 200 M anchor | 43 % | 1.33× | 82 % |
| roofline small `464/10/1120` | — | 1.55× | 69 % |
| grid largest | 48 % | — | — |
| grid smallest | **61 %** | **1.90×** | **47 %** |
| *grid sweep, all 64 points (cost-weighted)* | — | *1.59×* | — |
| *star sweep, all 18 points (cost-weighted)* | — | *1.17×* | — |

Three consequences:

- **The ballast problem is much worse at small scale.** At the grid's smallest corner, more than
  half the compute is the patchify stem, and most of that is the 3D stem. The dual fit (total and
  processor-only FLOPs, [SCALING_STRATEGY.md](SCALING_STRATEGY.md) §7.1) goes from strongly
  advised to indispensable. **Also log the 2D/3D FLOP split per run.** This needs a small addition
  to `FlopAccountant`: it currently accumulates only mixture-wide epoch totals, and dataset
  identity lives only in its geometry cache key. An epoch accumulator keyed by dimensionality
  (`len(token_shape)` non-singleton axes), all-reduced alongside the existing totals, is enough
  (§12).
- **3D shifts where processor FLOPs go.** Spatial attention is `O(N²)` per frame. It is 33 % of
  processor FLOPs on a TGC micro-batch at fullDims (52 % at the anchor), against 9 % (18 %) for
  shear flow. `mlp_dim` does not affect attention, so the mix of 2D and 3D data changes the
  marginal return of each shape knob (§1).
- **The sampler cannot be rebalanced without new code.**
  [`mixed_dset_sampler.py`](walrus/data/mixed_dset_sampler.py) draws the dataset with
  `torch.randint` over the available datasets, with no weights. The 3D share is reported, not
  corrected. Changing it would also depart from upstream WALRUS.

### 3.2 Data `D`

`T_unique_model` ≈ **3.88 G** train tokens (3.26 G 2D + 0.62 G 3D; all measured).
`FlopAccountant` computes it from the loaded datamodule, so it follows the mixture automatically.
A grid run (40 epochs, N1) sees 4.2 G tokens ≈ 1.1 × `T_unique`.

### 3.3 Parameters `N` — separate stems, `override_dimensionality=0`

With 3D data in the mixture, the reason `SCALING_STRATEGY_2D.md` §3.3 pinned `=3` no longer holds.
That reason was that under `=0` the 3D branch of a 2D-only mixture is dead weight. Here `=0` builds
a `"2"` and a `"3"` stem and **both are used**: 2D batches route to `embed["2"]` (32×32 tokens) and
3D batches to `embed["3"]` (17³ for fully open 64³/128³ grids, 16×16×17 for TRL3D). This was
confirmed by running real SN, TGC, TRL3D and SF batches through a `=0` model. It is the upstream
canonical setting
([pretrain_example_distributed_walrus.sh](walrus/run_scripts/pretrain_example_distributed_walrus.sh)).

Each stem carries full 3D kernels (`proj1.weight` is `(intermediate_dim, 67, 8, 8, 8)` even in the
`"2"` stem; 2D data sums the kernel over its singleton axis). So `=0` exactly doubles the
encoder/decoder parameters relative to `=3`. Counted by instantiating each model on the `meta`
device with the §4.1 pins:

| shape | total (`=0`) | encoder | processor | decoder | enc+dec share | total if `=3` |
|---|---:|---:|---:|---:|---:|---:|
| fullDims `1408/40/5632` | **1 286.2 M** | 87.6 M | 1 111.0 M | 87.6 M | 13.6 % | 1 198.6 M |
| HalfWalrus `1088/30/4352` | 644.0 M | 73.2 M | 497.7 M | 73.2 M | 22.7 % | 570.8 M |
| SoWalrus predicted `720/24/3616` | 306.7 M | 56.6 M | 193.5 M | 56.6 M | 36.9 % | 250.1 M |
| 200 M anchor `768/12/2048` | **202.6 M** | 58.8 M | 85.1 M | 58.8 M | 58.0 % | 143.8 M |
| grid largest `560/12/1344` | 142.5 M | 49.4 M | 43.7 M | 49.4 M | 69.3 % | 93.1 M |
| grid smallest `320/6/768` | 84.3 M | 38.6 M | 7.2 M | 38.6 M | **91.5 %** | 45.7 M |

fullDims comes to exactly the paper's published 1.286 B, and the anchor to the 202.6 M its name
comes from. Both parameter counts match the paper again. At the grid's small end, though,
**91.5 % of parameters are stem**. The requirement to report processor parameters alongside totals
is essential here.

Distributed correctness: under `distribution=fsdp` the device mesh is a single `fsdp` dimension,
so `sampling_rank = 0` and every rank draws the same dataset each micro-step
([training.py:357-376](walrus/trainer/training.py#L357-L376)). All ranks therefore run the same
stem, and the separate stems cannot desynchronise FSDP's per-module all-gathers. `FlopAccountant`
already hooks every module in `model.embed`, so both stems are accounted for.

### 3.4 Upstream metric `f_up`

60-step rollout VRMSE on the `valid` split, **reported three ways**:
- `f_up,2D`: equal-weight mean over the 9 2D sets
- `f_up,3D`: equal-weight mean over SN, TGC and TRL3D
- `f_up`: equal-weight mean over all 12. This is the primary metric, the same logged quantity as
  before.

The family-balanced mean of the 2D plan is kept as a secondary metric. A difference between
exponents fitted on `f_up,2D` and `f_up,3D` is a direct, cheap answer to "does 3D scale
differently?".

**Horizon caps.** The trainer rolls out `min(max_rollout_steps, available steps)`
([training.py:455-459](walrus/trainer/training.py#L455-L459)). After a 3-frame 3D history, TGC
allows 47 steps, SN 56, and TRL3D the full 60. `f_up,3D` therefore mixes horizons. Report
per-dataset horizons with every number.

**Saturation.** SN's best-model proxy (0.97 at T21–60) and TGC's (0.62) mean long-horizon VRMSE
is near its ceiling for those sets and will add a flat floor to `f_up`. Keep one-step and T1–10
VRMSE as secondary metrics for the 3D sets, and screen for flat per-dataset curves before fitting
(the saturation screen of [SCALING_STRATEGY.md](SCALING_STRATEGY.md) §11).

The full-validation spot-check of the 2D plan §3.4 still applies. Add one 3D set (TGC, the largest
valid split) to its priority list.

### 3.5 Downstream metric `f_down`

The four 2D targets are as in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) §3.5. The two 3D
targets:

| target | steps/traj | max rollout after 3-frame history | regime | primary `f_down` | valid set |
|---|---:|---:|---|---|---|
| MHD | 100 | 97 | chaotic + stochastically forced | one-step VRMSE, T1–10 VRMSE, binned spectral error | 10 traj |
| RT | 119 | 116 | chaotic onset | rollout VRMSE T1–60, plus spectral error | 10 traj |

MHD's forcing is random and never stored as a field, so no model can predict it pointwise beyond
the short term (categorisation doc §5.4.5). Its long-horizon VRMSE measures the forcing, not the
model. Both 3D targets have only 10 valid trajectories, so the trajectory bootstrap is mandatory,
as for TRL2 and VE.

---

## 4. Fixed experimental substrate

### 4.1 Held constant on every run in every tier

```
data=Leonardo_pretrain12_2d3d
data/field_index_map_override=full_well_field_index      # n_states = 67
model.intermediate_dim=352      model.projection_dim=48
model.groups=16                 model.override_dimensionality=0        # separate 2D/3D stems -- §3.3
model.processor.space_mixing.num_heads=16
model.processor.time_mixing.num_heads=16
model/processor/space_mixing=full_spatial_attention
model.causal_in_time=True       model.jitter_patches=True
model.drop_path=0.05
++model.use_periodic_fixed_jitter=True    ++model.input_field_drop=0.0    # upstream; every script already sets these
data.module_parameters.batch_size=2      n_steps_input=6      n_steps_output=1
data.module_parameters.min_dt_stride=1   max_dt_stride=5      max_samples=2000
trainer.grad_acc_steps=4        trainer.enable_amp=False     trainer.clip_gradient=10
trainer.prediction_type=delta
lr_scheduler=inv_sqrt_w_sqrt_ramps   (warmup_epochs=10, cooldown_epochs=10)
trainer.short_validation_length=20   trainer.val_frequency=2   trainer.rollout_val_frequency=5
trainer.max_rollout_steps=60         trainer.video_validation=False
++trainer.skip_spectral_metrics=True
distribution=fsdp               seed=<per run>
```

Three things differ from the 2D plan's pins:

- **`model.override_dimensionality=0`** (was `=3`), for the reasons in §3.3.
- **The two upstream pins are now listed.** `use_periodic_fixed_jitter=True` and
  `input_field_drop=0.0` are set by every generated `LeonardoRunScripts` script and by the
  canonical upstream script. Neither earlier document listed them, so a generator written from the
  documents alone would have silently picked up the class defaults (`False` and `0.1`).
- **`distribution=fsdp` is now a requirement, not an engineering choice.** In this codebase,
  `distribution=ddp` is FSDP with `ShardingStrategy.NO_SHARD` and per-module wrapping
  ([distribution_utils.py:86-90](walrus/utils/distribution_utils.py#L86-L90)), not PyTorch DDP.
  Its device mesh is named `ddp`, so each GPU gets its own sampling rank and draws its own dataset
  ([training.py:357-376](walrus/trainer/training.py#L357-L376)). With separate stems, one rank
  could run the 2D stem while another runs the 3D stem. FSDP's per-unit gradient reduction would
  then see mismatched participation, which is a likely hang. This is not verified, since it needs
  a multi-GPU job, but there is no reason to take the risk. Under `fsdp` the draw is
  world-synchronised, so every rank runs the same stem. The old plan's §4.1 allowed either
  setting; this plan does not.

[`Leonardo_pretrain12_2d3d.yaml`](walrus/configs/data/Leonardo_pretrain12_2d3d.yaml) gives SN, TGC
and TRL3D the upstream `all_2_3d.yaml` treatment: `step_downsample_factor=0.5`,
`batch_downsample_factor=0.5`, and log10 on density and temperature. Verified on real batches:
- T = 3 and B = 1 for 3D (the floor `int(2 × 0.5) = 1` is exact at `batch_size=2`)
- `log10_density` and `log10_temperature` appear with log-scale ranges
- TRL3D, which has no temperature field, silently skips that transform

Keep `batch_size` even, as the tokens doc §2.7 warns.

### 4.2 Storage

With every download complete, `cinQuota` reports **9.58 TB of the 20 TB** $WORK quota
(2026-10-02): ≈ 2.1 TB of other 2D data, ≈ 2.5 TB of 3D, and 5.34 TB of Euler. The 3D pretraining
train splits alone are ≈ 1.8 TB. Check `cinQuota` before any further download. Checkpoints grow with the
encoder/decoder doubling (§3.3): fullDims ≈ 16 GB per sharded AdamW checkpoint, and the anchor
≈ 2.5 GB. The $FAST budget of the old plan's §4.3 needs re-checking against the new counts.

---

## 5. Tier 0

### 5.1 FLOP counter — extend the real-data validation to 3D

`flops.py` needs no change. [`validate_flops_real_data.py`](LeonardoRunScripts/validate_flops_real_data.py)
must be extended to the 12 pretraining datasets and run with `override_dimensionality=0`. It
currently loops over the 2D-only smallest7 list with `=0`. The 3D encoder and decoder geometry
(8³/4³ kernels, 3D periodic padding in the decoder, 17-token open axes) has never been checked
against `FlopCounterMode` on real data. **This must pass before any Tier 0 job is submitted.**

### 5.2 Roofline

The design is as in the 2D plan §5.2, regenerated for this data config, with two changes:

- **3D I/O is the new dominant risk.** Approximate per-sample reads (4 frames, stored channels,
  fp32): **TRL3D ≈ 335 MB, SN ≈ 200 MB**, RT ≈ 134 MB, MHD ≈ 29 MB, TGC ≈ 25 MB. For comparison
  the largest 2D read is Euler at ≈ 37 MB. Two of the three 3D pretraining sets read 5–9× more per
  micro-batch than anything in the 2D plan. The roofline must report data-wait time **per
  dataset**, with 2D and 3D shown separately. A 3D data-wait floor would be invisible in an
  aggregate `sec/epoch`.
- **Staging is out of the question for 3D** (≈ 1.8 TB for the three 3D train splits alone). The
  paired staged-vs-unstaged subset test of the 2D plan should use one 3D set (SN, the heaviest
  per-sample read small enough to stage at 576 GB) and one 2D set.

GATE A gains a third outcome: **cost is FLOP-proportional for 2D but I/O-bound for 3D.** In that
case, report it and fall back to the 2D plan (§11).

### 5.3 – 5.4

- **Reproducibility post-conditions:** as in the 2D plan. The resolved dataset list must equal the
  twelve.
- **Checkpoint re-evaluation (§5.3b):** unchanged.
- **Noise floor:** as in the 2D plan, regenerated for this config, with the `f_down` noise
  measured on all **six** targets (18 short finetunes).

---

## 6. Tier 1 — hyperparameter reduction

Unchanged in design. The small-extreme leg now runs at 1.9× the 2D plan's per-epoch cost, because
that shape is where 3D costs most. The historical fullDims evidence (old §6.2) comes from 2D-only
mixtures (`6DS`, `smallest7`): it is partial evidence here, and weaker than in the 2D plan.

---

## 7. Tiers 2–3

### 7.0 Duration

The four bounds are as before. **The repetition bound (upper bound 3) partly binds again:**
- TRL3D reaches 16× near epoch 58 and 27.5× at 100.
- SN reaches 16× near epoch 98.
- Every 2D dataset stays below 9× even at 100 epochs.

This favours the lower half of the `[50, 100]` window allowed by the cooldown-fraction bound
(`max_epoch` ≈ 60). The final choice still depends on the saturation pilot, as before.

### 7.1 – 7.3

The grid and star shapes are unchanged. Two things change:

- **Costs:** FLOP-proportional GPU-h rise by 1.59× (grid sweep, cost-weighted) and 1.17× (star
  sweep) against the 2D plan (§3.1).
- **GATE C:** the star-centre re-run on this mixture is still required. It is now the most
  paper-like fullDims run in the study (the same stems and parameter count, a 2D + 3D mixture), so
  it is also the best available reference point for RQ2. Cost ≈ 470 × 1.14 ≈ **540 GPU-h**.

---

## 8. Tiers 4–5 — head-to-head

As in the 2D plan §8, with `f_down` on all **six** targets for both arms.

The repetition confound is larger than in the 2D plan, but only for the two small 3D sets. In the
SoWalrus arm TRL3D reaches ≈ 595× and SN ≈ 352×, while every 2D set stays at ≤ 179× (PS).
Report `f_up,2D` and `f_up,3D` separately for the head-to-head. If SoWalrus wins on 2D but loses on
3D, the repetition of the two small 3D sets is the first suspect, and RQ3 (§9) is the evidence to
check.

---

## 9. Tier 6 — RQ3 (repetition)

The anchor shape, 60 epochs, N1:

| level | knob | fraction | aggregate `R` | TRL3D `R` | PS `R` | GS `R` | TRL3D / PS trajectories left |
|---|---|---|---:|---:|---:|---:|---|
| 0 | — | 1 | 1.6× | 17× | 5× | 0.5× | 72 / 96 |
| 1 | trajectories | 1/4 | 6.5× | 66× | 20× | 2.0× | 18 / 24 |
| 2 | **samples** | 1/16 | 26× | 264× | 79× | 8.0× | all (windows thinned) |
| 3 | **samples** | 1/64 | 104× | 1 058× | 317× | 32× | all (windows thinned) |

TRL3D now has the same trajectory-granularity floor PS had in the 2D plan (72 train trajectories).
Level 2 therefore also uses `restrict_train_num_samples`; check its sampling semantics in
`multidataset.py` before use, as the 2D plan notes.

The change that helps: **level 0 already spans a 34× range in per-dataset `R`**, from GS at 0.5×
to TRL3D at 17×. So the per-dataset breakdown gives repetition-decay evidence from the
unrestricted runs alone. Eight runs at ≈ 320 × 1.33 ≈ **430 GPU-h**.

---

## 10. Tier 7 — RQ5 downstream finetuning

Protocol, checkpoint sets and analysis are as in [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md)
§10, over **six targets**. For the 3D targets:

- **Configs:** `Leonardo_MHD_64.yaml` (on-cluster base path) and upstream
  `rayleigh_taylor_instability.yaml`, whose base path is overridden by `server=leonardo`. Neither
  needs a new file.
- **Match the pretraining 3D regime:** `n_steps_input=3`, `batch_size=1`. The single-dataset
  configs' own defaults (`n_steps_input: 10`, `batch_size: 8`) must not be used; they would give a
  3D context the model never saw in pretraining.
- **Scarce regime:** RT has only 45 train trajectories and MHD 77, so "16 trajectories" is 36 % and
  21 % of them. For MHD, restrict per file (10 files, one per (Mach, Alfvénic Mach) pair) so every
  parameter setting stays represented.
- **Costs:** at the anchor a 3D finetune costs ≈ 2× a 2D one per micro-batch, 1.3–1.4× at fullDims,
  and 3.5–4.3× at the grid's small end.
- **Analysis:** report `f_down` in four groups: 2D in-family (HS, TRL2, VE), 2D far (ACT),
  3D across-dimension (RT), 3D far (MHD). Four targets from the 2D plan plus two here is still too
  few to *estimate* axis effects. As before, the categorisation doc's crossover study is the
  instrument for that.

---

## 11. Budget and sequencing

FLOP-proportional GPU-h, obtained by scaling each line of the 2D plan's §11 by the 12-mix cost
multiplier for the shapes that line runs (§3.1). Like every GPU-h figure since `max_rollout_steps`
went 20 → 60, these are provisional until the roofline re-measures.

| tier | content | GPU-h (2D plan) | multiplier | **GPU-h (this plan)** |
|---|---|---:|---:|---:|
| 0 | FLOP re-validation (2D + 3D), roofline incl. paired staging test | 400 | ≈ 1.4 | 550 |
| 0 | Noise floor 3 seeds × 60 ep + 18 `f_down` finetunes | 170 | 1.33 + extra targets | 260 |
| 1 | LR × WD sensitivity | 630 | ≈ 1.5 (small extreme 1.9, HalfWalrus 1.18) | 950 |
| 2 | Grid sweep 4×4×4 @ 40 ep (+1 extension round) | 580 | 1.59 | 920 |
| 3 | Star sweep 18 × 60 ep | 4 343 | 1.17 | 5 080 |
| 3 | Star-centre re-run on this mixture (GATE C) | 470 | 1.14 | 540 |
| 3 | Cooldown-branch validation (RQ4) | 250 | 1.17 | 290 |
| 6 | Repetition sub-study (RQ3) | 320 | 1.33 | 430 |
| 5 | Head-to-head, 2 arms at matched FLOPs | 6 240 | 1.14 | 7 110 |
| 7a | Downstream core, 6 targets | 1 500 | + 2 targets at 1.3–2.0× per µb | 2 700 |
| — | **Core total** | 14 900 | | **≈ 18 800** |
| — | Contingency / reruns (~25 %) | 3 700 | | ≈ 4 700 |
| — | **Core + contingency** | | | **≈ 23 500** |
| 7b | Downstream exponent fit (optional, GATE B) | 1 300 | 6 targets, ≈ 1.2 | ≈ 2 300 |
| 5 | Deviation checks (optional) | 3 120 | 1.24 (SoWalrus) | ≈ 3 870 |
| — | **Everything** | 23 000 | | **≈ 29 700** |

**Everything together lands at the ~30K budget ceiling**, unlike the 2D plan, which had ~7K to
spare. So **at most one of the two optional lines fits**, and the choice between Tier 7b and the
deviation checks moves forward to GATE B.

A pessimistic dataloader-floor column is not given. The 2D plan's floor figure (7.8 GPU-h per
epoch, measured on 2D-heavy mixtures) is not a safe floor once 200–335 MB 3D reads are in the mix.
It could be higher, and only GATE A can say by how much.

**Fallback:** if GATE A shows 3D I/O dominating, or the budget can't absorb the ≈ 6 700 GPU-h
difference, run [SCALING_STRATEGY_2D.md](SCALING_STRATEGY_2D.md) instead. Its 2D split is
identical, so everything built for this plan except the 3D-specific lines carries over. MHD and RT
can still be added to that plan as 3D downstream targets; the 3D stem would then be trained only
during finetuning.

### Sequencing

As in the 2D plan §11, with these prerequisites before GATE A:
- ✅ the Euler download completes, and a re-survey replaces every `~` (2026-10-02)
- `validate_flops_real_data.py` passes on all **12** datasets, including 3D geometry under `=0`
- the roofline's per-dataset data-wait shows the 3D sets are not I/O-bound — otherwise, fall back

---

## 12. Deliverables

- ✅ `walrus/configs/data/Leonardo_pretrain12_2d3d.yaml`, composition-checked. It loaded real
  batches from the 10 non-Euler datasets when written; Euler's two entries are identical to
  `Leonardo_pretrain9_2d.yaml`'s, whose datamodule now loads real Euler batches (2026-10-02).
- ✅ This document.
- ✅ Euler re-survey (§2.1, §2.6), 2026-10-02.
- 🔲 FLOP re-validation on 12 datasets, `=0`, and 3D (§5.1). The 9 2D datasets pass under `=3`
  (`LeonardoRunScripts/PT9_2d/validate_flops_pt9.py`, SCALING_STRATEGY_2D.md §5.1); a PT12 variant
  of that script under `=0` must still add the three 3D sets.
- 🔲 Generators switched to `data=Leonardo_pretrain12_2d3d` and `model.override_dimensionality=0`,
  and regenerated (roofline, noise floor, grid, star). **Note that every current generator pins
  `=3`**, correct for the 2D plan and wrong for this one. If both plans are kept, the generators
  should take the data config and `override_dimensionality` as a pair, so they can't drift apart.
- 🔲 `gen_downstream_runs.py` with per-target settings (the 3D `n_steps_input=3`, `batch_size=1`;
  per-file restriction for VE and MHD).
- 🔲 Per-run 2D/3D FLOP-split logging: a dimensionality-keyed epoch accumulator in
  `FlopAccountant` (`walrus/utils/flops.py`), plus a `tests/test_flops.py` case (§3.1).

**Figures:** as in the 2D plan, plus:
- (12) 2D vs 3D FLOP share and enc+dec share across the grid.
- (13) `s_k` fitted on `f_up,2D` vs `f_up,3D`.
- (14) If both plans run: `s_k` from the 2D-only pretraining vs this one.

---

## 13. Open risks

Every risk in the 2D plan §13 still applies. New or changed:

| risk | mitigation |
|---|---|
| 3D per-sample reads (up to ≈ 335 MB) make the study I/O-bound | per-dataset data-wait in the roofline (§5.2); fall back to the 2D plan (§11) |
| 3D stem ballast dominates small-shape compute (61 % of grid-corner FLOPs, 91.5 % of grid-corner parameters) | processor-only fit and the 2D/3D FLOP split on every point (§3.1) |
| 3D stem trained on only 3 datasets, all family II | stated; the 3D targets measure exactly this (MHD far, RT across dimension) |
| TRL3D and SN heavily repeated (16.5× / 9.8× at 60 ep; ≈ 595× / 352× in the SoWalrus arm) | `max_epoch` near 60 (§7.0); `f_up,2D` / `f_up,3D` split for the head-to-head (§8); RQ3 per-dataset breakdown (§9) |
| Budget at the ceiling with both optional lines | only one of Tier 7b and the deviation checks fits; choose at GATE B (§11) |
| Closed form unvalidated on real 3D data | extend `validate_flops_real_data.py` before Tier 0 (§5.1) |
| Current generators pin `override_dimensionality=3` | switch them together with the data config (§12) |
| `distribution=ddp` (FSDP `NO_SHARD`, per-GPU dataset draw) could run different stems on different ranks and hang | `distribution=fsdp` is mandatory (§4.1); add it to the §5.3 post-condition check |
| Mixed rollout horizons within `f_up,3D` (47 / 56 / 60 steps) | report per-dataset horizons; one-step and T1–10 secondaries (§3.4) |
| TRL2 and TRL3D density enter under different field names (`density` vs `log10_density`) | stated in §2.4; it makes the across-dimension test cleaner |
| No regular 3D dataset exists to hold out | stated in §2.4: 3D regime coverage is chaotic-only |
