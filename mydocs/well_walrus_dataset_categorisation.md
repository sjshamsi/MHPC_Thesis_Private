# How do we categorise The Well datasets

The Well says it has 16 datasets, but`acoustic_scattering` has three variants (discontinuous / inclusions / maze), `MHD` and `supernova_explosion` come with two resolutions, and `turbulent_radiative_layer` comes in 2D and 3D.

WALRUS pretrains on 19 datasets, 18 from The Well (acoustic variants and Euler boundary-condition variants treated seperately) plus `FBHarmonics` from FlowBench where a fluid flows pas an obstacle.

`post_neutron_star_merger` and `convective_envelope_rsg` are held out of pretraining and used as downstream tasks. Both 3D and expensive through numerical solvers.

---

## Short answers

| Question | Short answer |
|---|---|
| 1. Categorise by physics | Seven physical domains (§1), but The Well is better described by a multi-label set of physical ingredients (gravity, radiation, magnetic fields, rotation, shocks, and so on), because several datasets combine them. |
| 2. Categorise by equation form | Six equation families (§2). Most of The Well is one family: **compressible Euler plus source terms**. Several "equation twins" sit in very different physical contexts, e.g. `euler_multi_quadrants` ↔ `turbulent_radiative_layer`, and `shear_flow` ↔ `rayleigh_benard`. |
| 3. Do the papers categorise? | **The Well: no.** Datasets are listed alphabetically and it states explicitly that there is "no cross-linkage between different datasets". **WALRUS: yes, but loosely.** Its Table 1 groups datasets under seven headings that mix application domain with fluid-model type. **Neither paper categorises strictly by equation form.** |
| 4. Does WALRUS train or experiment by category? | **No.** Pretraining samples uniformly over *datasets*, not categories. The categories are used only to *present and interpret* per-dataset results after the fact. There is no category-held-out or category-balanced experiment. |
| 5. How should we split the data for our own experiments? | One crossover experiment per axis (equation family, linearity, dissipation, smoothness, predictability), plus a boundary-condition negative control and a data-driven axis. The axes are then compared by how well each one predicts transfer (§5). |

---

## 1. Categorisation by the physics described

### 1.1 By physical domain (my grouping)

| Domain | Datasets | Physics |
|---|---|---|
| **Acoustics / linear wave propagation** | `acoustic_scattering` (×3), `helmholtz_staircase` | Sound waves in heterogeneous media; scattering off a periodic, sound-hard surface |
| **Classical (laboratory-scale) fluid dynamics** | `shear_flow`, `rayleigh_benard`, `rayleigh_taylor_instability`, `euler_multi_quadrants` | Shear/Kelvin–Helmholtz instability, thermal convection, buoyancy-driven mixing, shock and rarefaction interactions |
| **Geophysical fluid dynamics** | `planetswe` | Rotating, forced shallow-water atmosphere on the sphere, initialised from ERA5 |
| **Complex / soft-matter fluids** | `viscoelastic_instability`, `active_matter` | Polymer solutions (elasto-inertial turbulence); self-propelled particles in a viscous fluid |
| **Chemistry / biology pattern formation** | `gray_scott_reaction_diffusion` | Two-species autocatalytic reaction–diffusion (Turing-type patterns) |
| **Astrophysics: interstellar medium** | `MHD`, `supernova_explosion`, `turbulence_gravity_cooling`, `turbulent_radiative_layer` (2D/3D) | ISM turbulence, supernova blast waves, self-gravitating cooling clouds, multiphase mixing layers |
| **Astrophysics: stars and compact objects** | `convective_envelope_rsg`, `post_neutron_star_merger` | Red-supergiant envelope convection with radiation; relativistic accretion disk after a neutron-star merger |

### 1.2 By physical ingredient (multi-label)

A single domain label hides a lot. `post_neutron_star_merger`, for instance, is simultaneously
MHD, relativistic, radiative and rotating. The matrix below comes from the governing equations in
[Well] Appendix C (pp. 31–44).

| Dataset | Compressible | Shocks / discontinuities | Turbulence | Gravity / buoyancy | Radiation / cooling | Magnetic field | Rotation | Microstructure tensor | Chemistry / composition |
|---|---|---|---|---|---|---|---|---|---|
| `acoustic_scattering` | (linear) | ✓ (in the medium ρ) | | | | | | | |
| `helmholtz_staircase` | (linear) | | | | | | | | |
| `euler_multi_quadrants` | ✓ | ✓ | | | | | | | |
| `turbulent_radiative_layer` | ✓ | | ✓ | | ✓ (cooling) | | | | |
| `supernova_explosion` | ✓ | ✓ (blast wave) | ✓ | ✓ (self-gravity) | ✓ (heat/cool) | | | | |
| `turbulence_gravity_cooling` | ✓ | | ✓ | ✓ (self-gravity) | ✓ (heat/cool) | | | | |
| `convective_envelope_rsg` | ✓ | | ✓ | ✓ | ✓ (transport) | | | | |
| `MHD` | ✓ (isothermal) | ✓ (supersonic runs) | ✓ (driven) | | | ✓ | | | |
| `post_neutron_star_merger` | ✓ | | ✓ (MRI) | ✓ (GR) | ✓ (neutrinos) | ✓ | ✓ (BH spin) | | ✓ (lepton no.) |
| `planetswe` | (depth-avg.) | | | ✓ (g) | | | ✓ (Coriolis) | | |
| `rayleigh_benard` | | | ✓ (high Ra) | ✓ (buoyancy) | | | | | |
| `rayleigh_taylor_instability` | (low-Mach, var. ρ) | | ✓ | ✓ | | | | | |
| `shear_flow` | | | ✓ | | | | | | |
| `viscoelastic_instability` | | | ✓ (EIT) | | | | | ✓ (conformation C) | |
| `active_matter` | | | (active turb.) | | | | | ✓ (orientation D) | |
| `gray_scott_reaction_diffusion` | | | | | | | | | ✓ |

---

## 2. Categorisation by the functional form of the governing equations

### 2.1 Governing system of each dataset

Condensed from [Well] Appendix C. The "Solver" column matters because the numerical method leaves
its own imprint on the data.

| Dataset | Governing system | Closure / EOS | Extra terms beyond the core system | Dissipation | Solver |
|---|---|---|---|---|---|
| `acoustic_scattering` | Linear 1st-order acoustic system: ∂p/∂t + K∇·u = 0, ∂u/∂t + ∇p/ρ = 0 | Variable ρ(x), K = 4 | none | none | Clawpack FV (TVD) |
| `helmholtz_staircase` | Wave equation ∂²U/∂t² − ΔU = δ(t)δ(x−x₀); solved as the Helmholtz BVP; time series is U = u·e^{−iωt} | constant medium | Neumann (sound-hard) boundary | none | Boundary integral + Floquet–Bloch |
| `euler_multi_quadrants` | 2D compressible **Euler** (mass, momentum, energy) | ideal gas, γ ∈ {1.13 … 1.76} | none | none (inviscid) | Clawpack FV |
| `turbulent_radiative_layer` | Compressible **Euler** | ideal gas, γ = 5/3 | energy sink −E/t_cool | none | Athena++ |
| `supernova_explosion` | Compressible **Euler**, Lagrangian form | ideal gas, γ = 5/3 | self-gravity −∇Φ, heating/cooling (Γ−Λ)/ρ | artificial viscosity | SPH (ASURA-FDPS), gridded |
| `turbulence_gravity_cooling` | *Same equations as `supernova_explosion`* ([Well] p. 43) | ideal gas, γ = 5/3 | self-gravity, metallicity-dependent heating/cooling | artificial viscosity | SPH, gridded |
| `convective_envelope_rsg` | Compressible **Euler** + radiative transfer for intensity I | gas + radiation | gravity, radiation four-force | none | Athena++ (Godunov) |
| `MHD` | Ideal compressible **MHD** (continuity, momentum with magnetic pressure/tension, induction) | isothermal, p = c_s²ρ | solenoidal forcing | none (ideal) | ENO FV |
| `post_neutron_star_merger` | **General-relativistic** ideal MHD + lepton-number conservation + neutrino transport | tabulated nuclear EOS (SFHo) | curved spacetime, neutrino four-force, lepton source | none (ideal) | νbhlight (FV + Monte Carlo) |
| `planetswe` | Rotating **shallow-water** equations on the sphere | h ↔ depth | Coriolis, seasonal/daily forcing F | hyperviscosity ν∇⁴ | Dedalus spectral |
| `shear_flow` | 2D incompressible **Navier–Stokes** + **passive** scalar advection–diffusion | incompressible | none | viscous (Re), diffusive (Sc) | Dedalus spectral |
| `rayleigh_benard` | 2D incompressible **Navier–Stokes** (Boussinesq) + **active** scalar (buoyancy) advection–diffusion | incompressible | buoyancy force b·e_z | viscous, diffusive (Ra, Pr) | Dedalus spectral |
| `rayleigh_taylor_instability` | Variable-density, low-Mach **Navier–Stokes**, with ∇·u = −κ∇·(∇ρ/ρ) | miscible, variable ρ | gravity ρg | viscous, molecular diffusion | TurMix3D FV |
| `viscoelastic_instability` | Incompressible **Navier–Stokes** + FENE-P **conformation-tensor** evolution | incompressible | polymer stress ∇·T(C) | solvent viscosity β, polymer diffusion ε | Dedalus spectral |
| `active_matter` | **Stokes** flow (no inertia) + Smoluchowski kinetic equation (moments: concentration, orientation tensor) | incompressible | active stress ∇·Σ | translational/rotational diffusion | pseudo-spectral |
| `gray_scott_reaction_diffusion` | **Reaction–diffusion**, 2 species; no velocity field | — | cubic reaction AB² | diffusion | Chebfun spectral (ETDRK4) |

### 2.2 Equation families (my analysis)

**I. Linear wave equations** (linear hyperbolic)
- `acoustic_scattering`: first-order acoustic system with variable coefficients
- `helmholtz_staircase`: second-order wave equation, constant medium, complex boundary, time-harmonic

**II. Compressible Euler plus source terms** (nonlinear hyperbolic conservation laws) — *the
largest family*
- IIa. Pure Euler: `euler_multi_quadrants`
- IIb. + optically thin cooling sink: `turbulent_radiative_layer` (2D, 3D)
- IIc. + self-gravity + heating/cooling (+ SPH artificial viscosity): `supernova_explosion`,
  `turbulence_gravity_cooling`
- IId. + gravity + full radiation transport: `convective_envelope_rsg`
- IIe. Depth-averaged analogue: `planetswe`. The shallow-water equations are mathematically
  identical to 2D **isentropic** Euler with γ = 2 (h plays the role of ρ, and gh²/2 the role of p).
  This is a standard result, not stated in either paper. On top of that come rotation, forcing,
  hyperviscosity and spherical geometry.

**III. Compressible MHD** (Euler + induction equation)
- IIIa. Isothermal, ideal, forced: `MHD`
- IIIb. General-relativistic, ideal, + lepton transport + neutrino radiation:
  `post_neutron_star_merger`

**IV. Incompressible / low-Mach Navier–Stokes coupled to a transported field**
- IVa. + passive scalar: `shear_flow`
- IVb. + active scalar that feeds back as a body force (Boussinesq buoyancy): `rayleigh_benard`
- IVc. Variable density + gravity (non-Boussinesq): `rayleigh_taylor_instability`
- IVd. + active second-order tensor that feeds back as a stress (polymer conformation):
  `viscoelastic_instability`

**V. Stokes flow coupled to a kinetic / moment equation for microstructure**
- `active_matter`: inertia-less cousin of IVd; an orientation tensor feeds back through an active
  stress

**VI. Reaction–diffusion** (nonlinear parabolic, no flow)
- `gray_scott_reaction_diffusion`: the only dataset with no velocity field and no advection

Cross-cutting mathematical labels:

| Property | Datasets |
|---|---|
| Linear | `acoustic_scattering`, `helmholtz_staircase` (everything else is nonlinear) |
| Ideal / non-dissipative | `acoustic_scattering`, `helmholtz_staircase`, `euler_multi_quadrants`, `turbulent_radiative_layer`, `convective_envelope_rsg`, `MHD`, `post_neutron_star_merger` |
| Elliptic constraint (incompressibility / Stokes) | `shear_flow`, `rayleigh_benard`, `rayleigh_taylor_instability`, `viscoelastic_instability`, `active_matter` |
| Curvilinear coordinates | `planetswe` (sphere), `convective_envelope_rsg` (r,θ,φ), `post_neutron_star_merger` (log r,θ,φ) |

### 2.3 How the families connect

Most of the families are limits or extensions of one another, so The Well is more tightly linked
at the equation level than its domain labels suggest:

```
                         Compressible Euler  (II)
                                 |
   ┌──────────────┬──────────────┼─────────────────┬───────────────────────┐
   │ linearise     │ add B-field   │ low-Mach limit   │ depth-average          │ add source terms
   │ about rest    │ + induction   │ + viscosity      │ (≅ isentropic, γ = 2)  │ (cooling, gravity,
   ▼               ▼               ▼                  ▼                        ▼  radiation)
 Linear acoustics  MHD (III)      Incompressible     Shallow water           TRL, SN, TGC, RSG
 (I)               └─ GR → PNS    Navier–Stokes (IV)  (planetswe)
                                   ├─ + passive scalar    → shear_flow
                                   ├─ + buoyancy          → rayleigh_benard
                                   │    (Boussinesq = small-Atwood limit of → rayleigh_taylor)
                                   └─ + polymer tensor    → viscoelastic
                                          └─ drop inertia, kinetic closure → active_matter (V)

 Reaction–diffusion (VI): isolated. Shares only the diffusion operator with IV.
```

Two of these links are worth stating explicitly:

- Linearising Euler about a fluid at rest gives *exactly* the acoustic system used in
  `acoustic_scattering`, with bulk modulus K = γp₀.
- The [Well] itself names the Boussinesq regime as Atwood < 0.1 (p. 41), which is the bridge from
  `rayleigh_taylor_instability` to `rayleigh_benard`-type equations.

### 2.4 "Equation twins" in different physical contexts

These are the pairs your question points at: different physical settings, essentially the same
governing equations.

| Pair | Physical contexts | What the equations share | What differs |
|---|---|---|---|
| `euler_multi_quadrants` ↔ `turbulent_radiative_layer` | Textbook gas-dynamics Riemann problem ↔ astrophysical hot/cold gas mixing layer | Identical compressible Euler system | TRL adds one energy sink (−E/t_cool); γ fixed at 5/3; shear-driven turbulence rather than shock interaction |
| `shear_flow` ↔ `rayleigh_benard` | Kelvin–Helmholtz-type shear layers ↔ thermal convection | 2D incompressible NS + one scalar advection–diffusion equation; same solver (Dedalus); same field shapes (scalar, pressure, velocity) | Whether the scalar feeds back (buoyancy) and the boundaries (fully periodic vs walls) |
| `planetswe` ↔ `euler_multi_quadrants` | Earth's atmosphere ↔ gas dynamics | Shallow water ≅ isentropic Euler with γ = 2 | Rotation, forcing, hyperviscosity, spherical geometry, no shocks by design |
| `viscoelastic_instability` ↔ `active_matter` | Industrial polymer solutions ↔ biological active suspensions | Velocity coupled to an evolving 2nd-order tensor that re-enters the momentum balance as a stress divergence | Inertia (Re = 1000 vs Stokes); passive relaxation vs active driving |
| `acoustic_scattering` ↔ `helmholtz_staircase` | Geophysical probing ↔ photonic/phononic crystal design | Linear acoustic wave equation | Variable medium vs constant medium with a complex boundary; transient vs time-harmonic |
| `MHD` ↔ `post_neutron_star_merger` | Diffuse ISM turbulence ↔ neutron-star merger remnant | Ideal MHD conservation laws + induction | GR, full energy equation, lepton and neutrino transport |
| `supernova_explosion` ↔ `turbulence_gravity_cooling` | Same domain (ISM), different scenario | Literally the same equations ([Well] p. 43) | Initial conditions: point energy injection vs turbulent self-gravitating cloud |
| `rayleigh_benard` ↔ `rayleigh_taylor_instability` | Heated layer ↔ two fluids of different density | Gravity-driven incompressible flow | Boussinesq vs variable density; 2D vs 3D |

### 2.5 Caveat: the same equations do not imply the same dynamics

Your question assumes that inherently similar equations give similar dynamics. That holds only
when the **regime** also matches. Within one equation family, the behaviour an ML model actually
sees is shaped by several further factors:

- **Dimensionless parameters.** Mach, Reynolds, Rayleigh, Atwood, Weissenberg. For example,
  `euler_multi_quadrants` is shock-dominated, while `turbulent_radiative_layer` is dominated by
  subsonic shear mixing.
- **Boundary conditions and geometry.** Periodic vs walls vs open; Cartesian vs spherical.
- **Initial-condition families.** `supernova_explosion` vs `turbulence_gravity_cooling` share
  equations but look nothing alike.
- **Which state variables are exposed.** `euler_multi_quadrants` stores density, energy, pressure
  and **momentum**; `turbulent_radiative_layer` stores density, pressure and **velocity**;
  `supernova_explosion` adds **temperature**. WALRUS embeds fields by name (the
  `field_to_index_map` in this repo), so the model is not told that momentum in one dataset is ρ×
  velocity in another.
- **Numerical method.** SPH-derived data resampled onto a grid (SN, TGC), TVD finite volume
  (Clawpack) and spectral methods (Dedalus) each leave different artefacts.

**Concrete evidence from The Well's own baselines** ([Well] Table 2, p. 8). `acoustic_scattering`
and `helmholtz_staircase` share the linear wave equation, yet they favour *opposite* architectures:

| One-step VRMSE | FNO | TFNO | U-net | CNextU-net |
|---|---|---|---|---|
| `acoustic_scattering` (maze) | 0.5062 | 0.5057 | 0.0351 | **0.0153** |
| `helmholtz_staircase` | **0.00046** | 0.00346 | 0.01931 | 0.02758 |

**My reading:** the sharp density discontinuities in the maze favour spatial-domain models, while
the smooth time-harmonic field favours spectral ones. That points to smoothness and discontinuities
mattering at least as much as equation form. The [Well] reports this spectral-vs-spatial split
across datasets (9/17 vs 8/17) and says "the reason for this is not immediately clear" (p. 8).

---

## 3. Do the papers categorise the datasets?

### 3.1 The Well: no formal categorisation

- **Table 1** (p. 3) and **§3.1** (pp. 3–7) list the datasets **alphabetically**, one subsection
  each. Table 1 records only technical attributes: coordinate system, resolution, number of steps
  and trajectories.
- The abstract and introduction name domains descriptively ("biological systems, fluid dynamics,
  acoustic scattering, as well as magneto-hydrodynamic simulations of extra-galactic fluids or
  supernova explosions", p. 1) but do not turn this into a taxonomy.
- **Appendix C** (pp. 31–44) gives the governing equations for every dataset, but in isolation,
  without comparing them. The only cross-reference is that `turbulence_gravity_cooling` "also
  follows the equations (40)–(43)" of `supernova_explosion` (p. 43).
- **Datasheet Q11** (p. 24) states directly: *"There is no cross-linkage between different
  datasets."*
- **Appendix D** (p. 45) does group datasets, but **by task suitability**, not physics: e.g.
  super-resolution (MHD, supernova), transfer across dimensionality (TRL 2D/3D), transfer across
  parameter ranges, steady-state prediction, long-term stability (`planetswe`), inverse scattering
  (acoustic, Helmholtz), and simulation acceleration (PNS, TGC).
- The **benchmark analysis** (p. 8) splits datasets empirically by which architecture wins (spatial
  vs spectral) and cannot explain the split with physics. The authors tested boundary conditions as
  an explanation and found "no clear trend".

### 3.2 WALRUS: a loose seven-way categorisation, used for presentation

The abstract lists domains: "astrophysics, geoscience, rheology, plasma physics, acoustics, and
classical fluids" (p. 1). The actual grouping is **Table 1** (p. 10), which presents cross-domain
results under seven headings:

| WALRUS heading | Datasets | Criterion implied |
|---|---|---|
| Biological and chemical behavior | Active Matter, Gray-Scott | application domain |
| Acoustics and wave propagation | Maze, Inclusions, Discontinuous, Staircase | physics / equation (linear waves) |
| Astrophysical and geoscience applications | Supernova (3D), TGC (3D), PlanetSWE | application domain |
| Plasmas | MHD (3D) | physics |
| Viscous fluids | Rayleigh-Benard, Shear Flow, FBHarmonics, RT Instability (3D) | fluid model |
| Inviscid fluids | MultiQuadrantsP, MultiQuadrantsO, TRL (2D), TRL (3D) | fluid model |
| Non-newtonian fluids | Viscoelastics | fluid model (constitutive law) |

The headings **mix criteria**: some are application domains, others are fluid-model or equation
classes. This produces inconsistencies when read as an equation taxonomy:

- **TRL** is astrophysical but sits under *Inviscid fluids*, grouped by equations with
  `euler_multi_quadrants`. **Supernova** and **TGC**, which solve the same compressible Euler +
  cooling family plus gravity, sit under *Astrophysical applications*, grouped by domain.
- **PlanetSWE** (geophysics) is grouped with astrophysics, not with the other fluid dynamics.
- **Active Matter** is grouped with Gray-Scott (both "biological"), although at the equation level
  it is closest to *Viscoelastics* (§2.4).

The paper does not define these headings; they organise a results table.

### 3.3 Neither paper categorises by equation form, though WALRUS reasons that way informally

Neither paper builds a categorisation from the functional form of the equations. WALRUS does,
however, **use equation-class reasoning to interpret results** (§5.2, p. 11):

- DPOT, whose core is the Fourier-based AFNO, is most competitive on *acoustics and linear wave
  propagation*, and even beats WALRUS at long horizons there. The authors connect this to Fourier
  methods being efficient numerical solvers "in smooth linear settings", and suggest "we may be
  able to derive intuition about architecture design from the classes of methods used for
  particular problems".
- Poseidon is most competitive on *inviscid fluids*, especially the Euler multi-quadrants problem.
  The authors explain this by Poseidon's pretraining data: 4/6 of its tasks were Euler solutions,
  2/6 specifically Euler quadrants.
- Appendix B.2.1 (p. 26) reasons by symmetry. Absolute position embeddings hurt translation-invariant
  systems like Euler and help systems with fixed spatial heterogeneity like Rayleigh–Bénard.

---

## 4. Does WALRUS pretrain or experiment with categories differentiated?

**No.** Categories appear only in how per-dataset results are *reported*.

### 4.1 Pretraining is category-agnostic

- **Sampling.** "a hierarchical sampling scheme where we uniformly sample a system S then uniformly
  select a starting frame" (§4.2, p. 6). The unit is the *dataset*, not the category.
- **Loss.** The per-field normalised L1 loss (Eq. 8) balances *fields*, not categories.
- **Distributed sampling.** Under HSDP, every rank in a sharding group samples the *same dataset*
  each step (p. 7). This is purely for compute load-balancing, not physics.
- **Implicit imbalance.** Because sampling is uniform over datasets, a category's share of
  pretraining equals how many datasets it happens to contain. Using the paper's figures of ~4M
  examples per 2D dataset and ~2M per 3D dataset (p. 8), approximate shares are:

| Grouping | Datasets | ≈ share of pretraining examples |
|---|---|---|
| *WALRUS headings* | | |
| Acoustics and wave propagation | 4 (2D) | 24% |
| Viscous fluids | 3 (2D) + 1 (3D) | 21% |
| Inviscid fluids | 3 (2D) + 1 (3D) | 21% |
| Astrophysical and geoscience | 1 (2D) + 2 (3D) | 12% |
| Biological and chemical | 2 (2D) | 12% |
| Non-newtonian | 1 (2D) | 6% |
| Plasmas | 1 (3D) | 3% |
| *My equation families* | | |
| II. Compressible Euler + sources (incl. `planetswe`) | 4 (2D) + 3 (3D) | 33% |
| IV. Incompressible NS + fields (incl. FBHarmonics) | 4 (2D) + 1 (3D) | 27% |
| I. Linear waves | 4 (2D) | 24% |
| V. Stokes + kinetic (`active_matter`) | 1 (2D) | 6% |
| VI. Reaction–diffusion (`gray_scott`) | 1 (2D) | 6% |
| III. MHD | 1 (3D) | 3% |

This imbalance is incidental rather than designed. The paper does not discuss it.

### 4.2 The experiments

| Experiment | What varies | Role of categories |
|---|---|---|
| §5.1 Downstream performance (pp. 8–10) | WALRUS vs MPP / Poseidon / DPOT, finetuned on held-out datasets | Results split only by **2D vs 3D** (Fig. 6, Fig. 7, Table 8) |
| §5.2 Cross-domain analysis (pp. 10–11) | Per-dataset finetuning on all 19 pretraining datasets | Results **grouped by the 7 headings** in Table 1, with qualitative per-category discussion (§3.3 above). No training by category. |
| §5.3 Pretraining strategy (pp. 11–12) | HalfWALRUS (full augmentation) vs Naive (strict 2D, no augmentation) vs Scratch | Pretrained on a fixed 2D subset (Maze, Discontinuous, Gray-Scott, Rayleigh-Benard, Shear Flow, TRL2D, Staircase, Viscoelastics), "chosen specifically for having strong anisotropic behavior". The **diversity manipulated is augmentation** (2D→3D embedding, rotations, time-striding), **not the mixture of physics categories**. |
| Patch-jitter ablation (Table 7, p. 33) | Jitter vs no jitter | Per dataset, no grouping |
| Finetuning hyperparameter search (App. B.3, p. 26) | Learning rate, batch size for baselines | Tuned on Euler periodic, acoustic discontinuous, Rayleigh-Benard and active matter. They appear to span categories, but no rationale is given. |

### 4.3 What was not done

Given the paper's headline claim of being "diversity-first", these are the category-level
experiments that are absent:

- **Leave-one-category-out pretraining**, e.g. hold out all compressible-Euler datasets and measure
  transfer to them.
- **Category-balanced vs dataset-balanced sampling**, which would correct the 24% acoustics vs 3%
  plasma imbalance.
- **Physics-diversity scaling**: fixing the number of examples and varying how many *categories or
  equation families* the pretraining mixture spans. The §5.3 ablation varies augmentation diversity
  at a fixed physics mixture, which is a different question.
- **Cross-family transfer**, e.g. whether pretraining on `euler_multi_quadrants` helps
  `turbulent_radiative_layer` more than pretraining on `shear_flow` does, i.e. whether equation-twin
  structure (§2.4) is actually exploited by the model.

The closest thing in the literature cited by WALRUS is the Poseidon observation in §3.3: strong
Euler performance explained by Euler-heavy pretraining. That is anecdotal evidence for equation-level
transfer, not a controlled test.

---

## 5. Proposed experimental splits

### 5.1 Design principle

The goal is to find out **which way of grouping datasets best predicts transfer**, instead of
assuming one. Each axis gets the same **crossover** design:

1. Split the datasets into two levels along the axis (e.g. *linear* vs *nonlinear*).
2. Pretrain one model on **two datasets from level A** and one on **two datasets from level B**,
   at an identical budget.
3. Finetune both models on **held-out targets from both levels**. The targets must be in neither
   pretraining set.
4. Compute the **in-level advantage** Δ: how much better each model does on its own level's
   targets than the other model does.

With transfer gain `g(run, t) = 1 − VRMSE(run, t) / VRMSE(scratch, t)` at a fixed finetuning
budget:

```
Δ_axis = ½ · [ mean_{t ∈ T_A} ( g(A,t) − g(B,t) )  +  mean_{t ∈ T_B} ( g(B,t) − g(A,t) ) ]
```

Δ > 0 means grouping along this axis captures something that transfers. The double difference
cancels out both target difficulty and the generic "any pretraining helps" effect, so no separate
mixed control is needed per axis. **The headline result is a comparison of Δ across axes.** For
multi-level axes (equation family), Δ is the mean over levels of (in-level gain − mean
other-level gain).

**Scope: 2D only.** In WALRUS, 2D and 3D use separate encoder/decoder weights, so transfer between
them mostly tests the encoder (§2.5). A 3D version is sketched in §5.6.

### 5.2 Dataset attributes (2D, on disk under `/leonardo_work/ICT26_MHPC_0/sshamsi/datasets/`)

Boundary conditions and field names below are read from each dataset's HDF5 metadata. The
predictability proxy is the **best** of the four models' median VRMSE averaged over steps 21–60
in [WALRUS] Table 1 (p. 10). Values ≳ 0.6 mean even the best model has decorrelated from the
truth by that horizon, which WALRUS notes is expected for chaotic systems (p. 9).

| Code | Dataset | Family (§2.2) | Linear | Explicit diffusion / viscosity | Sharp features | Predictability proxy (T21–60) | BCs (metadata) | Fields |
|---|---|---|---|---|---|---|---|---|
| AD | `acoustic_scattering_discontinuous` | I waves | ✓ | none | ✓ material interface | 0.04 predictable | mixed wall / open | pressure, velocity, (density†, speed_of_sound) |
| AI | `acoustic_scattering_inclusions` | I waves | ✓ | none | ✓ inclusions | 0.13 predictable | mixed wall / open | same as AD |
| AM | `acoustic_scattering_maze` | I waves | ✓ | none | ✓ maze walls | 0.05 predictable | mixed wall / open | same as AD |
| HS | `helmholtz_staircase` | I waves | ✓ | none | — smooth, time-harmonic | 0.003 predictable | open + wall (staircase) | pressure_re, pressure_im, mask |
| TRL2 | `turbulent_radiative_layer_2D` | II compressible | — | none (cooling sink only) | ✓ hot/cold phase interfaces | 0.84 **chaotic** | x periodic, y open | density, pressure, velocity |
| PS | `planetswe` | II (SWE analogue), **sphere** | — | hyperviscosity only | — | 0.02 predictable | φ periodic, θ open | height, velocity |
| SF | `shear_flow` | IV incompressible + field | — | viscous + diffusive | — | 0.08 predictable | fully periodic | pressure, tracer, velocity |
| RB | `rayleigh_benard` | IV incompressible + field | — | viscous + diffusive | — | 0.64 **chaotic** | x periodic, y wall | buoyancy, pressure, velocity |
| VE | `viscoelastic_instability` | IV incompressible + field | — | viscous + polymer diffusion | borderline (thin polymer-stress sheets) | n/a (short trajectories); contains chaotic attractors (CAR, EIT) | x periodic, y wall | pressure, velocity, C, c_zz |
| ACT | `active_matter` | V Stokes + kinetic | — | Stokes + diffusion | — | 1.25 **chaotic** | fully periodic | concentration, velocity, D, E |
| GS | `gray_scott_reaction_diffusion` | VI reaction–diffusion | — | diffusion | — | 0.13 predictable | fully periodic | A, B |
| EMQ-P\* | `euler_multi_quadrants` periodic | II compressible | — | none | ✓ shocks | 0.28 intermediate | periodic | density, energy, pressure, momentum |
| EMQ-O\* | `euler_multi_quadrants` open | II compressible | — | none | ✓ shocks | 0.28 intermediate | open | same as EMQ-P |

\* **Not on disk.** Both boundary-condition variants together are ~5 TB. The datasets on disk
currently total ~4.7 TB, against a 20 TB quota as of my 22 Sep note; check `cinQuota` before
downloading. Several splits below are much weaker without Euler, and each one says so.

† `density` is zeroed in the configs (`field_transforms: density: torch.zeros_like`), matching
[WALRUS] App. C.1.

**Caveat on the predictability proxy:** it also depends on step size and on model skill. Treat it
as a rough ordering, and replace it with the descriptors from §5.4 (A7) once they are computed.

### 5.3 Aliasing between axes on this data

With only ~11 datasets, several axes are partly confounded. It is better to state this up front
than to discover it in the results:

- **Linearity ≡ the wave family.** Every linear dataset is in family I. Axis A2 therefore
  reuses the family-I run; its only new information is whether a *multi-family* nonlinear model
  differs from the single-family incompressible one.
- **Dissipation ≈ sharpness.** Every dataset with sharp features is non-diffusive, and no
  diffusive dataset has shocks or interfaces. HS is the only smooth, non-diffusive dataset. Axes
  A3 and A4 therefore share their sharp / non-diffusive run and differ only in the contrasting
  run.
- **RB and VE share a representation**: same boundaries (x periodic, y wall) and overlapping
  fields (pressure, velocity + one more). Any in-family transfer from RB to VE may be partly
  representational. The negative control (A6) is designed to measure exactly this.
- **Shared field names.** `velocity` appears in every dataset except GS and HS, and `pressure` in
  most. To stop this acting as a free transfer channel, **reinitialise the field embeddings at
  finetuning for every target** as the primary setting; keeping them is a one-off ablation.

### 5.4 The splits

Each split is `pretrain set → held-out targets`. **Test** columns list targets held out of *both*
runs in that axis. "Without Euler" gives the fallback when `euler_multi_quadrants` isn't
downloaded.

#### A1. Equation family (3 levels)

| Level | Pretrain | In-level test | Without Euler |
|---|---|---|---|
| I Linear waves | AD, AM | HS (AI as a near-duplicate sanity check) | unchanged |
| II Compressible | EMQ-P, EMQ-O | TRL2, PS | {TRL2} → PS only (spherical, confounded). Better: drop level II and use TRL2 and PS as out-of-family probes |
| IV Incompressible + field | SF, RB | VE | unchanged |
| Out-of-family probes (all runs) | — | ACT, GS | ACT, GS (+ TRL2, PS) |

Tests whether the governing equation structure is what transfers. The HS and PS targets deliberately
share their level's equations while differing in geometry, boundaries and regime. This measures
your concern about coordinates and representation directly, rather than excluding it.

#### A2. Linear vs nonlinear

| Level | Pretrain | Test |
|---|---|---|
| Linear | AD, AM (reuses the A1-I run) | HS |
| Nonlinear | SF, EMQ-P (without Euler: SF, TRL2) | VE, ACT, GS, PS, TRL2 (drop TRL2 if pretrained on) |

Aliased with the wave family (§5.3). Its value is the comparison between the nonlinear run, which
mixes families, and the A1-IV run (incompressible only) on the nonlinear targets. Does
"nonlinearity in general" transfer, or only a specific family?

#### A3. Dissipation (explicit diffusion / viscosity operator present or not)

| Level | Pretrain | Test |
|---|---|---|
| Non-diffusive | AM, EMQ-P (without Euler: AM, TRL2) | HS, TRL2 (without Euler: HS only) |
| Diffusive | RB, GS | SF, VE, ACT |
| Intermediate probe | — | PS (hyperviscosity only) |

Boundary conditions are balanced: each level has one bounded and one periodic dataset.
Families are mixed within each level, so any Δ comes from dissipation rather than a single family.

#### A4. Sharp vs smooth

| Level | Pretrain | Test |
|---|---|---|
| Sharp | AM, EMQ-P (same run as A3 non-diffusive) | TRL2 (AI as sanity check) |
| Smooth | HS, SF | RB, ACT, GS, PS (VE excluded as borderline) |

Motivated by The Well's spectral-vs-spatial split and the acoustic/Helmholtz contrast (§2.5).
Putting HS in the *smooth* run splits the wave family across the two levels. Comparing this smooth
run with A3's diffusive run on the same smooth targets shows whether smoothness or diffusion is
the property that transfers. **Without Euler this axis is uninformative:** the only sharp target
left would be AI, a near-duplicate of the pretraining data.

#### A5. Predictable vs chaotic (regime)

| Level | Pretrain | Test |
|---|---|---|
| Predictable | AM, SF | HS, GS, PS, VE-AH |
| Chaotic | RB, TRL2 | ACT, VE-CAR, VE-EIT |

Family is partly controlled: each level contains one incompressible dataset (SF vs RB). Needs no
Euler. **Evaluate chaotic targets on one-step and T1–10 VRMSE plus spectral error** (The Well's
binned spectral metrics), not long-horizon VRMSE, which saturates for any model once it
decorrelates.

`viscoelastic_instability` is split by file, because its regimes differ (§5.4.5): VE-AH is the
steady arrowhead state, and VE-CAR / VE-EIT are chaotic. Select them with
`include_filters: ["_AH."]` or `include_filters: ["_CAR.", "_EIT."]`. The trailing `.` excludes the
`transtochaos_EIT_*` edge-state files. `shear_flow` sits in the predictable set only *over the
evaluated horizon*; it is chaotic in principle (§5.4.5).

**A5b, a cleaner within-dataset variant.** `rayleigh_benard` (one file per Rayleigh number),
`shear_flow` (per Reynolds number), `active_matter` (per activity α) and
`turbulent_radiative_layer_2D` (per t_cool) are stored with one file per parameter value. The
chaos level can therefore be varied *inside a dataset* with `include_filters`, keeping the
equations, fields and boundaries identical. For example, pretrain on low-Ra RB plus low-Re SF, and
compare with high-Ra RB plus high-Re SF, on held-out Ra and Re values. This removes both the family
and the representation confound from the regime question. First confirm the ordering by measuring
per-file decorrelation (the A7 descriptors): all these Ra values are far above the convective
threshold, so "low Ra" means *less* chaotic, not necessarily regular.

#### A6. Negative control: boundary topology (representation only)

| Level | Pretrain | Test |
|---|---|---|
| Fully periodic | SF, GS | ACT (+ EMQ-P if downloaded) |
| Bounded (walls / open) | RB, AM | VE, HS (+ EMQ-O if downloaded) |

Physics is deliberately mixed within each level. **If Δ here is as large as for the physics axes,
the model is grouping datasets by how they are presented, not by their dynamics.** This is the
direct test of whether the worry that "same equations can look very different" matters.

#### A7. Data-driven clusters (defined after measurement)

Compute label-free descriptors on each dataset's training split:

- persistence VRMSE (u_{t+1} = u_t)
- spectral slope and high-frequency energy fraction
- total variation / gradient sparsity
- decorrelation time in steps
- error of the best linear one-step predictor (a DMD fit)

Standardise the descriptors, cluster into two groups (k-means, k = 2), pretrain on the two most
central members of each cluster, and test on the rest. Comparing A7's Δ with A1–A6 answers
empirically whether labels or statistics group datasets better. Across all runs, regressing gain
on descriptor distance gives a second, finer-grained answer.

*Optional A0: WALRUS's domain headings as a baseline axis.* Not feasible cleanly with the 2D data
on disk: most headings have at most one available 2D dataset outside the acoustics group.

### 5.4.1 The equations behind A1, term by term

The equations below are copied from [Well] App. C (pp. 31–44), one per dataset in the equation-family
split, with each term labelled for the property it contributes. "Linear" here means the equations
are linear in the state variables — every term is a constant or a first derivative acting on a
state variable, with no state variable multiplying another. "Compressible" means the velocity
field is **not** divergence-free (∇·u ≠ 0): fluid parcels change volume, so density changes, and
disturbances travel at a finite sound speed. The density change may be carried explicitly
(∂ρ/∂t + ∇·(ρu) = 0) or implicitly through pressure, as in acoustics, where p′ = c²ρ′.
"Incompressible" means ∇·u = 0 is imposed as a constraint, which is equivalent to an infinite
sound speed. "Has a field" means the system
carries a variable beyond the core `(velocity, pressure)` pair (a scalar, tensor, or additional
component) that is transported by the flow and, in some cases, feeds back into the momentum
equation.

**I. Linear waves — `acoustic_scattering` (AD, AI, AM)**

$$
\frac{\partial p}{\partial t} + K(x,y)\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0,
\qquad
\frac{\partial u}{\partial t} + \frac{1}{\rho(x,y)}\frac{\partial p}{\partial x} = 0,
\qquad
\frac{\partial v}{\partial t} + \frac{1}{\rho(x,y)}\frac{\partial p}{\partial y} = 0
$$

- **Linear**: every term is a state variable (p, u, or v) multiplied by a coefficient that depends
  only on position (K(x,y), ρ(x,y)), never on another state variable. No term like u·∂u/∂x.
  [Well] calls this out directly: "These are simple linear dynamics" (p. 32).
- **Compressible, but linearised**: the first equation *is* mass conservation, written for the
  small density perturbation ρ′ and converted to pressure using p′ = c²ρ′. Start from the
  linearised continuity equation ∂ρ′/∂t + ρ₀∇·u = 0 and multiply by c²; this gives
  ∂p/∂t + K∇·u = 0 with bulk modulus K = ρ₀c². The equation says pressure changes *only because*
  ∇·u ≠ 0, which is compressibility itself. In an incompressible fluid sound waves could not
  exist at all.
- **Two different "densities"**: ρ(x,y) in the equations is the *background* density of the
  medium. It is fixed in time and acts as a material coefficient, which is why the configs can
  zero it out as an input field. The *acoustic* density perturbation ρ′ does evolve in time, but
  it is not stored separately because it is redundant with p.
- **What separates this from family II is amplitude, not compressibility.** These are the
  compressible Euler equations linearised about a fluid at rest (§2.3). Family II keeps the
  finite-amplitude nonlinear terms (ρu², u(ρE+p)); family I drops them.
- **No extra field**: the state is exactly (p, u, v); nothing else is carried.
- **What makes it hard**: ρ(x,y) and K jump sharply across material interfaces (walls at
  ρ = 10⁶, paths at ρ = 3 in the maze variant), so although the PDE is linear, its *coefficients*
  are discontinuous — this is the "sharp but non-diffusive" cell in §5.2's table.

**I. Linear waves — `helmholtz_staircase` (HS)**

$$
\frac{\partial^2 U(t,x)}{\partial t^2} - \Delta U(t,x) = \delta(t)\,\delta(x-x_0),
\qquad U_n(t,x) = 0 \text{ on } \partial\Omega
$$

solved in the frequency domain as the Helmholtz BVP $-(\Delta+\omega^2)u=\delta_{x_0}$.

- **Linear**: the only operator on U is the Laplacian Δ, a constant-coefficient linear operator;
  the source term δ(t)δ(x−x₀) does not depend on U at all.
- **Same family as `acoustic_scattering`**: both are the wave equation. `helmholtz_staircase` is
  the *constant-medium* limit — no ρ(x,y), no K(x,y) — with all the complexity moved into the
  boundary geometry (a periodic corrugated surface) instead of the coefficients. That is exactly
  the "same equation, different presentation" case in §2.4.
- **No extra field**: state is a single scalar acoustic pressure amplitude (here complex, so stored
  as `pressure_re`, `pressure_im`).
- **Compressible for the same reason as `acoustic_scattering`**: the wave equation is what the
  linearised compressible equations reduce to after eliminating velocity. Acoustic pressure U is
  proportional to the acoustic density perturbation, and the finite wave speed (c = 1 after
  rescaling) exists only because the medium is compressible.

**II. Compressible Euler — `euler_multi_quadrants` (EMQ-P, EMQ-O)**

$$
\frac{d}{dt}\iint_\Omega U\,dA + \oint_{\partial\Omega} (F\hat{\imath}+G\hat{\jmath})\cdot\hat{n}\,dS = 0,
\qquad
U = \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ \rho E\end{pmatrix},\quad
F = \begin{pmatrix}\rho u\\ \rho u^2+p\\ \rho uv\\ u(\rho E+p)\end{pmatrix},\quad
G = \begin{pmatrix}\rho v\\ \rho uv\\ \rho v^2+p\\ v(\rho E+p)\end{pmatrix}
$$

with $\rho E = p/(\gamma-1) + \tfrac12\rho(u^2+v^2)$.

- **Compressible, finite-amplitude**: ρ is the first component of the conserved state U and has
  its own flux ρu, so density is evolved rather than constrained. Compressibility separates this
  family from IV below. What separates it from family I is that the perturbations are not small,
  so the nonlinear terms are kept.
- **Nonlinear**: the fluxes contain products of state variables with themselves — ρu², ρuv,
  u(ρE+p) — so this is not linear in U.
- **No dissipation**: there is no viscous term (no ∇²u, no diffusion). Energy is conserved except
  across the shocks the equations themselves produce; the finite-volume solver adds numerical
  (not physical) dissipation only at discontinuities. This is why EMQ sits in the "non-diffusive"
  row of §5.2 despite having complex, turbulent-looking structure.
- **Sharp features by construction**: initial data is piecewise constant (a Riemann problem), so
  shocks and contact discontinuities are guaranteed, not merely possible.
- **No extra field**: state is exactly the standard Euler set (ρ, momentum, energy); pressure p is
  recovered algebraically via the ideal-gas closure, not carried as an independent dynamical field.

**II. Compressible — `turbulent_radiative_layer_2D` (TRL2)**

Same compressible Euler system as above (mass, momentum, energy conservation with ideal-gas
pressure, $E = P/(\gamma-1)$), **plus one added term** in the energy equation:

$$
\frac{\partial E}{\partial t} + \nabla\cdot\big((E+P)v\big) = -\frac{E}{t_{\rm cool}}
$$

- **Same compressibility and nonlinearity as EMQ**: ρ is still a conserved dynamical variable;
  fluxes are still products of state variables.
- **One extra source term, no extra field**: the cooling sink −E/t_cool is a function of the
  *existing* energy variable, not a new transported quantity — this is why TRL2 is grouped in the
  same equation family as EMQ (§2.4) rather than a different one, despite looking completely
  different (a smooth turbulent mixing layer rather than sharp shocks).
- **Still non-diffusive** in the classical sense — there is no ∇²(velocity) viscous term — but the
  cooling term removes energy locally in a way that is *dissipative* for the energy field
  specifically. §5.2/§5.3 treat TRL2 as non-diffusive by the "explicit viscosity/diffusion
  operator present" criterion, but flag it separately as the chaotic regime target.

**II. Compressible analogue — `planetswe` (PS)**

$$
\frac{\partial u}{\partial t} = -u\cdot\nabla u - g\nabla h - \nu\nabla^4 u - 2\Omega\times u,
\qquad
\frac{\partial h}{\partial t} = -H\nabla\cdot u - \nabla\cdot(hu) - \nu\nabla^4 h + F
$$

- **Compressible-equation analogue, not compressible fluid**: the shallow-water system is derived
  by depth-integrating incompressible Navier–Stokes, so physically the fluid is incompressible.
  Mathematically, however, height h plays exactly the role density ρ plays in 2D isentropic Euler
  with γ=2 — h has its own conservation law with a flux hu, the same structure as ρ's flux ρu in
  Euler. That is **my own observation**, not stated in [Well]; it is why PS is placed in family II
  rather than IV despite its physical derivation.
  It is nonlinear via the advective term u·∇u and the flux divergence ∇·(hu).
- **Extra terms not present in EMQ/TRL2**: Coriolis rotation −2Ω×u and a periodic seasonal forcing
  F(φ,θ,t) (daily/annual). These make PS the most heavily forced dataset in the split.
- **Hyperviscosity, not physical dissipation**: −ν∇⁴u is explicitly called "largely non-physical"
  by the authors (p. 37), included only for numerical stability at the resolved scale. This is why
  §5.2 lists PS as "none — hyperviscosity only" rather than counting it with the truly diffusive
  (viscous) datasets in family IV.
- **Curvilinear domain**: (θ,φ) on the sphere, not Cartesian — the representation confound flagged
  throughout §5.

**IV. Incompressible + transported field — `shear_flow` (SF)**

$$
\frac{\partial u}{\partial t} - \nu\Delta u + \nabla p = -u\cdot\nabla u,
\qquad \nabla\cdot u = 0,
\qquad
\frac{\partial s}{\partial t} - D\Delta s = -u\cdot\nabla s
$$

- **Incompressible**: the constraint ∇·u = 0 replaces density's own evolution equation — this is
  the structural difference from family II. Pressure p is a Lagrange multiplier enforcing the
  constraint, not a thermodynamic state variable with its own flux.
- **Dissipative**: the viscous term −ν∆u is present with a finite Reynolds number (Re ∈
  {10⁴,…,5×10⁵}), so this is genuinely diffusive, unlike EMQ/TRL2's numerical-only dissipation.
- **Has an extra field**: a passive tracer s, advected and diffused (Schmidt number D) but with
  **no feedback** into the momentum equation — it rides along with the flow without influencing it.
  This is the mildest way a "system with a field" can be built, and is why SF sits in the
  incompressible-but-simplest tier of family IV.

**IV. Incompressible + active field — `rayleigh_benard` (RB)**

$$
\frac{\partial b}{\partial t} - \kappa\Delta b = -u\cdot\nabla b,
\qquad
\frac{\partial u}{\partial t} - \nu\Delta u + \nabla p - b\,\hat{e}_z = -u\cdot\nabla u
$$

- **Same incompressible-NS skeleton as `shear_flow`** — identical viscous term −ν∆u, identical
  advection structure, same solver (Dedalus).
- **The one structural difference from SF**: the transported scalar here is buoyancy b, and it
  re-enters the momentum equation as the body force −b ê_z. This is an *active* scalar (it drives
  the flow) versus SF's *passive* tracer s (which does not). This single term is the entire
  mathematical distance between SF and RB, despite RB's dynamics (convection cells, chaotic
  regime, per §5.2) looking very different from SF's shear layers.
- **Dissipative**: both κ∆b (scalar diffusion) and ν∆u (viscosity) are present, parameterised by
  Rayleigh and Prandtl numbers.

**IV. Incompressible + tensor field — `viscoelastic_instability` (VE)**

$$
Re\,(\partial_t u + u\cdot\nabla u) + \nabla p = \beta\Delta u + (1-\beta)\,\nabla\cdot T(C),
\qquad \nabla\cdot u = 0,
$$
$$
\partial_t C + (u\cdot\nabla)C + T(C) = C\cdot\nabla u + (\nabla u)^T\cdot C + \varepsilon\Delta C
$$

- **Same incompressible skeleton again**: ∇·u = 0, advective nonlinearity u·∇u, a viscous term
  βΔu (here split between solvent viscosity β and a polymer contribution).
- **The field is now a second-order tensor, not a scalar**: C, the polymer conformation tensor,
  replaces RB's scalar buoyancy b. It couples to the flow in *both* directions — advected by u
  (∂_tC + u·∇C) *and* stretched by the velocity gradient (C·∇u + (∇u)ᵀ·C) *and* it exerts a stress
  back on the momentum equation via ∇·T(C). This is a strictly richer feedback loop than RB's
  scalar b → −bê_z: RB's field affects only the vertical momentum component with a fixed
  direction, while C's feedback is a full tensor stress that depends on the flow gradient itself.
  This graded structure (SF: no feedback → RB: scalar feedback → VE: tensor feedback) is the axis
  the A1 split tests when it holds out VE from a model pretrained on {SF, RB}: does "having some
  field that pushes back on the flow" transfer, or does it stop working once the field's own rank
  increases?
- **Dissipative**: βΔu on the velocity, εΔC on the tensor (polymer stress diffusivity).

### 5.4.2 What this changes about how to read the A1 split

Laying the equations out side by side sharpens two of the aliasing warnings from §5.3:

- **PS's placement in family II is a judgement call, not a fact from either paper.** The
  shallow-water ↔ Euler correspondence is real and standard, but [Well] itself calls the shallow
  water equations a "depth-integrated" approximation of an *incompressible* 3D flow (p. 36) — so
  PS can reasonably be argued into either family depending on whether "compressible" is read
  structurally (has its own conservation law, like h here) or physically (is the fluid actually
  compressible). If PS transfers better from {SF, RB} than from {EMQ-P, EMQ-O} in your results,
  that would be evidence for reading it physically instead.
- **The SF → RB → VE sequence is not three unrelated datasets but one continuous change**: no
  field → scalar field with no feedback → scalar field with feedback → tensor field with feedback
  (VE's C is the only one with feedback *and* rank 2). A1 holding out VE from a {SF, RB} pretrained
  model is really asking whether transfer survives one more step along this specific ladder, which
  is a narrower and more answerable question than "does incompressible flow transfer to
  incompressible flow".

### 5.4.3 What makes a dataset diffusive (axis A3)

**The rule:** a dataset is diffusive when its equations contain a *first derivative in time*
balanced by an *even-order derivative in space*, ∂ₜf = νΔf (or −ν∇⁴f). The type of each term
decides how it acts on a Fourier mode e^{ikx}:

| Term type | Example | Effect on a mode e^{ikx} | Character |
|---|---|---|---|
| ∂ₜf = νΔf (diffusion, viscosity) | νΔu, κΔb | amplitude × e^{−νk²t} | **Diffusive.** Small scales damped fastest; smooths; irreversible. |
| ∂ₜf = −ν∇⁴f (hyperviscosity) | `planetswe` | amplitude × e^{−νk⁴t} | Diffusive, concentrated at the smallest scales. |
| First-order space derivatives only (advection, fluxes) | ∂ₜp + K∇·u, Euler fluxes | phase shift e^{ik(x−ct)}, amplitude kept | **Non-diffusive.** Transport without spreading; energy conserved; sharp features persist. |
| Zeroth-order sink | −E/t_cool | amplitude × e^{−t/t_cool}, same at every k | Removes energy but does **not** smooth. |

**A Laplacian alone is not diffusion.** Whether it diffuses depends on what it is paired with:

- `helmholtz_staircase`, ∂²ₜU = ΔU: second order in *time*, so modes oscillate (e^{±ikt}) and never
  decay. This is a wave equation.
- Incompressible pressure, Δp = −∇·(u·∇u): there is no time derivative, so it is an instantaneous
  constraint, not diffusion.
- `active_matter`'s Stokes equation, −Δu + ∇P = ∇·Σ: also no time derivative; the velocity is set
  instantly by the stress.

**Diffusive datasets** (explicit ∂ₜ = νΔ-type term, [Well] App. C):

| Dataset | Diffusive terms | Strength set by |
|---|---|---|
| `shear_flow` | νΔu, DΔs | ν = 1/Re, D = 1/(Re·Sc); Re from 10⁴ to 5×10⁵ |
| `rayleigh_benard` | νΔu, κΔb | κ = (Ra·Pr)^(−1/2), ν = (Ra/Pr)^(−1/2); Ra from 10⁶ to 10¹⁰ |
| `viscoelastic_instability` | βΔu (solvent), εΔC (polymer stress) | Re = 1000; ε = 2×10⁻⁶ is tiny, mainly for numerical regularity |
| `gray_scott_reaction_diffusion` | δ_A ΔA, δ_B ΔB | δ_A = 2×10⁻⁵, δ_B = 10⁻⁵; the *difference* between the two drives pattern formation |
| `active_matter` | d_T and d_R (translational and rotational diffusion) in the kinetic equation | dimensionless constants; the Stokes flow is viscous as well |
| `rayleigh_taylor_instability` (3D) | viscous stress ∇·τ; molecular diffusivity κ in the ∇·u constraint | ν rescaled over time to keep the Kolmogorov scale at the grid spacing ([Well] Eq. 38) |
| `planetswe` | −ν∇⁴u, −ν∇⁴h (hyperviscosity) | ν = 1.76×10⁻¹⁰, tuned to act only near wavenumber 224; the authors call it "largely non-physical" (p. 37) |

**Non-diffusive datasets** (only first-order spatial derivatives):

| Dataset | Why |
|---|---|
| `acoustic_scattering` | only ∇·u and ∇p appear: pure wave transport |
| `helmholtz_staircase` | wave equation, second order in time (see above) |
| `euler_multi_quadrants` | flux divergence only; "inviscid" |
| `turbulent_radiative_layer` | Euler plus the −E/t_cool sink: energy is lost without smoothing |
| `MHD` | *ideal* MHD: no viscosity, no resistivity (η∇²B) |
| `post_neutron_star_merger` | ideal general-relativistic MHD |
| `supernova_explosion`, `turbulence_gravity_cooling` | Euler plus gravity plus heating/cooling, but see artificial viscosity below |

**Where the setup matters, beyond the equations:**

1. **The coefficient's value, not just its presence.** At Re = 5×10⁵ (`shear_flow`), viscosity
   matters only at the smallest scales; over large scales and short times the flow behaves almost
   inviscidly. "Diffusive" is really a continuum set by Re, Ra and the other dimensionless numbers.
   §5.2 classifies by whether the *operator is present*, not by how strong it is.
2. **Numerical dissipation.** Shock-capturing schemes smooth sharp features even when the equations
   are ideal: Clawpack's TVD flux limiter (acoustics, Euler), Athena++'s Godunov scheme (TRL, RSG),
   and the ENO scheme (MHD). Data from "ideal" datasets therefore contains some dissipation, but
   only at shocks and interfaces, and it comes from the discretisation rather than the physics.
   Dedalus spectral solvers (`shear_flow`, `rayleigh_benard`, `viscoelastic_instability`,
   `planetswe`) add almost none.
3. **SPH artificial viscosity.** `supernova_explosion` and `turbulence_gravity_cooling` include an
   explicit a_visc term in the momentum equation ([Well] Eq. 42). It is written into the equations
   but is a numerical device that switches on mainly at shocks. These two are "physically inviscid,
   numerically dissipative", a borderline case like `planetswe`'s hyperviscosity.
4. **Sinks and relaxation are not diffusion.** TRL's −E/t_cool and the polymer relaxation T(C) in
   `viscoelastic_instability` damp at the same rate at every scale. That is why TRL keeps sharp
   hot/cold interfaces while losing energy.
5. **Downsampling is not diffusion.** `MHD_64` was low-pass filtered from 256³. That removes small
   scales from the *data*, but the *dynamics* remain ideal.

### 5.4.4 What makes a dataset sharp or smooth (axis A4)

Sharpness is a competition. Some terms or setup choices **create** steep gradients, others **remove**
them. A dataset is sharp when a sharpening mechanism is present and nothing strong enough undoes it.

"Sharp" covers two different things, which need keeping apart:

| | Shock | Steep but smooth |
|---|---|---|
| What it is | a true discontinuity that forms in finite time | gradients grow very large but stay finite |
| Needs | nonlinear compressible flow and no diffusion | an amplifying or thinning mechanism and weak diffusion |
| Examples | `euler_multi_quadrants`, `supernova_explosion` | the polymer tensor C in `viscoelastic_instability`; the tracer in `shear_flow` |
| Visible in gridded data? | yes, as jumps over 1–2 cells | yes, once the gradient scale approaches the grid spacing |

A network trained on gridded data sees both as grid-scale structure, which is why axis A4 counts
both. But their mechanisms differ, and so do the constraints that rule them out.

**Mechanisms that create sharpness:**

1. **Nonlinear steepening in compressible flow (shocks).** In Euler, disturbances travel at u ± c,
   and both depend on the state. Compressed regions become denser and hotter, so c rises and the
   back of a wave outruns its front: the wave steepens into a shock in finite time, even from smooth
   initial data. This needs nonlinear fluxes (ρu², u(ρE+p)) *and* no diffusion to stop it.
   - `euler_multi_quadrants`: also sharp from t = 0, because the initial data is piecewise constant
     (a Riemann problem).
   - `supernova_explosion`: point energy injection drives a blast wave; [Well] describes "dense,
     sharp shells" (p. 6).
   - `MHD`: the supersonic runs (Mach up to 7) form shocks.
2. **Discontinuous coefficients or geometry.** A *linear* wave equation cannot create a
   discontinuity, but it faithfully carries and reflects one that the setup provides.
   - `acoustic_scattering`: ρ(x,y) jumps at interfaces (maze walls ρ = 10⁶ next to paths at ρ = 3).
     The pressure field develops reflections, kinks and gradient jumps at every interface. The
     equations are linear; the sharpness lives entirely in the coefficients.
   - `helmholtz_staircase`: the staircase corners produce gradient singularities ([Well] p. 36);
     away from the boundary the field is smooth. The stored `mask` field is a sharp binary image,
     but it never changes in time.
3. **Temperature-dependent cooling (thermal instability).** In `turbulent_radiative_layer`, cooling
   is fastest at intermediate temperatures ([Well] p. 44), even though the printed equation shows
   only −E/t_cool. Intermediate-temperature gas therefore cools quickly into the cold phase, the
   intermediate phase drains away, and the hot/cold interface stays thin, with no physical
   diffusion to widen it.
4. **Self-gravity (collapse).** In `turbulence_gravity_cooling`, gravity pulls dense regions
   together and makes them denser, the opposite of diffusion. Together with cooling this produces
   thin, dense filaments. The density range is wide enough that the configs log-transform it.
5. **Stretching with weak diffusion.** In `viscoelastic_instability`, C has its own evolution
   equation, ∂ₜC + (u·∇)C + T(C) = C·∇u + (∇u)ᵀ·C + εΔC, and it is not covered by the
   incompressibility constraint:
   - **Amplification.** The stretching term C·∇u + (∇u)ᵀ·C makes C grow exponentially wherever
     the flow is extensional. Polymers undergo the coil–stretch transition once relaxation cannot
     keep up (roughly, relaxation time × strain rate > ½ for dumbbell models).
   - **Slow relaxation.** At Weissenberg number 50 a polymer needs ~50 flow time-units to relax, so
     almost any appreciable strain stretches it faster than it recovers. Growth is finally capped
     by the FENE-P extensibility limit (L_max = 70), after C has grown by a factor of thousands.
   - **Almost no smoothing.** ε = 2×10⁻⁶ is about 1,000× weaker than the viscosity acting on
     velocity (β/Re ≈ 10⁻³). With ε removed, the C equation is pure transport plus stretching and
     can develop arbitrarily steep gradients. This is the well-known "high Weissenberg number
     problem" of viscoelastic simulation, and it is the reason ε is included at all.

   The velocity stays smooth even though C pushes back on it through (1−β)∇·T(C). The momentum
   equation carries solvent viscosity βΔu (β = 0.9), and incompressibility makes velocity and
   pressure respond to the whole stress field globally rather than point by point, so the velocity
   comes out roughly one derivative smoother than the stress. This is why §5.2 marks VE as
   borderline: smooth velocity, sharp C. The measurement below confirms it.

**Mechanisms that keep a dataset smooth:**

1. **Diffusion, ∂ₜf = νΔf.** Modes decay as e^{−νk²t}, so the steepest gradients are damped
   fastest, and gradients cannot get narrower than a scale set by the coefficient (`gray_scott`,
   `active_matter`, `rayleigh_benard`, `shear_flow`). Gray–Scott forms reaction fronts, but they
   have a finite width, roughly √(δ/reaction rate).
2. **Incompressibility, ∇·u = 0, prevents shocks in the flow variables.** With no density change
   there are no sound waves to steepen, and pressure comes from a Poisson equation, which smooths by
   nature. `shear_flow`, `rayleigh_benard` and `viscoelastic_instability` therefore cannot develop
   *shocks* at any Reynolds number. Incompressibility constrains only the **velocity**, though; it
   says nothing about fields that are carried by the flow. It even helps thin them: in 2D an
   incompressible flow preserves area, so a patch stretched in one direction must be squeezed by the
   same factor in the other. Transported fields therefore get drawn into sheets and filaments:
   - `shear_flow`'s tracer at Schmidt number up to 10 forms fine filaments. The only mechanism is
     thinning, and the diffusivity is moderate, so these stay resolved.
   - `viscoelastic_instability`'s C *also* has exponential amplification in the stretched regions
     and almost no diffusion (mechanism 5 above), so it becomes genuinely sharp.
3. **Linearity with smooth coefficients.** Linear waves do not steepen. `helmholtz_staircase` has a
   constant medium and is time-harmonic, so it stays smooth apart from its corners.
4. **Hyperviscosity together with gentle forcing.** The shallow-water equations are hyperbolic like
   Euler and *can* form shocks (hydraulic jumps, or bores). `planetswe` does not, because the forcing
   is gentle and −ν∇⁴u removes grid-scale structure. It is smooth because of its *setup*, even
   though its equation family allows sharpness.

**An empirical signal: the solver.** The solvers split almost perfectly along this line. Sharp
datasets use shock-capturing finite-volume or SPH codes (Clawpack, Athena++, ENO, SPH), which are
built to tolerate discontinuities. Smooth datasets use spectral codes (Dedalus, Chebfun), which need
smooth fields or become unstable. `viscoelastic_instability` uses Dedalus but needs its small ε
diffusion to stay stable, which is consistent with it being borderline. In choosing the solver, the
domain expert effectively recorded whether they expected discontinuities.

**Measured check on the stored data.** For each field, over trajectory 0 of each file, every 4th
snapshot, I computed two statistics of the squared gradient |∇f|²:

- **Flatness**, ⟨|∇f|⁴⟩ / ⟨|∇f|²⟩², which is ≈ 2 for Gaussian-like gradients and large when a few
  pixels dominate.
- **Top-1% share**: the fraction of total gradient energy contained in the 1% of pixels with the
  steepest gradient.

| Dataset (regime) | Field | Flatness | Top-1% share |
|---|---|---|---|
| `viscoelastic_instability` (CAR, chaotic arrowhead) | velocity_x | 2.9 | 5.9% |
| | C_yy | 950 | 75.5% |
| | C_yy, interior only (64-cell margin) | 444 | 69.6% |
| `viscoelastic_instability` (EIT, elasto-inertial turbulence) | velocity_x | 2.8 | 5.8% |
| | C_xy | 40 | 42.6% |
| | C_yy | 1804 | 78.4% |
| | C_yy, interior only (64-cell margin) | 788 | 73.7% |
| `shear_flow` (reference, smooth) | velocity_x | 12 | 25.4% |
| | tracer | 16 | 31.8% |
| `rayleigh_benard` (reference, smooth) | buoyancy | 17 | 32.2% |
| `acoustic_scattering_maze` (reference, sharp) | pressure | 210 | 46.5% |
| `turbulent_radiative_layer_2D` (reference, sharp) | density | 92 | 84.9% |

Unless marked, 4 cells are trimmed at every edge. In the chaotic regimes, `viscoelastic_instability`'s
velocity is the *smoothest* field measured, smoother even than `shear_flow`'s. Its C_yy is as
concentrated as the sharp references (TRL density) and far more intermittent than any of them. The
interior-only rows rule out near-wall boundary layers as the cause.

Two caveats on the metric:
1. It measures how *concentrated* the gradients are, which is not the same as how sharp they are. A
   smooth, localised bump on a flat background also scores high. That matters for the steady
   arrowhead regime (AH, not shown), where even pressure scores 85%, because the whole structure is
   a single localised feature. The chaotic regimes, where the velocity fills the domain, are the
   informative ones.
2. It uses one trajectory per file and five snapshots, so it is a spot check, not a full survey.
   The same statistics are natural candidates for the A7 descriptors.

| Dataset | Sharpening mechanism | Smoothing mechanism | Result |
|---|---|---|---|
| `euler_multi_quadrants` | discontinuous initial data + nonlinear steepening | none (numerical only) | **Sharp** (shocks) |
| `supernova_explosion` | blast wave + steepening | artificial viscosity, at shocks only | **Sharp** (shells) |
| `MHD` | steepening in supersonic runs | none (ideal) | **Sharp** at high Mach |
| `acoustic_scattering` | discontinuous ρ(x,y) | none | **Sharp** at interfaces |
| `turbulent_radiative_layer` | temperature-dependent cooling | none (numerical only) | **Sharp** interfaces |
| `turbulence_gravity_cooling` | self-gravity + cooling | artificial viscosity | **Sharp** filaments |
| `viscoelastic_instability` | polymer stretching | tiny ε diffusion | **Borderline** (sharp in C) |
| `shear_flow` | Kelvin–Helmholtz filamentation | viscosity + incompressibility | Smooth (thin but resolved) |
| `rayleigh_benard` | plume formation | viscosity + thermal diffusion + incompressibility | Smooth |
| `helmholtz_staircase` | corner singularities only | linear, constant medium | Smooth except at corners |
| `planetswe` | could form hydraulic jumps | hyperviscosity + gentle forcing | Smooth |
| `gray_scott_reaction_diffusion` | reaction fronts | diffusion | Smooth (finite-width fronts) |
| `active_matter` | active stresses | translational/rotational diffusion + Stokes | Smooth |

**Summary:** sharpness needs a mechanism that steepens gradients and the absence of anything to undo
it. Compressibility with nonlinearity, discontinuous coefficients, thermal instability,
gravitational collapse and stretching create sharpness; diffusion, linearity and hyperviscosity
prevent it. Incompressibility prevents **shocks** in velocity and pressure, but not steep gradients
in fields the flow carries: an amplified, weakly diffusive field such as C can be sharp inside a
perfectly smooth incompressible flow. Sharpness is therefore a property of each *field*, not only
of the dataset.

### 5.4.5 What makes a dataset chaotic or regular (axis A5)

**Terminology first.** Every dataset in The Well is *deterministic*: the equations contain no noise,
and the same initial state always gives the same trajectory. Chaos is a property of deterministic
systems. It means **sensitive dependence on initial conditions**: two almost identical states
diverge exponentially, so beyond some horizon the trajectory cannot be predicted pointwise, even
with perfect equations. The right contrast is therefore **chaotic vs regular** (steady, periodic or
decaying), not chaotic vs deterministic.

**What in the equations makes a system chaotic.** Four ingredients are needed together:

| Ingredient | Why it is needed | Where it appears |
|---|---|---|
| **1. Nonlinearity** | In a linear system, a small difference δ obeys the *same* linear equation, so it can only propagate or grow at a fixed modal rate; it cannot be stretched and folded. **Linear systems cannot be chaotic.** | u·∇u (advection), Euler fluxes ρu², stretching C·∇u, reaction AB², orientation coupling in the kinetic equation |
| **2. An instability** that amplifies small differences | Something must make nearby states separate | shear (Kelvin–Helmholtz), buoyancy (convection, Rayleigh–Taylor), activity, polymer elasticity, thermal instability, gravity |
| **3. A control parameter above threshold** | Diffusion damps perturbations. Chaos appears only when driving or nonlinearity beats damping by a large enough ratio. | Re, Ra, Wi, α, Mach, Atwood |
| **4. Sustained driving** (for lasting chaos) | Without energy input, a dissipative flow decays towards rest, so any chaos is *transient* | forcing terms, boundary heating, activity, imposed mean flow |

Diffusion works *against* chaos: it removes small scales, reduces the number of active degrees of
freedom and, at low Re/Ra/Wi, leaves flows steady or periodic.

**Three separate sources of unpredictability for an emulator.** Only the first is chaos, but all
three look alike in the error curves:

1. **Intrinsic chaos**: errors grow exponentially from the dynamics themselves.
2. **Unobserved forcing**: an external input that is not stored as a field. The random
   large-scale driving in `MHD` is unpredictable from the data whatever the dynamics.
   `planetswe`'s seasonal forcing is deterministic, but its phase (time of day and year) must be
   inferred from history.
3. **Unobserved parameters**: coefficients the model is never given, which it must infer from the
   history. These are γ in Euler, Re/Ra in the incompressible sets, and the medium density in
   acoustics (zeroed in the configs). This is the non-Markovian issue [WALRUS] App. D discusses.

**Dataset by dataset.** The proxy is the best model's T21–60 VRMSE from [WALRUS] Table 1; ≳ 0.6
means every model had decorrelated by then (§5.2).

| Dataset | Nonlinearity | Instability / driver | Control parameter (range) | Sustained? | Evidence | Classification |
|---|---|---|---|---|---|---|
| `acoustic_scattering` (×3) | none (linear) | none | — | energy leaves through open boundaries | proxy 0.04–0.13 | **Regular** |
| `helmholtz_staircase` | none (linear) | none | — | exactly time-periodic: U = u·e^{−iωt} | proxy 0.003 | **Regular** (the most predictable dataset) |
| `gray_scott_reaction_diffusion` | reaction AB² | diffusive (Turing) instability | f, k: six chosen pairs | patterns settle | "eventually reach a steady-state" ([Well] App. D, p. 45); proxy 0.13 | **Regular.** Gray–Scott has chaotic regimes, but the six chosen (f, k) pairs give stationary or regularly moving patterns |
| `euler_multi_quadrants`\* | flux terms | shock interactions; roll-up of contact discontinuities | γ = 1.13–1.76 | ideal, unforced | proxy 0.28 | **Intermediate** |
| `planetswe` | u·∇u, ∇·(hu) | shear (barotropic) instability | fixed Earth-like setup | yes: seasonal forcing | proxy 0.02 at 60 h; The Well frames long-term prediction as matching the *distribution* ([Well] App. D) | **Slowly chaotic**: predictable over the evaluated horizon, chaotic over long ones |
| `shear_flow` | u·∇u, u·∇s | Kelvin–Helmholtz | Re = 10⁴–5×10⁵ | no: unforced, decays | "highly sensitive to initial conditions" ([Well] C.12, p. 42); proxy 0.08 | **Chaotic in principle, slow**: predictable over 60 steps |
| `rayleigh_benard` | u·∇u, u·∇b | convective (buoyancy) instability | Ra = 10⁶–10¹⁰, far above onset (~1.7×10³ for rigid walls) | yes: fixed temperature difference at the walls | cell positions "highly sensitive to initial conditions" ([Well] C.10, App. D); proxy 0.64 | **Chaotic** |
| `turbulent_radiative_layer_2D` | flux terms + temperature-dependent cooling | Kelvin–Helmholtz shear + thermal instability | t_cool = 0.03–3.16 | shear-driven mixing | proxy 0.84 | **Chaotic** |
| `active_matter` | orientation coupling in the kinetic equation (*not* in Stokes, which is linear) | activity-driven instability of the aligned state | α = −1 to −5, ζ = 1–17 | yes: activity injects energy everywhere | proxy 1.25, the highest in 2D | **Chaotic**: "active turbulence" with no inertia at all |
| `viscoelastic_instability` | stretching C·∇u + u·∇u | elasto-inertial instability | Re = 1000, Wi = 50 (fixed) | yes: imposed bulk flow | multistable; see below | **Mixed, by file** |
| `MHD_64` (3D) | u·∇u, induction | driven turbulence | Mach 0.5–7 | yes: continuous forcing | proxy 1.23 | **Chaotic + stochastically forced** |
| `rayleigh_taylor_instability` (3D) | u·∇u | Rayleigh–Taylor (buoyancy) | Atwood 1/16–3/4 | gravity keeps releasing potential energy | proxy 0.48 | **Chaotic onset** (turbulent mixing develops) |
| `turbulence_gravity_cooling` (3D) | fluxes, gravity | turbulence + gravitational collapse | T₀, ρ₀, metallicity | turbulent initial conditions, collapse | proxy 0.62 | **Chaotic** |
| `supernova_explosion_128` (3D) | fluxes | blast wave | fixed | no | proxy 0.97 | **Saturates.** Probably shell instabilities; the paper does not discuss it |
| `turbulent_radiative_layer_3D` (3D) | as 2D | as 2D | as 2D | as 2D | proxy 0.78 | **Chaotic** |

\* not on disk.

**Four cases worth singling out:**

- **`viscoelastic_instability` contains every kind of behaviour, one file each.** The flow is
  multistable, with coexisting attractors ([Well] §3.3.2): a laminar state, a steady arrowhead
  (the `AH` file; a travelling wave, so regular), and the chaotic arrowhead (`CAR`) and
  elasto-inertial turbulence (`EIT`) states. The four `transtochaos_*` / `transtononchaos_*` files
  are **edge states**. They were found by *bisecting* between initial conditions that reach
  different attractors ([Well] C.16), so they sit on the boundary between basins with a single
  unstable direction. Paired trajectories start almost identically and then end up on different
  attractors. That is sensitive dependence by construction, and no emulator can predict which way
  such a trajectory will fall from a short history.
- **`active_matter` is chaotic without the usual nonlinearity.** Stokes flow has no u·∇u term,
  and the Reynolds number is effectively zero. The nonlinearity is in the particle orientation
  dynamics (the ṗ flux depends on ∇u, which depends on the orientation stress), and the energy
  comes from the activity α at every point. This is a different route to chaos from every fluid
  dataset.
- **`rayleigh_benard` is kept chaotic by its boundaries.** The equations have no forcing term. The
  fixed temperature difference between the walls, b(0) = L_z and b(L_z) = 0, supplies energy
  continuously, so the chaos persists instead of decaying. `shear_flow` has no such input: its
  shear layers roll up and then decay.
- **Chaos is relative to the horizon.** `planetswe` and `shear_flow` are most likely chaotic, but
  their error-doubling time is long compared with the 60 output steps evaluated (1 step = 1 model
  hour for `planetswe`). Whether a dataset "is chaotic" in a benchmark therefore depends on Δt and
  the evaluation window as well as on the physics.

**Measuring it directly.** The proxy above depends on model skill. A skill-free alternative for
A7: many datasets have multiple trajectories at identical parameters. Find pairs of snapshots from
different trajectories that are close to each other, then track how fast they separate over
subsequent steps. This is a nearest-neighbour estimate of the largest Lyapunov exponent. The
`viscoelastic_instability` edge-state pairs provide this directly. Report the result as
error-doubling time in output steps, the unit that matters for an emulator.

### 5.5 Runs, protocol and priorities

**Unique pretraining runs** (several axes reuse the same run):

| Run | Pretrain set | Used by |
|---|---|---|
| R1 | AD, AM | A1-I, A2-linear |
| R2 | EMQ-P, EMQ-O\* | A1-II |
| R3 | SF, RB | A1-IV |
| R4 | SF, EMQ-P\* (or SF, TRL2) | A2-nonlinear |
| R5 | AM, EMQ-P\* (or AM, TRL2) | A3-non-diffusive, A4-sharp |
| R6 | RB, GS | A3-diffusive |
| R7 | HS, SF | A4-smooth |
| R8 | AM, SF | A5-predictable |
| R9 | RB, TRL2 | A5-chaotic |
| R10 | SF, GS | A6-periodic |
| R11 | RB, AM | A6-bounded |
| R12–13 | from descriptors | A7 |
| Scratch | none | reference for every target |

That is 11 pretraining runs, plus 2 for A7. **Finetune every run on every 2D dataset outside its
own pretraining set**, not just its axis's targets. That yields one full run × target matrix, which
every Δ and the descriptor regression can be computed from, without extra pretraining.

**Protocol common to all runs:**

- 2D only; same model size, optimiser and schedule.
- Identical pretraining sample budget per run (uniform over its two datasets).
- Same augmentation policy in every run (WALRUS's, or none).
- Finetuning at 1k / 10k / 100k samples with identical hyperparameters; pretraining effects are
  largest when finetuning data is scarce ([WALRUS] Fig. 8).
- Field embeddings reinitialised at finetuning (§5.3), plus one ablation that keeps them.
- Metrics: VRMSE one-step, T1–10, T11–30; spectral error for chaotic targets.
- At least 3 seeds for Tier-1 runs; Δ reported with confidence intervals.

**Priorities, if compute is limited:**

| Tier | Runs | Answers |
|---|---|---|
| 1 | R1, R3 (+ R2 if Euler is downloaded), Scratch | Does equation family predict transfer? |
| 2 | R8, R9 · R5, R6, R7 | Does the regime (predictability, dissipation, smoothness) predict it better? |
| 3 | R10, R11 · R4 · A7 | Is it really representation (negative control)? Linearity? Data-driven grouping? |

### 5.6 3D variant (separate, smaller study)

On disk: a 3D compressible family (`supernova_explosion_128`, `turbulence_gravity_cooling`,
`turbulent_radiative_layer_3D`), plus the singletons `MHD_64` and `rayleigh_taylor_instability`.
Pretrain on the compressible family and test on MHD (the same equations plus a magnetic field;
fully periodic) and RT (incompressible, variable density; periodic + wall). This gives a one-level
family test with two representation-mismatched probes. Note that `supernova_explosion_128` and
`turbulence_gravity_cooling` are fully open while `turbulent_radiative_layer_3D` is periodic + open,
so boundary conditions are not balanced within this family.

---

## Page references

| Claim | Location |
|---|---|
| Well dataset list, technical attributes | [Well] Table 1, p. 3 |
| Per-dataset physics descriptions | [Well] §3.1, pp. 3–7 |
| Spectral vs spatial split, "not immediately clear" | [Well] §4 Analysis, p. 8 |
| "No cross-linkage between different datasets" | [Well] Datasheet Q11, p. 24 |
| Governing equations for each dataset | [Well] App. C, pp. 31–44 |
| TGC uses the supernova equations (40)–(43) | [Well] App. C.14, p. 43 |
| Boussinesq regime defined as Atwood < 0.1 | [Well] App. C.11, p. 41 |
| Task-based groupings | [Well] App. D, p. 45 |
| Domain list in abstract | [WALRUS] p. 1 |
| Uniform-over-datasets hierarchical sampling | [WALRUS] §4.2, p. 6 |
| HSDP same-dataset sampling | [WALRUS] §4.2, p. 7 |
| ~4M / ~2M examples per 2D / 3D dataset | [WALRUS] §5, p. 8 |
| Seven-heading categorisation | [WALRUS] Table 1, p. 10 |
| DPOT on acoustics, Poseidon on Euler | [WALRUS] §5.2, p. 11 |
| HalfWALRUS 2D subset, "strong anisotropic behavior" | [WALRUS] §5.3, pp. 11–12 |
| APE helps or hurts by symmetry | [WALRUS] App. B.2.1, p. 26 |
| Pretraining dataset table | [WALRUS] Table 5, p. 28 |
