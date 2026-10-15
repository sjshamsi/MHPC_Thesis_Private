# How might we categorise The Well datasets (if we can)

The Well paper says it has 16 datasets. We gotta note that `acoustic_scattering` has three variants (discontinuous / inclusions / maze), `MHD` and `supernova_explosion` come with two resolutions, and `turbulent_radiative_layer` comes in 2D and 3D.

WALRUS pretrains on 19 datasets, 18 from The Well (acoustic variants and Euler boundary-condition variants treated seperately) plus `FBHarmonics` from FlowBench where a fluid flows pas an obstacle.

In the Walrus paper, `post_neutron_star_merger` and `convective_envelope_rsg` are held out of pretraining and used as downstream tasks. Both 3D and expensive through numerical solvers.

---

## 0. At a glance

| Question | Short answer |
|---|---|
| 1. Categorise by physical domain | We can take a queue from the Walrus paper for physical domain categorisation (§1), but equations in two different settings might act similarly and vice versa. |
| 2. Categorise by equation form | Six families (§2.2); most datasets are compressible Euler plus source terms. Several "equation twins" exist (§2.4), e.g. `euler_multi_quadrants` and `turbulent_radiative_layer`, `shear_flow` and `rayleigh_benard`. The family split (§2.2) puts some members in pretraining and holds out others. |
| 3. What kind of categorisation is in the papers? | The Well doesn't categorise. WALRUS categorises datasets by physical domain in Table 1. |
| 5. How should WE split the data? | We can have one crossover experiment per axis of categorisation (equation family, linearity, dissipation, smoothness, predictability, whatever we deem a valid axis to split upon), plus a boundary-condition negative control and a data-driven axis. The axes are then compared by how well each one predicts transfer (§5). |

---

## 1. Categorisation by the physical domain

The Walrus paper abstract lists the domains, "astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids". Table 1 in the walrus paper says:

| Heading | Datasets |
|---|---|
| Biological and chemical behavior | Active Matter, Gray-Scott | application domain |
| Acoustics and wave propagation | Maze, Inclusions, Discontinuous, Staircase |
| Astrophysical and geoscience applications | Supernova (3D), TGC (3D), PlanetSWE |
| Plasmas | MHD (3D) |
| Viscous fluids | Rayleigh-Benard, Shear Flow, FBHarmonics, RT Instability (3D) |
| Inviscid fluids | MultiQuadrantsP, MultiQuadrantsO, TRL (2D), TRL (3D) |
| Non-newtonian fluids | Viscoelastics |

These headings kind of mix criteria, because some are application domains, and others identify classes like fluid-models or equation classes. Although these are simply for conveinience and are not defined explicitly. E.g.

- TRL is astrophysical domain but sits under Inviscid fluids (grouped by equations with
  `euler_multi_quadrants`). Supernova and TGC solve the same compressible Euler + cooling family plus gravity, but sit under Astrophysical applications (grouped by domain).
- PlanetSWE is geophysics and is grouped with astrophysics, not with the other fluid dynamics.
- Active Matter is grouped with Gray-Scott (both are "biological"), although at the equation level it is closest to Viscoelastics (§2.4).

### 1.2 Multi-label physics features

If we go by physical domain, we might want to look for not just one heading, but a collection of physical features. E.g. `post_neutron_star_merger` is magnetised, relativistic, radiating and rotating. So in this section, we can lay out several 'physical ingredients' and give each dataset a tick if it has the ingredient.

If we can see a certain physics feature that shows up as term in the equation of the dataset's system, or an extra equation in the dataset's system, we write it down. We're no doing throught all the dataset papers but the Well's paper's appendix C which has a lot of detail so we should be fine. Also noting if the dataset's system is turbulent.

| Column | Explanation | Source |
|---|---|---|
| Compressible | If the fluid density can change as it moves, and pressure disturbances travel at finite sound speed. If volume is fixed the fluid is incompressible | Maybe in equations $\rho$ has its own evolution equation. Then $\partial \rho / \partial t + \nabla\cdot(\rho u) = 0$ and $\nabla\cdot u \neq 0$ means compressable. Incompressible impose $\nabla\cdot u = 0$. |
| Shocks / discontinuities | Sharp jumps (a shock front, a contact surface) in the solution or the travelling medium (a wall, an inclusion). | We can have a conservation law equation like $\partial(\text{quantity}) / \partial t + \nabla\cdot(\text{flux}) = 0$. E.g. in Euler the mass flux is $\rho u$. In AS fluxes look like $K\,u$ or $p / \rho$ and all parts fof the wave travel at $c$. Although we may have a non-linear flux, maybe the flux depends on the solution itself (e.g. $\rho u^2$ or $u (\rho E + p)$ ). E,g. Burger's equation is $\partial u / \partial t + \partial (u^2 / 2) / \partial x = 0$. AND if we have no viscosity acting like a diffusion term (e.g. $\nu \nabla^2 u$ smooths steepest gradients and the shock's width then scales with $\nu$) then shocks are more probable. AND if the initial data is piecewise-constant in $\rho$, $p$, or $u$ then shocks might happen.
| Turbulence | Chaotic, multi-scale motion in which energy cascades from large eddies to small ones. | Needs the nonlinear advection term $u \cdot \nabla u$ and maybe a large Reynolds / Rayleigh / Mach number. |
| Gravity / buoyancy | Some body force from gravity, either uniform ($g$), from the fluid's own mass, or as buoyancy from density differences. | $\rho g$, $-\rho \nabla \Phi$, $b \hat{e}_z$ (buoyancy), $g \nabla h$, or, maybe curved spacetime in GR. |
| Radiation / cooling | Energy is exchanged with photons or neutrinos so the gas loses heat by radiating, or gains it by absorbing, or is pushed by radiation. | Needs a sink or source in the energy equation ($-E / t_{\rm cool}$, $\Gamma - \Lambda$), or a separate transport equation for radiation intensity $I$. |
| Magnetic field | Some magnetic field $B$ carried along by an electrically conducting fluid (e.g. plasma), which in turn pushes back on the fluid. | Some induction equation $\partial B / \partial t = \nabla \times (v \times B)$ and magnetic pressure / tension terms ($B^2$, $BB$) in the momentum equation. |
| Rotation | The fluid sits in a rotating frame or around a rotating body. | A Coriolis force $-2 \Omega \times u$, or maybe a spinning black hole in the spacetime metric. |
| Microstructure tensor | The fluid carries small-scale structure (polymer chains, rod-like swimmers) whose stretching and orientation are tracked by a tensor field, and that structure exerts a stress back on the flow. | An evolution equation for a tensor ($C$, $D$) plus a stress-divergence term $\nabla \cdot T$ or $\nabla \cdot \Sigma$ in the momentum equation. |
| Chemistry / composition | More than one chemical species, or a tracked composition variable that changes through reactions. | Reaction terms (e.g. $A B^2$) or a species-conservation equation with a source term. |

Some terms:
- MRI: magnetorotational instability, a magnetic instability in rotating disks that drives turbulence.
- EIT: elasto-inertial turbulence, chaotic flow driven jointly by fluid inertia and polymer elasticity.
- SAR: the steady "arrowhead" regime, a non-chaotic coherent state of the same viscoelastic flow.
- SPH: smoothed-particle hydrodynamics, a particle-based solver.
- Re, Ra, Ms, MA: Reynolds, Rayleigh, sonic Mach and Alfvénic Mach numbers.
- Ye: electron fraction.
- "—" means the ingredient is absent from the dataset's equations.

| Dataset | Compressible | Shocks / discontinuities | Turbulence | Gravity / buoyancy | Radiation / cooling | Magnetic field | Rotation | Microstructure tensor | Chemistry / composition |
|---|---|---|---|---|---|---|---|---|---|
| `acoustic_scattering` | Only in linearised form: small pressure waves in a fluid at rest | No shocks in solution; $\rho(x,y)$ jumps sharply (interfaces, inclusions, maze walls) | — | — | — | — | — | — | — |
| `helmholtz_staircase` | Only in linearised form: the wave equation for sound | None; the complexity is in the corrugated boundary shape | — | — | — | — | — | — | — |
| `euler_multi_quadrants` | Yes: full nonlinear compressible gas | Yes: Riemann initial data produce shocks, rarefactions and contact discontinuities | — | — | — | — | — | — | — |
| `turbulent_radiative_layer` | Yes: full nonlinear compressible gas, $\gamma = 5/3$ | — | Yes: a turbulent mixing layer between hot and cold gas | — | Yes: an optically thin cooling sink $-E/t_{\rm cool}$ in the energy equation | — | — | — | — |
| `supernova_explosion` | Yes: compressible gas, written in Lagrangian (particle) form | Yes: the expanding supernova blast wave is a shock | Yes: the surrounding medium is turbulent and filamentary | Yes: self-gravity, $-\nabla\Phi$ | Yes: radiative heating $\Gamma$ and cooling $\Lambda$ | — | — | — | — |
| `turbulence_gravity_cooling` | Yes: same equations as `supernova_explosion` | — | Yes: clouds start from a turbulent velocity field | Yes: self-gravity, $-\nabla\Phi$ | Yes: metallicity-dependent heating and cooling (switched off in the zero-metallicity runs) | — | — | — | — |
| `convective_envelope_rsg` | Yes: compressible gas coupled to radiation | — | Convective flow; [Well] does not call it turbulent *(our inference)* | Yes: the star's gravity, $-\rho\nabla\Phi$ with $\Phi$ set by the enclosed mass | Yes: full radiation transport, a separate equation for intensity $I$ | — | — (paper states rotation is neglected) | — | — |
| `MHD` | Yes, isothermal ($p = c_s^2\rho$, no energy equation) | Likely in the supersonic runs ($M_s$ up to 7) *(our inference)* | Yes: driven by large-scale random forcing $f$ | — | — | Yes: ideal MHD with an induction equation | — | — | — |
| `post_neutron_star_merger` | Yes: relativistic compressible gas, nuclear equation of state | — | Yes: MHD turbulence driven by the MRI | Yes: through GR, i.e. curved spacetime around a black hole | Yes: neutrino transport and neutrino–gas energy exchange | Yes: ideal GR-MHD | Yes: spinning black hole (spin a from 0 to 1) and an orbiting disk | — | Yes: electron fraction Ye with a lepton-number source term |
| `planetswe` | Not physically (water is incompressible), but layer depth $h$ obeys a density-like conservation law (§2.2) | — | — | Yes: gravity through $g\nabla h$ | — | — | Yes: Coriolis force $-2\Omega\times u$ on a rotating sphere | — | — |
| `rayleigh_benard` | No: incompressible (Boussinesq) | — | Chaotic convection at the highest Rayleigh numbers ($Ra$ up to $10^{10}$) *(our inference)* | Yes: buoyancy force $b\,\hat{e}_z$ from temperature differences | — | — | — | — | — |
| `rayleigh_taylor_instability` | Not acoustically: density varies because two fluids mix, and $\nabla\cdot u \neq 0$ only through molecular diffusion | — | Yes: the mixing layer becomes turbulent (paper expects a $k^{-5/3}$ spectrum) | Yes: gravity $\rho g$ on a heavy-over-light configuration | — | — | — | — | — |
| `shear_flow` | No: incompressible | — | Yes: Kelvin–Helmholtz instability and transition to turbulence ($Re$ up to $5\times10^5$) | — | — | — | — | — | — |
| `viscoelastic_instability` | No: incompressible | — | Yes: EIT is one of the states in the data | — | — | — | — | Yes: polymer conformation tensor C, feeding back as polymer stress | — |
| `active_matter` | No: incompressible Stokes flow | — | Not stated in [Well]. Stokes flow has no inertia, so any chaotic flow would be "active turbulence" driven by the swimmers *(our inference)* | — | — | — | — | Yes: orientation tensor D = ⟨pp⟩, feeding back as an active stress | — |
| `gray_scott_reaction_diffusion` | No fluid at all (no velocity field) | — | — | — | — | — | — | — | Yes: two reacting chemical species A and B |

#### Evidence for regime entries

Most entries above are single terms in the equations; §2.1 lists each with its equation number.
Turbulence and shocks are regimes, not terms, so they rest on the paper's descriptions:

| Dataset | Entry | Evidence ([Well] App. C) |
|---|---|---|
| `acoustic_scattering` | material jumps | Maze walls $\rho = 10^6$, paths $\rho = 3$ (p. 32). |
| `euler_multi_quadrants` | shocks | "A combination of shocks, rarefaction waves, or contact discontinuities" (p. 34). |
| `turbulent_radiative_layer` | turbulence | Mixing of hot and cold phases driven by their relative velocity (p. 44). |
| `supernova_explosion` | shocks, turbulence | "Shock waves (such as SN shells)"; "turbulence and cooling form a dense filamentary structure" (p. 43). |
| `turbulence_gravity_cooling` | turbulence | Initial clouds "with turbulence" (p. 43). |
| `convective_envelope_rsg` | turbulence *(inference)*; no rotation | Described as convective, not turbulent. "Neglect stellar rotation and magnetic fields" (p. 34). |
| `MHD` | turbulence; shocks *(inference)* | Continuous solenoidal driving (p. 37). Sonic Mach number up to 7, i.e. supersonic. |
| `post_neutron_star_merger` | turbulence, rotation | MRI "drives fluid motion"; "magnetohydrodynamic turbulence"; torus around a rotating black hole, spin $a \in [0, 1]$ (p. 39). |
| `rayleigh_benard` | chaotic *(inference)* | $Ra$ up to $10^{10}$; "highly sensitive to initial conditions" (p. 40). |
| `rayleigh_taylor_instability` | turbulence | "As turbulence develops … inertial cascade", $k^{-5/3}$ (p. 41). |
| `shear_flow` | turbulence | "Turbulent transition … Kelvin–Helmholtz instabilities, turbulent eddies" (p. 42). |
| `viscoelastic_instability` | turbulence | Data contain laminar, EIT and SAR states (p. 44). |
| `active_matter` | chaotic *(inference)* | Not stated. Stokes flow has no inertia, so any chaos is activity-driven. |

---

## 2. Categorisation by the functional form of governing equations

### 2.1 Governing system of each dataset

Equations and equation numbers are from [Well] App. C (pp. 31–44).

| Column | Meaning | Where to look |
|---|---|---|
| Governing system | The PDEs that are time-stepped. | Equations containing $\partial_t$. |
| Closure / EOS | The relation that fixes the remaining unknowns: an equation of state for a gas, $\nabla\cdot u = 0$ for an incompressible fluid, a stress law for polymers or particles. | Algebraic relations without $\partial_t$: $p = (\gamma-1)\rho e$, $p = c_s^2\rho$, $\nabla\cdot u = 0$. |
| Extra terms | Additions to the core system: body forces, energy sources and sinks, coupling to another field, forcing. | Non-zero right-hand sides: $\rho g$, $-\nabla\Phi$, $-E/t_{\rm cool}$, $f$, $\nabla\cdot T(C)$. |
| Dissipation | Terms that remove energy and smooth the solution. "None" means ideal equations: only the numerical scheme dissipates, at shocks (§5.4.1). | $\partial_t$ balanced by an even-order space derivative ($\nu\Delta u$, $-\nu\nabla^4 u$), or SPH's $a_{\rm visc}$. |
| Solver | Code and method. Shock-capturing FV smears jumps over a few cells; spectral methods need smooth fields; SPH data is resampled onto a grid. | "Simulation details" paragraphs; [Well] Table 4, p. 31. |

Abbreviations:
- FV: finite volume.
- TVD: total variation diminishing (no spurious oscillations at jumps).
- ENO / WENO: (weighted) essentially non-oscillatory reconstruction.
- SPH: smoothed-particle hydrodynamics.
- BVP: boundary value problem.
- BIE: boundary integral equation.
- IMEX: implicit–explicit time stepping.
- ETDRK4: exponential time-differencing fourth-order Runge–Kutta.

| Dataset | Governing system | Closure / EOS | Extra terms | Dissipation | Solver |
|---|---|---|---|---|---|
| `acoustic_scattering` | Linear first-order acoustics (eqs. 1–3): $\partial_t p + K(\partial_x u + \partial_y v) = 0$, $\partial_t u + \frac{1}{\rho}\partial_x p = 0$, same for $v$. | No EOS. Fixed medium: density $\rho(x,y)$, bulk modulus $K = 4$, sound speed $c = \sqrt{K/\rho}$. $\rho(x,y)$ is the background, not an evolving field, so the configs can zero it. | None (all right-hand sides $= 0$). Variation comes from the $\rho(x,y)$ maps. Boundaries open in $y$, reflective in $x$. | None: first derivatives only. The flux limiter smooths slightly at material jumps. | Clawpack: explicit FV, TVD, monotonized central-difference limiter (p. 32). |
| `helmholtz_staircase` | Wave equation (eq. 17): $\partial_t^2 U - \Delta U = \delta(t)\,\delta(x-x_0)$. Solved as the Helmholtz BVP $-(\Delta+\omega^2)u = \delta_{x_0}$ (eq. 19); time series built as $U = u\,e^{-i\omega t}$ (p. 36). | Constant medium, $c = 1$ after rescaling. | Point source $\delta(t)\,\delta(x-x_0)$. Sound-hard staircase boundary, $n\cdot\nabla U = 0$ (eq. 18). | None: $\Delta U$ pairs with $\partial_t^2$, so waves propagate without decay. | High-order BIE with the Floquet–Bloch transform (p. 36). |
| `euler_multi_quadrants` | 2D compressible Euler, integral form (eqs. 13–14): $\frac{d}{dt}\iint_\Omega U\,dA + \oint_{\partial\Omega}(F\hat\imath + G\hat\jmath)\cdot\hat n\,dS = 0$, with $U = (\rho, \rho u, \rho v, \rho E)$ and $F = (\rho u, \rho u^2+p, \rho uv, u(\rho E+p))$. | Ideal gas: $\rho E = \frac{p}{\gamma-1} + \frac12\rho(u^2+v^2)$. Ten values of $\gamma$ in $[1.13, 1.76]$. | None (right-hand side $= 0$). Boundary (periodic or open) and piecewise-constant initial data vary. | None: the fluxes contain no derivatives, so there is no viscous stress. | Clawpack: explicit FV with Riemann solvers (p. 34). |
| `turbulent_radiative_layer` | Compressible Euler (eqs. 45–47): $\partial_t\rho + \nabla\cdot(\rho v) = 0$, $\partial_t(\rho v) + \nabla\cdot(\rho vv + P) = 0$, $\partial_t E + \nabla\cdot((E+P)v) = -E/t_{\rm cool}$. | Ideal gas, $\gamma = 5/3$. Eq. 48 prints $E = P/(\gamma-1)$, omitting $\frac12\rho v^2$. | Energy sink $-E/t_{\rm cool}$ (eq. 47), $t_{\rm cool} \in [0.03, 3.16]$. Fastest at intermediate temperatures (p. 44). | None. The sink removes energy but does not smooth (§5.4.1). | Athena++ ([Well] Table 4). C.15 gives no method details. |
| `supernova_explosion` | Compressible Euler in Lagrangian form, following each parcel as SPH requires (eqs. 41–43): $\frac{d\rho}{dt} = -\rho\nabla\cdot v$, $\frac{d^2 r}{dt^2} = -\frac{\nabla P}{\rho} + a_{\rm visc} - \nabla\Phi$, $\frac{du}{dt} = -\frac{P}{\rho}\nabla\cdot v + \frac{\Gamma-\Lambda}{\rho}$. $r$ is position, $u$ specific internal energy. | Ideal monatomic gas: $P = (\gamma-1)\rho u$, $\gamma = 5/3$ (eq. 40). | Self-gravity $-\nabla\Phi$ (eq. 42). Radiative heating and cooling $(\Gamma-\Lambda)/\rho$ (eq. 43), at solar metallicity. | Artificial viscosity $a_{\rm visc}$ (eq. 42): numerical, for shocks. | ASURA-FDPS N-body/SPH (density-independent SPH), resampled to a uniform grid (pp. 31, 43). |
| `turbulence_gravity_cooling` | Same as `supernova_explosion` (eqs. 40–43, p. 43). | Same. | Same terms. Heating and cooling depend on metallicity $Z \in \{Z_0, 0.1Z_0, 0\}$; $Z = 0$ is adiabatic. | Same $a_{\rm visc}$. | Same. |
| `convective_envelope_rsg` | Radiation hydrodynamics (eqs. 8–11): Euler for the gas, plus transport of radiation intensity $I(t, x, n)$ in each direction $n$. $\partial_t\rho + \nabla\cdot(\rho v) = 0$; $\partial_t(\rho v) + \nabla\cdot(\rho vv + \mathsf{P}_{\rm gas}) = -\mathbf{G}_r - \rho\nabla\Phi$; $\partial_t E + \nabla\cdot[(E + P_{\rm gas})v] = -cG^0_r - \rho v\cdot\nabla\Phi$; $\partial_t I + c\,n\cdot\nabla I = S(I, n)$. | Ideal gas: $E = E_g + \frac12\rho v^2$ with $E_g = \frac32 P_{\rm gas}$ ($\gamma = 5/3$). OPAL opacities set the gas–radiation coupling in $S$ (eq. 12). | Gravity from the enclosed mass $m(r)$: $-\rho\nabla\Phi$ (eq. 9), $-\rho v\cdot\nabla\Phi$ (eq. 10). Radiation four-force: $-\mathbf{G}_r$ (eq. 9), $-cG^0_r$ (eq. 10). No rotation, no magnetic field (p. 34). | None. | Athena++: Godunov FV, spherical-polar grid, logarithmic radial spacing (p. 34). |
| `MHD` | Ideal compressible MHD, Gaussian units (eqs. 21–23): $\partial_t\rho + \nabla\cdot(\rho v) = 0$; $\partial_t(\rho v) + \nabla\cdot[\rho vv + (p + \frac{B^2}{8\pi})I - \frac{1}{4\pi}BB] = f$; $\partial_t B - \nabla\times(v\times B) = 0$. In eq. 22, $\frac{B^2}{8\pi}$ is magnetic pressure and $-\frac{1}{4\pi}BB$ magnetic tension. $\nabla\cdot B = 0$ is implied, not printed. | Isothermal, $p = c_s^2\rho$: no energy equation. | Forcing $f$ (eq. 22): random, solenoidal, continuous, at $k \approx 2.5$. | None (ideal): no $\nu\Delta v$, no $\eta\Delta B$. Field lines are frozen into the fluid. | Third-order hybrid ENO FV, periodic box (p. 37). |
| `post_neutron_star_merger` | General-relativistic ideal MHD (eqs. 26–28), plus lepton-number conservation (eq. 29) and neutrino transport (eq. 30). | Tabulated nuclear EOS (SFHo): pressure from density, temperature and electron fraction. | Gravity as spacetime curvature, $T^\kappa_{\ \lambda}\Gamma^\lambda_{\nu\kappa}$ (eq. 27). Neutrino four-force $G_\nu$ (eq. 27). Lepton source $G_{ye}$ (eq. 29). | None: "conductivities are assumed to be infinite" (p. 39). | νbhlight: FV GRMHD with constrained transport (keeps $\nabla\cdot B = 0$), operator-split with Monte Carlo neutrinos (p. 39). |
| `planetswe` | Rotating shallow water on the sphere (eqs. 24–25): $\partial_t u = -u\cdot\nabla u - g\nabla h - \nu\nabla^4 u - 2\Omega\times u$, $\partial_t h = -H\nabla\cdot u - \nabla\cdot(hu) - \nu\nabla^4 h + F$. | Hydrostatic: the pressure force is $-g\nabla h$. $h$ is layer height, $H$ mean depth. $h$ obeys a density-like conservation law (§2.2). | Coriolis $-2\Omega\times u$. Forcing $F$ with daily and annual cycles. Earth topography. | Hyperviscosity $-\nu\nabla^4 u$, $-\nu\nabla^4 h$, $\nu = 1.76\times10^{-10}$: "largely non-physical", for stability (p. 37). | Dedalus: spin-spherical-harmonic pseudospectral, IMEX Runge–Kutta, resampled to a latitude–longitude grid (p. 38). |
| `shear_flow` | 2D incompressible Navier–Stokes plus a passive scalar $s$: $\partial_t u - \nu\Delta u + \nabla p = -u\cdot\nabla u$, $\partial_t s - D\Delta s = -u\cdot\nabla s$. | Incompressible: $\nabla\cdot u = 0$ sets $p$, with gauge $\int p = 0$. | None: $s$ does not appear in the momentum equation (passive). | $-\nu\Delta u$ with $\nu = 1/Re$; $-D\Delta s$ with $D = 1/(Re\,Sc)$. | Dedalus: Fourier–Chebyshev pseudospectral, adaptive time step (p. 42). |
| `rayleigh_benard` | 2D incompressible Navier–Stokes (Boussinesq) plus an active scalar, buoyancy $b$: $\partial_t b - \kappa\Delta b = -u\cdot\nabla b$, $\partial_t u - \nu\Delta u + \nabla p - b\,e_z = -u\cdot\nabla u$. | Boussinesq: $\nabla\cdot u = 0$, with density differences kept only in the buoyancy force. Gauge $\int p = 0$. | Buoyancy $-b\,e_z$ in the momentum equation: $b$ drives the flow. Hot bottom wall, cold top wall. | $-\nu\Delta u$ and $-\kappa\Delta b$, with $\kappa = (Ra\,Pr)^{-1/2}$, $\nu = (Ra/Pr)^{-1/2}$. | Dedalus spectral, adaptive time step (p. 40). |
| `rayleigh_taylor_instability` | Variable-density, low-Mach Navier–Stokes for two miscible fluids (eqs. 32–35): $\partial_t\rho + \nabla\cdot(\rho u) = 0$, $\partial_t(\rho u) + \nabla\cdot(\rho uu) = -\nabla p + \nabla\cdot\tau + \rho g$, $\nabla\cdot u = -\kappa\nabla\cdot(\nabla\rho/\rho)$. | No EOS; eq. 34 replaces it. $\nabla\cdot u \neq 0$ only through interdiffusion, so there are no sound waves. | Gravity $\rho g$ (eq. 33), with heavy fluid above light. | Viscous stress $\nabla\cdot\tau$, $\tau = \rho\nu(\nabla u + (\nabla u)^T - \frac23(\nabla\cdot u)I)$ (eq. 35), $\nu$ rescaled in time to hold the Kolmogorov scale at grid spacing (eq. 38). Diffusion $\kappa$ (eq. 34). | TurMix3D: staggered-mesh FV, Lagrange + remap, TVD with Van Leer limiters, multigrid pressure solve (p. 41). |
| `viscoelastic_instability` | Incompressible Navier–Stokes plus the polymer conformation tensor $C$ (eqs. 49a–d): $Re(\partial_t u + u\cdot\nabla u) + \nabla p = \beta\Delta u + (1-\beta)\nabla\cdot T(C)$, $\nabla\cdot u = 0$, $\partial_t C + (u\cdot\nabla)C + T(C) = C\cdot\nabla u + (\nabla u)^T\cdot C + \varepsilon\Delta C$. | Incompressible flow. FENE-P stress $T(C) = \frac{1}{Wi}\big(\frac{C}{1-(\mathrm{tr}\,C-3)/L_{\max}^2} - I\big)$ (eq. 49c): springs that stiffen near maximum extension $L_{\max} = 70$. | Polymer stress $(1-\beta)\nabla\cdot T(C)$ (eq. 49a). In eq. 49d the flow stretches $C$ ($C\cdot\nabla u + (\nabla u)^T\cdot C$) and $T(C)$ relaxes it. | Solvent viscosity $\beta\Delta u$, $\beta = 0.9$. Stress diffusion $\varepsilon\Delta C$, $\varepsilon = 2\times10^{-6}$, mainly for numerical stability. | Dedalus (p. 44). |
| `active_matter` | Stokes flow (eq. 6) coupled to the Smoluchowski equation (eq. 4): $-\Delta u + \nabla P = \nabla\cdot\Sigma$, $\nabla\cdot u = 0$ (no $\partial_t u$, no $u\cdot\nabla u$); $\partial_t\Psi + \nabla_x\cdot(\dot x\Psi) + \nabla_p\cdot(\dot p\Psi) = 0$ conserves particles in position–orientation space. | No moment closure: the full $\Psi$ is resolved (256 orientational modes). Stored fields are its moments: concentration and $D = \langle pp\rangle$. | Active stress $\Sigma = \alpha D + \beta S:E - 2\zeta\beta(D\cdot D - S:D)$ (eq. 7): $\alpha D$ is self-propulsion, the $\zeta$ term steric alignment. | Diffusion from $-d_T\nabla_x\log\Psi$ and $-d_R\nabla_p\log\Psi$ in the fluxes (eq. 5). The Stokes flow is fully viscous. | Pseudo-spectral (Fourier in space and orientation), SBDF2 IMEX, periodic box (p. 33). |
| `gray_scott_reaction_diffusion` | Two-species reaction–diffusion (eqs. 15–16): $\partial_t A = \delta_A\Delta A - AB^2 + f(1-A)$, $\partial_t B = \delta_B\Delta B + AB^2 - (f+k)B$. No velocity. | None needed. | Reaction $\mp AB^2$ converts $A$ to $B$. Feed $f(1-A)$; kill $-(f+k)B$. | $\delta_A\Delta A$ and $\delta_B\Delta B$, with $\delta_A = 2\times10^{-5}$, $\delta_B = 10^{-5}$. The unequal rates allow patterns. | Chebfun: Fourier spectral, ETDRK4, periodic (p. 35). |

### 2.2 Equation families and the family split

Each dataset is assigned one family from its governing system (§2.1). Terms used:

- **Linear**: no term multiplies two state variables. Nonlinear examples: $u\cdot\nabla u$, $\rho u^2$, $AB^2$.
- **Compressible**: density has its own evolution equation, $\partial_t\rho + \nabla\cdot(\rho u) = 0$, so $\nabla\cdot u \neq 0$ and sound travels at finite speed.
- **Incompressible**: $\nabla\cdot u = 0$ is imposed. Pressure is whatever enforces it, so sound speed is infinite.
- **Field**: a transported variable besides velocity and pressure (a scalar or tensor). *Active* if it appears in the momentum equation, *passive* if not.

| Family | Deciding feature | Members |
|---|---|---|
| **I. Linear waves** | Linear, hyperbolic. Compressible Euler linearised about a fluid at rest: the nonlinear fluxes $\rho u^2$, $u(\rho E+p)$ are dropped, so waves do not steepen and cannot form shocks. | `acoustic_scattering`: variable medium $\rho(x,y)$.<br>`helmholtz_staircase`: constant medium, complex boundary. |
| **II. Compressible Euler + sources** | Nonlinear conservation laws for $\rho$, $\rho u$, $E$, with no physical viscosity. Members differ only in right-hand-side terms. | IIa `euler_multi_quadrants`: none.<br>IIb `turbulent_radiative_layer`: cooling sink.<br>IIc `supernova_explosion`, `turbulence_gravity_cooling`: self-gravity, heating and cooling.<br>IId `convective_envelope_rsg`: gravity, radiation transport.<br>IIe `planetswe`: shallow water (see note), plus Coriolis, forcing, hyperviscosity. |
| **III. Compressible MHD** | Family II plus the induction equation for $B$ and magnetic pressure and tension in the momentum equation. | IIIa `MHD`: isothermal, forced.<br>IIIb `post_neutron_star_merger`: general relativity, neutrinos, lepton number. |
| **IV. Incompressible Navier–Stokes + field** | $\nabla\cdot u = 0$ (or low-Mach), inertia ($u\cdot\nabla u$), viscosity, plus a transported field. | IVa `shear_flow`: passive scalar.<br>IVb `rayleigh_benard`: active scalar (buoyancy).<br>IVc `rayleigh_taylor_instability`: variable density with gravity.<br>IVd `viscoelastic_instability`: active tensor (polymer stress). |
| **V. Stokes + kinetic equation** | No inertia: the flow is set instantly by an active stress from the particle orientation tensor. | `active_matter` |
| **VI. Reaction–diffusion** | No velocity field and no advection. | `gray_scott_reaction_diffusion` |

**Placement notes.**

- **I vs II is amplitude, not physics.** Both are compressible gas; I keeps only small disturbances. `acoustic_scattering` is linearised Euler with bulk modulus $K = \gamma p_0$.
- **`planetswe` in II is a structural choice.** Shallow water matches 2D isentropic Euler with $\gamma = 2$: $h$ plays $\rho$ and $gh^2/2$ plays $p$. This is a standard result, not stated in either paper. Physically the fluid is incompressible; the equations are "derived from depth-integrating the incompressible Navier–Stokes equations" ([Well] p. 37). If PS transfers better from IV than from II, the physical reading wins.
- **`rayleigh_taylor_instability` in IV.** It has no equation of state and no sound waves; $\nabla\cdot u \neq 0$ only through interdiffusion (eq. 34). Below Atwood 0.1 it reduces to Boussinesq, i.e. to `rayleigh_benard`'s equations ([Well] p. 41).
- **IV is a ladder of feedback.** SF's scalar does not act on the flow. RB's scalar acts as a body force $-b\,e_z$. VE's tensor $C$ is stretched by the flow and pushes back as a stress $\nabla\cdot T(C)$.
- **V is IVd without inertia.** It has the same tensor-stress coupling as VE, but no $\partial_t u$ or $u\cdot\nabla u$, and is driven actively rather than relaxing passively.

**Cross-cutting labels.**

| Property | Datasets |
|---|---|
| Linear | Family I only |
| Ideal (no dissipation term) | `acoustic_scattering`, `helmholtz_staircase`, `euler_multi_quadrants`, `turbulent_radiative_layer`, `convective_envelope_rsg`, `MHD`, `post_neutron_star_merger` |
| Elliptic constraint (incompressibility or Stokes) | Families IV and V |
| Curvilinear coordinates | `planetswe` (sphere), `convective_envelope_rsg` $(r,\theta,\phi)$, `post_neutron_star_merger` $(\log r,\theta,\phi)$ |

#### The family split (experiment A1)

The crossover design is in §5.1 and the dataset codes in §5.2. The split is 2D only: family III and
the 3D members of II and IV have no 2D data (3D variant: §5.6).

| Family | Pretrain | In-family test | Without Euler |
|---|---|---|---|
| I | AD, AM | HS (AI as a near-duplicate sanity check) | unchanged |
| II | EMQ-P, EMQ-O | TRL2, PS | Drop the level; TRL2 and PS become probes. |
| IV | SF, RB | VE | unchanged |
| Probes (every run) | — | ACT (V), GS (VI) | ACT, GS, TRL2, PS |

What each target tests:

- **HS**: same equations as I, different presentation (constant medium, staircase boundary).
- **TRL2**: one extra sink term over EMQ.
- **PS**: II's structure on a sphere with rotation and forcing. It also tests the PS placement above.
- **VE**: one step up the IV ladder, from scalar feedback to tensor feedback.
- **ACT, GS**: no in-family pretraining, which gives the floor for out-of-family transfer.

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

### 2.4 "Equation twins" in different physical contexts

These are the pairs your question points at: different physical settings, essentially the same
governing equations.

| Pair | Physical contexts | What the equations share | What differs |
|---|---|---|---|
| `euler_multi_quadrants` ↔ `turbulent_radiative_layer` | Textbook gas-dynamics Riemann problem ↔ astrophysical hot/cold gas mixing layer | Identical compressible Euler system | TRL adds one energy sink ($-E/t_{\rm cool}$); $\gamma$ fixed at 5/3; shear-driven turbulence rather than shock interaction |
| `shear_flow` ↔ `rayleigh_benard` | Kelvin–Helmholtz-type shear layers ↔ thermal convection | 2D incompressible NS + one scalar advection–diffusion equation; same solver (Dedalus); same field shapes (scalar, pressure, velocity) | Whether the scalar feeds back (buoyancy) and the boundaries (fully periodic vs walls) |
| `planetswe` ↔ `euler_multi_quadrants` | Earth's atmosphere ↔ gas dynamics | Shallow water $\cong$ isentropic Euler with $\gamma = 2$ | Rotation, forcing, hyperviscosity, spherical geometry, no shocks by design |
| `viscoelastic_instability` ↔ `active_matter` | Industrial polymer solutions ↔ biological active suspensions | Velocity coupled to an evolving 2nd-order tensor that re-enters the momentum balance as a stress divergence | Inertia ($Re = 1000$ vs Stokes); passive relaxation vs active driving |
| `acoustic_scattering` ↔ `helmholtz_staircase` | Geophysical probing ↔ photonic/phononic crystal design | Linear acoustic wave equation | Variable medium vs constant medium with a complex boundary; transient vs time-harmonic |
| `MHD` ↔ `post_neutron_star_merger` | Diffuse ISM turbulence ↔ neutron-star merger remnant | Ideal MHD conservation laws + induction | GR, full energy equation, lepton and neutrino transport |
| `supernova_explosion` ↔ `turbulence_gravity_cooling` | Same domain (ISM), different scenario | Literally the same equations ([Well] p. 43) | Initial conditions: point energy injection vs turbulent self-gravitating cloud |
| `rayleigh_benard` ↔ `rayleigh_taylor_instability` | Heated layer ↔ two fluids of different density | Gravity-driven incompressible flow | Boussinesq vs variable density; 2D vs 3D |
| `turbulent_radiative_layer` ↔ `turbulence_gravity_cooling` | Hot/cold gas mixing layer in a galactic outflow ↔ collapsing, star-forming molecular cloud | Compressible Euler, $\gamma = 5/3$, with optically thin radiative cooling removing thermal energy (IIb $\subset$ IIc) | TGC adds self-gravity, heating and metallicity-dependent cooling; SPH (artificial viscosity) regridded vs Athena++ grid |
| `convective_envelope_rsg` ↔ `supernova_explosion` | Convection in a red-supergiant envelope ↔ blast wave in the ISM | Compressible Euler, $\gamma = 5/3$, with gravity and a radiative energy exchange (IIc ↔ IId) | Full radiation transport vs optically thin heating/cooling; spherical $(r,\theta,\phi)$ vs Cartesian; steady convection vs a single expanding shock |

**Trivial twins.** Some entries are one dataset under a different boundary condition, resolution or
dimension, not two physical settings, so the table leaves them out:
- the three `acoustic_scattering` variants (only the medium $\rho(x,y)$ differs)
- `euler_multi_quadrants` periodic/open
- `MHD` 64/256
- `supernova_explosion` 64/128
- `turbulent_radiative_layer` 2D/3D

#### Twins involving the non-Well datasets WALRUS uses

WALRUS ([WALRUS] App. C, Tables 5–6) adds one non-Well pretraining dataset (FBHarmonics, from
FlowBench) and several non-Well downstream tasks:
- FPOSkelenton (FlowBench)
- PoolBoil Subcooled (BubbleML 2.0)
- Conditioned Incompressible NS (PDEArena)
- CE-RM (PDEGym)
- CNS Turbulent/Random (PDEBench, 3D)

WALRUS names these datasets but does not give their equations. The descriptions below come from the
source benchmarks (FlowBench, BubbleML 2.0, PDEArena, Poseidon/PDEGym, PDEBench), which are not in
`PDFs/`, so check parameter values against those papers before quoting them.

| Pair | Physical contexts | What the equations share | What differs |
|---|---|---|---|
| FBHarmonics ↔ FPOSkelenton | Flow past a spherical-harmonic-shaped obstacle ↔ flow past a skeleton-shaped obstacle | Literally the same equations and solver: 2D incompressible NS, flow past an immersed obstacle (FlowBench "FPO") | Only the obstacle-geometry family. This is the FlowBench analogue of `supernova_explosion` ↔ `turbulence_gravity_cooling` (pretraining member ↔ downstream member) |
| FBHarmonics ↔ `shear_flow` | Engineering wake flow ↔ Kelvin–Helmholtz shear layer | 2D incompressible NS (family IV core, $\nabla\cdot u = 0$, $-\nu\Delta u$) | FB has no transported scalar, an inflow/outflow channel and an obstacle boundary; SF is fully periodic with a passive scalar |
| CE-RM ↔ `euler_multi_quadrants` | Shock hitting a perturbed density interface (Richtmyer–Meshkov) ↔ Riemann quadrant problem | Identical 2D compressible Euler (family IIa), ideal gas | Initial-condition family only (interface + shock vs piecewise-constant quadrants); single $\gamma$ vs a sweep of $\gamma$ |
| Conditioned Incompressible NS ↔ `rayleigh_benard` | Buoyant smoke plume ↔ thermal convection between plates | 2D incompressible NS + an active scalar that feeds back as a buoyancy body force (family IVb) | PDEArena's scalar is only advected (no diffusion), the buoyancy magnitude is the conditioning parameter, and there are no heated walls; RB's buoyancy diffuses and is set by Ra/Pr |
| PoolBoil Subcooled ↔ `rayleigh_benard` | Pool boiling of a subcooled liquid on a heater ↔ thermal convection | Incompressible NS + temperature advection–diffusion + gravity, heated from below | Two phases with a moving interface (level set), surface tension and phase change (evaporation/condensation latent heat) |
| CNS Turbulent/Random ↔ `turbulent_radiative_layer_3D` | Decaying compressible turbulence in a periodic box ↔ astrophysical mixing layer | 3D compressible fluid equations, $\gamma = 5/3$ (family II). PDEBench's 3D CNS uses very small viscosities, so it is close to pure Euler | CNS adds viscous stress; TRL adds the cooling sink; periodic box vs a shear layer |
| CNS Turbulent ↔ `MHD` | Hydrodynamic turbulence ↔ magnetised ISM turbulence | 3D compressible turbulence in a periodic box; MHD with B = 0 reduces to (isothermal) Euler | MHD carries B and the induction equation, is isothermal and is driven by forcing; CNS has a full energy equation and decays |

**Same physics, different equations (the inverse case).** CE-RM and
`rayleigh_taylor_instability` share a physical mechanism: an interface between fluids of different
density becomes unstable and mixes, and Richtmyer–Meshkov is the impulsive, shock-driven analogue of
Rayleigh–Taylor. They do not share equations, though: CE-RM is inviscid compressible Euler, while RTI
is low-Mach variable-density Navier–Stokes with diffusion. This pair is a test of whether transfer
follows the mechanism or the equation form.

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
  `field_to_index_map` in this repo), so the model is not told that momentum in one dataset is $\rho\,\times$
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

| Grouping | Datasets | Approx. share of pretraining examples |
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
4. Compute the **in-level advantage** $\Delta$: how much better each model does on its own level's
   targets than the other model does.

With transfer gain $g(\text{run}, t) = 1 - \mathrm{VRMSE}(\text{run}, t) / \mathrm{VRMSE}(\text{scratch}, t)$ at a fixed finetuning
budget:

$$
\Delta_{\rm axis} = \tfrac12\Big[\operatorname*{mean}_{t\in T_A}\big(g(A,t)-g(B,t)\big) + \operatorname*{mean}_{t\in T_B}\big(g(B,t)-g(A,t)\big)\Big]
$$

$\Delta > 0$ means grouping along this axis captures something that transfers. The double difference
cancels out both target difficulty and the generic "any pretraining helps" effect, so no separate
mixed control is needed per axis. **The headline result is a comparison of $\Delta$ across axes.** For
multi-level axes (equation family), $\Delta$ is the mean over levels of (in-level gain minus mean
other-level gain).

**Scope: 2D only.** In WALRUS, 2D and 3D use separate encoder/decoder weights, so transfer between
them mostly tests the encoder (§2.5). A 3D version is sketched in §5.6.

### 5.2 Dataset attributes (2D, on disk under `/leonardo_work/ICT26_MHPC_0/sshamsi/datasets/`)

Boundary conditions and field names below are read from each dataset's HDF5 metadata. The
predictability proxy is the **best** of the four models' median VRMSE averaged over steps 21–60
in [WALRUS] Table 1 (p. 10). Values $\gtrsim 0.6$ mean even the best model has decorrelated from the
truth by that horizon, which WALRUS notes is expected for chaotic systems (p. 9).

| Code | Dataset | Family (§2.2) | Linear | Explicit diffusion / viscosity | Sharp features | Predictability proxy (T21–60) | BCs (metadata) | Fields |
|---|---|---|---|---|---|---|---|---|
| AD | `acoustic_scattering_discontinuous` | I waves | ✓ | none | ✓ material interface | 0.04 predictable | mixed wall / open | pressure, velocity, (density†, speed_of_sound) |
| AI | `acoustic_scattering_inclusions` | I waves | ✓ | none | ✓ inclusions | 0.13 predictable | mixed wall / open | same as AD |
| AM | `acoustic_scattering_maze` | I waves | ✓ | none | ✓ maze walls | 0.05 predictable | mixed wall / open | same as AD |
| HS | `helmholtz_staircase` | I waves | ✓ | none | — smooth, time-harmonic | 0.003 predictable | open + wall (staircase) | pressure_re, pressure_im, mask |
| TRL2 | `turbulent_radiative_layer_2D` | II compressible | — | none (cooling sink only) | ✓ hot/cold phase interfaces | 0.84 **chaotic** | x periodic, y open | density, pressure, velocity |
| PS | `planetswe` | II (SWE analogue), **sphere** | — | hyperviscosity only | — | 0.02 predictable | $\phi$ periodic, $\theta$ open | height, velocity |
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

- **Linearity $\equiv$ the wave family.** Every linear dataset is in family I. Axis A2 therefore
  reuses the family-I run; its only new information is whether a *multi-family* nonlinear model
  differs from the single-family incompressible one.
- **Dissipation $\approx$ sharpness.** Every dataset with sharp features is non-diffusive, and no
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

#### A1. Equation family

Defined, with its pretrain and test sets, in [§2.2](#the-family-split-experiment-a1). Runs R1–R3 (§5.5).

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
Families are mixed within each level, so any $\Delta$ comes from dissipation rather than a single family.

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

`viscoelastic_instability` is split by file, because its regimes differ (§5.4.3): VE-AH is the
steady arrowhead state, and VE-CAR / VE-EIT are chaotic. Select them with
`include_filters: ["_AH."]` or `include_filters: ["_CAR.", "_EIT."]`. The trailing `.` excludes the
`transtochaos_EIT_*` edge-state files. `shear_flow` sits in the predictable set only *over the
evaluated horizon*; it is chaotic in principle (§5.4.3).

**A5b, a cleaner within-dataset variant.** `rayleigh_benard` (one file per Rayleigh number),
`shear_flow` (per Reynolds number), `active_matter` (per activity $\alpha$) and
`turbulent_radiative_layer_2D` (per $t_{\rm cool}$) are stored with one file per parameter value. The
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

Physics is deliberately mixed within each level. **If $\Delta$ here is as large as for the physics axes,
the model is grouping datasets by how they are presented, not by their dynamics.** This is the
direct test of whether the worry that "same equations can look very different" matters.

#### A7. Data-driven clusters (defined after measurement)

Compute label-free descriptors on each dataset's training split:

- persistence VRMSE ($u_{t+1} = u_t$)
- spectral slope and high-frequency energy fraction
- total variation / gradient sparsity
- decorrelation time in steps
- error of the best linear one-step predictor (a DMD fit)

Standardise the descriptors, cluster into two groups (k-means, $k = 2$), pretrain on the two most
central members of each cluster, and test on the rest. Comparing A7's $\Delta$ with A1–A6 answers
empirically whether labels or statistics group datasets better. Across all runs, regressing gain
on descriptor distance gives a second, finer-grained answer.

*Optional A0: WALRUS's domain headings as a baseline axis.* Not feasible cleanly with the 2D data
on disk: most headings have at most one available 2D dataset outside the acoustics group.

### 5.4.1 What makes a dataset diffusive (axis A3)

**The rule:** a dataset is diffusive when its equations contain a *first derivative in time*
balanced by an *even-order derivative in space*, $\partial_t f = \nu\Delta f$ (or $-\nu\nabla^4 f$). The type of each term
decides how it acts on a Fourier mode $e^{ikx}$:

| Term type | Example | Effect on a mode $e^{ikx}$ | Character |
|---|---|---|---|
| $\partial_t f = \nu\Delta f$ (diffusion, viscosity) | $\nu\Delta u$, $\kappa\Delta b$ | amplitude $\times\, e^{-\nu k^2 t}$ | **Diffusive.** Small scales damped fastest; smooths; irreversible. |
| $\partial_t f = -\nu\nabla^4 f$ (hyperviscosity) | `planetswe` | amplitude $\times\, e^{-\nu k^4 t}$ | Diffusive, concentrated at the smallest scales. |
| First-order space derivatives only (advection, fluxes) | $\partial_t p + K\nabla\cdot u$, Euler fluxes | phase shift $e^{ik(x-ct)}$, amplitude kept | **Non-diffusive.** Transport without spreading; energy conserved; sharp features persist. |
| Zeroth-order sink | $-E/t_{\rm cool}$ | amplitude $\times\, e^{-t/t_{\rm cool}}$, same at every $k$ | Removes energy but does **not** smooth. |

**A Laplacian alone is not diffusion.** Whether it diffuses depends on what it is paired with:

- `helmholtz_staircase`, $\partial_t^2 U = \Delta U$: second order in *time*, so modes oscillate ($e^{\pm ikt}$) and never
  decay. This is a wave equation.
- Incompressible pressure, $\Delta p = -\nabla\cdot(u\cdot\nabla u)$: there is no time derivative, so it is an instantaneous
  constraint, not diffusion.
- `active_matter`'s Stokes equation, $-\Delta u + \nabla P = \nabla\cdot\Sigma$: also no time derivative; the velocity is set
  instantly by the stress.

**Diffusive datasets** (explicit $\partial_t = \nu\Delta$-type term, [Well] App. C):

| Dataset | Diffusive terms | Strength set by |
|---|---|---|
| `shear_flow` | $\nu\Delta u$, $D\Delta s$ | $\nu = 1/Re$, $D = 1/(Re\,Sc)$; $Re$ from $10^4$ to $5\times10^5$ |
| `rayleigh_benard` | $\nu\Delta u$, $\kappa\Delta b$ | $\kappa = (Ra\,Pr)^{-1/2}$, $\nu = (Ra/Pr)^{-1/2}$; $Ra$ from $10^6$ to $10^{10}$ |
| `viscoelastic_instability` | $\beta\Delta u$ (solvent), $\varepsilon\Delta C$ (polymer stress) | $Re = 1000$; $\varepsilon = 2\times10^{-6}$ is tiny, mainly for numerical regularity |
| `gray_scott_reaction_diffusion` | $\delta_A\Delta A$, $\delta_B\Delta B$ | $\delta_A = 2\times10^{-5}$, $\delta_B = 10^{-5}$; the *difference* between the two drives pattern formation |
| `active_matter` | $d_T$ and $d_R$ (translational and rotational diffusion) in the kinetic equation | dimensionless constants; the Stokes flow is viscous as well |
| `rayleigh_taylor_instability` (3D) | viscous stress $\nabla\cdot\tau$; molecular diffusivity $\kappa$ in the $\nabla\cdot u$ constraint | $\nu$ rescaled over time to keep the Kolmogorov scale at the grid spacing ([Well] Eq. 38) |
| `planetswe` | $-\nu\nabla^4 u$, $-\nu\nabla^4 h$ (hyperviscosity) | $\nu = 1.76\times10^{-10}$, tuned to act only near wavenumber 224; the authors call it "largely non-physical" (p. 37) |

**Non-diffusive datasets** (only first-order spatial derivatives):

| Dataset | Why |
|---|---|
| `acoustic_scattering` | only $\nabla\cdot u$ and $\nabla p$ appear: pure wave transport |
| `helmholtz_staircase` | wave equation, second order in time (see above) |
| `euler_multi_quadrants` | flux divergence only; "inviscid" |
| `turbulent_radiative_layer` | Euler plus the $-E/t_{\rm cool}$ sink: energy is lost without smoothing |
| `MHD` | *ideal* MHD: no viscosity, no resistivity ($\eta\nabla^2 B$) |
| `post_neutron_star_merger` | ideal general-relativistic MHD |
| `supernova_explosion`, `turbulence_gravity_cooling` | Euler plus gravity plus heating/cooling, but see artificial viscosity below |

**Where the setup matters, beyond the equations:**

1. **The coefficient's value, not just its presence.** At $Re = 5\times10^5$ (`shear_flow`), viscosity
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
   explicit $a_{\rm visc}$ term in the momentum equation ([Well] Eq. 42). It is written into the equations
   but is a numerical device that switches on mainly at shocks. These two are "physically inviscid,
   numerically dissipative", a borderline case like `planetswe`'s hyperviscosity.
4. **Sinks and relaxation are not diffusion.** TRL's $-E/t_{\rm cool}$ and the polymer relaxation $T(C)$ in
   `viscoelastic_instability` damp at the same rate at every scale. That is why TRL keeps sharp
   hot/cold interfaces while losing energy.
5. **Downsampling is not diffusion.** `MHD_64` was low-pass filtered from $256^3$. That removes small
   scales from the *data*, but the *dynamics* remain ideal.

### 5.4.2 What makes a dataset sharp or smooth (axis A4)

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

1. **Nonlinear steepening in compressible flow (shocks).** In Euler, disturbances travel at $u \pm c$,
   and both depend on the state. Compressed regions become denser and hotter, so $c$ rises and the
   back of a wave outruns its front: the wave steepens into a shock in finite time, even from smooth
   initial data. This needs nonlinear fluxes ($\rho u^2$, $u(\rho E+p)$) *and* no diffusion to stop it.
   - `euler_multi_quadrants`: also sharp from $t = 0$, because the initial data is piecewise constant
     (a Riemann problem).
   - `supernova_explosion`: point energy injection drives a blast wave; [Well] describes "dense,
     sharp shells" (p. 6).
   - `MHD`: the supersonic runs (Mach up to 7) form shocks.
2. **Discontinuous coefficients or geometry.** A *linear* wave equation cannot create a
   discontinuity, but it faithfully carries and reflects one that the setup provides.
   - `acoustic_scattering`: $\rho(x,y)$ jumps at interfaces (maze walls $\rho = 10^6$ next to paths at $\rho = 3$).
     The pressure field develops reflections, kinks and gradient jumps at every interface. The
     equations are linear; the sharpness lives entirely in the coefficients.
   - `helmholtz_staircase`: the staircase corners produce gradient singularities ([Well] p. 36);
     away from the boundary the field is smooth. The stored `mask` field is a sharp binary image,
     but it never changes in time.
3. **Temperature-dependent cooling (thermal instability).** In `turbulent_radiative_layer`, cooling
   is fastest at intermediate temperatures ([Well] p. 44), even though the printed equation shows
   only $-E/t_{\rm cool}$. Intermediate-temperature gas therefore cools quickly into the cold phase, the
   intermediate phase drains away, and the hot/cold interface stays thin, with no physical
   diffusion to widen it.
4. **Self-gravity (collapse).** In `turbulence_gravity_cooling`, gravity pulls dense regions
   together and makes them denser, the opposite of diffusion. Together with cooling this produces
   thin, dense filaments. The density range is wide enough that the configs log-transform it.
5. **Stretching with weak diffusion.** In `viscoelastic_instability`, $C$ has its own evolution
   equation, $\partial_t C + (u\cdot\nabla)C + T(C) = C\cdot\nabla u + (\nabla u)^T\cdot C + \varepsilon\Delta C$, and it is not covered by the
   incompressibility constraint:
   - **Amplification.** The stretching term $C\cdot\nabla u + (\nabla u)^T\cdot C$ makes $C$ grow exponentially wherever
     the flow is extensional. Polymers undergo the coil–stretch transition once relaxation cannot
     keep up (roughly, relaxation time $\times$ strain rate $> 1/2$ for dumbbell models).
   - **Slow relaxation.** At Weissenberg number 50 a polymer needs ~50 flow time-units to relax, so
     almost any appreciable strain stretches it faster than it recovers. Growth is finally capped
     by the FENE-P extensibility limit ($L_{\max} = 70$), after $C$ has grown by a factor of thousands.
   - **Almost no smoothing.** $\varepsilon = 2\times10^{-6}$ is about $1000\times$ weaker than the viscosity acting on
     velocity ($\beta/Re \approx 10^{-3}$). With $\varepsilon$ removed, the $C$ equation is pure transport plus stretching and
     can develop arbitrarily steep gradients. This is the well-known "high Weissenberg number
     problem" of viscoelastic simulation, and it is the reason $\varepsilon$ is included at all.

   The velocity stays smooth even though $C$ pushes back on it through $(1-\beta)\nabla\cdot T(C)$. The momentum
   equation carries solvent viscosity $\beta\Delta u$ ($\beta = 0.9$), and incompressibility makes velocity and
   pressure respond to the whole stress field globally rather than point by point, so the velocity
   comes out roughly one derivative smoother than the stress. This is why §5.2 marks VE as
   borderline: smooth velocity, sharp $C$. The measurement below confirms it.

**Mechanisms that keep a dataset smooth:**

1. **Diffusion, $\partial_t f = \nu\Delta f$.** Modes decay as $e^{-\nu k^2 t}$, so the steepest gradients are damped
   fastest, and gradients cannot get narrower than a scale set by the coefficient (`gray_scott`,
   `active_matter`, `rayleigh_benard`, `shear_flow`). Gray–Scott forms reaction fronts, but they
   have a finite width, roughly $\sqrt{\delta/\text{reaction rate}}$.
2. **Incompressibility, $\nabla\cdot u = 0$, prevents shocks in the flow variables.** With no density change
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
   is gentle and $-\nu\nabla^4 u$ removes grid-scale structure. It is smooth because of its *setup*, even
   though its equation family allows sharpness.

**An empirical signal: the solver.** The solvers split almost perfectly along this line. Sharp
datasets use shock-capturing finite-volume or SPH codes (Clawpack, Athena++, ENO, SPH), which are
built to tolerate discontinuities. Smooth datasets use spectral codes (Dedalus, Chebfun), which need
smooth fields or become unstable. `viscoelastic_instability` uses Dedalus but needs its small $\varepsilon$
diffusion to stay stable, which is consistent with it being borderline. In choosing the solver, the
domain expert effectively recorded whether they expected discontinuities.

**Measured check on the stored data.** For each field, over trajectory 0 of each file, every 4th
snapshot, I computed two statistics of the squared gradient $\lvert\nabla f\rvert^2$:

- **Flatness**, $\langle\lvert\nabla f\rvert^4\rangle / \langle\lvert\nabla f\rvert^2\rangle^2$, which is $\approx 2$ for Gaussian-like gradients and large when a few
  pixels dominate.
- **Top-1% share**: the fraction of total gradient energy contained in the 1% of pixels with the
  steepest gradient.

| Dataset (regime) | Field | Flatness | Top-1% share |
|---|---|---|---|
| `viscoelastic_instability` (CAR, chaotic arrowhead) | velocity_x | 2.9 | 5.9% |
| | $C_{yy}$ | 950 | 75.5% |
| | $C_{yy}$, interior only (64-cell margin) | 444 | 69.6% |
| `viscoelastic_instability` (EIT, elasto-inertial turbulence) | velocity_x | 2.8 | 5.8% |
| | $C_{xy}$ | 40 | 42.6% |
| | $C_{yy}$ | 1804 | 78.4% |
| | $C_{yy}$, interior only (64-cell margin) | 788 | 73.7% |
| `shear_flow` (reference, smooth) | velocity_x | 12 | 25.4% |
| | tracer | 16 | 31.8% |
| `rayleigh_benard` (reference, smooth) | buoyancy | 17 | 32.2% |
| `acoustic_scattering_maze` (reference, sharp) | pressure | 210 | 46.5% |
| `turbulent_radiative_layer_2D` (reference, sharp) | density | 92 | 84.9% |

Unless marked, 4 cells are trimmed at every edge. In the chaotic regimes, `viscoelastic_instability`'s
velocity is the *smoothest* field measured, smoother even than `shear_flow`'s. Its $C_{yy}$ is as
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
| `acoustic_scattering` | discontinuous $\rho(x,y)$ | none | **Sharp** at interfaces |
| `turbulent_radiative_layer` | temperature-dependent cooling | none (numerical only) | **Sharp** interfaces |
| `turbulence_gravity_cooling` | self-gravity + cooling | artificial viscosity | **Sharp** filaments |
| `viscoelastic_instability` | polymer stretching | tiny $\varepsilon$ diffusion | **Borderline** (sharp in $C$) |
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

### 5.4.3 What makes a dataset chaotic or regular (axis A5)

**Terminology first.** Every dataset in The Well is *deterministic*: the equations contain no noise,
and the same initial state always gives the same trajectory. Chaos is a property of deterministic
systems. It means **sensitive dependence on initial conditions**: two almost identical states
diverge exponentially, so beyond some horizon the trajectory cannot be predicted pointwise, even
with perfect equations. The right contrast is therefore **chaotic vs regular** (steady, periodic or
decaying), not chaotic vs deterministic.

**What in the equations makes a system chaotic.** Four ingredients are needed together:

| Ingredient | Why it is needed | Where it appears |
|---|---|---|
| **1. Nonlinearity** | In a linear system, a small difference $\delta$ obeys the *same* linear equation, so it can only propagate or grow at a fixed modal rate; it cannot be stretched and folded. **Linear systems cannot be chaotic.** | $u\cdot\nabla u$ (advection), Euler fluxes $\rho u^2$, stretching $C\cdot\nabla u$, reaction $AB^2$, orientation coupling in the kinetic equation |
| **2. An instability** that amplifies small differences | Something must make nearby states separate | shear (Kelvin–Helmholtz), buoyancy (convection, Rayleigh–Taylor), activity, polymer elasticity, thermal instability, gravity |
| **3. A control parameter above threshold** | Diffusion damps perturbations. Chaos appears only when driving or nonlinearity beats damping by a large enough ratio. | $Re$, $Ra$, $Wi$, $\alpha$, Mach, Atwood |
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
   history. These are $\gamma$ in Euler, $Re$/$Ra$ in the incompressible sets, and the medium density in
   acoustics (zeroed in the configs). This is the non-Markovian issue [WALRUS] App. D discusses.

**Dataset by dataset.** The proxy is the best model's T21–60 VRMSE from [WALRUS] Table 1; $\gtrsim 0.6$
means every model had decorrelated by then (§5.2).

| Dataset | Nonlinearity | Instability / driver | Control parameter (range) | Sustained? | Evidence | Classification |
|---|---|---|---|---|---|---|
| `acoustic_scattering` (×3) | none (linear) | none | — | energy leaves through open boundaries | proxy 0.04–0.13 | **Regular** |
| `helmholtz_staircase` | none (linear) | none | — | exactly time-periodic: $U = u\,e^{-i\omega t}$ | proxy 0.003 | **Regular** (the most predictable dataset) |
| `gray_scott_reaction_diffusion` | reaction $AB^2$ | diffusive (Turing) instability | $f$, $k$: six chosen pairs | patterns settle | "eventually reach a steady-state" ([Well] App. D, p. 45); proxy 0.13 | **Regular.** Gray–Scott has chaotic regimes, but the six chosen $(f, k)$ pairs give stationary or regularly moving patterns |
| `euler_multi_quadrants`\* | flux terms | shock interactions; roll-up of contact discontinuities | $\gamma = 1.13$–$1.76$ | ideal, unforced | proxy 0.28 | **Intermediate** |
| `planetswe` | $u\cdot\nabla u$, $\nabla\cdot(hu)$ | shear (barotropic) instability | fixed Earth-like setup | yes: seasonal forcing | proxy 0.02 at 60 h; The Well frames long-term prediction as matching the *distribution* ([Well] App. D) | **Slowly chaotic**: predictable over the evaluated horizon, chaotic over long ones |
| `shear_flow` | $u\cdot\nabla u$, $u\cdot\nabla s$ | Kelvin–Helmholtz | $Re = 10^4$–$5\times10^5$ | no: unforced, decays | "highly sensitive to initial conditions" ([Well] C.12, p. 42); proxy 0.08 | **Chaotic in principle, slow**: predictable over 60 steps |
| `rayleigh_benard` | $u\cdot\nabla u$, $u\cdot\nabla b$ | convective (buoyancy) instability | $Ra = 10^6$–$10^{10}$, far above onset ($\sim 1.7\times10^3$ for rigid walls) | yes: fixed temperature difference at the walls | cell positions "highly sensitive to initial conditions" ([Well] C.10, App. D); proxy 0.64 | **Chaotic** |
| `turbulent_radiative_layer_2D` | flux terms + temperature-dependent cooling | Kelvin–Helmholtz shear + thermal instability | $t_{\rm cool} = 0.03$–$3.16$ | shear-driven mixing | proxy 0.84 | **Chaotic** |
| `active_matter` | orientation coupling in the kinetic equation (*not* in Stokes, which is linear) | activity-driven instability of the aligned state | $\alpha = -1$ to $-5$, $\zeta = 1$–$17$ | yes: activity injects energy everywhere | proxy 1.25, the highest in 2D | **Chaotic**: "active turbulence" with no inertia at all |
| `viscoelastic_instability` | stretching $C\cdot\nabla u$ + $u\cdot\nabla u$ | elasto-inertial instability | $Re = 1000$, $Wi = 50$ (fixed) | yes: imposed bulk flow | multistable; see below | **Mixed, by file** |
| `MHD_64` (3D) | $u\cdot\nabla u$, induction | driven turbulence | Mach 0.5–7 | yes: continuous forcing | proxy 1.23 | **Chaotic + stochastically forced** |
| `rayleigh_taylor_instability` (3D) | $u\cdot\nabla u$ | Rayleigh–Taylor (buoyancy) | Atwood 1/16–3/4 | gravity keeps releasing potential energy | proxy 0.48 | **Chaotic onset** (turbulent mixing develops) |
| `turbulence_gravity_cooling` (3D) | fluxes, gravity | turbulence + gravitational collapse | $T_0$, $\rho_0$, metallicity | turbulent initial conditions, collapse | proxy 0.62 | **Chaotic** |
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
- **`active_matter` is chaotic without the usual nonlinearity.** Stokes flow has no $u\cdot\nabla u$ term,
  and the Reynolds number is effectively zero. The nonlinearity is in the particle orientation
  dynamics (the $\dot{p}$ flux depends on $\nabla u$, which depends on the orientation stress), and the energy
  comes from the activity $\alpha$ at every point. This is a different route to chaos from every fluid
  dataset.
- **`rayleigh_benard` is kept chaotic by its boundaries.** The equations have no forcing term. The
  fixed temperature difference between the walls, $b(0) = L_z$ and $b(L_z) = 0$, supplies energy
  continuously, so the chaos persists instead of decaying. `shear_flow` has no such input: its
  shear layers roll up and then decay.
- **Chaos is relative to the horizon.** `planetswe` and `shear_flow` are most likely chaotic, but
  their error-doubling time is long compared with the 60 output steps evaluated (1 step = 1 model
  hour for `planetswe`). Whether a dataset "is chaotic" in a benchmark therefore depends on $\Delta t$ and
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
every $\Delta$ and the descriptor regression can be computed from, without extra pretraining.

**Protocol common to all runs:**

- 2D only; same model size, optimiser and schedule.
- Identical pretraining sample budget per run (uniform over its two datasets).
- Same augmentation policy in every run (WALRUS's, or none).
- Finetuning at 1k / 10k / 100k samples with identical hyperparameters; pretraining effects are
  largest when finetuning data is scarce ([WALRUS] Fig. 8).
- Field embeddings reinitialised at finetuning (§5.3), plus one ablation that keeps them.
- Metrics: VRMSE one-step, T1–10, T11–30; spectral error for chaotic targets.
- At least 3 seeds for Tier-1 runs; $\Delta$ reported with confidence intervals.

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
