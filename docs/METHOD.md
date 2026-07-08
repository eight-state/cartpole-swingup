# Method: how we got an empirically validated n=5 cart-pole swing-up + balance

This document covers the result, the method and why it works where
general-purpose trajectory optimization fails, and the novelty with its boundary.

---

## 1. The result: n=5, empirically validated in saturated simulation

> **Verification boundary.** Throughout this document, "validated" / "verified"
> means the committed scripts reproduce the stated success **counts** in the
> force-saturated simulator from perturbed ICs (documented
> distribution, fixed seed), judged against the committed predicate `v1`. That is
> an empirical script-verification: not a formal proof, not hardware. Every count
> is backed by a committed JSON in `results/`. The full definition lives in the
> README verification-boundary box.

A quintuple (5-link) inverted pendulum on a single underactuated cart:

- **Full swing-up**: start hanging (all 5 link angles = π exactly, cart at rest).
  The cart drives all five links **through** the unstable region to upright
  (all angles → 0), then balances. End state: all 5 links within ~0.5° of
  upright (wrapped, max |θ| = 0.50°), rates ~0.004 to 0.009, cart ~0.016 m,
  ẋ ~ −0.009. Every link inverts π → 0, including the inner ones.
- **1 ms-consistent nominal**: the saved trajectory
  (`results/nom_n5_gluck_cont.npz`, 6.0 s, 6000 nodes) is dynamically consistent
  under the simulator's ZOH step (4 RK4 substeps per 1 ms control tick, matching
  `rollout_zoh`). Independently re-checked: max defect ~1.2e-13 state units, mean
  ~2.5e-15, zero ticks above 0.05° drift. (A single one-step RK4 per tick, a
  coarser integrator than the simulator uses, gives ~1.1e-8; the 1.2e-13 figure
  is against the simulator's own substep integration.) The boundary jumps present
  in the raw inversion plan are **gone**.
- **TVLQR tracks it**: whole-trajectory time-varying LQR linearized along the
  nominal (terminal cost = the upright CARE solution `P_static`, scale 1) tracks
  the swing-up and hands off to the static upright LQR. Closed-loop monodromy
  spectral radius **ρ ≈ 0.030** (≪ 1), an empirical contractivity indicator
  along the validated nominal (not a global proof). ρ is computed once from the
  nominal linearization (force- and IC-independent), so the identical ρ quoted in
  each per-leg JSON is that single number, not a per-leg re-verification.
- **Closed-loop empirically validated in the saturated sim**
  (`rollout_zoh` with hard `np.clip(u, ±F)` and RK4 sub-stepping, a genuinely
  saturated plant rather than a linear surrogate), under perturbed ICs,
  script-verified against predicate `v1`:
  - **64/64** at σ=0.02 (~1.1°/link) across two seeds (12345: 24/24; 999: 40/40),
    60 N bound, peak force 28.4 N (`results/clvalidate_n5_F60_banked_seed12345.json`
    + `...banked_seed999.json`).
  - **24/24** fresh (seed 7777, σ=0.02) **plus 24/24** 5× stress (seed 2024,
    σ=0.10, ~5.7°/link, initial-angle offsets up to ~18° (3σ of the σ=0.10 draw)).
  - These are two distinct claims at two distinct amplitudes: **88/88 at σ=0.02**
    (64 banked + 24 fresh) and **24/24 at σ=0.10** (5× stress). The two legs have
    no single pooled success rate, so they are never merged into one number.
- **Peak cart force**: nominal 20.18 N, default-σ validation ~28 N. Both sit far
  inside the 60 N validation bound and the 150 N model spec.
- **Cart excursion**: the predicate caps `|x| ≤ 2 m` only during the final 5 s
  hold; during swing-up the cart peaks at ~3.7 m (n=5 nominal |x| = 3.69 m, range
  3.71 m), well inside the ±10 m rail. See the README and VALIDATION_REPORTS for
  the per-leg figures and the hardware-rail implication.

### The success predicate

A run counts as success only if **all n** links satisfy `|wrap(θ_i)| ≤ 5°` **and**
`|θ̇_i| ≤ 0.5` **and** `|x| ≤ 2 m` **and** `|ẋ| ≤ 0.5`, held **continuously** for
the **final 5 s**, with the cart on-track over the **whole** rollout, including the
swing-up before the hold window. The actuator force is clipped to the bound by
construction, so "force within bound" is trivially true and gates nothing; the
meaningful gates are the track limit and the continuous 5 s hold. The actuator is
allowed to ride saturation, disclosed per leg via `n_saturated_ics` and
`max_abs_force_demanded`.

### Robustness scope (per amplitude)

Large margin at σ≈1°; zero margin (saturation-limited) at σ=0.10. At σ≈1° there is
~30 to 40 N of force headroom. The 5× stress leg (σ=0.10, ≈5.7°) is robust by
riding the 60 N saturation: **11 of its 24 ICs** hit 60 N to recover (committed
`n_saturated_ics`; raw demanded force peaks at 139.9 N pre-clip), so headroom there
is zero. The actuator is allowed to ride saturation (force is clipped to the bound
by construction, so it gates nothing; the meaningful gates are the track limit and
the continuous 5 s hold), which makes the stress leg legitimate saturated-actuator
robustness, but a stronger initial-angle offset or a tighter bound would start
producing clip-limited failures. The robustness is to validated initial-angle
offsets, not unbounded.

---

## 2. The method, and why it works where general-purpose solvers fail

### 2.1 Architecture (two stages + feedback)

```
  Stage 1: Glueck/Kugi exact I/O inversion  ──►  in-basin swing-up PLAN
           (saturated Fourier-coeff BVP,           (physical accel/force,
            multiple-shooting, n-continuation)       but boundary jumps)
                              │
                              ▼  (warm start, in-basin)
  Stage 2: direct-collocation continuity polish  ──►  CONTINUOUS 1 ms nominal
           (per-node defect, grid homotopy             (defect ~1e-13,
            5ms → 2.5ms → 1ms)                           jumps removed)
                              │
                              ▼
  Feedback: whole-trajectory TVLQR (terminal Λ = P_static)  ──►  closed-loop catch
```

### 2.2 Decision-space size versus node-wise collocation, iLQG, and multi-start

The cart position is the output with **relative degree r = 2** (the input is the
cart **acceleration**, `u = ÿ` exactly). So the feedforward is recovered
*algebraically* once the **internal dynamics** are solved: the pendulum angles,
an n-DOF **unstable, non-minimum-phase** ODE driven by ÿ. We never discretize the
whole state into thousands of decision variables. We parametrize the *input* with
a handful of Fourier coefficients (≈ 2n) plus the transition time T and close the
resulting small two-point boundary-value problem. The decision space is
O(2n + #coeffs), **tens** of unknowns rather than ~thousands × n_states. That is
why it stays well-conditioned where a direct-collocation NLP's Jacobian degrades
with the unstable internal dynamics.

Method origin: Glück, Eder & Kugi, "Swing-up control of a triple pendulum on a
cart with experimental validation," *Automatica* 49(3):801-808, 2013. The
inversion-based feedforward lineage is Graichen-Hagenmeyer-Zeitz (*Automatica*
2005) and Graichen-Treuer-Zeitz (double 2007, triple side-stepping CDC-ECC 2005).
**Those authors stopped at the triple (n=3).** The n≥4 extension and the
collocation-polish synthesis below are this project's contribution.

### 2.3 Failure modes of the cold solves

From the project's own negative results (these are real, measured failures, not
hypotheticals):

- **Cold direct collocation / iLQG / multi-start fail.** On the n≥4 unstable
  non-minimum-phase plant, naive cold solves give singular/ill-conditioned
  Jacobians, absurd accelerations (the original n=3 attempt railed to
  ~1115 m/s² with `solve_bvp` status=2 "singular Jacobian", force → ∞), and
  wild non-upright local optima. The repo's own committed dense n=4 trajopt
  nominal (`nom_n4_4msF150`, at 150 N) catches **0/24** perturbed ICs.
- **Saturation alone makes it worse.** A global `tanh` accel saturation is an
  *attractor for the degenerate bang-bang solution*: large coefficients trivially
  match the boundary conditions, and where the `tanh` rails its Jacobian wrt the
  coefficients collapses → singular. Saturation is a cure only *inside* the
  stable-inversion structure, not bolted onto a global solve.
- **Continuation warm-start alone is insufficient.** Copying link n−1 onto a
  coarse mesh is far from the n manifold; the first global Newton step blows up
  the unstable internal dynamics → NaN Jacobian.
- **Single-shooting / global-Fourier closure is singular.** Integrating the
  inversion plan's coefficients continuously from φ(0)=π over the full 5 to 6 s
  gives terminal ~164° (it falls over): the multiple-shooting plan **relies on
  its segment-boundary snaps** to suppress the unstable modes. The same
  coefficients without snaps diverge, and the forward sensitivity of the unstable
  plant over the full horizon saturates (~5e9 first-order optimality), so the
  optimizer cannot move.

### 2.4 The two fixes: stable multiple-shooting inversion and collocation polish

**Fix 1, stable multiple-shooting inversion + n-continuation (Stage 1).** Split
[0,T] into M segments. The unknowns are the segment-boundary angle states plus
the saturated Fourier accel coefficients. RK4-integrate each **short** segment
(the short horizon caps the unstable modes' exponential growth, keeping it
well-conditioned), with residuals = continuity gaps + the 16 (here 2(n+1)·2)
boundary conditions, solved by `least_squares` with a block sparsity pattern.
Warm-start each n from the converged n−1. This **fixes the singular-Jacobian /
absurd-accel failure completely**: the ladder reaches upright at physical
accel/force (n=2: 12.9 N; n=3: 24.3 N; n=4: ~25 N; n=5: ~28 N peak), versus ~300 g
and status=2 for the cold global solve. The one remaining defect is that the saved
plan carries **boundary-jump artifacts** (~1° one-step RK4 steps at the segment
seams) that an unstable n-link amplifies, so TVLQR about the raw plan diverges.
Stage 2 removes exactly that.

**Fix 2, direct-collocation continuity polish (Stage 2, `zoh_consistent=True`).**
Keep **every**
node as a decision variable (well-conditioned, like multiple shooting) **and**
enforce the dynamics defect at **every** node, so the result is continuous and
1 ms-consistent by construction with no seams to jump at. The Stage-1 trajectory
is the **in-basin warm start** that cold collocation lacked, so it now converges.
Grid homotopy 5 ms → 2.5 ms → 1 ms (IPOPT, collocation defect driven to ~1e-13).
The polish **redistributes** the trajectory onto a uniformly
dynamically-consistent path *without leaving the basin* (terminal ~0.5°, force
still ~20 N). For n=5: stage-0 5 ms term 0.234°, stage-1 2.5 ms term 0.331°,
stage-2 1 ms term 0.500°, peak force 20.18 N, one-step RK4 boundary jumps 21 → 0.

**Feedback, whole-trajectory TVLQR.** Linearize the unsaturated continuous
dynamics along the nominal (saturation is applied during rollout, not in the
EOM). Solve the discrete time-varying Riccati equation backward with
terminal cost = the infinite-horizon upright CARE solution `P_static`, so the
gains converge to the steady-state upright LQR at t=T. With no step injections the
closed loop **contracts** (ρ≈0.030) and the n=5 basin is wide enough for the
64/64 catches.

### 2.5 The ladder (n=2 → n=5)

The 2→3→4 stages reproduced the published-style n=4 numbers exactly, confirming
the pipeline is unchanged across n. The n=4 case (ρ=0.036, 64/64 at 60 N, peak
~24 N) was observed in development; no n=4 JSON is committed in this repo, so it is
reported as a development observation, not a committed result. n=5 is the headline.
The ladder
stops at n=5: a raw n=6 inversion plan exists upstream but is not part of this
repo's committed validated artifact (and travels even further on the rail than
n=5), so it is excluded here.

---

## 3. Novelty and its boundary

The claim is precise and positive:

> **First public, code-reproducible n=5 cart-pole swing-up-and-balance
> artifact**. It ships a saved nominal trajectory, validates the closed loop in a
> saturated simulator against a strict committed predicate, and reproduces in one
> command.

Concretely: a 1 ms-consistent full n=5 swing-up that TVLQR catches 88/88 at σ=0.02
(64 banked across two seeds + 24 fresh) and 24/24 at σ=0.10 under 5× stress in the
saturated 60 N sim against predicate `v1` (two distinct amplitude claims, never
pooled), ρ≈0.030, every count backed by a committed JSON
(`results/clvalidate_n5_F60_*.json` + `combined_validation_report.json`). The
qualifiers **public / code-reproducible / artifact** are load-bearing and travel
with the claim: no prior public artifact known to the author combines cart + n=5 +
swing-up + released reproducible code.

The boundary that makes the claim narrow: an empirical
validation, not a formal proof; and prior n=5 work exists, none of it a
reproducible cart swing-up artifact. The full accounting, and how each prior item
differs (the code-less `WwR92kx6tcA` video, Kotelovych 2024 Isaac-Sim
stabilization, Lam & Davison base-torque balance, the Glück et al. hardware
triple), is in **[PRIOR_ART.md](PRIOR_ART.md)**. Two further items it does not
displace: Oh et al. (*IJCAS* 2025, "QIP") is RL n=4 hardware on a cart (single-motor
linear rail; the paper is paywalled, so the actuator is inferred from the
abstract, from "QIP" nomenclature, and from the authors' cart-QIP lineage,
not quoted verbatim); and the yacine/@yacineMTB n=6 social-media
RL clips are in-progress sim with no reproducible artifact.

---

## 4. Sources

- Glück, Eder, Kugi. "Swing-up control of a triple pendulum on a cart with
  experimental validation." *Automatica* 49(3):801-808, 2013.
  (https://www.acin.tuwien.ac.at/fileadmin/cds/pre_post_print/glueck2013.pdf)
- Graichen, Treuer, Zeitz. "Swing-up of the double pendulum on a cart by
  feedforward and feedback control with experimental validation."
  *Automatica* 43(1):63-71, 2007.
- Graichen, Treuer, Zeitz. "Fast side-stepping of the triple inverted pendulum."
  *44th IEEE CDC-ECC*, 2005.
- Graichen, Hagenmeyer, Zeitz. "A new approach to inversion-based feedforward
  control design for nonlinear systems." *Automatica* 41(12):2033-2041, 2005.
- Kaheman, Fasel, Bramburger, Strom, Kutz, Brunton et al. "The experimental
  multi-arm pendulum on a cart." *HardwareX* 15:e00465, 2023. (CAD + datasets +
  parameter-ID; no swing-up controller code.)
- Lam, Davison. "The real stabilizability radius of the multi-link inverted
  pendulum." *ACC*, 2006. (Bottom-pivot torque chain, **not a cart**;
  stabilization only, sim, v=1…7.)
- Oh, Lee, Ryoo, Koh, Han, Lee. "Reinforcement learning to achieve real-time
  control of a quadruple inverted pendulum (QIP)." *IJCAS* 23(9):2797-2806, 2025.
  (n=4, RL hardware; cart-mounted single-motor QIP, strongly inferred; full
  text paywalled.)
- "The quintuple inverted pendulum." YouTube video id `WwR92kx6tcA`. (Matlab/
  Simulink per the description; claims swing-up + stabilization; **no public
  code**. The claim is taken from the title and description only.)
- Kotelovych et al. "Stabilization of a Quintuple Inverted Pendulum System in
  Isaac Sim." Springer, 2024. (n=5 **stabilization** in Isaac Sim: near-upright
  LQR balance, **not** swing-up.)
