# METHOD: what is new at n=8 (almost nothing, which is the point)

Delta-doc. The full method (including the n=7 impossibility verdict and its
refutation, the NCR analysis, the discrete-TVLQR fix, densification, and the
gate evidence-construction history across three adversarial review rounds)
lives in the n=7 release (eight-state/septuple-cartpole, docs/METHOD.md).
This document records only the n=8 specifics.

## 1. Seed: Glück MS continuation (the n5→n6→n7 recipe, one more rung)

`gluck_n8_from_n7.py` is the exact analog of the n6→n7 continuation: sample the
saved n=7 MS trajectory at the n=8 segment boundaries, copy link 7's
angle/rate columns onto the new link 8, extend T 8s→9s, run the same
MultiShoot solver. Result (`nom_n8_gluck.npz`): terminal 0.063°, peak force
39.9 N, rates ~3e-4, a genuine open-loop n=8 swing-up plan (whose open-loop
replay diverges, as every raw MS seed's does; that is what the polish and
feedback are for).

## 2. Polish: ONE-SHOT 4 ms collocation (`_n8_oneshot.py`)

2250-node, 4 ms RK4 collocation NLP (IPOPT/MUMPS, single continuous solve,
max_iter 4000), warm-started from the seed. Converged at iteration 804
(11 h): transcription defect **2.1e-12**, peak feedforward **23.2 N** of
150, terminal **0.0115°**.

The one n=8-specific lesson, negative and useful: **chunked warm-restart
IPOPT (40-iteration chunks with checkpointing) diverged on every attempt at
BOTH 4 ms and 2 ms grids** (constraint violation grew 1e2 → 5e2), while the
identical problem solved cleanly as a single continuous descent. The
barrier-restart pattern (not the transcription grid, not the physics) was
the failure mode. The same pattern held at n=7. Treat chunked IPOPT
restarts as broken for these polishes.

## 3. Densify + control: unchanged from n=7

Densification onto the exact 1 ms sim grid: max node-boundary seam
**4.233e-3**, ~50× n=7's 8.34e-5, because the n=8 trajectory is stiffer
(upright spectrum reaches λ≈33 vs 29.3, and the swing passes through
higher-acceleration configurations). The seams are absorbed by the per-tick
feedback: peak closed-loop demand over the whole maneuver is 23.2 N, equal
to the feedforward peak. Committed as a test with the measured bound.

Exact-ZOH discrete-time TVLQR along the dense nominal: **monodromy
rho = 0.156** (n=7: 0.197; n=8's loop is, if anything, more contractive).

## 4. Unperturbed closed-loop result

Real saturated 1 ms sim from exact hanging: handoff 0.0122°, swing peak
23.2 N, static-LQR hold PASS (peak 77.7 N), locked predicate v1. This is
the same milestone the n=7 release calls V6.

## 5. Perturbed composite gate + the fast stack

The perturbed-IC gate (sigma=0.02, 24 ICs, two seeds) runs the n=7 composite
policy (replan-at-t0 → discrete TVLQR → steering-NLP catch → static LQR hold,
with the stage-B pre-roll fallback) under the same locked predicate and
iteration-only budgets. The **physics, simulator, perturbation model, seeds,
and success predicate are byte-for-byte the n=7 gate's.** Only the solver
internals changed, for speed and one robustness fix:

- **Stage-A dual warm start.** Each per-IC replan re-enters IPOPT primal AND
  dual warm-started from the unperturbed nominal's KKT multipliers (harvested
  once, shipped as `results/warmpack_n8.npz`). This collapses the replan
  iteration count and, more importantly, its *variance*: on the reference
  laptop the worst-case IC dropped from ~12 h to ~30 min of solve, and the
  per-IC iteration spread shrank ~3 to 5×. The optimum is unchanged (same
  objective to ~13 digits); only the path to it is shorter.
- **Steer primal warm start (the robustness fix).** The catch NLP has a tight
  2e-5 terminal ball and is ill-conditioned; a cold linear-interpolation start
  can wander to the 1500-iteration budget and *fail to converge* even though
  the handoff is excellent (~2e-4 rad). Warm-starting the catch from the
  nominal steer plan converges it in ~13 to 38 iterations. This is
  **verdict-changing**: with a cold steer, several handoff-perfect ICs fail
  only on catch non-convergence (6 on seed 777, 2 on seed 12345); warm-started,
  they pass. It lifts seed 777 to 24/24.
- **Pure-speed, bit-identical pieces.** Per-step flat SX RK4 (`F_step`) with a
  thread-parallel defect map, mapaccum densification, and a
  batched-linearization discrete TVLQR, all verified bit-identical to the
  serial reference (`_dtvlqr.py`).

**What did NOT work, recorded so it isn't re-tried:** an 8 ms replan grid
(half the nodes) solves fast but produces plans that diverge 2.7 to 3.1 rad
when densified and tracked in the real 1 ms sim (8 links need fine time
resolution; 4 ms is the floor). Whole-NLP `expand`/JIT (memory/compile blowup)
and L-BFGS (non-convergent) were also dead ends. The binding limit on this
hardware is **memory bandwidth**, not cores: the gate's wall time is set by
the ~2M-nonzero Jacobian / ~1M-nonzero Hessian streamed per IPOPT iteration.

**Reproducibility is conditioned on the shipped warmpack.** The dual warm
start makes the *tightest* ICs depend on the exact warm-start iterate, so the
warmpack is treated as a fixed, shipped input (`results/warmpack_n8.npz`), not
regenerated at reproduce time (`ensure_warmpack()` reuses it if present). With
the shipped code + shipped warmpack, both seeds reproduce 24/24. One knife-edge
case is disclosed: seed 12345 tag 4 passes as banked but failed
`A_track_diverged` under a *different* warm start in development: its replan
lands a trackable plan with the shipped warmpack and an untrackable one
otherwise. We report the banked 24/24 and the edge rather than either
overclaim robustness or hide the sensitivity. Seed 777 has no comparable case.

**Fixed-nominal leg (the honest like-for-like).** Running the n5/n6-equivalent
controller (fixed dense nominal + discrete TVLQR, no replan,
`cl_validate_n8_fixed.py`) gives **8/24 (seed 12345)** and **16/24 (seed
777)**, vs 18/24 at n=7. n=8's catch is materially tighter; fixed feedback
alone does not reach the headline, the per-IC replanning does.

**Hold-predicate correction (vs the sibling releases).** The gate's continuous
5 s hold is measured as elapsed time `(run-1)*dt`, identical to
`rollout.static_hold_rollout`, so a pass requires a genuine 5.0 s (5001 in-set
1 ms samples). The n=5/6/7 release gate scripts accept at `run >= int(5/dt)`
(5000 samples = 4.999 s), a 1 ms-lenient implementation of the same predicate v1;
this n=8 release corrects it. The correction is not verdict-changing at n=8
(passing ICs hold upright indefinitely under the contractive discrete TVLQR,
rho=0.156), but the banked counts here are under the *exact* 5.0 s predicate.
The same one-line fix should be back-ported to the n=5/6/7 release gates.

Results are banked with full provenance (commit_sha + git_dirty + nominal
sha256) in `results/clvalidate_n8_{composite,fixed}_seed{777,12345}.json`.

## 6. Cross-n statement, updated

Four consecutive rungs (n=5, 6, 7, 8) now support it: through n=8, the
binding constraints are transcription fidelity, gain discretization, basin
realization, and solver mode (controller numerics, all), not single-input
actuator authority. Peak feedforward force has been FLAT across the ladder
(21.6, 23.3, 23.2 N at n=6/7/8) while the spectrum stiffens and the
unstable-mode count grows. The 150 N actuator is nowhere near its limit.
