# METHOD: what is new at n=10 (almost nothing, which is the point)

Delta-doc. The full method lives in the n=7/n=8 METHODs, and the pre-roll gate
architecture and NCR analysis in the n=9 release. This document records the
n=10 specifics — which are deliberately thin: **the n=9 pre-roll gate ported to
n=10 with zero re-tuning.** The only real n=10 lesson is a solver-exit-rule one.

## 1. Seed: Glück MS continuation (one more rung)

`gluck_n10_from_n9.py` samples the saved n=9 MS trajectory at the n=10 segment
boundaries, clones link 9's angle/rate onto the new link 10, and re-runs the
MultiShoot solver (`nom_n10_gluck.npz`). ~2.9 h. Same recipe as every rung.

## 2. Polish: 4 ms `w_v` collocation, and the tight-exit lesson (`_n10_oneshot_wv.py`)

2500-node, 4 ms RK4 collocation warm-started from the gluck seed, with the
running velocity penalty carried over from n=9 (which used `w_v = 6e-4`), here
tuned to `w_v = 1e-3`. Converged:
RK4-4ms transcription defect **1.361e-07**, peak feedforward **35.97 N**,
terminal **0.0115°**.

**The one n=10-specific lesson (a solver-exit rule, negative and useful).** An
`acceptable_tol=1e-4` shortcut *looked* converged but left transcription defect
**5.6e-6**, and tracking that nominal blew through the stiff t≈2.23 s window:
the defect kicks were amplified ~5000× and drove the controller to saturation.
The fix was not a new controller — it was a **tighter exit rule** on the SAME
solve: require `defect < 1e-6` and `dual < 5e-4`. The re-solved nominal (the
`t`/tight suffix, `nom_n10_*_wv1en3t.npz`) is the shipped one. Lesson: at n=10
the defect must genuinely reach ~1e-7; acceptable-tol is not acceptable.

## 3. Densify + control

Densification onto the exact 1 ms grid: max node-boundary seam **8.195e-6**.
Exact-ZOH discrete-time TVLQR: **monodromy rho = 0.1042** (contractive).

## 4. Unperturbed closed-loop result

Real saturated 1 ms sim from exact hanging: handoff **0.0115°**, swing peak
36.0 N, static-LQR hold → **PASS**, holding un-saturated. The unperturbed catch
peaks at **98.6 N** and never hits the 150 N clip — **cleaner than n=9's
saturating catch**. n=10's swing-up is gentler than n=9's on every axis
measured (peak force 36 vs 41 N, peak link rate 11.1 vs 12.5 rad/s).
`reproduce_n10.py` reproduces it; the hold is watched over a 10 s window as at
n=9.

## 5. Pre-roll gate — identical to n=9, zero re-tuning

`scripts/gate_preroll.py` is the n-generic form of the n=9 gate (parametrized by
`NLINKS` and `NOM_PATH`): per-IC fixed **LQR-about-down** pre-roll (settle the
σ=0.02 hanging perturbation back to `x_nom[0]`) → fixed-nominal discrete TVLQR
track → static-LQR hold. **No per-IC NLP.** The n=9 and n=10 gates run the same
code with the same config; nothing was re-tuned for the extra link.

Result: **24/24 on each of seeds 12345, 777, 2024** (72/72), ~3.5 min/seed,
Wilson-95 lower bound 0.862. Same plant / sim / predicate / perturbation model
(N(0,0.02) on all states at hanging) as the n=5..9 releases. Banked with
provenance in `results/gate_n10_preroll_seed{12345,777,2024}.json`.

## 6. Cross-n statement, updated

Six rungs (n=5..10) now support it: through n=10, the binding constraints are
transcription fidelity, gain discretization, basin realization, and solver mode
(controller numerics), not single-input actuator authority. The pre-roll gate
architecture has now produced release-grade σ=0.02 results at **two consecutive
n with no per-IC NLP**, and n=10 needed *gentler* control than n=9. Peak
feedforward force is still ≈ a quarter of the 150 N actuator.
