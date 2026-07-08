# Method: the n=6 deltas

The full method is documented in the n=5 release at
https://github.com/eight-state/quintuple-cartpole. n=6 uses the same
three stages: Glück/Kugi exact input-output inversion to get an in-basin
swing-up plan, a direct-collocation continuity polish to remove the boundary
jumps, and whole-trajectory TVLQR for the closed loop. The n=5 METHOD covers why
general-purpose trajectory optimization fails on the unstable non-minimum-phase
plant, the relative-degree-2 inversion structure, the stable multiple-shooting
BVP, and the failure modes the architecture avoids. This document covers only
what is new at n=6.

> **What "validated" means here** (verification boundary): the committed scripts
> reproduce the stated success counts in the force-saturated simulator from
> perturbed ICs drawn from the documented distribution at a fixed seed, judged
> against the committed predicate `v1`. It is not a formal stability/robustness
> proof and not hardware. Every count below is backed by a committed JSON in
> `results/`.

---

## 1. n-continuation from n=5 to n=6

The 6-link nominal is reached by continuation from the saved n=5 trajectory
rather than a fresh chain. The Stage-1 multiple-shooting inversion was
warm-started directly from the n=5 trajectory by sampling its angle and rate
columns at the n=6 segment-boundary times and copying link 5 onto the new link 6.
The n=6 solve (T=7 s, a_max=90, M=60 segments, nsub=12, NP=12 Fourier
coefficients) converged in about 1805 s to a reconstructed terminal max-angle of
0.104° with peak cart force 35.7 N. As at every n, the raw multiple-shooting plan
carries boundary-jump artifacts at the segment seams that an unstable 6-link chain
amplifies, so TVLQR about the raw plan diverges (closed-loop replay on the raw
plan reaches ~178°). The collocation polish removes those jumps. The pipeline is
otherwise unchanged from n=5.

## 2. The cc-iLQG 1 ms refinement

This is the one new piece of machinery in the n=6 release. The collocation polish
brought the nominal down a grid homotopy to a 2.5 ms seed. Pushing that seed to
the native 1 ms grid with the n=5 IPOPT polish was slow: the 1 ms collocation has
7000 nodes at 14-state and runs about 5× slower per IPOPT iteration than n=5, and
a single 1 ms solve repeatedly exceeded the available compute window.

The 1 ms grid was instead reached by a cell-correction iLQG (cc-iLQG) pass that
refines the 2.5 ms seed node by node against a native MSVC-compiled 14-state
Jacobian. Compiling the Jacobian to native code makes each per-node linearization
fast enough that the full 2.5 ms-to-1 ms refinement finishes in about 8 s. The
output is bit-exact 1 ms-consistent: the saved trajectory satisfies the discrete
1 ms ZOH dynamics exactly, link by link, so its 1-step ZOH defect is 0.0.

The collocation homotopy that fed the refinement:

| stage | dt     | nodes | result   | colloc defect | terminal |
|-------|--------|-------|----------|---------------|----------|
| 0     | 10 ms  | 700   | solved   | 3.9e-10       | 0.162°   |
| 1     | 5 ms   | 1400  | solved   | 3.7e-13       | 0.232°   |
| 2     | 2.5 ms | 2800  | solved   | 3.0e-13       | 0.329°   |
| 3     | 1 ms   | 7000  | cc-iLQG  | 0.0 (1-step ZOH) | 0.246° |

`nom_n6_gluck_cont.npz` is the 1 ms stage: terminal 0.246°, peak feedforward
21.56 N, 7000 intervals. This is parity with the n=4 and n=5 nominals, which are
native 1 ms.

## 3. The 6-link nominal and the closed loop

A sextuple inverted pendulum on a single underactuated cart:

- **Full swing-up.** All 6 link angles start at π exactly (hanging, cart at rest).
  The cart drives all six links through the unstable region to upright, then
  balances. The polished nominal ends with all links within 0.246° of upright.
  Every link inverts π → 0, including the inner ones.
- **1 ms-consistent nominal** (`results/nom_n6_gluck_cont.npz`, 7.0 s, 7000
  intervals, 14-state), 1-step ZOH defect 0.0. The raw inversion plan's boundary
  jumps are gone.
- **TVLQR tracks it.** Whole-trajectory time-varying LQR linearized along the
  nominal, terminal cost the upright CARE solution `P_static` at scale 1, tracks
  the swing-up and hands off to the static upright LQR. Closed-loop monodromy
  spectral radius **ρ ≈ 0.0270** (≪ 1), an empirical contractivity indicator along
  the validated nominal, not a global proof. ρ is computed once from the nominal
  linearization (force- and IC-independent), so the identical ρ quoted in each
  per-leg JSON is that single number, not a per-leg re-verification.
- **Closed-loop empirically validated** in the saturated sim (`rollout_zoh` with
  hard `np.clip(u, ±F)` and RK4 sub-stepping), under perturbed ICs, against
  predicate `v1`: **48/48** at σ=0.02 (~1.1°/link) across two seeds (12345: 24/24;
  999: 24/24), 60 N bound, peak force ≤ 38.6 N, zero ticks clipped
  (`results/clvalidate_n6_F60_banked_seed12345.json` and `...banked_seed999.json`).
- **Peak cart force.** Nominal 21.56 N, validation ≤ 38.6 N, both far inside the
  60 N validation bound and the 150 N model spec.
- **Cart excursion.** The predicate caps `|x| ≤ 2 m` only during the final 5 s
  hold. The peak cart excursion during swing-up is 4.58 m (n=6 nominal peak `|x|` =
  4.582 m, range 4.625 m), larger than n=5 (3.69 m). A real n=6 rig would need
  about 5 m of usable travel, or a re-solve under a tighter track constraint.

### Robustness scope

Robust to validated kicks with margin: at σ≈1° the peak validation force is
≤ 38.6 N of the 60 N bound, about 20 N of headroom, zero ticks clipped. We did not
run a 5× stress leg for n=6, so the claim is robust to roughly 1° IC noise with
margin and is not characterized beyond that. A stronger kick or a tighter bound
would eventually produce clip-limited failures, as observed for n=5. We do not
claim unbounded robustness.

### The ladder

The n=4 case (ρ=0.036, 64/64 at 60 N, peak ~24 N) was observed in development; no
n=4 JSON is committed in this repo, so it is reported as a development observation,
not a committed result. n=5 is closed-loop validated there (ρ=0.0298, 64/64). n=6
is this release: ρ=0.0270, 48/48, native 1 ms grid. The
2→3→4 stages reproduced the published-style n=4 numbers exactly (n=2 0.81°/12.9 N,
n=3 1.28°/24.3 N, n=4 0.091°/25.4 N), confirming the pipeline is unchanged across
n.

---

## 4. Novelty

See [PRIOR_ART.md](PRIOR_ART.md) for the full table. The defensible claim:

> First open-source, code-reproducible, non-RL n=6 cart-pole
> swing-up-and-balance artifact that we found: a saved nominal plus saturated
> closed-loop validation against a strict committed predicate plus one-command
> reproduction, built by exact-inversion trajectory design rather than learning.

This is not the first 6-link solve. yacine (@yacineMTB) posted a 6-pendulum
cartpole RL solve publicly first on 2026-06-09 (pufferlib, its standard PPO, in
MuJoCo, no reproducible code artifact released). We concede the
public-first 6-link demonstration and make no priority claim; we hold a local
build timestamp but no public proof. The contribution is the reproducible, non-RL
artifact, not its date. It is also not hardware and not a formal robustness proof.
The qualifiers open-source, code-reproducible, and non-RL are load-bearing and
travel with the claim.

The inversion-based feedforward lineage (Glück/Kugi, Graichen/Zeitz) tops out at
the triple on a cart, to our knowledge. The n≥4 extension and the
collocation-polish synthesis are documented in the n=5 repo; reaching n=6 by
continuation and the cc-iLQG 1 ms refinement are this release.

## 5. Future work: higher link counts

The single cart's control authority tightens with each link, and the n=6 nominal
already travels further on the rail than n=5 (4.58 m versus 3.69 m). Higher link
counts are open future work. We make no claim that n=6 is a ceiling or that n=7 is
unreachable.

## 6. Sources

The shared-method sources (Glück/Kugi 2013, the Graichen/Zeitz inversion lineage,
the HardwareX multi-arm pendulum, Lam & Davison, the Isaac-Sim and video n=5
prior art) are listed in the n=5 repo and in [PRIOR_ART.md](PRIOR_ART.md). The
n=6-specific reference:

- yacine (@yacineMTB), public post, 2026-06-09: a 6-pendulum cartpole swing-up/
  balance trained with RL (pufferlib, its standard PPO) in MuJoCo (posted
  publicly first; no reproducible code artifact released alongside it).
