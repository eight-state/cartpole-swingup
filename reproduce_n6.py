"""ONE-COMMAND reproduction of the headline n=6 (sextuple) cart-pole result.

    uv run python reproduce_n6.py

What this does (all from the validated nominal selected in ``configs/nominal.py``,
no re-solving):

  1. Loads the n=6 nominal via the SINGLE path constant in ``configs/nominal.py``
     (the native 1 ms-grid ``nom_n6_gluck_cont.npz``; see the README
     "Nominal rigor status" section and that file's header).
  2. Builds whole-trajectory TVLQR linearized ALONG the nominal (terminal cost =
     the upright CARE solution) and computes the closed-loop monodromy spectral
     radius ``rho`` along the nominal at the 1 ms control rate.
  3. Runs the closed-loop perturbed-initial-condition ensemble in the REAL
     saturated simulator (``dynamics.rollout_zoh`` with hard force clipping), at
     the 60 N validation bound, across the two banked validation seeds
     (sigma=0.02, ~1.1 deg / link). Each run must hold ALL 6 links in the locked
     success set (|theta|<=5deg, |thetadot|<=0.5, |x|<=2, |xdot|<=0.5)
     continuously for the final 5 s, with force and track respected over the
     WHOLE rollout (predicate v1).
  4. Regenerates the demo GIF (``results/demo_sextuple.gif``) from one
     deterministic swing-up + 5 s balance rollout.

Prints the success fraction for each seed. Expected: 24/24 + 24/24 = 48/48 at
sigma=0.02, F=60 N, rho ~ 0.0270 (< 1), matching the committed validation JSONs.

Determinism note: the perturbed-IC ensemble runs across a process pool; the RNG
seed fixes the ICs, and each rollout is itself deterministic (fixed-step RK4,
ZOH). The success COUNTS are reproducible; wall-time is not.

NOTE on the nominal grid: the shipped n=6 nominal is on the native 1 ms grid
(1-step ZOH defect 0.0, bit-exact 1 ms-consistent), at parity with the n=4 / n=5
nominals. Closed-loop validation runs the REAL plant at the 1 ms control rate and
catches 48/48. The active nominal is pinned in configs/nominal.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Single-threaded BLAS so the process pool does not oversubscribe / spike RAM.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

import r2_validate as V  # noqa: E402
import nominal as NOM     # noqa: E402  (the single source of truth)

N = 6
FORCE_BOUND = 60.0
# Two banked validation seeds (sigma=0.02 each) -> 24/24 + 24/24 = 48/48.
SEEDS = (12345, 999)
N_IC = 24


def _load_nominal():
    spec = NOM.NOMINAL
    d = np.load(spec.path)
    x_nom = d["x"]
    u_nom = d["u"]
    horizon = float(d["horizon"])
    t_nom = np.linspace(0.0, horizon, len(x_nom))
    return spec, t_nom, x_nom, u_nom, horizon


def main(n_ic: int = N_IC, workers: int = 16) -> int:
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec

    spec = NOM.NOMINAL
    if not spec.path.exists():
        print(f"[reproduce] MISSING nominal: {spec.path}", file=sys.stderr)
        return 2

    nspec, t_nom, x_nom, u_nom, horizon = _load_nominal()
    print(f"[reproduce] n={N} nominal: {nspec.file}  horizon={horizon:.2f}s  "
          f"nodes={len(u_nom)}  grid={nspec.grid_ms:.1f}ms  "
          f"(native_1ms={nspec.is_native_1ms})  "
          f"validation force bound={FORCE_BOUND:.0f}N", flush=True)
    print(f"[reproduce] nominal rigor: {nspec.label}", flush=True)

    # --- monodromy (contractivity of the closed loop along the nominal) ------
    model = NLinkCartPole(CartPoleSpec().with_n_links(N))
    u_pad = (np.append(u_nom, u_nom[-1])
             if len(u_nom) == len(t_nom) - 1 else u_nom)
    tv, _ = V.build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    dt_grid = horizon / (len(x_nom) - 1)
    rho, _ = V.closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, dt_grid)
    print(f"[reproduce] closed-loop monodromy rho = {rho:.4g} "
          f"({'contracting' if rho < 1 else 'NOT contracting'})", flush=True)

    # --- closed-loop validation across the two banked seeds ------------------
    total_success = 0
    total_trials = 0
    peak_force_all = 0.0
    peak_x_all = 0.0
    for seed in SEEDS:
        print(f"\n[reproduce] validation seed={seed}  "
              f"(sigma=0.02, {n_ic} ICs, F={FORCE_BOUND:.0f}N)...", flush=True)
        st = V.perturbed_ic_study(
            N, t_nom, x_nom, u_nom, FORCE_BOUND,
            n_ic=n_ic, pos_sigma=0.02, ang_sigma=0.02, vel_sigma=0.02,
            max_workers=workers, seed=seed)
        total_success += st["n_success"]
        total_trials += st["n_ic"]
        peak_force_all = max(peak_force_all, st["max_force_over_runs"])
        peak_x_all = max(peak_x_all, st["max_abs_x_over_runs"])
        print(f"[reproduce] seed {seed}: {st['n_success']}/{st['n_ic']} "
              f"= {st['frac']:.3f}  CI[{st['wilson_lo']:.3f},{st['wilson_hi']:.3f}]  "
              f"peakF={st['max_force_over_runs']:.1f}N  "
              f"max|x|={st['max_abs_x_over_runs']:.2f}m  {st['wall']:.0f}s",
              flush=True)

    # --- regenerate the GIF --------------------------------------------------
    print("\n[reproduce] regenerating demo GIF...", flush=True)
    try:
        import importlib
        demo = importlib.import_module("demo_sextuple")
        demo.run_demo(make_plots=True, make_animation=True)
    except Exception as exc:  # pragma: no cover
        print(f"[reproduce] GIF/plots step skipped ({exc})", flush=True)

    # --- verdict -------------------------------------------------------------
    ok = (total_success == total_trials and rho < 1.0)
    print("\n" + "=" * 70, flush=True)
    print(f"  RESULT n=6 sextuple cart-pole swing-up + balance", flush=True)
    print(f"    nominal grid    : {nspec.grid_ms:.1f}ms "
          f"({'1 ms native (parity with n=4 / n=5)' if nspec.is_native_1ms else '2.5 ms collocation grid'})",
          flush=True)
    print(f"    monodromy rho   : {rho:.4g}  (<1 => contracting)", flush=True)
    print(f"    closed-loop     : {total_success}/{total_trials} "
          f"(two seeds, sigma=0.02, F=60 N)", flush=True)
    print(f"    peak force      : {peak_force_all:.1f} N  (60 N bound; never binds)",
          flush=True)
    print(f"    peak cart |x|   : {peak_x_all:.2f} m  (rail bound +/-10 m)", flush=True)
    print(f"  VERDICT: {'REPRODUCED' if ok else 'MISMATCH -- investigate'}", flush=True)
    print("=" * 70, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
