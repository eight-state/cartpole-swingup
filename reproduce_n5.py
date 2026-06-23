"""ONE-COMMAND reproduction of the headline n=5 (quintuple) cart-pole result.

    uv run python reproduce_n5.py

What this does (all from the validated nominal ``results/nom_n5_gluck_cont.npz``,
no re-solving):

  1. Builds whole-trajectory TVLQR linearized ALONG the nominal (terminal cost
     = the upright CARE solution) and computes the closed-loop monodromy
     spectral radius ``rho`` along the nominal at the 1 ms control rate.
  2. Runs the closed-loop perturbed-initial-condition ensemble in the REAL
     saturated simulator (``dynamics.rollout_zoh`` with hard force clipping),
     at the 60 N validation bound:
       - FRESH validation: 24 ICs, sigma=0.02 (~1.1 deg / link).
       - 5x STRESS:        24 ICs, sigma=0.10 (~5.7 deg / link, initial-angle
                           offsets up to ~18 deg / link (3-sigma of the
                           sigma=0.10 draw)). This regime RIDES the 60 N actuator
                           saturation (see README robustness caveat).
     Each run must hold ALL 5 links in the locked success set
     (|theta|<=5deg, |thetadot|<=0.5, |x|<=2, |xdot|<=0.5) continuously for the
     final 5 s, with the track respected over the WHOLE rollout. The actuator
     force is clipped to the bound by construction, so it is trivially within
     bound and gates nothing; the meaningful gates are track + the 5 s hold.
  3. Regenerates the demo GIF (``results/demo_quintuple.gif``) from one
     deterministic swing-up + 5 s balance rollout.

Prints the success fraction for each study. Expected: 24/24 fresh and
24/24 under 5x stress, matching the banked 64/64 (two seeds) at default sigma.

Determinism note: the perturbed-IC ensemble runs across a process pool; the
RNG seed fixes the ICs, and each rollout is itself deterministic (fixed-step
RK4, ZOH). The success COUNTS are reproducible; wall-time is not.
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

import r2_validate as V  # noqa: E402

N = 5
FORCE_BOUND = 60.0
RESULTS = REPO / "results"
NOM_PATH = RESULTS / "nom_n5_gluck_cont.npz"


def _load_nominal():
    d = np.load(NOM_PATH)
    x_nom = d["x"]
    u_nom = d["u"]
    horizon = float(d["horizon"])
    t_nom = np.linspace(0.0, horizon, len(x_nom))
    return t_nom, x_nom, u_nom, horizon


def main(n_ic: int = 24, workers: int = 16) -> int:
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec

    if not NOM_PATH.exists():
        print(f"[reproduce] MISSING nominal: {NOM_PATH}", file=sys.stderr)
        return 2

    t_nom, x_nom, u_nom, horizon = _load_nominal()
    print(f"[reproduce] n={N} nominal: {NOM_PATH.name}  horizon={horizon:.2f}s  "
          f"nodes={len(u_nom)}  validation force bound={FORCE_BOUND:.0f}N",
          flush=True)

    # --- monodromy (contractivity of the closed loop along the nominal) ------
    spec = CartPoleSpec().with_n_links(N)
    model = NLinkCartPole(spec)
    u_pad = (np.append(u_nom, u_nom[-1])
             if len(u_nom) == len(t_nom) - 1 else u_nom)
    tv, _ = V.build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    dt_grid = horizon / (len(x_nom) - 1)
    rho, _ = V.closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, dt_grid)
    print(f"[reproduce] closed-loop monodromy rho = {rho:.4g} "
          f"({'contracting' if rho < 1 else 'NOT contracting'})", flush=True)

    # --- FRESH validation: sigma 0.02 (~1.1 deg) -----------------------------
    print(f"\n[reproduce] FRESH validation  (sigma=0.02, {n_ic} ICs, F={FORCE_BOUND:.0f}N)...",
          flush=True)
    fresh = V.perturbed_ic_study(
        N, t_nom, x_nom, u_nom, FORCE_BOUND,
        n_ic=n_ic, pos_sigma=0.02, ang_sigma=0.02, vel_sigma=0.02,
        max_workers=workers, seed=7777)
    print(f"[reproduce] FRESH:  {fresh['n_success']}/{fresh['n_ic']} "
          f"= {fresh['frac']:.3f}  CI[{fresh['wilson_lo']:.3f},{fresh['wilson_hi']:.3f}]  "
          f"peakF={fresh['max_force_over_runs']:.1f}N  {fresh['wall']:.0f}s", flush=True)

    # --- 5x STRESS: sigma 0.10 (~5.7 deg) ------------------------------------
    print(f"\n[reproduce] 5x STRESS  (sigma=0.10, {n_ic} ICs, F={FORCE_BOUND:.0f}N)...",
          flush=True)
    stress = V.perturbed_ic_study(
        N, t_nom, x_nom, u_nom, FORCE_BOUND,
        n_ic=n_ic, pos_sigma=0.10, ang_sigma=0.10, vel_sigma=0.10,
        max_workers=workers, seed=2024)
    rides = stress["max_force_over_runs"] >= FORCE_BOUND - 1e-3
    print(f"[reproduce] STRESS: {stress['n_success']}/{stress['n_ic']} "
          f"= {stress['frac']:.3f}  CI[{stress['wilson_lo']:.3f},{stress['wilson_hi']:.3f}]  "
          f"peakF={stress['max_force_over_runs']:.1f}N  {stress['wall']:.0f}s", flush=True)
    if rides:
        print("[reproduce] NOTE: 5x stress RIDES the 60 N saturation "
              "(no force headroom); robust to validated initial-angle offsets, "
              "not unbounded.",
              flush=True)

    # --- regenerate the GIF --------------------------------------------------
    print("\n[reproduce] regenerating demo GIF...", flush=True)
    try:
        import importlib
        demo = importlib.import_module("demo_quintuple")
        demo.run_demo(make_plots=True, make_animation=True)
    except Exception as exc:  # pragma: no cover
        print(f"[reproduce] GIF/plots step skipped ({exc})", flush=True)

    # --- verdict -------------------------------------------------------------
    ok = (fresh["n_success"] == fresh["n_ic"]
          and stress["n_success"] == stress["n_ic"]
          and rho < 1.0)
    print("\n" + "=" * 70, flush=True)
    print(f"  RESULT n=5 quintuple cart-pole swing-up + balance", flush=True)
    print(f"    monodromy rho   : {rho:.4g}  (<1 => contracting)", flush=True)
    print(f"    fresh  (1.1deg) : {fresh['n_success']}/{fresh['n_ic']}", flush=True)
    print(f"    5x stress(5.7deg): {stress['n_success']}/{stress['n_ic']} "
          f"(peakF {stress['max_force_over_runs']:.0f}N)", flush=True)
    print(f"    banked (two seeds, sigma=0.02): 64/64  "
          f"(see results/combined_validation_report.json)",
          flush=True)
    print(f"  VERDICT: {'REPRODUCED' if ok else 'MISMATCH -- investigate'}", flush=True)
    print("=" * 70, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
