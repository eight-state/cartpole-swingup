"""Generate machine-readable closed-loop validation reports for ALL legs.

Backs every success count quoted in the README/METHOD with a committed JSON
report. Runs the four perturbed-IC legs on the verified n=5 nominal at the 60 N
bound and writes one JSON per leg plus a combined report.

Legs (all on results/nom_n5_gluck_cont.npz, F=60 N):
  - seed 12345, n_ic 24, sigma 0.02   (banked leg A of the 64/64)
  - seed   999, n_ic 40, sigma 0.02   (banked leg B of the 64/64)
  - seed  7777, n_ic 24, sigma 0.02   (fresh)
  - seed  2024, n_ic 24, sigma 0.10   (5x stress)

Each report records: commit_sha, python_version, nominal sha256, seed, sigma,
n_trials, n_success, force_limit, max_abs_force (clipped force applied),
max_abs_force_demanded (raw pre-clip demand), n_saturated_ics (ICs that hit the
bound), MAX_ABS_X over the whole rollout, nominal_horizon_s and
rollout_duration_s, monodromy_rho, predicate version. Numbers are reported AS
RUN; if a leg does not reproduce a previously quoted count, the actual count is
written.

    uv run python scripts/gen_validation_reports.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import r2_validate as V  # noqa: E402

N = 5
FORCE = 60.0
RESULTS = REPO / "results"
NOM_PATH = RESULTS / "nom_n5_gluck_cont.npz"

# Locked success predicate (mirrors funnels.in_success_set defaults). Bumping
# any threshold MUST bump this version string.
PREDICATE = {
    "version": "v1",
    "theta_tol_deg": 5.0,
    "thetadot_tol_rad_s": 0.5,
    "x_tol_m": 2.0,
    "xdot_tol_m_s": 0.5,
    "hold_time_s": 5.0,
    "rail_bound_m": 10.0,
    "note": ("all n links within tolerances held continuously for the final "
             "5 s; force and track respected over the WHOLE rollout"),
}

LEGS = [
    {"name": "banked_seed12345", "seed": 12345, "n_ic": 24, "sigma": 0.02},
    {"name": "banked_seed999",   "seed": 999,   "n_ic": 40, "sigma": 0.02},
    {"name": "fresh_seed7777",   "seed": 7777,  "n_ic": 24, "sigma": 0.02},
    {"name": "stress_seed2024",  "seed": 2024,  "n_ic": 24, "sigma": 0.10},
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    if not NOM_PATH.exists():
        print(f"MISSING nominal: {NOM_PATH}", file=sys.stderr)
        return 2

    d = np.load(NOM_PATH)
    x_nom = d["x"]
    u_nom = d["u"]
    horizon = float(d["horizon"])
    t_nom = np.linspace(0.0, horizon, len(x_nom))

    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    spec = CartPoleSpec().with_n_links(N)
    model = NLinkCartPole(spec)
    u_pad = (np.append(u_nom, u_nom[-1])
             if len(u_nom) == len(t_nom) - 1 else u_nom)
    tv, _ = V.build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    dt_grid = horizon / (len(x_nom) - 1)
    rho, _ = V.closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, dt_grid)

    commit_sha = _git_sha()
    try:
        import subprocess as _sp
        _porc = _sp.run(["git", "status", "--porcelain"], cwd=str(REPO),
                        capture_output=True, text=True, timeout=10).stdout
        git_dirty = bool(_porc.strip())
    except Exception:
        git_dirty = None
    py_version = platform.python_version()
    nom_sha = _sha256(NOM_PATH)
    print(f"commit={commit_sha[:10]} python={py_version} "
          f"nominal_sha256={nom_sha[:12]}... rho={rho:.6g}", flush=True)

    leg_reports = []
    for leg in LEGS:
        s = leg["sigma"]
        st = V.perturbed_ic_study(
            N, t_nom, x_nom, u_nom, FORCE,
            n_ic=leg["n_ic"], pos_sigma=s, ang_sigma=s, vel_sigma=s,
            max_workers=16, seed=leg["seed"])
        rep = {
            "leg": leg["name"],
            "commit_sha": commit_sha,
            "git_dirty": git_dirty,
            "python_version": py_version,
            "nominal": "results/nom_n5_gluck_cont.npz",
            "nominal_sha256": nom_sha,
            "n_links": N,
            # The nominal swing-up horizon (6.0 s) is distinct from the
            # closed-loop rollout duration, which is swing-up + 5 s hold + a
            # 1 s settle budget (see rollout.simulate_handoff).
            "nominal_horizon_s": horizon,
            "rollout_duration_s": horizon + 5.0 + 1.0,
            "nodes": len(x_nom) - 1,
            "seed": leg["seed"],
            "sigma": s,
            "n_trials": st["n_ic"],
            "n_success": st["n_success"],
            "frac": st["frac"],
            "wilson_lo": st["wilson_lo"],
            "wilson_hi": st["wilson_hi"],
            "force_limit": FORCE,
            "max_abs_force": st["max_force_over_runs"],
            "max_abs_force_demanded": st["max_force_demanded_over_runs"],
            "n_saturated_ics": st["n_saturated_ics"],
            "max_abs_x": st["max_abs_x_over_runs"],
            "monodromy_rho": rho,
            "predicate": PREDICATE,
            "wall_s": st["wall"],
        }
        out = RESULTS / f"clvalidate_n5_F60_{leg['name']}.json"
        out.write_text(json.dumps(rep, indent=2))
        leg_reports.append(rep)
        print(f"[{leg['name']}] {st['n_success']}/{st['n_ic']} "
              f"peakF={st['max_force_over_runs']:.1f}N "
              f"rawF={st['max_force_demanded_over_runs']:.1f}N "
              f"sat={st['n_saturated_ics']}/{st['n_ic']} "
              f"max|x|={st['max_abs_x_over_runs']:.3f}m  -> {out.name}",
              flush=True)

    # The README/METHOD no-pooling rule: the two perturbation-amplitude legs
    # (sigma=0.02 and sigma=0.10) are distinct claims and MUST NOT be summed into
    # one pooled success count. So we report success counts grouped by sigma and
    # never emit an all-legs pooled total.
    by_sigma: "OrderedDict[str, dict]" = OrderedDict()
    for r in leg_reports:
        key = f"{r['sigma']:.2f}"
        g = by_sigma.setdefault(key, {"trials": 0, "success": 0})
        g["trials"] += r["n_trials"]
        g["success"] += r["n_success"]
    banked = [r for r in leg_reports if r["leg"].startswith("banked")]
    banked_trials = sum(r["n_trials"] for r in banked)
    banked_success = sum(r["n_success"] for r in banked)
    combined = {
        "commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "python_version": py_version,
        "nominal": "results/nom_n5_gluck_cont.npz",
        "nominal_sha256": nom_sha,
        "n_links": N,
        "force_limit": FORCE,
        "monodromy_rho": rho,
        "predicate": PREDICATE,
        "totals": {
            # Per-amplitude success counts, reported separately. The README
            # forbids merging these into a single pooled number.
            "by_sigma": by_sigma,
            "banked_two_seed_trials": banked_trials,
            "banked_two_seed_success": banked_success,
            "max_abs_force_over_all_legs": max(
                r["max_abs_force"] for r in leg_reports),
            "max_abs_force_demanded_over_all_legs": max(
                r["max_abs_force_demanded"] for r in leg_reports),
            "stress_n_saturated_ics": next(
                (r["n_saturated_ics"] for r in leg_reports
                 if r["leg"] == "stress_seed2024"), 0),
            "stress_n_trials": next(
                (r["n_trials"] for r in leg_reports
                 if r["leg"] == "stress_seed2024"), 0),
            "max_abs_x_over_all_legs": max(
                r["max_abs_x"] for r in leg_reports),
            "no_pool_note": (
                "Per the README/METHOD no-pooling rule, the two "
                "perturbation-amplitude legs (sigma=0.02 and sigma=0.10) are "
                "reported separately under by_sigma and are never summed into a "
                "single pooled success count."),
        },
        "legs": leg_reports,
    }
    out = RESULTS / "combined_validation_report.json"
    out.write_text(json.dumps(combined, indent=2))
    by_sigma_str = ", ".join(
        f"sigma={k} {g['success']}/{g['trials']}"
        for k, g in by_sigma.items())
    print(f"\ncombined: banked {banked_success}/{banked_trials}, "
          f"{by_sigma_str} (reported separately, never pooled), "
          f"stress saturated ICs = "
          f"{combined['totals']['stress_n_saturated_ics']}/"
          f"{combined['totals']['stress_n_trials']}, "
          f"max|x| over all legs = "
          f"{combined['totals']['max_abs_x_over_all_legs']:.3f} m  -> {out.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
