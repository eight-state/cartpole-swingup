"""16-wide closed-loop validation wrapper reusing r2_validate machinery.

Validates a SPECIFIC saved nominal (explicit path) with a 16-worker perturbed-IC
pool in the REAL saturated sim (simulate_handoff -> rollout_zoh with clip).
Usage: cl_validate16.py <nom_path> <n> <force> [n_ic] [seed]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_validate as V  # noqa: E402


def main():
    nom_path = Path(sys.argv[1])
    n = int(sys.argv[2])
    force = float(sys.argv[3])
    n_ic = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 12345

    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec

    d = np.load(nom_path)
    x_nom = d["x"]
    u_nom = d["u"]
    horizon = float(d["horizon"])
    spec = CartPoleSpec().with_n_links(n)
    model = NLinkCartPole(spec)
    t_nom = np.linspace(0.0, horizon, len(x_nom))
    u_pad = (np.append(u_nom, u_nom[-1])
             if len(u_nom) == len(t_nom) - 1 else u_nom)

    tv, P = V.build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    dt_grid = horizon / (len(x_nom) - 1)
    rho, _ = V.closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, dt_grid)

    st = V.perturbed_ic_study(n, t_nom, x_nom, u_nom, force,
                              n_ic=n_ic, max_workers=16, seed=seed)
    rep = {"n": n, "nominal": str(nom_path), "horizon": horizon,
           "nodes": len(x_nom) - 1, "monodromy_rho": rho,
           "force": force, "study": st}
    print(f"[CL n={n} F={force:.0f}] {st['n_success']}/{st['n_ic']} "
          f"= {st['frac']:.3f} CI[{st['wilson_lo']:.3f},{st['wilson_hi']:.3f}] "
          f"rho={rho:.4g} peakF={st['max_force_over_runs']:.1f}N "
          f"{st['wall']:.1f}s", flush=True)
    out = (Path(__file__).resolve().parent.parent / "results"
           / f"clvalidate_n{n}_F{int(force)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
