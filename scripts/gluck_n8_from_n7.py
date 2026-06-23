"""Standalone n=8 Gluck MS swing-up, warm-started from the SAVED n=7 MS nominal.

Exact analog of scripts/gluck_n7_from_n6.py (which produced the n=7 MS plan),
bumped one link: loads results/nom_n7_gluck.npz, samples its angle/rate
columns at the n=8 segment-boundary times, copies link 7 onto the new link 8,
and runs the SAME MultiShoot solver + evaluate() as the proven recipe.

Usage: gluck_n8_from_n7.py [T] [a_max] [M] [nsub] [maxfev]
Saves results/nom_n8_gluck.npz on success.

NOTE: this is the continuation-SEED generator and is NOT on the one-command
reproduction path (`reproduce_n8.py` uses the byte-pinned shipped nominal). It
requires the n=7 release's ``nom_n7_gluck.npz`` (from
github.com/eight-state/septuple-cartpole), which is NOT shipped in this repo —
so this seed-generation step is reproducible only with that file present, not
from a clean clone of this repo alone.
"""
import os, sys, time, json, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gluck_swingup2 as G
from gluck_swingup2 import make_model, build_fns, MultiShoot, evaluate, log


def warm_from_prev_traj(Xp, tp, np_links, new_n, new_ts, prev_p, new_NP):
    """n -> n+1 boundary-state warm start (same construction as n6->n7)."""
    M1 = len(new_ts)
    Z = np.zeros((M1, 2 * new_n))
    for j in range(new_n):
        src = min(j, np_links - 1)
        ang_col = 1 + src
        rate_col = (np_links + 1) + 1 + src
        Z[:, j] = np.interp(new_ts, tp, Xp[:, ang_col])
        Z[:, new_n + j] = np.interp(new_ts, tp, Xp[:, rate_col])
    p = np.zeros(new_NP)
    p[:len(prev_p)] = prev_p
    return Z, p


def main():
    # continuation schedule: extend the proven n=7 settings (T=8, a_max=100,
    # M=64) one link with the same longer-T / slightly heavier-mesh rationale.
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
    a_max = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 72
    nsub = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    maxfev = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    n = 8; NP = 2 * n

    d7 = np.load("results/nom_n7_gluck.npz", allow_pickle=True)
    X7 = d7["states"]; t7 = d7["t"]; p7 = d7["p"]
    n7 = int(d7["n"])

    m = make_model(n); fang, fforce = build_fns(m)
    ts = np.linspace(0, T, M + 1)
    Z0, p0 = warm_from_prev_traj(X7, t7, n7, n, ts, p7, NP)

    G.LOG = open("results/gluck_n8_run.log", "w")
    log(f"=== n=8 MS warm from n=7 traj: T={T} a_max={a_max} M={M} nsub={nsub} "
        f"maxfev={maxfev} NP={NP} ===")
    ms = MultiShoot(m, fang, T, a_max, NP, M, nsub)
    t0 = time.time()
    res, Z, p = ms.solve(Z0, p0, maxfev=maxfev)
    el = time.time() - t0
    log(f"solve done nfev={res.nfev} cost={res.cost:.3e} wall={el:.0f}s")
    rec = evaluate(m, fang, fforce, Z, p, ts, T, a_max, NP,
                   f"n=8 MS T={T} a_max={a_max} M={M}",
                   save_path="results/nom_n8_gluck.npz")
    rec["nfev"] = int(res.nfev); rec["solve_s"] = round(el, 1)
    log("\n=== N8 MS RESULT ===")
    log(json.dumps(rec))
    G.LOG.close()
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
