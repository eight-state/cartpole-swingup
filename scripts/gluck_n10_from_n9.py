"""n=10 Gluck multiple-shooting swing-up, warm-started from the saved n=9 MS
nominal. Exact analog of gluck_n9_from_n8.py bumped one link: loads
runs/r2/nom_n9_gluck.npz, samples its angle/rate columns at the n=10 segment
boundaries, copies link 9 onto new link 10, runs the SAME MultiShoot solver.

Continuation: T 10s, mesh M 80 -> 88, a_max/nsub unchanged.
Usage: gluck_n10_from_n9.py [T] [a_max] [M] [nsub] [maxfev]
Saves runs/r2/nom_n10_gluck.npz on success.
"""
import os, sys, time, json, warnings
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gluck_swingup2 as G
from gluck_swingup2 import make_model, build_fns, MultiShoot, evaluate, log
from gluck_n9_from_n8 import warm_from_prev_traj


def main():
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    a_max = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 88
    nsub = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    maxfev = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    n = 10; NP = 2 * n

    d9 = np.load("runs/r2/nom_n9_gluck.npz", allow_pickle=True)
    X9 = d9["states"]; t9 = d9["t"]; p9 = d9["p"]; n9 = int(d9["n"])

    m = make_model(n); fang, fforce = build_fns(m)
    ts = np.linspace(0, T, M + 1)
    Z0, p0 = warm_from_prev_traj(X9, t9, n9, n, ts, p9, NP)

    G.LOG = open("runs/r2/gluck_n10_run.log", "w")
    log(f"=== n=10 MS warm from n=9 traj: T={T} a_max={a_max} M={M} nsub={nsub} "
        f"maxfev={maxfev} NP={NP} ===")
    ms = MultiShoot(m, fang, T, a_max, NP, M, nsub)
    t0 = time.time()
    res, Z, p = ms.solve(Z0, p0, maxfev=maxfev)
    el = time.time() - t0
    log(f"solve done nfev={res.nfev} cost={res.cost:.3e} wall={el:.0f}s")
    rec = evaluate(m, fang, fforce, Z, p, ts, T, a_max, NP,
                   f"n=10 MS T={T} a_max={a_max} M={M}",
                   save_path="runs/r2/nom_n10_gluck.npz")
    rec["nfev"] = int(res.nfev); rec["solve_s"] = round(el, 1)
    log("\n=== N10 MS RESULT ===")
    log(json.dumps(rec))
    G.LOG.close()
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
