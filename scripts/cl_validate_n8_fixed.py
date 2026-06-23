"""n=8 FIXED-nominal leg — the n=5/n=6-equivalent controller (the honest
like-for-like robustness number).

ONE fixed dense nominal + exact-ZOH discrete-time TVLQR feedback + static-LQR
hold. NO per-IC replanning. This is exactly the controller architecture the
n=5 (88/88) and n=6 (48/48) releases used; at n=8 the catch is tighter, so
this leg is expected to be well below the composite gate's count — that gap
IS the honest statement (fixed feedback alone does not reach n=8's headline;
the composite leg's per-IC replanning does).

Same physics / simulator / perturbation model / seeds / predicate as the
composite gate and the n=5/6/7 releases.

Usage: cl_validate_n8_fixed.py [n_ic] [seed]
Writes: results/clvalidate_n8_fixed_seed<seed>.json
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
RESULTS = REPO / "results"
NOM_DENSE = str(RESULTS / "nom_n8_dense1ms.npz")
HOLD_S = 5.0


def main():
    n_ic = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345

    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.lqr import static_lqr, wrap_state_error
    from cartpole_race.funnels import in_success_set
    from fast_pieces import FastDTVLQR

    n = 8
    nx = 2 * (n + 1)
    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0,
                        link_masses_kg=[0.10] * n, link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    m = NLinkCartPole(spec)
    dt = spec.control_dt_s
    track = spec.track_half_length_m
    fb = spec.force_bound_n
    xup = m.x_equilibrium("up")
    K, _P = static_lqr(m)
    Krow = np.asarray(K).reshape(-1)

    dd = np.load(NOM_DENSE)
    Xn, Un = dd["x"], dd["u"]
    Tn = float(dd["horizon"])
    # ONE fixed-nominal discrete TVLQR (built once, used for every IC)
    tv = FastDTVLQR(m, Xn, Un, dt)

    rng = np.random.default_rng(seed)
    results = []
    t0 = time.time()
    for tag in range(n_ic):
        dx = np.zeros(nx)
        dx[0] = rng.normal(0, 0.02)
        dx[1:1 + n] = rng.normal(0, 0.02, n)
        dx[1 + n] = rng.normal(0, 0.02)
        dx[2 + n:] = rng.normal(0, 0.02, n)
        x0 = Xn[0] + dx

        max_demand = [0.0]

        def track_pol(x, t):
            u = float(tv.policy(x, t))
            max_demand[0] = max(max_demand[0], abs(u))
            return float(np.clip(u, -fb, fb))

        # track the FIXED nominal (no replan), then static-LQR hold
        _t, x1, u1 = m.rollout_zoh(x0, track_pol, Tn, dt, spec.rk4_max_step_s)
        xh = x1[-1]
        rec = {"tag": tag}
        if np.any(np.isnan(xh)):
            rec.update(success=False, fail_stage="track_diverged")
            results.append(rec)
            continue

        def hold_pol(x, t):
            u = -float(Krow @ wrap_state_error(x, xup, n))
            max_demand[0] = max(max_demand[0], abs(u))
            return float(np.clip(u, -fb, fb))

        _t, x3, u3 = m.rollout_zoh(xh, hold_pol, HOLD_S + 1.0, dt,
                                   spec.rk4_max_step_s)
        in_set = [in_success_set(m, xx) for xx in x3]
        run = 0
        for v_ in in_set:
            run = run + 1 if v_ else 0
        # (run-1) ticks of elapsed hold, matching rollout.static_hold_rollout
        # (a genuine 5.0 s = 5001 samples, not the 4.999 s of run>=5000).
        hold_ok = bool(max(0, run - 1) * dt >= HOLD_S - 1e-9)
        peakF = float(max(np.max(np.abs(u1)), np.max(np.abs(u3))))
        track_ok = bool(max(np.max(np.abs(x1[:, 0])),
                            np.max(np.abs(x3[:, 0]))) <= track)
        rec.update(success=bool(hold_ok and track_ok),
                   peakF=round(peakF, 3),
                   max_force_demanded=round(max_demand[0], 3),
                   saturated=bool(max_demand[0] > fb), track_ok=track_ok)
        if not rec["success"]:
            rec.setdefault("fail_stage", "hold")
        results.append(rec)
        print("  ", json.dumps(rec, default=str), flush=True)

    k = sum(1 for r in results if r.get("success"))
    n_sat = sum(1 for r in results if r.get("saturated"))
    mfd = max((r.get("max_force_demanded", 0.0) for r in results), default=0.0)
    print(f"[CL-FIXED n=8] {k}/{n_ic} success (fixed nominal + TVLQR, no "
          f"replan)  max_force_demanded={mfd:.1f}N  ({time.time()-t0:.0f}s)",
          flush=True)
    nom_sha = hashlib.sha256(Path(NOM_DENSE).read_bytes()).hexdigest()
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain",
                                     "--untracked-files=no"],
                                    cwd=str(REPO), capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except Exception:
        commit = None
        dirty = None
    out = RESULTS / f"clvalidate_n8_fixed_seed{seed}.json"
    out.write_text(json.dumps(
        {"controller": "fixed_nominal_tvlqr_no_replan", "n_success": k,
         "n_ic": n_ic, "seed": seed, "commit_sha": commit, "git_dirty": dirty,
         "nominal_sha256": nom_sha, "max_force_demanded_over_runs": mfd,
         "n_saturated_ics": n_sat,
         "results": sorted(results, key=lambda r: r["tag"])},
        indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
