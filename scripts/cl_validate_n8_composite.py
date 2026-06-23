"""n=8 closed-loop composite gate (fast stack) — ONE program, ONE uniform
policy, self-describing output. Physics & predicate identical to the
n=5/n=6/n=7 releases; the solver stack is re-engineered for speed + steer
robustness (see docs/METHOD.md "Fast stack").

THE COMPOSITE CONTROLLER (one policy, applied uniformly to every IC):
  stage A  replan-at-t0: re-solve the 9 s swing-up NLP from the measured
           perturbed state (DUAL warm-started from the unperturbed nominal),
           densify, track with exact-ZOH discrete TVLQR.
  stage B  pre-roll fallback, triggered ONLY by stage A's NLP failing its
           iteration budget AT t=0 (a signal causally available before any
           motion). Tracks the FIXED nominal for the benign first 2 s, then
           replans the remaining horizon from the measured mid-state.
  then     steering-NLP catch from the measured handoff -> static LQR hold.
           The steer is PRIMAL warm-started from the nominal steer plan (the
           real handoffs cluster ~2e-4 rad from the nominal arrival): the
           catch NLP converges in ~25 iters where a cold start can stall.

REAL saturated sim throughout (rollout_zoh, hard 150 N clip, 1 ms ZOH, RK4
substeps); sigma=0.02 perturbed ICs at hanging; locked predicate v1
(|theta|<=5 deg, |thetad|<=0.5, |x|<=2 m, |xd|<=0.5, continuous 5 s hold).

Budget is ITERATION-ONLY by design (no wall/CPU caps): with single-threaded
BLAS the iteration path, stage taken, and every rollout are machine-
independent; only wall time varies. Success COUNTS are reproducible; per-IC
solve wall-times are not.

Fast-stack solver changes (speed/robustness only; the honest judge remains the
downstream saturated-sim hold predicate, which a sloppy plan cannot fake):
  - per-step flat SX RK4 (F_step) + thread-parallel defect map (bit-identical
    to the serial graph)
  - stage-A dual warm start from the nominal's KKT multipliers
  - steer primal warm start from the nominal steer plan
  - mapaccum densify + batched-linearization discrete TVLQR

Usage: cl_validate_n8_composite.py [n_ic] [seed] [workers]
Writes: results/clvalidate_n8_composite_seed<seed>.json
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

SOLVER_TIER = {
    "CARTPOLE_MU_STRATEGY": "adaptive",
    "CARTPOLE_ACCEPTABLE_TOL": "1e-4",
    "CARTPOLE_ACCEPTABLE_ITER": "8",
}
for _k, _v in SOLVER_TIER.items():
    os.environ[_k] = _v
os.environ.pop("CARTPOLE_MAX_CPU_S", None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
RESULTS = REPO / "results"
NOM_DENSE = str(RESULTS / "nom_n8_dense1ms.npz")
NOM_4MS = str(RESULTS / "nom_n8_4ms.npz")
WARMPACK = str(RESULTS / "warmpack_n8.npz")

T_STEER = 2.0
N_STEER_NODES = 500
U_PLAN = 100.0
TERM_TOL = 2e-5
HOLD_S = 5.0
T_PREROLL = 2.0
MAX_ITER = 1500
MAP_THREADS = int(os.environ.get("CARTPOLE_MAP_THREADS", "1"))
GATE_VERSION = "faststack-v1"

_W = {}  # per-worker cache (built once)


def _make_model():
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    n = 8
    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0,
                        link_masses_kg=[0.10] * n, link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    return NLinkCartPole(spec), spec


def ensure_warmpack():
    """Harvest the nominal warm-start pack (stage-A KKT multipliers + nominal
    steer plan) ONCE and cache to results/.

    The SHIPPED ``results/warmpack_n8.npz`` is a fixed, load-bearing input of
    this release: the banked counts are a property of code + THIS warmpack.
    Regeneration (the fallback below, only if the file is missing) re-solves
    the nominal and may land slightly different KKT multipliers, which can flip
    the disclosed knife-edge IC (seed 12345 tag 4) — so regenerating is NOT
    guaranteed equivalent to the shipped pack. Keep the shipped file in place
    to reproduce the banked result."""
    if os.path.exists(WARMPACK):
        return
    from fast_trajopt import FastColloc
    m, spec = _make_model()
    dc = np.load(NOM_4MS)
    Xc, Uc = dc["x"], dc["u"]
    Tn = float(dc["horizon"])
    dd = np.load(NOM_DENSE)
    print("[WARMPACK] harvesting stage-A duals + nominal steer...", flush=True)
    fcA = FastColloc(m, horizon_s=Tn, n_nodes=len(Uc), force_bound=U_PLAN,
                     terminal_tol_rad=2e-4, w_u=1e-4, max_iter=400,
                     warm_duals=True, map_threads=MAP_THREADS)
    rA = fcA.solve(dd["x"][0], x_init=Xc, u_init=Uc)
    assert rA.success, "nominal stage-A harvest failed"
    fcS = FastColloc(m, horizon_s=T_STEER, n_nodes=N_STEER_NODES,
                     force_bound=U_PLAN, terminal_tol_rad=TERM_TOL,
                     w_u=1e-4, max_iter=MAX_ITER, map_threads=MAP_THREADS)
    rS = fcS.solve(dd["x"][-1])
    assert rS.success, "nominal steer harvest failed"
    np.savez(WARMPACK, A_lam_g=rA.lam_g, A_lam_x=rA.lam_x,
             S_x=rS.x, S_u=rS.u)
    print("[WARMPACK] saved", WARMPACK, flush=True)


class _ForceLogged:
    """Clip at fb, record pre-clip demanded force (saturation honesty)."""
    def __init__(self, raw, fb):
        self.raw = raw; self.fb = fb
        self.max_demand = 0.0; self.saturated = False

    def __call__(self, x, t):
        u = float(self.raw(x, t)); a = abs(u)
        if a > self.max_demand:
            self.max_demand = a
        if a > self.fb:
            self.saturated = True
        return float(np.clip(u, -self.fb, self.fb))


def _worker_init():
    import numpy as _np
    from cartpole_race.lqr import static_lqr
    from fast_trajopt import FastColloc
    from fast_pieces import make_densifier
    m, spec = _make_model()
    n = m.n
    dd = _np.load(NOM_DENSE); dc = _np.load(NOM_4MS); wp = _np.load(WARMPACK)
    Xc, Uc = dc["x"], dc["u"]; Tn = float(dc["horizon"]); Ncoarse = len(Uc)
    dt = spec.control_dt_s
    n_sub = max(1, int(_np.ceil(dt / spec.rk4_max_step_s)))
    K, _P = static_lqr(m)
    _W.update(
        m=m, spec=spec, n=n, dt=dt, n_sub=n_sub, stride=4,
        Xn_f=dd["x"], Un_f=dd["u"], Xc=Xc, Uc=Uc, Tn=Tn, Ncoarse=Ncoarse,
        Krow=_np.asarray(K).reshape(-1), xup=m.x_equilibrium("up"), wp=wp,
        fcA=FastColloc(m, horizon_s=Tn, n_nodes=Ncoarse, force_bound=U_PLAN,
                       terminal_tol_rad=2e-4, w_u=1e-4, max_iter=MAX_ITER,
                       warm_duals=True, map_threads=MAP_THREADS),
        fcS=FastColloc(m, horizon_s=T_STEER, n_nodes=N_STEER_NODES,
                       force_bound=U_PLAN, terminal_tol_rad=TERM_TOL,
                       w_u=1e-4, max_iter=MAX_ITER, map_threads=MAP_THREADS),
        densA=make_densifier(m, dt, n_sub, 4, Ncoarse),
        densS=make_densifier(m, dt, n_sub, 4, N_STEER_NODES),
    )


def one_ic(args):
    (dx, tag) = args
    import numpy as _np
    if not _W:
        _worker_init()
    from cartpole_race.lqr import wrap_state_error
    from cartpole_race.funnels import in_success_set
    from fast_pieces import FastDTVLQR, make_densifier
    from fast_trajopt import FastColloc

    m = _W["m"]; spec = _W["spec"]; n = _W["n"]; dt = _W["dt"]
    stride = _W["stride"]; Xc = _W["Xc"]; Uc = _W["Uc"]; Tn = _W["Tn"]
    wp = _W["wp"]; fb = spec.force_bound_n; track = spec.track_half_length_m
    xup = _W["xup"]
    x0 = _W["Xn_f"][0] + _np.asarray(dx)
    rec = {"tag": tag, "gate_version": GATE_VERSION}
    demands = []; logs_xu = []

    # ---- stage A: replan-at-t0 (dual warm start) ----
    t0 = time.time()
    rp = _W["fcA"].solve(x0, x_init=Xc, u_init=Uc,
                         lam_g0=wp["A_lam_g"], lam_x0=wp["A_lam_x"])
    rec["stageA_solve_s"] = round(time.time() - t0, 1)
    rec["stageA_iters"] = rp.iter_count
    if rp.success:
        rec["stage"] = "A_replan_t0"
        Xd, Ud = _W["densA"](rp.x, rp.u)
        tv1 = FastDTVLQR(m, Xd, Ud, dt)
        pol1 = _ForceLogged(tv1.policy, fb)
        _t, x1, u1 = m.rollout_zoh(x0, pol1, Tn, dt, spec.rk4_max_step_s)
        xh = x1[-1]
        if _np.any(_np.isnan(xh)) or float(_np.max(
                _np.abs(xh[1:1 + n] - Xd[-1, 1:1 + n]))) > 0.5:
            rec.update(success=False, fail_stage="A_track_diverged")
            return rec
        demands.append(pol1); logs_xu.append((x1, u1))
    else:
        rec["stage"] = "B_preroll"
        tvf = FastDTVLQR(m, _W["Xn_f"], _W["Un_f"], dt)
        polf = _ForceLogged(tvf.policy, fb)
        _t, xA, uA = m.rollout_zoh(x0, polf, T_PREROLL, dt, spec.rk4_max_step_s)
        demands.append(polf); logs_xu.append((xA, uA))
        x_mid = xA[-1]; T_rem = Tn - T_PREROLL
        N_rem = int(round(T_rem / (stride * dt)))
        kc = int(round(T_PREROLL / (stride * dt)))
        fcB = _W.get("fcB")
        if fcB is None:
            fcB = FastColloc(m, horizon_s=T_rem, n_nodes=N_rem,
                             force_bound=U_PLAN, terminal_tol_rad=2e-4,
                             w_u=1e-4, max_iter=MAX_ITER, map_threads=MAP_THREADS)
            _W["fcB"] = fcB
            _W["densB"] = make_densifier(m, dt, _W["n_sub"], stride, N_rem)
        t0 = time.time()
        rp2 = fcB.solve(x_mid, x_init=Xc[kc:kc + N_rem + 1],
                        u_init=Uc[kc:kc + N_rem])
        rec["stageB_solve_s"] = round(time.time() - t0, 1)
        rec["stageB_iters"] = rp2.iter_count
        if not rp2.success:
            rec.update(success=False, fail_stage="B_replan_nlp")
            return rec
        Xd, Ud = _W["densB"](rp2.x, rp2.u)
        tv2 = FastDTVLQR(m, Xd, Ud, dt)
        pol2 = _ForceLogged(tv2.policy, fb)
        _t, x1, u1 = m.rollout_zoh(x_mid, pol2, T_rem, dt, spec.rk4_max_step_s)
        demands.append(pol2); logs_xu.append((x1, u1))
        xh = x1[-1]
        if _np.any(_np.isnan(xh)):
            rec.update(success=False, fail_stage="B_track")
            return rec

    hdev = float(_np.max(_np.abs(((xh[1:1 + n] - xup[1:1 + n] + _np.pi)
                                  % (2 * _np.pi)) - _np.pi)))
    rec["handoff_dev_rad"] = round(hdev, 8)

    # ---- steering catch (PRIMAL warm start from nominal steer plan) ----
    t0 = time.time()
    st = _W["fcS"].solve(xh, x_init=wp["S_x"], u_init=wp["S_u"])
    rec["steer_solve_s"] = round(time.time() - t0, 1)
    rec["steer_iters"] = st.iter_count
    if not st.success:
        rec.update(success=False, fail_stage="steer_nlp")
        return rec
    Xs, Us = _W["densS"](st.x, st.u)
    tv3 = FastDTVLQR(m, Xs, Us, dt)
    pol3 = _ForceLogged(tv3.policy, fb)
    _t, x2, u2 = m.rollout_zoh(xh, pol3, T_STEER, dt, spec.rk4_max_step_s)
    demands.append(pol3); logs_xu.append((x2, u2))

    # ---- static hold ----
    Krow = _W["Krow"]

    def _static_raw(x, t):
        return -float(Krow @ wrap_state_error(x, xup, n))

    pol4 = _ForceLogged(_static_raw, fb)
    _t, x3, u3 = m.rollout_zoh(x2[-1], pol4, HOLD_S + 1.0, dt,
                               spec.rk4_max_step_s)
    demands.append(pol4); logs_xu.append((x3, u3))
    in_set = [in_success_set(m, xx) for xx in x3]
    run = 0
    for v_ in in_set:
        run = run + 1 if v_ else 0
    # Elapsed continuous in-set time is (run-1) ticks, not `run` samples
    # (gaps, not points) — byte-for-byte the convention in
    # rollout.static_hold_rollout. Requires a genuine 5.0 s hold (5001 samples
    # at 1 ms), not the 4.999 s that `run >= int(HOLD_S/dt)` (5000) would pass.
    hold_ok = bool(max(0, run - 1) * dt >= HOLD_S - 1e-9)
    peakF = float(max(_np.max(_np.abs(u)) for _, u in logs_xu))
    track_ok = bool(max(float(_np.max(_np.abs(x[:, 0]))) for x, _ in logs_xu)
                    <= track)
    rec.update(
        success=bool(hold_ok and track_ok),
        peakF=round(peakF, 3),
        max_force_demanded=round(max(d.max_demand for d in demands), 3),
        saturated=bool(any(d.saturated for d in demands)), track_ok=track_ok)
    if not rec["success"]:
        rec.setdefault("fail_stage", "hold")
    return rec


def main():
    n_ic = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n = 8; nx = 2 * (n + 1)
    ensure_warmpack()
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_ic):
        dx = np.zeros(nx)
        dx[0] = rng.normal(0, 0.02)
        dx[1:1 + n] = rng.normal(0, 0.02, n)
        dx[1 + n] = rng.normal(0, 0.02)
        dx[2 + n:] = rng.normal(0, 0.02, n)
        jobs.append((dx, i))
    print(f"[CL-COMPOSITE n=8] {GATE_VERSION}  n_ic={n_ic} seed={seed} "
          f"sigma=0.02 workers={workers} solver_tier={SOLVER_TIER}", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_ic, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            print("  ", json.dumps(r, default=str), flush=True)
    k = sum(1 for r in results if r.get("success"))
    n_sat = sum(1 for r in results if r.get("saturated"))
    mfd = max((r.get("max_force_demanded", 0.0) for r in results), default=0.0)
    el = time.time() - t0
    print(f"[CL-COMPOSITE n=8] {k}/{n_ic} success  "
          f"max_force_demanded={mfd:.1f}N n_saturated_ics={n_sat} "
          f"({el:.0f}s = {el/3600:.2f}h)", flush=True)
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        # Tracked-only: untracked result artifacts (this run's own JSONs) must
        # NOT count as a dirty CODE state for provenance.
        dirty = bool(subprocess.run(["git", "status", "--porcelain",
                                     "--untracked-files=no"], cwd=REPO,
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:
        commit = None; dirty = None
    nom_sha = hashlib.sha256(Path(NOM_DENSE).read_bytes()).hexdigest()
    z = 1.959963984540054
    ph = k / n_ic; den = 1 + z * z / n_ic
    ctr = (ph + z * z / (2 * n_ic)) / den
    hw = z * ((ph * (1 - ph) / n_ic + z * z / (4 * n_ic * n_ic)) ** 0.5) / den
    wilson = [max(0.0, ctr - hw), min(1.0, ctr + hw)]
    suffix = f"_n{n_ic}" if n_ic != 24 else ""
    out = RESULTS / f"clvalidate_n8_composite_seed{seed}{suffix}.json"
    out.write_text(json.dumps(
        {"gate_version": GATE_VERSION, "n_success": k, "n_ic": n_ic,
         "seed": seed, "wilson_95": wilson, "commit_sha": commit,
         "git_dirty": dirty, "nominal_sha256": nom_sha,
         "steer_fix": "primal_warm_start_from_nominal",
         "solver_tier": SOLVER_TIER, "max_iter": MAX_ITER,
         "max_force_demanded_over_runs": mfd, "n_saturated_ics": n_sat,
         "results": sorted(results, key=lambda r: r["tag"])},
        indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
