"""n=8 polish, ONE-SHOT (the n7-proven pattern): a single continuous
solve_trajopt call (no chunking - the chunked PersistentColloc runs diverged
at BOTH 4 ms and 2 ms grids, implicating the per-chunk barrier restarts, and
that mode has never converged on these polishes).

4 ms grid, warm-started from the n=8 Glueck MS seed. On success: densify,
discrete TVLQR, unperturbed closed-loop + hold.
Usage: _n8_oneshot.py [h_node_ms]

This runs a multi-hour solve, so it is guarded by ``if __name__ == "__main__"``
and is safe to import for inspection without launching the solver.
"""
import os, sys, time
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.collocation import solve_trajopt
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.rollout import static_hold_rollout
from _dtvlqr import DiscreteTVLQR


def main():
    h_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    n = 8
    T = 9.0
    N = int(round(T / (h_ms * 1e-3)))
    stride = int(round(h_ms))

    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10] * n,
                        link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    m = NLinkCartPole(spec)
    nx = m.nx

    d8 = np.load("results/nom_n8_gluck.npz", allow_pickle=True)
    X8 = d8["states"]; U8 = d8["forces"]
    stride_w = int(round((T / N) / 0.001))
    Xw = X8[::stride_w][: N + 1].copy()
    Uw = U8[: N * stride_w].reshape(N, stride_w).mean(axis=1)
    Xw[0] = m.x_equilibrium("down")

    print(f"[N8-ONESHOT] T={T}s N={N} ({h_ms}ms nodes), warm from Glueck seed",
          flush=True)
    t0 = time.time()
    res = solve_trajopt(m, m.x_equilibrium("down"), horizon_s=T, n_nodes=N,
                        terminal_tol_rad=2e-4, force_bound=100.0, w_u=1e-4,
                        x_init_guess=Xw, u_init_guess=Uw,
                        zoh_consistent=False, max_iter=4000, print_level=5)
    el = time.time() - t0
    print(f"[N8-ONESHOT] {res.solver_status} defect={res.max_defect:.3e} "
          f"peakF={np.abs(res.u).max():.1f}N {el:.0f}s", flush=True)
    ang = res.x[-1, 1:1 + n]
    term = np.rad2deg(np.max(np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi)))
    print(f"[N8-ONESHOT] terminal {term:.4f} deg", flush=True)
    np.savez(f"results/nom_n8_{int(h_ms)}ms.npz", x=res.x, u=res.u, horizon=T,
             n=n, force=150.0, n_nodes=N)
    if not res.success:
        print("[N8-ONESHOT] NLP did not converge - probe outcome recorded",
              flush=True)
        sys.exit(0)

    n_sub = max(1, int(np.ceil(spec.control_dt_s / spec.rk4_max_step_s)))
    dt_sub = spec.control_dt_s / n_sub
    Xd = [res.x[0]]; Ud = []
    for k in range(N):
        xx = res.x[k].astype(float).copy()
        for _ in range(stride):
            for _ in range(n_sub):
                xx = m.rk4_step(xx, float(res.u[k]), dt_sub)
            Xd.append(xx.copy()); Ud.append(float(res.u[k]))
    Xd = np.array(Xd); Ud = np.array(Ud)
    np.savez("results/nom_n8_dense1ms.npz", x=Xd, u=Ud, horizon=T, n=n,
             force=150.0)
    tv = DiscreteTVLQR(m, Xd, Ud, spec.control_dt_s)
    print(f"[N8-ONESHOT] DTVLQR rho={tv.monodromy():.4g}", flush=True)
    K, P = static_lqr(m)
    sp_ = StaticLQRPolicy(m, K); sp_.P = P
    x0 = m.x_equilibrium("down")
    t1, x1, u1 = m.rollout_zoh(x0, lambda x, t: float(np.clip(tv.policy(x, t),
                                                              -150, 150)),
                               T, spec.control_dt_s, spec.rk4_max_step_s)
    xup = m.x_equilibrium("up")
    xh = x1[-1]
    hdev = np.rad2deg(np.max(np.abs(((xh[1:1 + n] - xup[1:1 + n] + np.pi)
                                     % (2 * np.pi)) - np.pi)))
    print(f"[N8-ONESHOT] CL handoff dev {hdev:.5f} deg peakF "
          f"{np.abs(u1).max():.1f} N", flush=True)
    succ, info = static_hold_rollout(m, xh, sp_, hold_time_s=5.0)
    print(f"[N8-ONESHOT] HOLD success={succ} maxF={info.get('max_force'):.1f}",
          flush=True)
    if succ:
        print("\n*** n=8 SWING-UP + BALANCE: UNPERTURBED CLOSED-LOOP PASS ***",
              flush=True)


if __name__ == "__main__":
    main()
