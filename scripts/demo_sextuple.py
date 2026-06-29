"""Clean, reproducible demo: closed-loop swing-up + balance of the n=6
(SEXTUPLE) inverted pendulum on a cart, in the REAL saturated simulator.

One command, deterministic:

    uv run python scripts/demo_sextuple.py

What it does
------------
1. Loads the validated n=6 nominal selected in ``configs/nominal.py`` (the native
   1 ms-grid ``results/nom_n6_gluck_cont.npz``, a 7.0 s Glueck-inversion +
   collocation-polish swing-up).
2. Builds whole-trajectory TVLQR linearized ALONG that nominal, with terminal
   cost ``S(tf) = P_static`` (the upright CARE solution), exactly as in
   ``scripts/r2_validate.py:build_tvlqr_along_nominal``.
3. Runs ONE closed-loop rollout in the real saturated sim
   (``dynamics.rollout_zoh``): TVLQR feedback over [0, 7.0 s] hands off to the
   static LQR, which holds upright for a further 5 s.
4. Asserts SUCCESS: every link stays continuously inside the LOCKED success set
   (|theta|<=5deg, |thetadot|<=0.5, |x|<=2, |xdot|<=0.5) for the final 5 s, the
   cart stays on the track, and the force never exceeds the bound.
5. Saves the animated GIF (and plots if matplotlib's writer is available) to
   ``results/``.

Nothing here re-solves a trajectory or runs the perturbed-IC ensemble; this is a
single light rollout (one swing-up + a 5 s hold) plus plotting.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep BLAS single-threaded so this stays light on the shared memory bus.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "configs"))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.funnels import in_success_set
from cartpole_race.lqr import StaticLQRPolicy, static_lqr, wrap_state_error
from cartpole_race.tvlqr import TVLQR

import nominal as NOM  # the single source of truth for the nominal file/grid

N_LINKS = 6
HOLD_S = 5.0
# Settling allowance after the swing-up nominal ends, BEFORE the continuous 5 s
# in-set hold is required to fit. The closed loop enters the locked success set a
# fraction of a second after handoff (the native 1 ms nominal terminal is 0.246
# deg, well inside the 5 deg gate, but a brief TVLQR -> static settle remains).
# This only lengthens the rollout window so a genuine continuous 5.0 s in-set tail
# can form; the success predicate (continuous 5.0 s in-set, force/track over the
# whole rollout) is unchanged. This mirrors the validator's handoff hold phase.
SETTLE_S = 1.0
FORCE_BOUND = 60.0  # matches the closed-loop validation bound; peak use is ~22 N
OUT = REPO / "results"


def build_tvlqr_along_nominal(model, t_nom, x_nom, u_nom):
    """TVLQR linearized along the nominal, terminal Lambda = P_static.

    Identical construction to scripts/r2_validate.py: terminal cost is the
    infinite-horizon upright CARE solution so the time-varying gains
    converge to the steady-state LQR at t=T.
    """
    _, P = static_lqr(model)
    tv = TVLQR(model, t_nom, x_nom, u_nom, Qf=P, n_eval=400)
    return tv, P


def run_demo(make_plots: bool = True, make_animation: bool = True):
    spec = CartPoleSpec().with_n_links(N_LINKS)
    spec = spec.model_copy(update={"force_bound_n": FORCE_BOUND})
    model = NLinkCartPole(spec)
    n = model.n
    control_dt = spec.control_dt_s
    rk4 = spec.rk4_max_step_s

    # --- load the validated nominal (via the single source of truth) ----------
    nspec = NOM.NOMINAL
    d = np.load(nspec.path)
    x_nom = d["x"]                      # (N+1, nx)
    u_nom = d["u"]                      # (N,)
    horizon = float(d["horizon"])      # 7.0 s
    t_nom = np.linspace(0.0, horizon, len(x_nom))
    # Pad u to the knot grid (N+1) so TVLQR's interpolation is in-bounds.
    u_pad = np.append(u_nom, u_nom[-1]) if len(u_nom) == len(t_nom) - 1 else u_nom

    print(f"[demo] n={n} nominal: {nspec.file}  horizon={horizon:.2f}s  "
          f"nodes={len(u_nom)}  grid={nspec.grid_ms:.1f}ms  "
          f"force_bound={FORCE_BOUND:.0f}N", flush=True)

    # --- build the controllers -----------------------------------------------
    tv, P = build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    K_static, _ = static_lqr(model)
    static_pol = StaticLQRPolicy(model, K_static)

    # --- ONE deterministic closed-loop rollout in the real saturated sim -----
    # Start hanging (the down equilibrium), exactly where the nominal begins.
    x0 = model.x_equilibrium("down")

    def policy(x, t):
        # Whole-trajectory TVLQR over the swing-up, then hand off to static LQR.
        if t < horizon:
            return tv.policy(x, t)
        return static_pol(x, t)

    total = horizon + SETTLE_S + HOLD_S
    t_log, x_log, u_log, _u_raw = model.rollout_zoh(x0, policy, total, control_dt, rk4)
    print(f"[demo] rollout: {len(u_log)} ticks over {total:.2f}s "
          f"(swing-up {horizon:.1f}s + settle {SETTLE_S:.1f}s + hold "
          f"{HOLD_S:.1f}s)", flush=True)

    # --- evaluate success (the LOCKED predicate) -----------------------------
    track = spec.track_half_length_m
    in_set = np.array([in_success_set(model, xx) for xx in x_log])
    # continuous in-set tail
    tail = 0
    for j in range(len(in_set) - 1, -1, -1):
        if in_set[j]:
            tail += 1
        else:
            break
    # Exact 5.0 s hold: elapsed is (tail-1) ticks, matching the gate
    # (rollout.simulate_handoff / static_hold_rollout), not the 4.999 s of
    # tail * control_dt (5000 in-set samples).
    hold_achieved = max(0, tail - 1) * control_dt
    peak_force = float(np.max(np.abs(u_log)))
    track_ok = bool(np.all(np.abs(x_log[:, 0]) <= track))
    force_ok = bool(peak_force <= FORCE_BOUND + 1e-6)
    finite_ok = bool(np.all(np.isfinite(x_log)))
    peak_x = float(np.max(np.abs(x_log[:, 0])))

    # final per-link angle residuals (deg) for reporting
    final_err = wrap_state_error(x_log[-1], model.x_equilibrium("up"), n)
    final_ang_deg = np.rad2deg(np.abs(final_err[1:1 + n]))

    print(f"[demo] in-set hold (final continuous) = {hold_achieved:.2f}s "
          f"(need {HOLD_S:.1f}s)", flush=True)
    print(f"[demo] peak |force| = {peak_force:.2f} N  (bound {FORCE_BOUND:.0f} N)",
          flush=True)
    print(f"[demo] peak cart |x| over rollout = {peak_x:.2f} m  "
          f"(rail bound +/-{track:.0f} m)", flush=True)
    print(f"[demo] cart on track = {track_ok}  forces in bound = {force_ok}",
          flush=True)
    print(f"[demo] final link angle residuals (deg) = "
          f"{np.array2string(final_ang_deg, precision=5)}", flush=True)

    success = bool(finite_ok and track_ok and force_ok
                   and hold_achieved >= HOLD_S - 1e-9)

    # --- VISUALS -------------------------------------------------------------
    saved = []
    if make_plots:
        saved += _save_plots(t_log, x_log, u_log, n, horizon, FORCE_BOUND)
    if make_animation:
        anim_path = _save_animation(t_log, x_log, model, horizon)
        if anim_path:
            saved.append(anim_path)

    # --- the assertion (deterministic acceptance gate) -----------------------
    assert success, (
        f"DEMO FAILED: hold={hold_achieved:.2f}s (need {HOLD_S}s), "
        f"track_ok={track_ok}, force_ok={force_ok}, finite={finite_ok}")

    print("\n[demo] SUCCESS: sextuple swung up and held upright for "
          f"{hold_achieved:.1f}s; all links in the locked set; "
          f"peak force {peak_force:.1f} N <= {FORCE_BOUND:.0f} N.", flush=True)
    if saved:
        print("[demo] saved visuals:", flush=True)
        for s in saved:
            print(f"         {Path(s).resolve()}", flush=True)
    return {
        "success": success,
        "hold_achieved_s": hold_achieved,
        "peak_force_n": peak_force,
        "peak_abs_x_m": peak_x,
        "final_ang_deg": final_ang_deg.tolist(),
        "saved": [str(Path(s).resolve()) for s in saved],
    }


def _save_plots(t_log, x_log, u_log, n, horizon, force_bound):
    """Publication-quality 3-panel figure: angles, force, cart position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    # control ticks for force align with t_log[:-1]
    tu = t_log[:-1]
    # wrap angles to (-pi, pi] for display, in degrees
    ang = (x_log[:, 1:1 + n] + np.pi) % (2 * np.pi) - np.pi
    ang_deg = np.rad2deg(ang)

    plt.rcParams.update({
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 120, "savefig.dpi": 150,
    })
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # (a) link angles
    ax = axes[0]
    colors = ["#1b6ca8", "#e07a5f", "#3d9970", "#9b59b6", "#d4ac0d", "#c0392b"]
    for i in range(n):
        ax.plot(t_log, ang_deg[:, i], color=colors[i % len(colors)], lw=1.4,
                label=f"link {i + 1}")
    ax.axhline(180, color="gray", ls=":", lw=0.8)
    ax.axhline(-180, color="gray", ls=":", lw=0.8)
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvspan(horizon, t_log[-1], color="green", alpha=0.06)
    ax.axvline(horizon, color="green", ls="--", lw=1.0, alpha=0.7,
               label="handoff -> static LQR")
    ax.set_ylabel("link angle (deg)\n(0 = upright)")
    ax.set_title("Sextuple inverted pendulum on a cart: closed-loop "
                 "swing-up + 5 s balance (n=6, ~22 N peak / 60 N bound)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    # (b) cart force
    ax = axes[1]
    ax.plot(tu, u_log, color="#9b59b6", lw=1.0)
    ax.axhline(force_bound, color="red", ls="--", lw=1.0,
               label=f"+/- {force_bound:.0f} N bound")
    ax.axhline(-force_bound, color="red", ls="--", lw=1.0)
    peak = float(np.max(np.abs(u_log)))
    ax.set_ylim(-force_bound * 1.1, force_bound * 1.1)
    ax.set_ylabel("cart force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.01, 0.04, f"peak |force| = {peak:.1f} N",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

    # (c) cart position
    ax = axes[2]
    ax.plot(t_log, x_log[:, 0], color="#2c3e50", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_ylabel("cart position (m)")
    ax.set_xlabel("time (s)")

    fig.tight_layout()
    path = OUT / "demo_sextuple_plots.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    # Also a zoomed force panel to show how far inside the bound we sit.
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(tu, u_log, color="#9b59b6", lw=1.0)
    pad = max(5.0, peak * 1.25)
    ax2.set_ylim(-pad, pad)
    ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("cart force (N)")
    ax2.set_title(f"Cart force stays within +/-{peak:.0f} N "
                  f"(60 N validation bound never approached)")
    fig2.tight_layout()
    path2 = OUT / "demo_sextuple_force_zoom.png"
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)

    return [str(path), str(path2)]


def _save_animation(t_log, x_log, model, horizon):
    """Light matplotlib GIF of the sextuple swinging up and balancing.

    Subsamples to ~25 fps so the GIF is small and the render stays light.
    Returns the path, or None if no writer is available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:  # pragma: no cover
        print(f"[demo] animation skipped (matplotlib import: {exc})", flush=True)
        return None

    n = model.n
    lengths = np.asarray(model.spec.link_lengths_m, dtype=float)

    # Subsample to ~25 fps.
    fps = 25
    dt = t_log[1] - t_log[0]
    stride = max(1, int(round((1.0 / fps) / dt)))
    idx = np.arange(0, len(t_log), stride)
    frames = len(idx)

    total_len = float(np.sum(lengths))
    xs_cart = x_log[:, 0]
    xmin = float(np.min(xs_cart)) - total_len - 0.3
    xmax = float(np.max(xs_cart)) + total_len + 0.3
    ymax = total_len + 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.6", lw=1.0)
    ax.set_title("Sextuple inverted pendulum: closed-loop swing-up + balance")

    cart_w, cart_h = 0.4, 0.2
    cart = plt.Rectangle((0, 0), cart_w, cart_h, fc="#2c3e50", ec="k", zorder=5)
    ax.add_patch(cart)
    (line,) = ax.plot([], [], "-o", color="#e07a5f", lw=3, ms=6, zorder=6)
    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle="round", fc="white", ec="0.7",
                                 alpha=0.85))

    def link_points(state):
        xc = state[0]
        thetas = state[1:1 + n]  # absolute angles from upright
        px, py = [xc], [0.0]
        x, y = xc, 0.0
        for i in range(n):
            # theta measured from upright (0 = straight up). Pendulum tip:
            x = x + lengths[i] * np.sin(thetas[i])
            y = y + lengths[i] * np.cos(thetas[i])
            px.append(x)
            py.append(y)
        return px, py

    def init():
        line.set_data([], [])
        time_txt.set_text("")
        return line, cart, time_txt

    def update(fi):
        k = idx[fi]
        st = x_log[k]
        cart.set_xy((st[0] - cart_w / 2, -cart_h / 2))
        px, py = link_points(st)
        line.set_data(px, py)
        phase = "swing-up" if t_log[k] < horizon else "BALANCE (static LQR)"
        time_txt.set_text(f"t = {t_log[k]:5.2f} s   [{phase}]")
        col = "#3d9970" if t_log[k] >= horizon else "#e07a5f"
        line.set_color(col)
        return line, cart, time_txt

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=True, interval=1000 / fps)
    path = OUT / "demo_sextuple.gif"
    try:
        anim.save(str(path), writer=PillowWriter(fps=fps))
    except Exception as exc:  # pragma: no cover
        print(f"[demo] animation skipped (writer: {exc})", flush=True)
        plt.close(fig)
        return None
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    run_demo()
