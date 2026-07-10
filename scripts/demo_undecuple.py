"""Render the released N11 unperturbed swing-up and hold as a GIF.

    uv run python scripts/demo_undecuple.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "r2"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO / "configs")]

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from cartpole_race.funnels import in_success_set  # noqa: E402
from cartpole_race.lqr import static_lqr, wrap_state_error  # noqa: E402
from fast_pieces import FastDTVLQR  # noqa: E402
from nominal import NOMINAL  # noqa: E402

N_LINKS = 11
HOLD_S = 5.0
HOLD_WINDOW_S = 12.0
GIF_NAME = "demo_undecuple.gif"


def _trailing_success_seconds(model: NLinkCartPole, states: np.ndarray) -> float:
    run = 0
    for state in states:
        run = run + 1 if in_success_set(model, state) else 0
    return max(0, run - 1) * model.spec.control_dt_s


def main() -> int:
    spec = CartPoleSpec().with_n_links(N_LINKS)
    model = NLinkCartPole(spec)
    with np.load(NOMINAL.path, allow_pickle=False) as data:
        nominal_x = np.asarray(data["x"], dtype=float)
        nominal_u = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
    tracker = FastDTVLQR(model, nominal_x, nominal_u, spec.control_dt_s)
    force_bound = spec.force_bound_n

    def tracking_policy(state: np.ndarray, elapsed_s: float) -> float:
        return float(np.clip(tracker.policy(state, elapsed_s), -force_bound, force_bound))

    tracked_t, tracked_x, tracked_u = model.rollout_zoh(
        nominal_x[0],
        tracking_policy,
        horizon_s,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    x_up = model.x_equilibrium("up")
    hold_gain, _ = static_lqr(model)
    hold_gain = np.asarray(hold_gain).reshape(-1)

    def hold_policy(state: np.ndarray, _: float) -> float:
        return float(
            np.clip(
                -float(hold_gain @ wrap_state_error(state, x_up, N_LINKS)),
                -force_bound,
                force_bound,
            )
        )

    held_t, held_x, held_u = model.rollout_zoh(
        tracked_x[-1],
        hold_policy,
        HOLD_WINDOW_S,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    states = np.vstack([tracked_x, held_x[1:]])
    times = np.concatenate([tracked_t, horizon_s + held_t[1:]])
    controls = np.concatenate([tracked_u, held_u])
    hold_s = _trailing_success_seconds(model, held_x)
    passed = bool(
        hold_s >= HOLD_S
        and np.max(np.abs(states[:, 0])) <= spec.track_half_length_m
    )
    print(
        f"[demo n=11] hold {hold_s:.1f} s, peak force {np.max(np.abs(controls)):.6f} N "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        raise RuntimeError("demo rollout failed the release predicate")
    _save_gif(times, states, controls, model, horizon_s)
    return 0


def _save_gif(
    times: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    model: NLinkCartPole,
    horizon_s: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fps = 25
    frame_step = int(round(1.0 / (fps * 0.001)))
    frames = range(0, len(states), frame_step)
    link_length = model.spec.link_lengths_m[0]

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.5, 6.5)
    axis.set_ylim(-6.4, 6.6)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        x_coordinates = [float(state[0])]
        y_coordinates = [0.0]
        for index in range(model.n):
            x_coordinates.append(x_coordinates[-1] + link_length * np.sin(state[1 + index]))
            y_coordinates.append(y_coordinates[-1] + link_length * np.cos(state[1 + index]))
        return x_coordinates, y_coordinates

    def update(frame_index: int):
        state = states[frame_index]
        x_coordinates, y_coordinates = points(state)
        cart.set_data([x_coordinates[0]], [0.0])
        chain.set_data(x_coordinates, y_coordinates)
        elapsed_s = times[frame_index]
        phase = "swing-up" if elapsed_s < horizon_s else "balance"
        title.set_text(f"n=11 cart-pole: {phase}, t={elapsed_s:5.2f} s")
        control_index = min(frame_index, len(controls) - 1)
        force_text.set_text(f"force {controls[control_index]:+6.1f} N, |u| <= 150")
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    output = RUNS / GIF_NAME
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"[demo] saved {output.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
