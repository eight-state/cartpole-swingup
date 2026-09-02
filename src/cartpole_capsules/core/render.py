"""Render a rollout log to a GIF without touching frozen evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_cartpole_gif(
    output: Path,
    t_log: np.ndarray,
    x_log: np.ndarray,
    u_log: np.ndarray,
    *,
    n_links: int,
    link_length_m: float,
    swingup_horizon_s: float,
    force_bound_n: float,
) -> None:
    """Render one already-verified rollout to an explicit local output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    dt = float(t_log[1] - t_log[0])
    step = max(1, int(round(1.0 / (fps * dt))))
    frames = range(0, len(x_log), step)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-4.0, 4.2)
    ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=1)
    title = ax.set_title("")
    (cart,) = ax.plot([], [], "s", ms=14, color="#1f4e9c")
    (chain,) = ax.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    ftxt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        xs = [float(state[0])]
        ys = [0.0]
        for index in range(n_links):
            xs.append(xs[-1] + link_length_m * np.sin(state[1 + index]))
            ys.append(ys[-1] + link_length_m * np.cos(state[1 + index]))
        return xs, ys

    def update(frame: int):
        state = x_log[frame]
        xs, ys = points(state)
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        t = t_log[frame]
        phase = "swing-up" if t < swingup_horizon_s else "balance"
        title.set_text(f"n={n_links} cart-pole - {phase}  t={t:5.2f} s")
        control_index = min(frame, len(u_log) - 1)
        ftxt.set_text(f"applied force {u_log[control_index]:+6.1f} N (|u|<={force_bound_n:g})")
        return cart, chain, title, ftxt

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(str(output), writer=PillowWriter(fps=fps))
    plt.close(fig)
