"""The single authoritative fresh n=7 rollout and renderer.

The committed dense nominal is an immutable input. This module rebuilds the
exact-ZOH discrete TVLQR and terminal static LQR, then runs the saturated plant
from the exact hanging equilibrium. It never synthesizes a nominal or reruns a
perturbation gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec, load_spec
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.predicate import final_hold_s

REPO = Path(__file__).resolve().parents[2]
WORKING = REPO / ".working"
CONFIG_PATH = REPO / "configs" / "env-base.yaml"
NOMINAL_PATH = REPO / "results" / "nom_n7_dense1ms.npz"
NOMINAL_SHA256 = "fe192b9eefb19540af782ef8163d1ce2b54ef76faf94cfa9e789c95c367c5b13"
N_LINKS = 7
HOLD_S = 5.0


@dataclass(frozen=True)
class ReleaseStack:
    """The loaded nominal and freshly rebuilt controllers."""

    model: NLinkCartPole
    states: np.ndarray
    controls: np.ndarray
    horizon_s: float
    tracker: DiscreteTVLQR
    static_policy: StaticLQRPolicy


@dataclass(frozen=True)
class LiveRun:
    """One fresh simulator trace and the metrics derived from it."""

    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    metrics: dict[str, Any]


def sha256(path: Path) -> str:
    """Return a raw-byte SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release_stack() -> ReleaseStack:
    """Load the one fixed nominal and rebuild the exact-ZOH controller stack."""
    if sha256(NOMINAL_PATH) != NOMINAL_SHA256:
        raise ValueError("dense nominal bytes do not match the released authority")

    spec: CartPoleSpec = load_spec(CONFIG_PATH)
    if spec.n_links != N_LINKS:
        raise ValueError(f"release config has {spec.n_links} links, expected {N_LINKS}")
    model = NLinkCartPole(spec)
    with np.load(NOMINAL_PATH, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(archive["horizon"])
    if states.shape != (8001, model.nx) or controls.shape != (8000,):
        raise ValueError("dense nominal shape is not the released 8,000-tick grid")
    if horizon_s != 8.0:
        raise ValueError("dense nominal horizon is not 8.0 seconds")

    tracker = DiscreteTVLQR(model, states, controls, spec.control_dt_s)
    static_gain, static_p = static_lqr(model)
    static_policy = StaticLQRPolicy(model, static_gain)
    static_policy.P = static_p
    return ReleaseStack(model, states, controls, horizon_s, tracker, static_policy)


def run_live(stack: ReleaseStack | None = None) -> LiveRun:
    """Run the authoritative unperturbed rollout through the live simulator."""
    stack = build_release_stack() if stack is None else stack
    model = stack.model
    spec = model.spec

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < stack.horizon_s:
            return float(
                np.clip(
                    stack.tracker.policy(state, time_s),
                    -spec.force_bound_n,
                    spec.force_bound_n,
                )
            )
        return stack.static_policy(state, time_s)

    times, states, controls = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        stack.horizon_s + HOLD_S + 1.0,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    handoff = states[len(stack.controls)]
    upright = model.x_equilibrium("up")
    angles = ((handoff[1 : 1 + N_LINKS] - upright[1 : 1 + N_LINKS] + np.pi) % (2 * np.pi)) - np.pi
    hold_s = final_hold_s(model, states, spec.control_dt_s)
    track_abs_max_m = float(np.max(np.abs(states[:, 0])))
    metrics: dict[str, Any] = {
        "format": "n7-live-demo-v1",
        "nominal_file": NOMINAL_PATH.name,
        "nominal_sha256": NOMINAL_SHA256,
        "n_links": N_LINKS,
        "horizon_s": stack.horizon_s,
        "control_ticks": len(stack.controls),
        "rho": stack.tracker.monodromy(),
        "swing_handoff_dev_deg": float(np.rad2deg(np.max(np.abs(angles)))),
        "swing_peak_force_n": float(np.max(np.abs(controls[: len(stack.controls)]))),
        "hold_peak_force_n": float(np.max(np.abs(controls[len(stack.controls) :]))),
        "track_abs_max_m": track_abs_max_m,
        "final_hold_s": hold_s,
        "success": bool(
            hold_s >= HOLD_S - 1e-9
            and track_abs_max_m <= spec.track_half_length_m
        ),
    }
    if not metrics["success"] or metrics["rho"] >= 1.0:
        raise RuntimeError("fresh n=7 rollout failed the released predicate")
    return LiveRun(times, states, controls, metrics)


def render(run: LiveRun, model: NLinkCartPole, horizon_s: float, output: Path) -> None:
    """Render the supplied fresh trace without running another simulation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    step = int(round(1.0 / (fps * model.spec.control_dt_s)))
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.2, 6.2)
    axis.set_ylim(-4.0, 4.2)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        xs = [float(state[0])]
        ys = [0.0]
        length = model.spec.link_lengths_m[0]
        for link in range(model.n):
            xs.append(xs[-1] + length * np.sin(state[1 + link]))
            ys.append(ys[-1] + length * np.cos(state[1 + link]))
        return xs, ys

    def update(frame: int):
        xs, ys = points(run.states[frame])
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        time_s = run.times[frame]
        phase = "swing-up" if time_s < horizon_s else "balance"
        title.set_text(f"n={model.n} cart-pole — {phase}  t={time_s:5.2f} s")
        control_index = min(frame, len(run.controls) - 1)
        force_text.set_text(
            f"force {run.controls[control_index]:+6.1f} N (|u|<={model.spec.force_bound_n:g})"
        )
        return cart, chain, title, force_text

    animation = FuncAnimation(
        figure, update, frames=range(0, len(run.states), step), blit=False
    )
    animation.save(str(output), writer=PillowWriter(fps=fps))
    plt.close(figure)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a release output beneath the ignored working directory."""
    if WORKING not in path.resolve().parents:
        raise ValueError("release outputs must be written under .working/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def demo_main() -> int:
    """Run and render the single fresh release rollout."""
    stack = build_release_stack()
    run = run_live(stack)
    output_dir = WORKING / "n7-demo"
    render(run, stack.model, stack.horizon_s, output_dir / "n7-demo.gif")
    write_json(output_dir / "live-metrics.json", run.metrics)
    print(
        f"[n7-demo] PASS rho={run.metrics['rho']:.4g} "
        f"handoff={run.metrics['swing_handoff_dev_deg']:.4f}deg "
        f"hold={run.metrics['final_hold_s']:.3f}s"
    )
    return 0
