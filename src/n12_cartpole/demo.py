"""Render the deterministic N12 unperturbed rollout as a GIF."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error
from n12_cartpole.fast_pieces import FastDTVLQR, make_densifier
from n12_cartpole.success import N_LINKS, in_success_set
from n12_cartpole.verifier import (
    ARTIFACT_PATH,
    CONTROL_DT_S,
    FORCE_BOUND_N,
    RK4_SUBSTEP_S,
    SWITCH_TICK,
    TOTAL_TICKS,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY / "runs" / "r2" / "demo_n12.gif"
FRAME_STEP_TICKS = 50
FRAME_DURATION_MS = 50
CANVAS_SIZE = (720, 500)
PALETTE = (
    (244, 246, 248),
    (35, 43, 54),
    (29, 96, 145),
    (199, 83, 57),
    (66, 137, 79),
    (196, 150, 45),
    (165, 173, 185),
)


def _rollout() -> tuple[NLinkCartPole, np.ndarray, np.ndarray, np.ndarray]:
    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1] * N_LINKS,
        link_lengths_m=[0.5] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND_N,
        track_half_length_m=10.0,
        control_rate_hz=1000.0,
        rk4_max_step_s=RK4_SUBSTEP_S,
    )
    model = NLinkCartPole(spec)
    with np.load(ARTIFACT_PATH, allow_pickle=False) as data:
        source_states = np.asarray(data["x"], dtype=float)
        source_controls = np.asarray(data["u"], dtype=float).reshape(-1)
    dense_states, dense_controls = make_densifier(
        model, CONTROL_DT_S, 4, 4, len(source_controls)
    )(source_states, source_controls)
    tracking_q = make_Q(N_LINKS)
    tracking_q[N_LINKS + 2 :, N_LINKS + 2 :] *= 0.25
    _, tracking_terminal_p = static_lqr(model, Q=tracking_q, R=make_R())
    tracker = FastDTVLQR(
        model,
        dense_states,
        dense_controls,
        CONTROL_DT_S,
        Qf=tracking_terminal_p,
        Q=tracking_q,
        R=make_R(),
    )
    static_gain, _ = static_lqr(model)
    static_gain = np.asarray(static_gain).reshape(-1)
    upright = model.x_equilibrium("up")

    def live_policy(state: np.ndarray, time_s: float) -> float:
        tick = int(round(time_s / CONTROL_DT_S))
        if tick < SWITCH_TICK:
            return float(tracker.policy(state, time_s))
        return -float(static_gain @ wrap_state_error(state, upright, N_LINKS))

    times, states, applied_forces = model.rollout_zoh(
        model.x_equilibrium("down"),
        live_policy,
        TOTAL_TICKS * CONTROL_DT_S,
        CONTROL_DT_S,
        RK4_SUBSTEP_S,
    )
    if not all(in_success_set(model, state) for state in states[SWITCH_TICK:]):
        raise RuntimeError("demo rollout left the locked success set during static hold")
    if np.max(np.abs(applied_forces)) > FORCE_BOUND_N:
        raise RuntimeError("demo rollout exceeded the applied force bound")
    return model, times, states, applied_forces


def _palette_bytes() -> bytes:
    values = [component for color in PALETTE for component in color]
    values.extend([0] * (768 - len(values)))
    return bytes(values)


def _link_points(model: NLinkCartPole, state: np.ndarray) -> list[tuple[int, int]]:
    center_x = CANVAS_SIZE[0] // 2
    cart_y = 420
    pixels_per_metre = 36.0
    link_length = model.spec.link_lengths_m[0] * pixels_per_metre
    x_coordinate = center_x + state[0] * pixels_per_metre
    y_coordinate = float(cart_y)
    points = [(round(x_coordinate), round(y_coordinate))]
    for link_index in range(model.n):
        angle = state[1 + link_index]
        x_coordinate += link_length * np.sin(angle)
        y_coordinate -= link_length * np.cos(angle)
        points.append((round(x_coordinate), round(y_coordinate)))
    return points


def _frame(
    model: NLinkCartPole,
    state: np.ndarray,
    applied_force: float,
    tick: int,
) -> Image.Image:
    image = Image.new("P", CANVAS_SIZE, color=0)
    image.putpalette(_palette_bytes())
    draw = ImageDraw.Draw(image)
    width, _ = CANVAS_SIZE
    ground_y = 420
    draw.line((0, ground_y, width, ground_y), fill=6, width=2)
    points = _link_points(model, state)
    cart_x, cart_y = points[0]
    phase_colour = 3 if tick < SWITCH_TICK else 4
    draw.line(points, fill=phase_colour, width=4)
    for point_x, point_y in points[1:]:
        draw.ellipse((point_x - 4, point_y - 4, point_x + 4, point_y + 4), fill=1)
    draw.rectangle((cart_x - 18, cart_y - 10, cart_x + 18, cart_y + 10), fill=2)
    draw.ellipse((cart_x - 15, cart_y + 8, cart_x - 5, cart_y + 18), fill=1)
    draw.ellipse((cart_x + 5, cart_y + 8, cart_x + 15, cart_y + 18), fill=1)
    arrow_end = round(cart_x + applied_force)
    force_colour = 5 if abs(applied_force) > 149.999 else 1
    draw.line((cart_x, cart_y + 42, arrow_end, cart_y + 42), fill=force_colour, width=3)
    arrow_direction = 1 if applied_force >= 0 else -1
    draw.polygon(
        [
            (arrow_end, cart_y + 42),
            (arrow_end - 8 * arrow_direction, cart_y + 36),
            (arrow_end - 8 * arrow_direction, cart_y + 48),
        ],
        fill=force_colour,
    )
    return image


def render(output: Path) -> Path:
    """Regenerate the committed GIF from the frozen nominal and live policy."""
    model, _, states, applied_forces = _rollout()
    frames = [
        _frame(
            model,
            states[tick],
            applied_forces[min(tick, len(applied_forces) - 1)],
            tick,
        )
        for tick in range(0, len(states), FRAME_STEP_TICKS)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output


def cli(argv: Sequence[str] | None = None) -> int:
    """Render the N12 release demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPOSITORY / args.output
    rendered = render(output)
    try:
        display_path = rendered.relative_to(REPOSITORY)
    except ValueError:
        display_path = rendered
    print(display_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
