"""Render one freshly recomputed N12 rollout to an ignored local GIF."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cartpole_race.dynamics import NLinkCartPole
from n12_cartpole.simulator import (
    FORCE_BOUND_N,
    N_LINKS,
    REPOSITORY,
    SWITCH_TICK,
    run_live_rollout,
)
from n12_cartpole.success import in_success_set

DEFAULT_OUTPUT = REPOSITORY / ".working" / "n12-demo.gif"
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


def _palette_bytes() -> bytes:
    values = [component for colour in PALETTE for component in colour]
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
    for index in range(N_LINKS):
        angle = state[1 + index]
        x_coordinate += link_length * np.sin(angle)
        y_coordinate -= link_length * np.cos(angle)
        points.append((round(x_coordinate), round(y_coordinate)))
    return points


def _frame(
    model: NLinkCartPole, state: np.ndarray, applied_force: float, tick: int
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
    force_colour = 5 if abs(applied_force) > FORCE_BOUND_N - 1e-3 else 1
    draw.line((cart_x, cart_y + 42, arrow_end, cart_y + 42), fill=force_colour, width=3)
    direction = 1 if applied_force >= 0 else -1
    draw.polygon(
        [
            (arrow_end, cart_y + 42),
            (arrow_end - 8 * direction, cart_y + 36),
            (arrow_end - 8 * direction, cart_y + 48),
        ],
        fill=force_colour,
    )
    return image


def render(output: Path) -> Path:
    """Run the shared live stack and render its applied-force trajectory."""
    rollout = run_live_rollout()
    if not all(
        in_success_set(rollout.model, state)
        for state in rollout.states[SWITCH_TICK:]
    ):
        raise RuntimeError("live rollout left the locked success set during static hold")
    if np.max(np.abs(rollout.applied_forces)) > FORCE_BOUND_N:
        raise RuntimeError("live rollout exceeded the simulator force bound")
    frames = [
        _frame(
            rollout.model,
            rollout.states[tick],
            rollout.applied_forces[min(tick, len(rollout.applied_forces) - 1)],
            tick,
        )
        for tick in range(0, len(rollout.states), FRAME_STEP_TICKS)
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
    """Render N12 to ``.working/n12-demo.gif`` unless an output is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPOSITORY / args.output
    rendered = render(output)
    try:
        display_path = rendered.relative_to(REPOSITORY)
    except ValueError:
        display_path = rendered
    print(f"rendered {display_path.as_posix()} from the live N12 stack")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
