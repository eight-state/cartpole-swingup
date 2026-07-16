"""The authoritative elapsed-time success predicate for the n=8 rollout."""
from __future__ import annotations

from typing import Any

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error

ANGLE_TOLERANCE_RAD = float(np.deg2rad(5.0))
ANGULAR_RATE_TOLERANCE_RAD_S = 0.5
CART_POSITION_TOLERANCE_M = 2.0
CART_SPEED_TOLERANCE_M_S = 0.5


def in_hold_set(model: NLinkCartPole, state: np.ndarray) -> bool:
    """Return whether one state satisfies every upright hold-set limit."""
    error = wrap_state_error(state, model.x_equilibrium("up"), model.n)
    angles = error[1 : 1 + model.n]
    angular_rates = state[model.n + 2 :]
    return bool(
        np.all(np.abs(angles) <= ANGLE_TOLERANCE_RAD)
        and np.all(np.abs(angular_rates) <= ANGULAR_RATE_TOLERANCE_RAD_S)
        and abs(state[0]) <= CART_POSITION_TOLERANCE_M
        and abs(state[model.n + 1]) <= CART_SPEED_TOLERANCE_M_S
    )


def evaluate_success_predicate(
    model: NLinkCartPole,
    states: np.ndarray,
    applied_controls: np.ndarray,
    hold_time_s: float = 5.0,
) -> dict[str, Any]:
    """Evaluate the sole release predicate from a saturated simulator log.

    States include both endpoints and controls span their intervening intervals.
    A suffix of ``N`` in-set samples covers ``(N - 1) * dt`` elapsed seconds,
    so a 5.0 s hold at 1 ms needs 5001 consecutive samples.
    """
    states = np.asarray(states, dtype=float)
    applied_controls = np.asarray(applied_controls, dtype=float).reshape(-1)
    if states.ndim != 2 or states.shape[1] != model.nx or len(states) == 0:
        raise ValueError(f"states must have shape (ticks, {model.nx})")
    if len(applied_controls) != len(states) - 1:
        raise ValueError("applied_controls must have one entry per state transition")

    held = np.fromiter((in_hold_set(model, state) for state in states), dtype=bool)
    suffix_samples = 0
    for value in held[::-1]:
        if not value:
            break
        suffix_samples += 1

    spec = model.spec
    elapsed_hold_s = max(0, suffix_samples - 1) * spec.control_dt_s
    max_force = float(np.max(np.abs(applied_controls))) if len(applied_controls) else 0.0
    max_cart_position = float(np.max(np.abs(states[:, 0])))
    track_ok = max_cart_position <= spec.track_half_length_m
    force_ok = max_force <= spec.force_bound_n + 1e-6
    return {
        "success": bool(track_ok and force_ok and elapsed_hold_s >= hold_time_s - 1e-9),
        "tail_hold_s": elapsed_hold_s,
        "tail_hold_samples": suffix_samples,
        "max_applied_force_n": max_force,
        "min_track_margin_m": float(spec.track_half_length_m - max_cart_position),
        "track_ok": track_ok,
        "force_ok": force_ok,
        "final_state": states[-1].tolist(),
    }
