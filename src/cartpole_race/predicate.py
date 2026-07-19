"""The sole sampled success predicate for the N5 release."""

from __future__ import annotations

from typing import Any

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error

ANGLE_TOLERANCE_RAD = float(np.deg2rad(5.0))
ANGULAR_RATE_TOLERANCE_RAD_S = 0.5
CART_POSITION_TOLERANCE_M = 2.0
CART_SPEED_TOLERANCE_M_S = 0.5


def in_success_set(model: NLinkCartPole, state: np.ndarray) -> bool:
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
    """Check full-track compliance and the elapsed final in-set hold.

    A suffix of N in-set samples spans (N - 1) control intervals, so a five
    second hold at 1 kHz requires 5,001 consecutive samples.
    """
    states = np.asarray(states, dtype=float)
    applied_controls = np.asarray(applied_controls, dtype=float).reshape(-1)
    if states.ndim != 2 or states.shape != (len(states), model.nx):
        raise ValueError(f"states must have shape (ticks, {model.nx})")
    if len(applied_controls) != len(states) - 1:
        raise ValueError("applied_controls must span state transitions")

    held = np.fromiter((in_success_set(model, state) for state in states), dtype=bool)
    suffix_samples = 0
    for value in held[::-1]:
        if not value:
            break
        suffix_samples += 1

    elapsed_hold_s = max(0, suffix_samples - 1) * model.spec.control_dt_s
    peak_force_n = float(np.max(np.abs(applied_controls))) if len(applied_controls) else 0.0
    peak_cart_m = float(np.max(np.abs(states[:, 0])))
    track_ok = peak_cart_m <= model.spec.track_half_length_m
    force_ok = peak_force_n <= model.spec.force_bound_n + 1e-9
    return {
        "success": bool(track_ok and force_ok and elapsed_hold_s >= hold_time_s - 1e-9),
        "final_hold_s": elapsed_hold_s,
        "final_hold_samples": suffix_samples,
        "peak_applied_force_n": peak_force_n,
        "peak_cart_m": peak_cart_m,
        "track_ok": track_ok,
        "force_ok": force_ok,
    }
