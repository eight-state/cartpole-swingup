"""Sampled upright success predicate with both release-locked hold metrics.

One predicate, two hold measurements over 1 kHz sampled states:

- ``trailing_hold_s``: elapsed time spanned by the final consecutive in-set
  suffix (n05/n06/n07/n08/n12 release semantics).
- ``longest_hold_s``: elapsed time of the longest consecutive in-set run
  (n09/n10/n11 release semantics).

Locked hold set: wrapped angles within 5 deg, angular rates within 0.5 rad/s,
cart within 2.0 m, cart speed within 0.5 m/s. A run of ``N`` in-set samples
spans ``(N - 1)`` control intervals, so a five second hold at 1 kHz requires
5,001 consecutive samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.lqr import wrap_state_error

ANGLE_TOLERANCE_RAD = float(np.deg2rad(5.0))
ANGULAR_RATE_TOLERANCE_RAD_S = 0.5
CART_POSITION_TOLERANCE_M = 2.0
CART_SPEED_TOLERANCE_M_S = 0.5


def in_success_set(
    model: NLinkCartPole,
    state: np.ndarray,
    theta_tol: float = ANGLE_TOLERANCE_RAD,
    theta_rate_tol: float = ANGULAR_RATE_TOLERANCE_RAD_S,
    cart_position_tol: float = CART_POSITION_TOLERANCE_M,
    cart_rate_tol: float = CART_SPEED_TOLERANCE_M_S,
) -> bool:
    """Return whether one state satisfies every upright hold-set limit.

    Angle errors are wrapped to ``(-pi, pi]`` before comparison; the angular
    rates, cart position, and cart speed are compared directly.
    """
    n = model.n
    state = np.asarray(state, dtype=float).reshape(-1)
    error = wrap_state_error(state, model.x_equilibrium("up"), n)
    angles = error[1 : 1 + n]
    angular_rates = state[n + 2 :]
    return bool(
        np.all(np.abs(angles) <= theta_tol)
        and np.all(np.abs(angular_rates) <= theta_rate_tol)
        and abs(state[0]) <= cart_position_tol
        and abs(state[n + 1]) <= cart_rate_tol
    )


def trailing_hold_s(model: NLinkCartPole, states: np.ndarray, control_dt_s: float) -> float:
    """Elapsed time spanned by the final consecutive in-set sampled states."""
    samples = 0
    for state in np.asarray(states)[::-1]:
        if not in_success_set(model, state):
            break
        samples += 1
    return max(0, samples - 1) * control_dt_s


def longest_hold_s(model: NLinkCartPole, states: np.ndarray, control_dt_s: float) -> float:
    """Elapsed time of the longest consecutive in-set run across the log."""
    run = best = 0
    for state in np.asarray(states):
        run = run + 1 if in_success_set(model, state) else 0
        best = max(best, run)
    return max(0, best - 1) * control_dt_s


def evaluate_success_predicate(
    model: NLinkCartPole,
    states: np.ndarray,
    applied_controls: np.ndarray,
    hold_time_s: float = 5.0,
) -> dict[str, Any]:
    """Check full-track compliance and the elapsed trailing in-set hold.

    Also gates on the track half-length and the force bound (epsilon
    ``1e-9`` above the bound). Returns a dict with ``success``,
    ``final_hold_s`` (trailing semantics), ``final_hold_samples``,
    ``peak_applied_force_n``, ``peak_cart_m``, ``track_ok``, ``force_ok``.
    """
    states = np.asarray(states, dtype=float)
    applied_controls = np.asarray(applied_controls, dtype=float).reshape(-1)
    if states.ndim != 2 or states.shape[1] != model.nx:
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
