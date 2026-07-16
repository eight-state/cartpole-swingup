"""Locked success predicate for the n=10 release."""

from __future__ import annotations

import numpy as np

from cartpole_race.lqr import wrap_state_error


def in_success_set(
    model,
    x: np.ndarray,
    theta_tol: float = np.deg2rad(5.0),
    thetad_tol: float = 0.5,
    x_tol: float = 2.0,
    xdot_tol: float = 0.5,
) -> bool:
    """Return whether one state is inside the release's upright hold set."""
    n = model.n
    state = np.asarray(x, dtype=float).reshape(-1)
    error = wrap_state_error(state, model.x_equilibrium("up"), n)
    return bool(
        np.all(np.abs(error[1 : 1 + n]) <= theta_tol)
        and np.all(np.abs(state[2 + n :]) <= thetad_tol)
        and abs(state[0]) <= x_tol
        and abs(state[1 + n]) <= xdot_tol
    )


def longest_continuous_hold_s(model, x_log: np.ndarray, dt_s: float) -> float:
    """Return the longest continuous time span inside :func:`in_success_set`."""
    run = best = 0
    for state in x_log:
        run = run + 1 if in_success_set(model, state) else 0
        best = max(best, run)
    return max(0, best - 1) * dt_s
