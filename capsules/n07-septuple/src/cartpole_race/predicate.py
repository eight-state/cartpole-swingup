"""Locked n=7 upright success predicate."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error


def in_success_set(
    model: NLinkCartPole,
    x: np.ndarray,
    theta_tol: float = np.deg2rad(5.0),
    thetad_tol: float = 0.5,
    x_tol: float = 2.0,
    xdot_tol: float = 0.5,
) -> bool:
    """Return whether one state satisfies the locked upright hold set."""
    n = model.n
    x = np.asarray(x).reshape(-1)
    x_up = model.x_equilibrium("up")
    e = wrap_state_error(x, x_up, n)
    theta_err = e[1 : 1 + n]
    thetad = x[1 + n + 1 :]
    return bool(
        np.all(np.abs(theta_err) <= theta_tol)
        and np.all(np.abs(thetad) <= thetad_tol)
        and abs(x[0]) <= x_tol
        and abs(x[1 + n]) <= xdot_tol
    )


def final_hold_s(
    model: NLinkCartPole, states: np.ndarray, control_dt_s: float
) -> float:
    """Return the elapsed time spanned by final sampled 1 kHz in-set states."""
    samples = 0
    for state in np.asarray(states)[::-1]:
        if not in_success_set(model, state):
            break
        samples += 1
    return max(0, samples - 1) * control_dt_s
