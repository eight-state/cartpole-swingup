"""Locked N12 upright success predicate."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error

N_LINKS = 12
MAX_WRAPPED_LINK_ANGLE_DEG = 5.0
MAX_LINK_RATE_RAD_S = 0.5
MAX_ABS_CART_M = 2.0
MAX_ABS_CART_RATE_M_S = 0.5


def in_success_set(model: NLinkCartPole, state: np.ndarray) -> bool:
    """Return whether ``state`` satisfies the locked upright hold predicate."""
    up = model.x_equilibrium("up")
    error = wrap_state_error(state, up, N_LINKS)
    angle_error = error[1 : N_LINKS + 1]
    link_rates = state[N_LINKS + 2 :]
    return bool(
        np.all(np.abs(angle_error) <= np.deg2rad(MAX_WRAPPED_LINK_ANGLE_DEG))
        and np.all(np.abs(link_rates) <= MAX_LINK_RATE_RAD_S)
        and abs(state[0]) <= MAX_ABS_CART_M
        and abs(state[N_LINKS + 1]) <= MAX_ABS_CART_RATE_M_S
    )
