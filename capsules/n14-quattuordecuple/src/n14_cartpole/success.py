"""Locked N14 success-set predicate."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

N_LINKS = 14
MAX_WRAPPED_LINK_ANGLE_DEG = 5.0
MAX_LINK_RATE_RAD_S = 0.5
MAX_ABS_CART_M = 2.0
MAX_ABS_CART_RATE_M_S = 0.5


def wrap_to_pi(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap angles to ``[-pi, pi)``."""
    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def in_success_set(state: npt.ArrayLike) -> bool:
    """Return whether one N14 state satisfies every locked success bound."""
    value = np.asarray(state, dtype=np.float64).reshape(-1)
    if value.shape != (2 * (N_LINKS + 1),) or not np.isfinite(value).all():
        return False
    angles = wrap_to_pi(value[1 : N_LINKS + 1])
    return bool(
        abs(float(value[0])) <= MAX_ABS_CART_M
        and abs(float(value[N_LINKS + 1])) <= MAX_ABS_CART_RATE_M_S
        and np.max(np.abs(angles)) <= np.deg2rad(MAX_WRAPPED_LINK_ANGLE_DEG)
        and np.max(np.abs(value[N_LINKS + 2 :])) <= MAX_LINK_RATE_RAD_S
    )
