from __future__ import annotations

import numpy as np

from n14_cartpole.success import N_LINKS, in_success_set


def upright_state() -> np.ndarray:
    return np.zeros(2 * (N_LINKS + 1), dtype=np.float64)


def test_success_predicate_accepts_inclusive_boundaries() -> None:
    state = upright_state()
    state[0] = 2.0
    state[1] = np.deg2rad(5.0)
    state[N_LINKS + 1] = 0.5
    state[N_LINKS + 2] = 0.5
    assert in_success_set(state)


def test_success_predicate_rejects_each_exceeded_boundary() -> None:
    indexes_and_values = [
        (0, 2.0 + 1e-12),
        (1, np.deg2rad(5.0) + 1e-12),
        (N_LINKS + 1, 0.5 + 1e-12),
        (N_LINKS + 2, 0.5 + 1e-12),
    ]
    for index, value in indexes_and_values:
        state = upright_state()
        state[index] = value
        assert not in_success_set(state)


def test_success_predicate_wraps_angles() -> None:
    state = upright_state()
    state[1] = 2.0 * np.pi - np.deg2rad(4.0)
    assert in_success_set(state)
