"""The public N6 predicate is sampled at 1 kHz; clipping occurs at the plant boundary."""

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.n6 import (
    CONTROL_DT_S,
    FORCE_BOUND_N,
    _fixed_spec,
    continuous_hold_s,
    in_success_set,
)


def test_final_hold_counts_intervals_not_samples() -> None:
    """Five seconds requires 5,001 consecutive 1 ms state samples."""
    assert continuous_hold_s(np.ones(5001, dtype=bool), CONTROL_DT_S) == 5.0
    assert continuous_hold_s(np.ones(5000, dtype=bool), CONTROL_DT_S) == 4.999
    assert continuous_hold_s(np.array([False, True, True]), CONTROL_DT_S) == 0.001


def test_predicate_and_saturated_zoh_boundary() -> None:
    """The upright state is in-set and applied force is clipped by rollout_zoh."""
    model = NLinkCartPole(_fixed_spec())
    assert in_success_set(model, model.x_equilibrium("up"))
    state = model.x_equilibrium("up")
    state[1] = np.deg2rad(5.1)
    assert not in_success_set(model, state)

    _, _, applied, raw = model.rollout_zoh(
        model.x_equilibrium("down"),
        lambda _state, _time: 2.0 * FORCE_BOUND_N,
        0.003,
        model.spec.control_dt_s,
        model.spec.rk4_max_step_s,
    )
    assert np.all(applied == FORCE_BOUND_N)
    assert np.all(raw == 2.0 * FORCE_BOUND_N)
