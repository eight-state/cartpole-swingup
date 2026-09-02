"""The release predicate measures elapsed intervals, never sample count."""
from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.verification import evaluate_success_predicate


def _upright_log(model: NLinkCartPole, samples: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.repeat(model.x_equilibrium("up")[None, :], samples, axis=0)
    return states, np.zeros(samples - 1)


def test_predicate_requires_5001_in_set_1ms_samples() -> None:
    model = NLinkCartPole(CartPoleSpec().with_n_links(1))
    states, controls = _upright_log(model, 5000)
    short = evaluate_success_predicate(model, states, controls, hold_time_s=5.0)
    assert short["tail_hold_s"] == 4.999
    assert short["tail_hold_samples"] == 5000
    assert not short["success"]

    states, controls = _upright_log(model, 5001)
    exact = evaluate_success_predicate(model, states, controls, hold_time_s=5.0)
    assert exact["tail_hold_s"] == 5.0
    assert exact["tail_hold_samples"] == 5001
    assert exact["success"]
