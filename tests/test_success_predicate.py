"""The public final-hold predicate measures elapsed simulator time."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.predicate import evaluate_success_predicate


def test_five_second_hold_needs_5001_one_millisecond_samples() -> None:
    model = NLinkCartPole(CartPoleSpec())
    upright = model.x_equilibrium("up")
    short_states = np.repeat(upright[None, :], 5000, axis=0)
    short = evaluate_success_predicate(model, short_states, np.zeros(4999))
    assert short["final_hold_s"] == 4.999
    assert not short["success"]

    exact_states = np.repeat(upright[None, :], 5001, axis=0)
    exact = evaluate_success_predicate(model, exact_states, np.zeros(5000))
    assert exact["final_hold_s"] == 5.0
    assert exact["success"]
