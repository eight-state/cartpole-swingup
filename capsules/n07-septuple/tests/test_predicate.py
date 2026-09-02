"""Boundary tests for the sampled final-hold calculation."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.predicate import final_hold_s


def test_final_hold_spans_control_intervals_between_samples() -> None:
    model = NLinkCartPole(CartPoleSpec().with_n_links(1))
    upright = model.x_equilibrium("up")

    assert final_hold_s(model, np.repeat(upright[None, :], 5000, axis=0), 0.001) == 4.999
    assert final_hold_s(model, np.repeat(upright[None, :], 5001, axis=0), 0.001) == 5.0
