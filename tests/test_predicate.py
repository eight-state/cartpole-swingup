"""The public five-second claim uses elapsed time, not sample count alone."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.release import load_release_spec
from cartpole_race.predicate import longest_continuous_hold_s


def test_success_predicate_measures_a_continuous_five_second_window() -> None:
    spec = load_release_spec()
    model = NLinkCartPole(spec)
    upright = model.x_equilibrium("up")
    states = np.repeat(upright[None, :], 5_001, axis=0)

    assert longest_continuous_hold_s(model, states, spec.control_dt_s) == 5.0
    states[2_500, 1] = np.deg2rad(6.0)
    assert longest_continuous_hold_s(model, states, spec.control_dt_s) == 2.499
