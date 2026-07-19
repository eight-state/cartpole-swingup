"""The five-second acceptance threshold uses 1 kHz logged-state samples."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import load_spec
from cartpole_race.n10_release import N10_SPEC_PATH
from cartpole_race.predicate import longest_continuous_hold_s


def test_five_seconds_requires_5001_one_millisecond_states() -> None:
    model = NLinkCartPole(load_spec(N10_SPEC_PATH))
    upright = model.x_equilibrium("up")
    short = np.repeat(upright[None, :], 5_000, axis=0)
    exact = np.repeat(upright[None, :], 5_001, axis=0)

    assert longest_continuous_hold_s(model, short, 0.001) == 4.999
    assert longest_continuous_hold_s(model, exact, 0.001) == 5.0
