"""Unit tests for cartpole_capsules.core.predicate."""

from __future__ import annotations

import numpy as np
import pytest

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec
from cartpole_capsules.core.predicate import (
    evaluate_success_predicate,
    in_success_set,
    longest_hold_s,
    trailing_hold_s,
)


@pytest.fixture(scope="module")
def model() -> NLinkCartPole:
    spec = CartPoleSpec(
        n_links=1,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        damping_links_n_m_s_rad=[0.0],
    )
    return NLinkCartPole(spec)


def make_state(model: NLinkCartPole, **overrides) -> np.ndarray:
    """Upright rest state with named components overridden."""
    x = model.x_equilibrium("up")
    values = {
        "x": 0.0,
        "theta": 0.0,
        "xdot": 0.0,
        "thetadot": 0.0,
    } | overrides
    x[0] = values["x"]
    x[1] = values["theta"]
    x[2] = values["xdot"]
    x[3] = values["thetadot"]
    return x


def test_upright_rest_state_is_in_set(model: NLinkCartPole):
    assert in_success_set(model, make_state(model))


def test_angle_tolerance_uses_wrapping(model: NLinkCartPole):
    """Angles beyond one turn wrap back into the hold set."""
    x = make_state(model, theta=2.0 * np.pi + 0.05)  # wraps to 0.05 rad
    assert in_success_set(model, x)
    x = make_state(model, theta=-2.0 * np.pi - 0.05)  # wraps to -0.05 rad
    assert in_success_set(model, x)
    x = make_state(model, theta=0.1)  # over the 5 deg = 0.0873 rad limit
    assert not in_success_set(model, x)


def test_rate_and_cart_limits(model: NLinkCartPole):
    assert not in_success_set(model, make_state(model, thetadot=0.6))
    assert in_success_set(model, make_state(model, thetadot=0.4))
    assert not in_success_set(model, make_state(model, x=2.5))
    assert in_success_set(model, make_state(model, x=1.5))
    assert not in_success_set(model, make_state(model, xdot=0.6))
    assert in_success_set(model, make_state(model, xdot=0.4))


def test_tolerance_overrides(model: NLinkCartPole):
    x = make_state(model, theta=0.1)
    assert not in_success_set(model, x)
    assert in_success_set(model, x, theta_tol=0.2)


def test_trailing_hold_counts_suffix(model: NLinkCartPole):
    good = make_state(model)
    bad = make_state(model, theta=0.5)
    states = np.stack([good, bad, good, good, good])
    # 3 trailing samples span 2 intervals at 1 ms.
    assert trailing_hold_s(model, states, 0.001) == pytest.approx(0.002)


def test_longest_hold_ignores_later_failures(model: NLinkCartPole):
    good = make_state(model)
    bad = make_state(model, theta=0.5)
    # Longest run is 3 samples (0.002 s) even though the tail is in-set too.
    states = np.stack([good, good, good, bad, good])
    assert longest_hold_s(model, states, 0.001) == pytest.approx(0.002)
    # Trailing metric sees only the final single-sample run: 0 s.
    assert trailing_hold_s(model, states, 0.001) == 0.0


def test_empty_and_all_bad_states(model: NLinkCartPole):
    bad = make_state(model, theta=0.5)
    states = np.stack([bad, bad])
    assert trailing_hold_s(model, states, 0.001) == 0.0
    assert longest_hold_s(model, states, 0.001) == 0.0


def test_evaluate_success_predicate_gates(model: NLinkCartPole):
    spec = CartPoleSpec(
        n_links=1,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        damping_links_n_m_s_rad=[0.0],
    )
    m = NLinkCartPole(spec)
    n_states = 11  # 10 transitions, 10 ms
    states = np.tile(m.x_equilibrium("up"), (n_states, 1))
    controls = np.zeros(n_states - 1)
    result = evaluate_success_predicate(m, states, controls, hold_time_s=0.005)
    assert result["success"] is True
    assert result["final_hold_s"] == pytest.approx(0.01)
    assert result["final_hold_samples"] == n_states
    assert result["track_ok"] is True and result["force_ok"] is True

    # Break the force bound (spec bound is 150 N).
    result = evaluate_success_predicate(m, states, np.full(n_states - 1, 200.0))
    assert result["force_ok"] is False and result["success"] is False

    # Break the track bound.
    off_track = states.copy()
    off_track[:, 0] = 20.0
    result = evaluate_success_predicate(m, off_track, controls)
    assert result["track_ok"] is False and result["success"] is False

    # Insufficient trailing hold.
    states_short = np.tile(m.x_equilibrium("up"), (5, 1))
    result = evaluate_success_predicate(m, states_short, np.zeros(4), hold_time_s=0.005)
    assert result["success"] is False


def test_evaluate_success_predicate_validates_shapes(model: NLinkCartPole):
    states = np.zeros((5, model.nx))
    with pytest.raises(ValueError, match="span state transitions"):
        evaluate_success_predicate(model, states, np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        evaluate_success_predicate(model, np.zeros((5, 3)), np.zeros(4))
