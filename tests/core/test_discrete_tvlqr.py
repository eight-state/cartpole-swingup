"""Unit tests for cartpole_capsules.core.discrete_tvlqr. Tiny nominals only."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from cartpole_capsules.core.discrete_tvlqr import DiscreteTVLQR, zoh_ab
from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec


@pytest.fixture(scope="module")
def n1_model() -> NLinkCartPole:
    spec = CartPoleSpec(
        n_links=1,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        damping_links_n_m_s_rad=[0.0],
    )
    return NLinkCartPole(spec)


def test_zoh_ab_matches_block_expm(n1_model: NLinkCartPole):
    """Reference construction of the exact ZOH discretization."""
    model = n1_model
    x = model.x_equilibrium("up")
    dt = 0.001
    A, B = model.linearize(x, 0.0)
    Ad, Bd = zoh_ab(model, x, 0.0, dt)
    nx = A.shape[0]
    ref = sla.expm(np.block([[A * dt, (B * dt)], [np.zeros((1, nx + 1))]]))
    assert np.allclose(Ad, ref[:nx, :nx])
    assert np.allclose(Bd, ref[:nx, nx])


def test_zoh_ab_continuous_limit(n1_model: NLinkCartPole):
    """For small dt, (Ad - I)/dt approximates A and Bd/dt approximates B."""
    model = n1_model
    x = model.x_equilibrium("up")
    A, B = model.linearize(x, 0.0)
    dt = 1e-5
    Ad, Bd = zoh_ab(model, x, 0.0, dt)
    # First-order consistency: error stays O(dt) against the exact matrices.
    assert np.max(np.abs((Ad - np.eye(A.shape[0])) / dt - A)) < 1e-3
    assert np.max(np.abs(Bd / dt - B.reshape(-1))) < 1e-3


def test_requires_one_more_state_than_controls(n1_model: NLinkCartPole):
    model = n1_model
    states = np.zeros((5, model.nx))
    controls = np.zeros(5)
    with pytest.raises(ValueError, match="one more state"):
        DiscreteTVLQR(model, states, controls, 0.001)


def test_policy_returns_nominal_on_nominal(n1_model: NLinkCartPole):
    model = n1_model
    x_up = model.x_equilibrium("up")
    states = np.stack([x_up, x_up, x_up])
    controls = np.array([0.0, 0.0])
    tracker = DiscreteTVLQR(model, states, controls, 0.001)
    assert tracker.policy(x_up, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert tracker.policy(x_up, 0.0015) == pytest.approx(0.0, abs=1e-9)


def test_shapes_and_monodromy_short_horizon(n1_model: NLinkCartPole):
    """Per-tick closed loop is nearly neutral near upright; the full-horizon
    contraction gate is a per-rung property of the shipped nominal (the n10
    release gate asserts rho < 1 on its banked trajectory). Here we check the
    structures and the one-step Riccati identity, not a contraction claim."""
    model = n1_model
    x_up = model.x_equilibrium("up")
    n_ticks = 20
    states = np.tile(x_up, (n_ticks + 1, 1))
    controls = np.zeros(n_ticks)
    tracker = DiscreteTVLQR(model, states, controls, 0.001)
    assert tracker.gains.shape == (n_ticks, model.nx)
    assert tracker.ad.shape == (n_ticks, model.nx, model.nx)
    assert tracker.bd.shape == (n_ticks, model.nx)
    rho = tracker.monodromy()
    assert rho > 0.0


def test_riccati_identity_single_tick(n1_model: NLinkCartPole):
    """One backward step must satisfy S = Q + R kk' + Acl' Qf Acl."""
    from cartpole_capsules.core.lqr import make_Q, static_lqr

    model = n1_model
    x_up = model.x_equilibrium("up")
    _, qf = static_lqr(model)
    q = make_Q(1)
    r_scalar = 0.02
    tracker = DiscreteTVLQR(model, np.stack([x_up, x_up]), np.zeros(1), 0.001, qf=qf, q=q)
    k = tracker.gains[0]
    a_cl = tracker.ad[0] - np.outer(tracker.bd[0], k)
    rhs = q + r_scalar * np.outer(k, k) + a_cl.T @ qf @ a_cl
    assert np.allclose(tracker.initial_cost, rhs, atol=1e-6)


def test_policy_feedback_pulls_toward_nominal(n1_model: NLinkCartPole):
    """A tilted state demands force in the stabilizing direction."""
    model = n1_model
    x_up = model.x_equilibrium("up")
    states = np.tile(x_up, (3, 1))
    tracker = DiscreteTVLQR(model, states, np.zeros(2), 0.001)
    tilted = x_up.copy()
    tilted[1] = 0.05
    demand = tracker.policy(tilted, 0.0)
    # Positive tilt (link leaning right) needs positive cart force.
    assert demand > 0.0
    # Tick index clamps to the last tick for late times.
    late = tracker.policy(tilted, 10.0)
    last = float(tracker.controls[-1] - tracker.gains[-1] @ _wrap(model, tilted))
    assert late == pytest.approx(last)


def _wrap(model: NLinkCartPole, x: np.ndarray) -> np.ndarray:
    from cartpole_capsules.core.lqr import wrap_state_error

    return wrap_state_error(x, model.x_equilibrium("up"), model.n)
