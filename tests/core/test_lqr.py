"""Unit tests for cartpole_capsules.core.lqr."""

from __future__ import annotations

import helpers
import numpy as np
import pytest

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec
from cartpole_capsules.core.lqr import (
    Q_ANG_VEL,
    Q_ANGLE,
    Q_CART_POS,
    Q_CART_VEL,
    R_STATIC,
    StaticLQRPolicy,
    make_Q,
    make_R,
    static_lqr,
    wrap_state_error,
    wrap_to_pi,
)

n1_spec = pytest.fixture(lambda: helpers.make_n1_spec())


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    # Convention: the interval is (-pi, pi], so +pi wraps to -pi.
    assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)
    assert wrap_to_pi(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_to_pi(-3.0 * np.pi) == pytest.approx(-np.pi)
    wrapped = np.asarray(wrap_to_pi(np.array([2.0 * np.pi, -2.0 * np.pi])))
    assert np.allclose(wrapped, [0.0, 0.0])


def test_wrap_state_error_wraps_only_angles(n1_spec: CartPoleSpec):
    x = np.array([1.5, np.pi + 0.2, 0.3, 0.4])
    ref = np.zeros(4)
    e = wrap_state_error(x, ref, 1)
    assert e[0] == pytest.approx(1.5)  # cart position: plain difference
    assert e[1] == pytest.approx(-np.pi + 0.2)  # angle: wrapped
    assert e[2] == pytest.approx(0.3)  # cart velocity: plain
    assert e[3] == pytest.approx(0.4)  # angular velocity: plain


def test_make_q_and_r_locked_values():
    Q = make_Q(2)
    assert np.allclose(np.diag(Q), [Q_CART_POS, Q_ANGLE, Q_ANGLE, Q_CART_VEL, Q_ANG_VEL, Q_ANG_VEL])
    assert Q_CART_POS == 1.0 and Q_ANGLE == 80.0
    assert Q_CART_VEL == 1.0 and Q_ANG_VEL == 5.0
    R = make_R()
    assert R.shape == (1, 1)
    assert R[0, 0] == R_STATIC == 0.02


def test_static_lqr_shapes(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    K, P = static_lqr(model)
    assert K.shape == (1, 4)
    assert P.shape == (4, 4)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_static_policy_zero_at_upright(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    policy = StaticLQRPolicy(model)
    assert policy(model.x_equilibrium("up"), 0.0) == pytest.approx(0.0, abs=1e-9)


def test_static_policy_saturates(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    policy = StaticLQRPolicy(model)
    x = model.x_equilibrium("up")
    x[1] = 3.0  # large tilt error demands a huge correction
    demand = policy(x, 0.0)
    assert demand == model.spec.force_bound_n


def test_static_policy_value_function(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    policy = StaticLQRPolicy(model)
    x_up = model.x_equilibrium("up")
    assert policy.value(x_up) == pytest.approx(0.0, abs=1e-12)
    x = x_up.copy()
    x[0] = 0.1
    assert policy.value(x) > 0.0


def test_policy_accepts_precomputed_gain(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    K, _ = static_lqr(model)
    a = StaticLQRPolicy(model, K=K)
    b = StaticLQRPolicy(model)
    x = np.array([0.2, 0.1, -0.05, 0.02])
    assert a(x, 0.0) == pytest.approx(b(x, 0.0))
