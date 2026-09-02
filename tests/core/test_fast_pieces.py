"""Tests for cartpole_capsules.core.fast_pieces. Small grids only."""

from __future__ import annotations

import numpy as np
import pytest

from cartpole_capsules.core.discrete_tvlqr import DiscreteTVLQR
from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec
from cartpole_capsules.core.fast_pieces import (
    FastDTVLQR,
    check_bitexact_densify,
    make_densifier,
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


def test_densifier_is_bit_exact(model: NLinkCartPole):
    """The mapaccum graph composes the same RK4 calls as the Python loop."""
    control_dt = 0.004
    n_sub = 2
    stride = 3
    n_coarse = 5
    densify = make_densifier(model, control_dt, n_sub, stride, n_coarse)
    rng = np.random.default_rng(7)
    x0 = model.x_equilibrium("down")
    Xp = np.tile(x0, (n_coarse, 1))
    Xp[:, 1] = rng.uniform(-0.3, 0.3, size=n_coarse)
    Up = rng.uniform(-50.0, 50.0, size=n_coarse)
    Xd, Ud = densify(Xp, Up)
    assert Xd.shape == (n_coarse * stride + 1, model.nx)
    assert Ud.shape == (n_coarse * stride,)
    assert np.all(Ud == np.repeat(Up, stride))
    assert np.array_equal(Xd[0], Xp[0])
    same, md = check_bitexact_densify(
        model, densify, Xp, Up, control_dt, n_sub, stride, n_check=n_coarse
    )
    assert same and md == 0.0


def test_fast_dtvlqr_matches_discrete_tvlqr(model: NLinkCartPole):
    """Batched linearization must reproduce the looped tracker exactly."""
    rng = np.random.default_rng(11)
    x0 = model.x_equilibrium("up")
    n_ticks = 15
    states = np.tile(x0, (n_ticks + 1, 1))
    states[:, 1] = rng.uniform(-0.05, 0.05, size=n_ticks + 1)
    controls = rng.uniform(-20.0, 20.0, size=n_ticks)
    dt = 0.004
    slow = DiscreteTVLQR(model, states, controls, dt)
    fast = FastDTVLQR(model, states, controls, dt)
    assert np.array_equal(slow.ad, fast.Ad)
    assert np.array_equal(slow.bd, fast.Bd)
    assert np.array_equal(slow.gains, fast.K)
    assert np.array_equal(slow.initial_cost, fast.S0)
    # Policy agreement on and off the nominal.
    probe = x0.copy()
    probe[1] = 0.02
    for t in (0.0, 0.008, 0.05):
        assert slow.policy(probe, t) == pytest.approx(fast.policy(probe, t), abs=1e-12)


def test_fast_dtvlqr_validates_shapes(model: NLinkCartPole):
    x0 = model.x_equilibrium("up")
    with pytest.raises(ValueError, match="one more state"):
        FastDTVLQR(model, np.tile(x0, (4, 1)), np.zeros(4), 0.004)


def test_fast_dtvlqr_q_override(model: NLinkCartPole):
    """An explicit Q flows into the Riccati like the reference tracker."""
    from cartpole_capsules.core.lqr import make_Q

    x0 = model.x_equilibrium("up")
    n_ticks = 6
    states = np.tile(x0, (n_ticks + 1, 1))
    controls = np.zeros(n_ticks)
    q = make_Q(1) * 2.0
    slow = DiscreteTVLQR(model, states, controls, 0.004, q=q)
    fast = FastDTVLQR(model, states, controls, 0.004, Q=q)
    assert np.array_equal(slow.gains, fast.K)
