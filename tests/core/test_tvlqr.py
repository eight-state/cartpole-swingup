"""Unit tests for cartpole_capsules.core.tvlqr. Short horizons only."""

from __future__ import annotations

import numpy as np
import pytest

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec
from cartpole_capsules.core.lqr import static_lqr
from cartpole_capsules.core.tvlqr import TVLQR, build_upright_tvlqr


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


def test_constant_upright_terminal_gain_matches_scaled_static(n1_model: NLinkCartPole):
    """build_upright_tvlqr contracts Qf = 25 * P_static; K(tf) must match."""
    model = n1_model
    horizon = 0.5
    catch = build_upright_tvlqr(model, horizon)
    K_static, P = static_lqr(model)
    # At tf the backward Riccati solution is S(tf) = Qf = 25 P, so the
    # terminal gain is exactly 25x the static gain (R^-1 B' S(tf)).
    K_tf = catch.K_at(horizon)
    assert np.allclose(K_tf, 25.0 * K_static, rtol=1e-4)
    # Early in the horizon the gain is smaller than the terminal value.
    K_t0 = catch.K_at(0.0)
    assert np.linalg.norm(K_t0) < np.linalg.norm(K_tf)


def test_tvlqr_policy_zero_on_nominal(n1_model: NLinkCartPole):
    model = n1_model
    x_up = model.x_equilibrium("up")
    t_nom = np.array([0.0, 0.5])
    x_nom = np.vstack([x_up, x_up])
    u_nom = np.array([0.0, 0.0])
    _, P = static_lqr(model)
    tv = TVLQR(model, t_nom, x_nom, u_nom, Qf=P, n_eval=50)
    for t in (0.0, 0.25, 0.5):
        assert tv.policy(x_up, t) == pytest.approx(0.0, abs=1e-9)
        assert tv.value(t, x_up) == pytest.approx(0.0, abs=1e-12)


def test_tvlqr_interpolation_clamps(n1_model: NLinkCartPole):
    model = n1_model
    catch = build_upright_tvlqr(model, 0.5, n_eval=50)
    before = catch.K_at(-1.0)
    at_start = catch.K_at(0.0)
    after = catch.K_at(2.0)
    at_end = catch.K_at(0.5)
    assert np.allclose(before, at_start)
    assert np.allclose(after, at_end)


def test_tvlqr_s_stays_symmetric(n1_model: NLinkCartPole):
    model = n1_model
    catch = build_upright_tvlqr(model, 0.5, n_eval=50)
    for S in catch.S_grid:
        assert np.allclose(S, S.T, atol=1e-8)


def test_tvlqr_constant_nominal_fast_path(n1_model: NLinkCartPole):
    """A two-sample constant nominal returns the exact endpoint values."""
    model = n1_model
    x_up = model.x_equilibrium("up")
    tv = build_upright_tvlqr(model, 1.0, n_eval=40)
    x, u = tv._nom_at(0.37)
    assert np.array_equal(x, x_up)
    assert u == 0.0
