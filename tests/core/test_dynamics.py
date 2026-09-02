"""Unit tests for cartpole_capsules.core.dynamics. No long simulations."""

from __future__ import annotations

import helpers
import numpy as np
import pytest

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec

n1_spec = pytest.fixture(lambda: helpers.make_n1_spec())


def test_equilibria(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    up = model.x_equilibrium("up")
    down = model.x_equilibrium("down")
    assert up.shape == (4,)
    assert up[1] == 0.0
    assert down[1] == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="kind"):
        model.x_equilibrium("sideways")


def test_zero_dynamics_at_hanging_equilibrium(n1_spec: CartPoleSpec):
    """With zero force the hanging state must be an exact fixed point."""
    model = NLinkCartPole(n1_spec)
    x_down = model.x_equilibrium("down")
    xdot = model.f_num(x_down, 0.0)
    assert np.allclose(xdot, 0.0, atol=1e-12)


def test_upright_linearization_matches_hand_derived_n1(n1_spec: CartPoleSpec):
    """Check A, B against physics invariants of the upright n=1 system.

    Signs and couplings are asserted from the moment equation about the
    pivot; magnitudes are cross-checked through the symbolic energy.
    """
    model = NLinkCartPole(n1_spec)
    A, B = model.linearize(model.x_equilibrium("up"), 0.0)
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)
    mc, m, ll, g = 1.0, 0.1, 0.5, 9.81
    # Rows 0 and 1 are kinematic (xdot = vx, thetadot = omega).
    assert A[0, 2] == pytest.approx(1.0)
    assert A[1, 3] == pytest.approx(1.0)
    theta_ddot_theta = A[3, 1]
    theta_ddot_u = B[3, 0]
    # Gravity destabilizes the upright (positive feedback on theta).
    assert theta_ddot_theta > 0.0
    # Cart force pushes the base; the upright link tips relative to vertical
    # with negative theta acceleration per unit force.
    assert theta_ddot_u < 0.0
    # Closed-form values from the same Lagrangian (rod, COM at l/2,
    # I = m l^2 / 12), solving the 2x2 mass-matrix system:
    #   theta_ddot/theta = 6 (mc + m) g / (l (4 mc + m))
    #   theta_ddot/u     = -6 / (l (4 mc + m))
    expected = 6.0 * (mc + m) * g / (ll * (4.0 * mc + m))
    assert theta_ddot_theta == pytest.approx(expected, rel=1e-9)
    assert theta_ddot_u == pytest.approx(-6.0 / (ll * (4.0 * mc + m)), rel=1e-9)
    # Cart row: xddot/theta = -3 m g / (4 mc + m), xddot/u = 4 / (4 mc + m).
    assert A[2, 1] == pytest.approx(-3.0 * m * g / (4.0 * mc + m), rel=1e-9)
    assert B[2, 0] == pytest.approx(4.0 / (4.0 * mc + m), rel=1e-9)
    # Energy: upright is the potential maximum, so a small tilt lowers V by
    # m g (l/2) (1 - cos eps) ~ m g (l/2) eps^2 / 2.
    x_up = model.x_equilibrium("up")
    eps = 1e-3
    x_tilt = x_up.copy()
    x_tilt[1] = eps
    dE = model.energy(x_tilt) - model.energy(x_up)
    assert dE == pytest.approx(-m * g * (ll / 2.0) * eps**2 / 2.0, rel=1e-6)


def test_rollout_zoh_shapes_and_determinism(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    x0 = model.x_equilibrium("down")
    policy = lambda x, t: 0.0  # noqa: E731
    out1 = model.rollout_zoh(x0, policy, 0.05, 0.001, 0.00025)
    out2 = model.rollout_zoh(x0, policy, 0.05, 0.001, 0.00025)
    t_log, x_log, u_log = out1
    assert t_log.shape == (51,)
    assert x_log.shape == (51, 4)
    assert u_log.shape == (50,)
    assert t_log[0] == 0.0 and t_log[-1] == pytest.approx(0.05)
    assert np.array_equal(x_log, out2[1])
    assert np.array_equal(u_log, out2[2])


def test_rollout_return_raw_flag(n1_spec: CartPoleSpec):
    """return_raw=True appends the pre-clip demanded force."""
    model = NLinkCartPole(n1_spec)
    x0 = model.x_equilibrium("down")
    # Demand far beyond the 150 N bound; every tick must saturate.
    policy = lambda x, t: 999.0  # noqa: E731
    t_log, x_log, u_log = model.rollout_zoh(x0, policy, 0.01, 0.001, 0.00025)
    t2, x2, u2, u_raw = model.rollout_zoh(x0, policy, 0.01, 0.001, 0.00025, return_raw=True)
    assert np.array_equal(t_log, t2) and np.array_equal(x_log, x2)
    assert np.all(u_log == model.spec.force_bound_n)
    assert np.all(u_raw == 999.0)
    # Unclipped policy demand below the bound leaves both logs identical.
    policy_small = lambda x, t: 12.0  # noqa: E731
    _, _, u_small = model.rollout_zoh(x0, policy_small, 0.01, 0.001, 0.00025)
    _, _, _, u_small_raw = model.rollout_zoh(
        x0, policy_small, 0.01, 0.001, 0.00025, return_raw=True
    )
    assert np.array_equal(u_small, u_small_raw)
    assert np.all(u_small == 12.0)


def test_rollout_clips_negative_demand(n1_spec: CartPoleSpec):
    model = NLinkCartPole(n1_spec)
    x0 = model.x_equilibrium("down")
    policy = lambda x, t: -999.0  # noqa: E731
    _, _, u_log = model.rollout_zoh(x0, policy, 0.005, 0.001, 0.00025)
    assert np.all(u_log == -model.spec.force_bound_n)


def test_rk4_step_energy_conservation_swing_free():
    """Unforced, undamped pendulum preserves energy to RK4 accuracy."""
    spec = CartPoleSpec(
        n_links=1,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        damping_links_n_m_s_rad=[0.0],
    )
    model = NLinkCartPole(spec)
    x = model.x_equilibrium("down")
    x[1] = 0.5  # displaced from the hanging minimum
    e0 = model.energy(x)
    for _ in range(50):
        x = model.rk4_step(x, 0.0, 0.001)
    assert model.energy(x) == pytest.approx(e0, rel=1e-9)
