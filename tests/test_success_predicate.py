from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from n12_cartpole.success import in_success_set


def make_model() -> NLinkCartPole:
    return NLinkCartPole(
        CartPoleSpec(
            n_links=12,
            link_masses_kg=[0.1] * 12,
            link_lengths_m=[0.5] * 12,
            damping_links_n_m_s_rad=[0.0] * 12,
        )
    )


def test_upright_equilibrium_is_in_the_locked_success_set() -> None:
    model = make_model()
    assert in_success_set(model, model.x_equilibrium("up"))


def test_each_locked_success_boundary_rejects_a_violation() -> None:
    model = make_model()
    state = model.x_equilibrium("up")

    angle_violation = state.copy()
    angle_violation[1] = np.deg2rad(5.1)
    assert not in_success_set(model, angle_violation)

    link_rate_violation = state.copy()
    link_rate_violation[14] = 0.5001
    assert not in_success_set(model, link_rate_violation)

    cart_violation = state.copy()
    cart_violation[0] = 2.0001
    assert not in_success_set(model, cart_violation)

    cart_rate_violation = state.copy()
    cart_rate_violation[13] = 0.5001
    assert not in_success_set(model, cart_rate_violation)
