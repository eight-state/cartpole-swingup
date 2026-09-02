"""Small shared fixtures for numerical core tests."""

from cartpole_capsules.core.dynamics import NLinkCartPole
from cartpole_capsules.core.env_spec import CartPoleSpec


def make_n1_spec() -> CartPoleSpec:
    """Return a one-link spec with the shared physical defaults."""
    return CartPoleSpec(
        n_links=1,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        gravity_m_s2=9.81,
        damping_links_n_m_s_rad=[0.0],
        force_bound_n=150.0,
    )


def make_n1_model() -> NLinkCartPole:
    """Build the one-link test model."""
    return NLinkCartPole(make_n1_spec())
