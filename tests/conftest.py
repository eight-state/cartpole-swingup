"""Shared fixtures for deterministic cart-pole rigor tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec, load_spec

N_LINKS_SWEEP = [1, 2, 3, 11]
_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "env-n11.yaml"
_BASE_SPEC = load_spec(_CONFIG)


def spec_for(n: int) -> CartPoleSpec:
    """Return the release plant resized to ``n`` links for core tests."""
    return _BASE_SPEC.with_n_links(n)


@pytest.fixture(params=N_LINKS_SWEEP, ids=lambda n: f"n{n}")
def n_links(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def model(n_links: int) -> NLinkCartPole:
    return NLinkCartPole(spec_for(n_links))
