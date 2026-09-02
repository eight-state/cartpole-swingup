"""Unit tests for cartpole_capsules.core.env_spec."""

from __future__ import annotations

import pytest

from cartpole_capsules.core.env_spec import CartPoleSpec, load_spec


def test_defaults_are_frozen():
    spec = CartPoleSpec()
    assert spec.n_links == 6
    assert spec.nx == 2 * 7
    assert spec.nq == 7
    assert spec.force_bound_n == 150.0
    assert spec.control_dt_s == pytest.approx(0.001)
    with pytest.raises((TypeError, ValueError)):
        spec.n_links = 5  # type: ignore[misc]


def test_per_link_lists_must_match_n_links():
    with pytest.raises(ValueError, match="link_masses_kg"):
        CartPoleSpec(n_links=3, link_masses_kg=[0.1, 0.1])


def test_n_links_must_be_positive():
    with pytest.raises(ValueError, match="n_links"):
        CartPoleSpec(n_links=0)


def test_with_n_links_broadcasts_first_link_values():
    base = CartPoleSpec(
        n_links=6,
        link_masses_kg=[0.2] * 6,
        link_lengths_m=[0.3] * 6,
        damping_links_n_m_s_rad=[0.05] * 6,
        force_bound_n=60.0,
    )
    small = base.with_n_links(2)
    assert small.n_links == 2
    assert small.link_masses_kg == [0.2, 0.2]
    assert small.link_lengths_m == [0.3, 0.3]
    assert small.damping_links_n_m_s_rad == [0.05, 0.05]
    assert small.force_bound_n == 60.0
    assert small.nx == 2 * 3


def test_timing_validation():
    with pytest.raises(ValueError, match="control_rate_hz"):
        CartPoleSpec(control_rate_hz=0.0)
    with pytest.raises(ValueError, match="rk4_max_step_s"):
        CartPoleSpec(rk4_max_step_s=-1.0)


def test_load_spec_round_trip(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text(
        "n_links: 2\n"
        "link_masses_kg: [0.1, 0.1]\n"
        "link_lengths_m: [0.5, 0.5]\n"
        "damping_links_n_m_s_rad: [0.0, 0.0]\n"
        "force_bound_n: 60.0\n"
        "hold_time_s: 5.0\n",
        encoding="utf-8",
    )
    spec = load_spec(path)
    assert spec.n_links == 2
    assert spec.force_bound_n == 60.0


def test_load_spec_rejects_unknown_keys(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_field"):
        load_spec(path)
