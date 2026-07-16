"""Focused gate for the shipped n=7 exact-ZOH discrete TVLQR."""

from __future__ import annotations

from cartpole_race.release import build_release_stack


def test_discrete_tvlqr_monodromy_contracts() -> None:
    stack = build_release_stack()
    rho = stack.tracker.monodromy()
    assert rho < 0.5, f"rho unexpectedly large: {rho:.4g}"
