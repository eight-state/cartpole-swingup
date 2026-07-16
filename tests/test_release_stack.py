"""Focused checks for the frozen inputs and exact-ZOH live controller."""

from __future__ import annotations

import numpy as np

from cartpole_race.release import audit_nominal_artifacts, build_release_stack


def test_frozen_nominals_and_discrete_tvlqr_release_stack() -> None:
    stack = build_release_stack()
    nominal = audit_nominal_artifacts(stack)
    rho = stack.tracker.monodromy()

    assert nominal["n_ticks"] == 10_000
    assert nominal["parent_rk4_4ms_defect"] < 2e-5
    assert nominal["dense_simulator_defect"] < 1e-6
    assert np.isclose(rho, 0.122926, rtol=0.0, atol=1e-6)
