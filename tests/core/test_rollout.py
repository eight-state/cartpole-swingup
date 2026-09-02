"""Tests for shared policy and controls-only rollout."""

from __future__ import annotations

import helpers
import numpy as np
import pytest

from cartpole_capsules.core.rollout import replay_controls, run_policy


class ConstantStrategy:
    def __init__(self, force: float) -> None:
        self.force = force

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
        return self.force


def test_policy_rollout_records_raw_and_applied_force() -> None:
    model = helpers.make_n1_model()
    record = run_policy(
        model,
        ConstantStrategy(999.0),
        3,
        0.001,
        0.00025,
        model.x_equilibrium("down"),
        phase_label=lambda tick: "test",
    )
    assert record.states.shape == (4, model.nx)
    assert np.array_equal(record.raw, np.full(3, 999.0))
    assert np.array_equal(record.applied, np.full(3, model.spec.force_bound_n))
    assert record.phases == ("test",) * 3


def test_controls_replay_preserves_in_bound_controls() -> None:
    model = helpers.make_n1_model()
    controls = np.array([1.0, -2.0, 3.0])
    record = replay_controls(model, controls, 0.001, 0.00025, quarter_metrics=True)
    assert np.array_equal(record.raw, controls)
    assert np.array_equal(record.applied, controls)
    assert record.quarter_cart_peak_m is not None


def test_controls_replay_rejects_over_limit_input() -> None:
    model = helpers.make_n1_model()
    with pytest.raises(ValueError, match="over-limit"):
        replay_controls(model, np.array([model.spec.force_bound_n + 1.0]), 0.001, 0.00025)
