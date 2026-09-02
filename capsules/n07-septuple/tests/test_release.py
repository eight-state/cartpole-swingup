"""Focused tests for the sole supported fresh n=7 rollout."""

from __future__ import annotations

import math

from cartpole_race.release import build_release_stack, run_live


def test_fresh_rollout_matches_the_baseline_live_metrics() -> None:
    run = run_live(build_release_stack())
    metrics = run.metrics
    assert metrics["success"] is True
    assert metrics["control_ticks"] == 8000
    assert math.isclose(metrics["rho"], 0.19714933550375185, abs_tol=1e-9)
    assert math.isclose(metrics["swing_handoff_dev_deg"], 0.011464552073950984, abs_tol=1e-9)
    assert math.isclose(metrics["swing_peak_force_n"], 23.292757063117687, abs_tol=1e-9)
    assert math.isclose(metrics["hold_peak_force_n"], 27.125402075050488, abs_tol=1e-7)
    assert math.isclose(metrics["track_abs_max_m"], 5.079856145198866, abs_tol=1e-9)
    assert math.isclose(metrics["final_hold_s"], 8.073, abs_tol=1e-12)
