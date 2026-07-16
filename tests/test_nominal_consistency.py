"""The released nominal remains consistent with the public simulator."""

from __future__ import annotations

from cartpole_race.release import audit_nominal_consistency


def test_fixed_nominal_is_zoh_consistent() -> None:
    audit = audit_nominal_consistency()
    assert audit["control_ticks"] == 6000
    assert audit["horizon_s"] == 6.0
    assert audit["max_zoh_defect"] < 1e-10
    assert audit["peak_feedforward_force_n"] == 20.175585913426993
