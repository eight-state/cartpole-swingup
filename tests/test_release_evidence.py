"""Checks for the immutable historical evidence boundary."""

from __future__ import annotations

from cartpole_race.n10_release import audit_banked_gate_evidence


def test_banked_gate_records_are_complete_and_internally_consistent() -> None:
    audit = audit_banked_gate_evidence()
    assert audit["total_successes"] == 72
    assert audit["total_trials"] == 72
    assert [row["seed"] for row in audit["files"]] == [12345, 777, 2024]
    assert all(row["wilson95"] == [0.862, 1.0] for row in audit["files"])
