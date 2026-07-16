"""Checks for the immutable historical N11 evidence boundary."""

from __future__ import annotations

from cartpole_race.release import audit_banked_gate_evidence, audit_source_identity


def test_frozen_release_sources_match_their_canonical_git_bytes() -> None:
    records = audit_source_identity()
    assert len(records) == 5


def test_banked_gate_records_are_complete_and_internally_consistent() -> None:
    audit = audit_banked_gate_evidence()
    assert audit["total_successes"] == 72
    assert audit["total_trials"] == 72
    assert [record["seed"] for record in audit["files"]] == [12345, 777, 2024]
    assert all(record["successes"] == record["trials"] == 24 for record in audit["files"])
    assert all(record["wilson95"] == [0.862, 1.0] for record in audit["files"])
