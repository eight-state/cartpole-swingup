"""Checks for the immutable historical N11 evidence boundary."""

from __future__ import annotations

import json

from cartpole_race.release import (
    audit_banked_gate_evidence,
    audit_source_identity,
    run_live,
)


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


def test_live_metrics_schema_uses_the_sampled_hold_key() -> None:
    metrics = json.loads(json.dumps(run_live().metrics))
    closed_loop = metrics["live_closed_loop"]

    assert "longest_sampled_hold_s" in closed_loop
    assert "longest_continuous_hold_s" not in closed_loop
