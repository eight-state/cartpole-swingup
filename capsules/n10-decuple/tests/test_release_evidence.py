"""Checks for the immutable historical evidence boundary."""

from __future__ import annotations

import json

from cartpole_race.n10_release import audit_banked_gate_evidence, run_unperturbed


def test_banked_gate_records_are_complete_and_internally_consistent() -> None:
    audit = audit_banked_gate_evidence()

    assert set(audit) == {"status", "files", "total_successes", "total_trials"}
    assert audit["status"] == (
        "stored historical records hashed; metadata validated; stored success flags "
        "counted; Wilson intervals recomputed; historical outcomes not re-evaluated; "
        "perturbations not rerun"
    )
    assert audit["total_successes"] == 72
    assert audit["total_trials"] == 72
    assert [row["seed"] for row in audit["files"]] == [12345, 777, 2024]
    assert all(
        set(row)
        == {
            "file",
            "sha256",
            "seed",
            "successes",
            "trials",
            "wilson95",
            "recorded_nominal_label",
        }
        for row in audit["files"]
    )
    assert all(row["wilson95"] == [0.862, 1.0] for row in audit["files"])


def test_live_metric_schema_uses_the_sampled_hold_key() -> None:
    metrics = json.loads(json.dumps(run_unperturbed().metrics))
    closed_loop = metrics["live_closed_loop"]

    assert "longest_sampled_hold_s" in closed_loop
    assert "longest_continuous_hold_s" not in closed_loop
