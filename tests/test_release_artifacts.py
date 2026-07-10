"""Checks that the five banked N11 release artifacts remain internally valid."""
from __future__ import annotations

from release_audit import audit_release_artifacts


def test_banked_artifacts_match_hashes_and_invariants() -> None:
    report = audit_release_artifacts()
    assert report["aggregate_successes"] == 72
    assert report["aggregate_trials"] == 72
