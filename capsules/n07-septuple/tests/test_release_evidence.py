"""Focused audit test for the immutable banked evidence boundary."""

from __future__ import annotations

from cartpole_race.release_audit import audit_authority_bytes, audit_banked_evidence


def test_authorities_and_banked_evidence_are_intact() -> None:
    authorities = audit_authority_bytes()
    evidence = audit_banked_evidence()
    assert len(authorities) == 6
    assert [(record["seed"], record["successes"]) for record in evidence["records"]] == [
        (12345, 24),
        (777, 24),
        (12345, 18),
    ]
    assert len(evidence["logs"]) == 4
