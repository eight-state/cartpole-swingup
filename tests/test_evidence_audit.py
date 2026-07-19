"""The frozen evidence manifest remains internally consistent."""
from __future__ import annotations

from pathlib import Path

from cartpole_race.evidence import audit_frozen_evidence
from cartpole_race.n8 import _banked_evidence_authority


def test_frozen_evidence_audit() -> None:
    audit = audit_frozen_evidence(Path(__file__).resolve().parents[1])
    assert len(audit["artifacts"]) == 6
    assert audit["banked_gate_audit"]["clvalidate_n8_composite_seed12345.json"]["n_success"] == 24
    assert audit["banked_gate_audit"]["clvalidate_n8_fixed_seed12345.json"]["n_success"] == 8


def test_banked_source_provenance_is_unverified() -> None:
    audit = audit_frozen_evidence(Path(__file__).resolve().parents[1])
    assert _banked_evidence_authority(audit)["banked_source_provenance"] == "unverified"
