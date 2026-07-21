from __future__ import annotations

import json
from pathlib import Path

from n14_cartpole.release_audit import audit_release


def test_release_hash_locks_pass() -> None:
    result = audit_release()
    assert result["verdict"] == "PASS"
    assert result["artifact_count"] >= 3
    assert result["source_count"] >= 8


def test_retained_verification_report_passes() -> None:
    report = json.loads(Path("artifacts/verification.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["failures"] == []
    assert report["expected_witness"]["all_assertions_pass"] is True
