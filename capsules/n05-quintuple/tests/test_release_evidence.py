"""The historical N5 evidence remains byte-stable and internally consistent."""

from __future__ import annotations

from pathlib import Path

from cartpole_race.evidence import audit_authority_bytes, audit_historical_reports

REPO = Path(__file__).resolve().parents[1]


def test_authority_and_historical_ledgers() -> None:
    authority = audit_authority_bytes(REPO)
    historical = audit_historical_reports(REPO)
    assert len(authority) == 12
    assert historical["by_sigma"] == {
        "0.02": {"successes": 88, "trials": 88},
        "0.10": {"successes": 24, "trials": 24},
    }
