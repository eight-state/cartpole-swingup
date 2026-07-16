"""The retained gate JSONs are internally consistent historical evidence."""

from cartpole_race.n6 import EVIDENCE_COMMIT, audit_historical_evidence


def test_historical_gate_counts_and_provenance_boundary() -> None:
    """Two declared 24/24 gates aggregate to 48/48 without source verification."""
    audit = audit_historical_evidence()
    assert not audit.errors
    assert [(leg.seed, leg.successes, leg.trials) for leg in audit.legs] == [
        (999, 24, 24),
        (12345, 24, 24),
    ]
    assert (audit.total_successes, audit.total_trials) == (48, 48)
    assert not audit.row_records_available
    assert audit.provenance.startswith(f"unavailable: embedded commit {EVIDENCE_COMMIT}")
