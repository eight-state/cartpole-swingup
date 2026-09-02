"""The retained gate JSONs are internally consistent historical evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cartpole_race.n6 as n6
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


def _valid_gate_record(seed: int) -> dict[str, object]:
    return {
        "rows": [{}],
        "seed": seed,
        "n_success": 24,
        "n_trials": 24,
        "frac": 1.0,
        "commit_sha": EVIDENCE_COMMIT,
        "nominal": "results/nom_n6_gluck_cont.npz",
        "nominal_sha256": n6.NOMINAL_SHA256,
        "n_links": n6.N_LINKS,
        "force_limit": n6.FORCE_BOUND_N,
    }


@pytest.mark.parametrize(
    "failure",
    ["missing", "unreadable", "malformed", "empty_rows", "non_list_rows"],
)
def test_row_records_are_unavailable_when_an_expected_gate_is_not_readable_as_rows(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = tuple(tmp_path / name for name in n6.EVIDENCE_SHA256)
    for path, seed in zip(paths, (12345, 999), strict=True):
        path.write_text(json.dumps(_valid_gate_record(seed)), encoding="utf-8")

    target = paths[0]
    if failure == "missing":
        target.unlink()
    elif failure == "malformed":
        target.write_text("{", encoding="utf-8")
    elif failure in {"empty_rows", "non_list_rows"}:
        record = _valid_gate_record(12345)
        record["rows"] = [] if failure == "empty_rows" else {}
        target.write_text(json.dumps(record), encoding="utf-8")

    expected_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()
    }
    monkeypatch.setattr(n6, "REPO", tmp_path)
    monkeypatch.setattr(n6, "EVIDENCE_PATHS", paths)
    monkeypatch.setattr(n6, "EVIDENCE_SHA256", expected_hashes)
    monkeypatch.setattr(n6, "_source_provenance", lambda: "unverified")

    if failure == "unreadable":
        sha256 = n6._sha256

        def unreadable_sha256(path: Path) -> str:
            if path == target:
                raise OSError("read denied")
            return sha256(path)

        monkeypatch.setattr(n6, "_sha256", unreadable_sha256)

    assert not audit_historical_evidence().row_records_available
