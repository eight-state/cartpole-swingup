from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from n14_cartpole.release_audit import (
    EXPECTED_ARTIFACT_PATHS,
    EXPECTED_SOURCE_PATHS,
    REPOSITORY,
    audit_release,
    sha256,
)

IMMUTABLE_SHA256 = {
    "artifacts/n14-witness.npz": "f10fc56e854050e6091f0ac7ce406772875ab70f38829d2a177d134e97ed0b29",
    "artifacts/expected-witness.json": "493ff90986a09453761490fc0cbc5761f602d2b0b9e42e02b7f0078508046b41",
    "artifacts/provenance.json": "cdf8a86cd76fcc8bc7244c1234ee3f1d4a09d4657aed0bfcc04c0eae5bd655cc",
    "artifacts/verification.json": "0b9289b3243bef5e9d4cf8f5687f0fe1c5cb01d6bff064872223fa93d3b7b6a9",
}


def _copy_capsule(tmp_path: Path) -> Path:
    copied_root = tmp_path / "capsule"
    shutil.copytree(
        REPOSITORY,
        copied_root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            ".working",
            "__pycache__",
        ),
    )
    return copied_root


def _read_lock(root: Path, name: str) -> dict[str, object]:
    return json.loads((root / "artifacts" / name).read_text(encoding="utf-8"))


def _write_lock(root: Path, name: str, payload: dict[str, object]) -> None:
    (root / "artifacts" / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_release_hash_locks_pass_with_exact_authority_sets() -> None:
    result = audit_release()
    assert result["verdict"] == "PASS"
    assert result["artifact_count"] == 3
    assert result["source_count"] == 21
    assert set(result["artifacts"]) == EXPECTED_ARTIFACT_PATHS
    assert set(result["sources"]) == EXPECTED_SOURCE_PATHS


def test_copied_capsule_verifier_drift_is_reported(tmp_path: Path) -> None:
    copied_root = _copy_capsule(tmp_path)
    verifier_path = copied_root / "src" / "n14_cartpole" / "verifier.py"
    verifier_path.write_text(
        verifier_path.read_text(encoding="utf-8") + "\n# audit drift probe\n",
        encoding="utf-8",
    )

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert {failure["path"] for failure in result["failures"]} >= {
        "src/n14_cartpole/verifier.py"
    }


def test_malformed_lock_returns_structured_failure(tmp_path: Path) -> None:
    copied_root = _copy_capsule(tmp_path)
    (copied_root / "artifacts" / "source-sha256.json").write_text("{", encoding="utf-8")

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert result["failures"]


def test_missing_locked_file_returns_structured_failure(tmp_path: Path) -> None:
    copied_root = _copy_capsule(tmp_path)
    (copied_root / "README.md").unlink()

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert {failure["path"] for failure in result["failures"]} >= {"README.md"}


def test_invalid_digest_returns_structured_failure(tmp_path: Path) -> None:
    copied_root = _copy_capsule(tmp_path)
    source_lock = _read_lock(copied_root, "source-sha256.json")
    rows = source_lock["sha256"]
    assert isinstance(rows, dict)
    rows["README.md"] = "not-a-digest"
    _write_lock(copied_root, "source-sha256.json", source_lock)

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert result["failures"]


@pytest.mark.parametrize("escape", ("../outside.py", "..\\outside.py", "C:\\outside.py"))
def test_path_traversal_returns_structured_failure(tmp_path: Path, escape: str) -> None:
    copied_root = _copy_capsule(tmp_path)
    source_lock = _read_lock(copied_root, "source-sha256.json")
    rows = source_lock["sha256"]
    assert isinstance(rows, dict)
    rows[escape] = "0" * 64
    _write_lock(copied_root, "source-sha256.json", source_lock)

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert any("unsafe_path" in failure["reason"] for failure in result["failures"])


def test_lock_row_removal_returns_structured_failure(tmp_path: Path) -> None:
    copied_root = _copy_capsule(tmp_path)
    source_lock = _read_lock(copied_root, "source-sha256.json")
    rows = source_lock["sha256"]
    assert isinstance(rows, dict)
    rows.pop("README.md")
    _write_lock(copied_root, "source-sha256.json", source_lock)

    result = audit_release(root=copied_root)

    assert result["verdict"] == "FAIL"
    assert any("path_set_mismatch" in failure["reason"] for failure in result["failures"])


@pytest.mark.parametrize("relative, expected", IMMUTABLE_SHA256.items())
def test_immutable_release_bytes_are_frozen(relative: str, expected: str) -> None:
    assert sha256(REPOSITORY / relative) == expected


def test_retained_verification_report_passes() -> None:
    report = json.loads((REPOSITORY / "artifacts" / "verification.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["failures"] == []
    assert report["expected_witness"]["all_assertions_pass"] is True
