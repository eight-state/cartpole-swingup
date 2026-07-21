"""Integrity audit for the immutable N14 release files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from n14_cartpole.verifier import REPOSITORY, sha256

MANIFEST_PATH = REPOSITORY / "artifacts" / "MANIFEST.json"
SOURCE_LOCK_PATH = REPOSITORY / "artifacts" / "source-sha256.json"


def _load_map(path: Path, key: str) -> dict[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[key]
    if not isinstance(rows, dict) or not all(
        isinstance(relative, str) and isinstance(digest, str)
        for relative, digest in rows.items()
    ):
        raise TypeError(f"{path.name}:{key} must be a string map")
    return rows


def _audit_map(rows: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in rows.items():
        path = REPOSITORY / relative
        if not path.is_file():
            raise AssertionError(f"missing release file: {relative}")
        digest = sha256(path)
        if digest != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")
        actual[relative] = digest
    return actual


def audit_release() -> dict[str, Any]:
    """Verify every frozen artifact and source digest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("release") != "N14":
        raise AssertionError("artifact manifest identity differs")
    if source_lock.get("schema_version") != 1 or source_lock.get("release") != "N14":
        raise AssertionError("source lock identity differs")
    artifacts = _audit_map(_load_map(MANIFEST_PATH, "sha256"))
    sources = _audit_map(_load_map(SOURCE_LOCK_PATH, "sha256"))
    return {
        "verdict": "PASS",
        "artifact_count": len(artifacts),
        "source_count": len(sources),
        "artifacts": artifacts,
        "sources": sources,
    }
