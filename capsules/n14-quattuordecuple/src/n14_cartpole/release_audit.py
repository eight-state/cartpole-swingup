"""Lowest-layer integrity audit for the immutable N14 source capsule."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]

MANIFEST_RELATIVE_PATH = "artifacts/MANIFEST.json"
SOURCE_LOCK_RELATIVE_PATH = "artifacts/source-sha256.json"
MANIFEST_PATH = REPOSITORY / MANIFEST_RELATIVE_PATH
SOURCE_LOCK_PATH = REPOSITORY / SOURCE_LOCK_RELATIVE_PATH

CAPSULE_REQUIRED_PATHS = frozenset(
    {
        "pyproject.toml",
        MANIFEST_RELATIVE_PATH,
        SOURCE_LOCK_RELATIVE_PATH,
        "artifacts/n14-witness.npz",
        "artifacts/expected-witness.json",
        "artifacts/provenance.json",
        "src/n14_cartpole/verifier.py",
    }
)
EXPECTED_ARTIFACT_PATHS = frozenset(
    {
        "artifacts/expected-witness.json",
        "artifacts/n14-witness.npz",
        "artifacts/provenance.json",
    }
)
EXPECTED_SOURCE_PATHS = frozenset(
    {
        ".gitattributes",
        ".github/workflows/verify.yml",
        ".gitignore",
        "LICENSE",
        "PROVENANCE.md",
        "README.md",
        "docs/METHOD.md",
        "docs/RELEASE_EVIDENCE.md",
        "pyproject.toml",
        "src/cartpole_race/__init__.py",
        "src/cartpole_race/dynamics.py",
        "src/cartpole_race/env_spec.py",
        "src/n14_cartpole/__init__.py",
        "src/n14_cartpole/release.py",
        "src/n14_cartpole/release_audit.py",
        "src/n14_cartpole/success.py",
        "src/n14_cartpole/verifier.py",
        "tests/test_integrity.py",
        "tests/test_success.py",
        "tests/test_verifier.py",
        "uv.lock",
    }
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_from_posix(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _is_safe_relative_path(relative: str) -> bool:
    """Accept only a normalized POSIX file path that is safe on Windows too."""
    if not relative or "\\" in relative or "\x00" in relative:
        return False
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    components = relative.split("/")
    return not (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or any(component in {"", ".", ".."} for component in components)
    )


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def locate_source_capsule(root: Path | None = None) -> Path:
    """Return a complete source-capsule root or raise ``ValueError``.

    ``root`` exists for copied-capsule audit tests. Production callers derive the
    root from this module and separately enforce their executing-module identity.
    """
    try:
        capsule_root = (REPOSITORY if root is None else Path(root)).resolve()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("source capsule root is unavailable") from error
    if not capsule_root.is_dir():
        raise ValueError("source capsule root is not a directory")

    for relative in sorted(CAPSULE_REQUIRED_PATHS):
        candidate = _path_from_posix(capsule_root, relative)
        try:
            resolved = candidate.resolve()
        except OSError as error:
            raise ValueError(f"source capsule path is unavailable: {relative}") from error
        if not _is_within(capsule_root, resolved) or not resolved.is_file():
            raise ValueError(f"source capsule path is missing: {relative}")
    return capsule_root


def _failure(path: str, reason: str) -> dict[str, str]:
    return {"path": path, "reason": reason}


def _load_lock(
    capsule_root: Path,
    relative_path: str,
    failures: list[dict[str, str]],
) -> dict[str, str] | None:
    path = _path_from_posix(capsule_root, relative_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(_failure(relative_path, "missing_lock"))
        return None
    except (OSError, UnicodeDecodeError):
        failures.append(_failure(relative_path, "unreadable_lock"))
        return None
    except json.JSONDecodeError:
        failures.append(_failure(relative_path, "invalid_json"))
        return None

    if not isinstance(payload, dict):
        failures.append(_failure(relative_path, "lock_must_be_object"))
        return None
    if payload.get("schema_version") != 1 or payload.get("release") != "N14":
        failures.append(_failure(relative_path, "lock_identity_mismatch"))
        return None

    raw_rows = payload.get("sha256")
    if not isinstance(raw_rows, dict) or not raw_rows:
        failures.append(_failure(relative_path, "sha256_must_be_nonempty_string_map"))
        return None

    rows: dict[str, str] = {}
    valid = True
    for locked_path, digest in raw_rows.items():
        if not isinstance(locked_path, str):
            failures.append(_failure(relative_path, "sha256_path_must_be_string"))
            valid = False
            continue
        if not _is_safe_relative_path(locked_path):
            failures.append(_failure(relative_path, f"unsafe_path:{locked_path!r}"))
            valid = False
            continue
        if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
            failures.append(_failure(relative_path, f"invalid_sha256:{locked_path}"))
            valid = False
            continue
        rows[locked_path] = digest
    return rows if valid else None


def _audit_entries(
    capsule_root: Path,
    rows: dict[str, str],
    actual: dict[str, str],
    failures: list[dict[str, str]],
) -> None:
    for relative, expected in rows.items():
        candidate = _path_from_posix(capsule_root, relative)
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            failures.append(_failure(relative, "unreadable_path"))
            continue
        if not _is_within(capsule_root, resolved):
            failures.append(_failure(relative, "path_escapes_capsule"))
            continue
        if not resolved.is_file():
            failures.append(_failure(relative, "missing_file"))
            continue
        try:
            observed = sha256(resolved)
        except (OSError, ValueError):
            failures.append(_failure(relative, "unreadable_file"))
            continue
        actual[relative] = observed
        if observed != expected:
            failures.append(_failure(relative, "sha256_mismatch"))


def _path_set_failure(
    lock_path: str,
    actual_paths: set[str],
    expected_paths: frozenset[str],
    failures: list[dict[str, str]],
) -> None:
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected)}")
        failures.append(_failure(lock_path, f"path_set_mismatch:{';'.join(details)}"))


def _audit_result(
    artifacts: dict[str, str],
    sources: dict[str, str],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verdict": "PASS" if not failures else "FAIL",
        "artifact_count": len(artifacts),
        "source_count": len(sources),
        "artifacts": artifacts,
        "sources": sources,
    }
    if failures:
        result["failures"] = sorted(failures, key=lambda failure: (failure["path"], failure["reason"]))
    return result


def audit_release(root: Path | None = None) -> dict[str, Any]:
    """Verify every immutable artifact and source digest without raising authority errors."""
    artifacts: dict[str, str] = {}
    sources: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    try:
        capsule_root = locate_source_capsule(root)
    except ValueError as error:
        failures.append(_failure("source_capsule", str(error)))
        return _audit_result(artifacts, sources, failures)

    manifest = _load_lock(capsule_root, MANIFEST_RELATIVE_PATH, failures)
    source_lock = _load_lock(capsule_root, SOURCE_LOCK_RELATIVE_PATH, failures)
    if manifest is not None:
        _path_set_failure(
            MANIFEST_RELATIVE_PATH,
            set(manifest),
            EXPECTED_ARTIFACT_PATHS,
            failures,
        )
        _audit_entries(capsule_root, manifest, artifacts, failures)
    if source_lock is not None:
        _path_set_failure(
            SOURCE_LOCK_RELATIVE_PATH,
            set(source_lock),
            EXPECTED_SOURCE_PATHS,
            failures,
        )
        _audit_entries(capsule_root, source_lock, sources, failures)
    return _audit_result(artifacts, sources, failures)
