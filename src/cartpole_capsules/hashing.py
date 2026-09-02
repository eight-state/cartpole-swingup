"""Cross-platform Git-canonical hashing for imported capsule trees."""

from __future__ import annotations

import hashlib
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


class TreeHashError(ValueError):
    """Raised when a capsule tree cannot be hashed safely."""


@dataclass(frozen=True)
class TreeDigest:
    """Canonical digest and tracked file count for one capsule directory."""

    sha256: str
    file_count: int


def hash_tree(root: Path) -> TreeDigest:
    """Hash tracked paths, modes, and clean-filtered Git blobs under *root*."""
    root = root.resolve()
    if not root.is_dir():
        raise TreeHashError(f"capsule directory not found: {root}")

    repository = _git_text(root, "rev-parse", "--show-toplevel")
    repository_root = Path(repository).resolve()
    try:
        prefix = root.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise TreeHashError(f"capsule is outside its Git repository: {root}") from exc

    indexed = _git_bytes(
        repository_root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        prefix,
    )
    records: list[tuple[str, str, str]] = []
    for raw_record in indexed.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, blob, stage = metadata.decode("ascii").split()
            repository_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise TreeHashError("Git returned a malformed index record") from exc
        if stage != "0":
            raise TreeHashError(f"capsule contains an unmerged index entry: {repository_path}")
        if mode == "160000":
            raise TreeHashError(f"capsule contains a nested Git repository: {repository_path}")
        expected_prefix = f"{prefix}/"
        if not repository_path.startswith(expected_prefix):
            raise TreeHashError(f"indexed path escaped capsule prefix: {repository_path}")
        relative = repository_path[len(expected_prefix) :]
        working_path = repository_root / Path(repository_path)
        if not working_path.exists():
            raise TreeHashError(f"tracked capsule file is missing: {repository_path}")

        observed_blob = _git_text(
            repository_root,
            "hash-object",
            f"--path={repository_path}",
            "--",
            str(working_path),
        )
        if observed_blob != blob:
            raise TreeHashError(f"tracked capsule file changed: {repository_path}")
        records.append((relative, mode, blob))

    untracked = _git_bytes(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        prefix,
    )
    untracked_paths = [path.decode("utf-8") for path in untracked.split(b"\0") if path]
    if untracked_paths:
        raise TreeHashError(f"capsule contains untracked files: {', '.join(untracked_paths)}")

    records.sort(key=lambda record: record[0])
    digest = hashlib.sha256()
    for relative, mode, blob in records:
        for value in (relative, mode, blob):
            encoded = value.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)

    return TreeDigest(sha256=digest.hexdigest(), file_count=len(records))


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8").strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repository), *arguments),
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise TreeHashError(f"Git authority check failed: {detail or exc.returncode}") from exc
