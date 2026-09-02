"""Deterministic filesystem hashing for imported capsule trees."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path


class TreeHashError(ValueError):
    """Raised when a capsule tree cannot be hashed safely."""


@dataclass(frozen=True)
class TreeDigest:
    """Digest and file count for one capsule directory."""

    sha256: str
    file_count: int


def hash_tree(root: Path) -> TreeDigest:
    """Hash every regular file under *root* in sorted relative-path order."""
    root = root.resolve()
    if not root.is_dir():
        raise TreeHashError(f"capsule directory not found: {root}")

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TreeHashError(f"symbolic links are not supported in capsules: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise TreeHashError(f"unsupported filesystem entry in capsule: {path}")

    files.sort(key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack(">Q", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", size))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)

    return TreeDigest(sha256=digest.hexdigest(), file_count=len(files))
