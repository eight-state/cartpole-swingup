"""Authority checks for original standalone releases preserved by Git tags."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Protocol


class LegacySource(Protocol):
    """Source identity fields required by legacy checks."""

    legacy_tag: str
    source_commit: str
    source_tree: str


class LegacyError(ValueError):
    """Raised when a preserved release tag or file does not match its registry."""


def audit_legacy_source(root: Path, source: LegacySource) -> dict[str, str]:
    """Verify that one tag resolves to the registered source commit and tree."""
    commit = _git_text(root, "rev-list", "-n", "1", source.legacy_tag)
    if commit != source.source_commit:
        raise LegacyError(
            f"legacy tag {source.legacy_tag} resolves to {commit}, expected {source.source_commit}"
        )
    tree = _git_text(root, "rev-parse", f"{commit}^{{tree}}")
    if tree != source.source_tree:
        raise LegacyError(
            f"legacy tree {source.legacy_tag} is {tree}, expected {source.source_tree}"
        )
    return {"tag": source.legacy_tag, "commit": commit, "tree": tree}


def legacy_file_sha256(root: Path, source: LegacySource, relative_path: str) -> str:
    """Hash one file from the original standalone release tree."""
    value = _git_bytes(root, "show", f"{source.legacy_tag}:{relative_path}")
    return hashlib.sha256(value).hexdigest()


def _git_text(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("utf-8").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ("git", "-C", str(root), *arguments),
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise LegacyError(f"Git legacy check failed: {detail or exc.returncode}") from exc
