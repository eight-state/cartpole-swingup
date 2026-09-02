"""Shared layout helpers for the cartpole-capsules registry.

The repository root is located by walking upward from a start path until a
directory containing ``pyproject.toml`` and a ``capsules`` directory is found.
This keeps every entry point usable both from an installed console script and
from an arbitrary current working directory.
"""

from __future__ import annotations

from pathlib import Path

MARKERS = ("pyproject.toml", "capsules")


class LayoutError(RuntimeError):
    """Raised when the repository root cannot be located."""


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (default: cwd) to find the repository root."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in MARKERS):
            return candidate
    raise LayoutError(
        "Could not locate the repository root (a directory holding pyproject.toml "
        f"and a capsules directory), starting from {current}."
    )


def manifest_path(root: Path) -> Path:
    """Return the path of the capsule manifest inside *root*."""
    return root / "capsules" / "imports.json"
