"""Registry checks and capsule registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cartpole_capsules.hashing import TreeDigest, hash_tree
from cartpole_capsules.manifest import (
    CapsuleEntry,
    ManifestError,
    capsule_directory,
    load_manifest,
    write_manifest,
)


@dataclass(frozen=True)
class CapsuleCheck:
    """Observed authority state for one capsule."""

    entry: CapsuleEntry
    digest: TreeDigest


def check_registry(root: Path) -> list[CapsuleCheck]:
    """Verify the manifest, registered bytes, and capsule directory set."""
    root = root.resolve()
    entries = load_manifest(root)
    capsules_root = root / "capsules"
    if not capsules_root.is_dir():
        raise ManifestError(f"capsules directory not found: {capsules_root}")

    expected = {entry.slug for entry in entries}
    actual: set[str] = set()
    for child in capsules_root.iterdir():
        if child.name == "imports.json":
            if not child.is_file():
                raise ManifestError("capsules/imports.json must be a regular file")
            continue
        if not child.is_dir():
            raise ManifestError(f"unexpected file in capsules directory: {child.name}")
        actual.add(child.name)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ManifestError(f"registered capsule directories are missing: {', '.join(missing)}")
    if unexpected:
        raise ManifestError(f"unregistered capsule directories found: {', '.join(unexpected)}")

    checks: list[CapsuleCheck] = []
    for entry in entries:
        digest = hash_tree(capsule_directory(root, entry))
        if digest.sha256 != entry.content_hash:
            raise ManifestError(
                f"capsule {entry.slug} content hash changed: "
                f"expected {entry.content_hash}, observed {digest.sha256}"
            )
        if digest.file_count != entry.file_count:
            raise ManifestError(
                f"capsule {entry.slug} file count changed: "
                f"expected {entry.file_count}, observed {digest.file_count}"
            )
        checks.append(CapsuleCheck(entry=entry, digest=digest))
    return checks


def register_capsule(
    root: Path,
    *,
    rung: int,
    slug: str,
    source_url: str,
    source_commit: str,
    source_tree: str,
    verification_command: str,
    demo_command: str | None,
    runner: str,
    evidence_class: str,
) -> CapsuleEntry:
    """Register the current bytes of one newly imported capsule."""
    root = root.resolve()
    entries = load_manifest(root)
    if any(entry.rung == rung or entry.slug == slug for entry in entries):
        raise ManifestError(f"rung or slug already registered: n={rung}, {slug}")

    capsule_path = f"capsules/{slug}"
    directory = root / capsule_path
    digest = hash_tree(directory)
    entry = CapsuleEntry.from_mapping(
        {
            "rung": rung,
            "slug": slug,
            "source_url": source_url,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "capsule_path": capsule_path,
            "content_hash": digest.sha256,
            "file_count": digest.file_count,
            "verification_command": verification_command,
            "demo_command": demo_command,
            "runner": runner,
            "evidence_class": evidence_class,
        }
    )
    write_manifest(root, [*entries, entry])
    return entry
