from __future__ import annotations

from pathlib import Path

import pytest
from conftest import stage

from cartpole_capsules.hashing import TreeHashError, hash_tree


def test_tree_hash_is_deterministic_across_creation_order(registry_root: Path) -> None:
    first = registry_root / "capsules" / "first"
    second = registry_root / "capsules" / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_bytes(b"two")
    (first / "a.txt").write_bytes(b"one")
    (second / "a.txt").write_bytes(b"one")
    (second / "b.txt").write_bytes(b"two")
    stage(registry_root, "capsules/first", "capsules/second")

    assert hash_tree(first) == hash_tree(second)
    assert hash_tree(first).file_count == 2


def test_tree_hash_detects_unstaged_content_drift(registry_root: Path) -> None:
    capsule = registry_root / "capsules" / "capsule"
    capsule.mkdir()
    artifact = capsule / "artifact.bin"
    artifact.write_bytes(b"original")
    stage(registry_root, "capsules/capsule")
    original = hash_tree(capsule)

    artifact.write_bytes(b"changed")
    with pytest.raises(TreeHashError, match="changed"):
        hash_tree(capsule)

    stage(registry_root, "capsules/capsule")
    assert hash_tree(capsule).sha256 != original.sha256


def test_tree_hash_normalizes_checkout_line_endings(registry_root: Path) -> None:
    capsule = registry_root / "capsules" / "capsule"
    capsule.mkdir()
    (capsule / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    text = capsule / "record.txt"
    text.write_bytes(b"first\nsecond\n")
    stage(registry_root, "capsules/capsule")
    expected = hash_tree(capsule)

    text.write_bytes(b"first\r\nsecond\r\n")
    assert hash_tree(capsule) == expected


def test_tree_hash_rejects_untracked_files(registry_root: Path) -> None:
    capsule = registry_root / "capsules" / "capsule"
    capsule.mkdir()
    (capsule / "tracked.txt").write_text("tracked", encoding="utf-8")
    stage(registry_root, "capsules/capsule")
    (capsule / "extra.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(TreeHashError, match="untracked"):
        hash_tree(capsule)
