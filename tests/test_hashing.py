from __future__ import annotations

from pathlib import Path

from cartpole_capsules.hashing import hash_tree


def test_tree_hash_is_deterministic_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_bytes(b"two")
    (first / "a.txt").write_bytes(b"one")
    (second / "a.txt").write_bytes(b"one")
    (second / "b.txt").write_bytes(b"two")

    assert hash_tree(first) == hash_tree(second)
    assert hash_tree(first).file_count == 2


def test_tree_hash_detects_path_and_content_drift(tmp_path: Path) -> None:
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    artifact = capsule / "artifact.bin"
    artifact.write_bytes(b"original")
    original = hash_tree(capsule)

    artifact.write_bytes(b"changed")
    changed = hash_tree(capsule)
    assert changed.sha256 != original.sha256

    artifact.rename(capsule / "renamed.bin")
    renamed = hash_tree(capsule)
    assert renamed.sha256 != changed.sha256
