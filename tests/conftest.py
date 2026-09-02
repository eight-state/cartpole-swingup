from __future__ import annotations

from pathlib import Path

import pytest

from cartpole_capsules.hashing import hash_tree
from cartpole_capsules.manifest import CapsuleEntry, write_manifest


@pytest.fixture
def registry_root(tmp_path: Path) -> Path:
    (tmp_path / "capsules").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    write_manifest(tmp_path, [])
    return tmp_path


def create_capsule(root: Path, *, rung: int = 5, slug: str = "n05-quintuple") -> CapsuleEntry:
    directory = root / "capsules" / slug
    directory.mkdir()
    (directory / "artifact.txt").write_text("frozen\n", encoding="utf-8")
    digest = hash_tree(directory)
    entry = CapsuleEntry.from_mapping(
        {
            "rung": rung,
            "slug": slug,
            "source_url": "https://github.com/eight-state/quintuple-cartpole",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "capsule_path": f"capsules/{slug}",
            "content_hash": digest.sha256,
            "file_count": digest.file_count,
            "verification_command": "uv run n5-verify",
            "demo_command": "uv run n5-demo",
            "runner": "ubuntu-latest",
            "evidence_class": "historical-ledger-audit+fresh-rerun",
        }
    )
    write_manifest(root, [entry])
    return entry
