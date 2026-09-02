from __future__ import annotations

import json
from pathlib import Path

import pytest

from cartpole_capsules.manifest import CapsuleEntry, ManifestError, load_manifest, write_manifest


def valid_entry() -> dict[str, object]:
    return {
        "rung": 5,
        "slug": "n05-quintuple",
        "source_url": "https://github.com/eight-state/quintuple-cartpole",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "capsule_path": "capsules/n05-quintuple",
        "content_hash": "c" * 64,
        "file_count": 2,
        "verification_command": "uv run n5-verify",
        "demo_command": None,
        "runner": "ubuntu-latest",
        "evidence_class": "historical-ledger-audit+fresh-rerun",
    }


def test_valid_entry_accepts_sortable_slug() -> None:
    entry = CapsuleEntry.from_mapping(valid_entry())
    assert entry.rung == 5
    assert entry.slug == "n05-quintuple"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rung", 4),
        ("rung", True),
        ("slug", "quintuple"),
        ("slug", "n06-sextuple"),
        ("source_url", "http://github.com/eight-state/quintuple-cartpole"),
        ("source_url", "https://example.com/eight-state/quintuple-cartpole"),
        ("source_commit", "a" * 39),
        ("source_tree", "G" * 40),
        ("capsule_path", "capsules/../escape"),
        ("content_hash", "c" * 63),
        ("file_count", 0),
        ("verification_command", "python verify.py"),
        ("runner", "macos-latest"),
    ],
)
def test_invalid_fields_are_rejected(field: str, value: object) -> None:
    mapping = valid_entry()
    mapping[field] = value
    with pytest.raises(ManifestError):
        CapsuleEntry.from_mapping(mapping)


def test_unknown_fields_are_rejected() -> None:
    mapping = valid_entry()
    mapping["extra"] = "not allowed"
    with pytest.raises(ManifestError, match="unknown"):
        CapsuleEntry.from_mapping(mapping)


def test_manifest_writer_uses_lf_on_every_platform(registry_root: Path) -> None:
    write_manifest(registry_root, [])
    raw = (registry_root / "capsules" / "imports.json").read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw


def test_duplicate_rungs_are_rejected(registry_root: Path) -> None:
    first = valid_entry()
    second = valid_entry()
    second["slug"] = "n05-other"
    second["capsule_path"] = "capsules/n05-other"
    payload = {"schema_version": 1, "capsules": [first, second]}
    (registry_root / "capsules" / "imports.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate rung"):
        load_manifest(registry_root)
