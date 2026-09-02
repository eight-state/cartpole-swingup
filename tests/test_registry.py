from __future__ import annotations

from pathlib import Path

import pytest
from conftest import create_capsule

from cartpole_capsules.manifest import ManifestError, load_manifest
from cartpole_capsules.registry import check_registry, register_capsule


def test_empty_registry_passes(registry_root: Path) -> None:
    assert check_registry(registry_root) == []


def test_registered_capsule_passes(registry_root: Path) -> None:
    entry = create_capsule(registry_root)
    checks = check_registry(registry_root)
    assert [check.entry for check in checks] == [entry]


def test_content_drift_fails(registry_root: Path) -> None:
    create_capsule(registry_root)
    (registry_root / "capsules" / "n05-quintuple" / "artifact.txt").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="content hash changed"):
        check_registry(registry_root)


def test_unregistered_directory_fails(registry_root: Path) -> None:
    (registry_root / "capsules" / "n06-sextuple").mkdir()
    with pytest.raises(ManifestError, match="unregistered"):
        check_registry(registry_root)


def test_unexpected_file_fails(registry_root: Path) -> None:
    (registry_root / "capsules" / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ManifestError, match="unexpected file"):
        check_registry(registry_root)


def test_registration_records_observed_bytes(registry_root: Path) -> None:
    directory = registry_root / "capsules" / "n05-quintuple"
    directory.mkdir()
    (directory / "artifact.txt").write_text("frozen\n", encoding="utf-8")

    entry = register_capsule(
        registry_root,
        rung=5,
        slug="n05-quintuple",
        source_url="https://github.com/eight-state/quintuple-cartpole",
        source_commit="a" * 40,
        source_tree="b" * 40,
        verification_command="uv run n5-verify",
        demo_command="uv run n5-demo",
        runner="ubuntu-latest",
        evidence_class="historical-ledger-audit+fresh-rerun",
    )

    assert entry.file_count == 1
    assert load_manifest(registry_root) == [entry]
    assert check_registry(registry_root)[0].entry == entry
