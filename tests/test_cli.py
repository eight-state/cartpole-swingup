from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from conftest import create_capsule

from cartpole_capsules.cli import main, verification_steps


def test_list_empty_registry(registry_root: Path, capsys: object) -> None:
    assert main(("--root", str(registry_root), "list")) == 0
    assert "No capsules registered." in capsys.readouterr().out


def test_check_reports_count(registry_root: Path, capsys: object) -> None:
    create_capsule(registry_root)
    assert main(("--root", str(registry_root), "check")) == 0
    assert "Verified 1 registered capsule(s)." in capsys.readouterr().out


def test_verification_steps_are_locked_then_capsule_command(registry_root: Path) -> None:
    entry = create_capsule(registry_root)
    assert verification_steps(entry) == (
        ("uv", "sync", "--locked"),
        ("uv", "run", "n5-verify"),
    )


def test_verify_runs_two_safe_subprocesses(registry_root: Path, monkeypatch: object) -> None:
    entry = create_capsule(registry_root)
    calls: list[tuple[tuple[str, ...], Path, bool]] = []

    def fake_run(command: tuple[str, ...], *, cwd: Path, check: bool) -> SimpleNamespace:
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("cartpole_capsules.cli.subprocess.run", fake_run)
    assert main(("--root", str(registry_root), "verify", "5")) == 0
    expected_directory = registry_root / entry.capsule_path
    assert calls == [
        (("uv", "sync", "--locked"), expected_directory, False),
        (("uv", "run", "n5-verify"), expected_directory, False),
    ]


def test_verify_unknown_rung_fails(registry_root: Path, capsys: object) -> None:
    assert main(("--root", str(registry_root), "verify", "15")) == 1
    assert "no capsule registered for n=15" in capsys.readouterr().out
