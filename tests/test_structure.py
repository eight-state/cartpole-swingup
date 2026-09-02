"""Lean-monorepo structure and code-budget tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(("git", "ls-files", "-z"))
    return [Path(value.decode("utf-8")) for value in raw.split(b"\0") if value]


def test_one_project_and_one_lockfile() -> None:
    files = tracked_files()
    assert [path for path in files if path.name == "pyproject.toml"] == [Path("pyproject.toml")]
    assert [path for path in files if path.name == "uv.lock"] == [Path("uv.lock")]


def test_rungs_contain_data_not_python_projects() -> None:
    rung_files = [path for path in tracked_files() if path.parts[0] == "rungs"]
    assert not [path for path in rung_files if path.suffix == ".py"]
    assert not [
        path
        for path in rung_files
        if any(part in {"src", "tests", ".github"} for part in path.parts)
    ]


def test_rung_evidence_disables_text_conversion() -> None:
    path = "rungs/n12-duodecuple/artifacts/n12-evidence.json"
    result = subprocess.check_output(("git", "check-attr", "text", "--", path), text=True)
    assert result.strip().endswith("text: unset")


def test_total_tracked_python_budget() -> None:
    python_files = [path for path in tracked_files() if path.suffix == ".py"]
    lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in python_files)
    assert lines <= 5_500
