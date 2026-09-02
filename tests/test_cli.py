from __future__ import annotations

from typing import Any

import pytest

from cartpole_capsules import cli


def test_list_reports_all_rungs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(("list",)) == 0
    output = capsys.readouterr().out
    assert "n=5" in output
    assert "n=14" in output
    assert "n=15" not in output


def test_check_uses_shared_authority_for_every_rung(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[int] = []

    def fake_check(config: Any) -> dict[str, Any]:
        calls.append(config.rung)
        return {"banked_gates": {"files": []}}

    monkeypatch.setattr(cli, "check_rung_authority", fake_check)
    assert cli.main(("check",)) == 0
    assert calls == list(range(5, 15))
    assert "n=14: authority PASS" in capsys.readouterr().out


def test_verify_all_uses_one_shared_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, bool]] = []

    def fake_verify(config: Any, *, no_replay: bool) -> dict[str, Any]:
        calls.append((config.rung, no_replay))
        return {"verdict": "AUDITED"}

    monkeypatch.setattr(cli, "verify_rung", fake_verify)
    monkeypatch.setattr(cli, "_write_json", lambda path, value: None)
    assert cli.main(("verify", "all", "--audit-only")) == 0
    assert calls == [(rung, True) for rung in range(5, 15)]


def test_verify_unknown_rung_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(("verify", "15")) == 1
    assert "unknown rung 15" in capsys.readouterr().out
