from __future__ import annotations

from n12_cartpole import release
from n12_cartpole.release_audit import (
    PRE_ROLL_CAP_S,
    REFERENCE_DENSIFY_STRIDE,
    SWITCH_TICK,
    TRACKER_LINK_RATE_Q_SCALE,
    audit_release_artifacts,
    wilson95,
)


def test_n12_banked_release_audit_derives_the_full_gate_total() -> None:
    audit = audit_release_artifacts()

    assert [gate["successes"] for gate in audit["gates"]] == [24, 24, 24]
    assert [gate["trials"] for gate in audit["gates"]] == [24, 24, 24]
    assert sum(gate["successes"] for gate in audit["gates"]) == 72
    assert sum(gate["trials"] for gate in audit["gates"]) == 72


def test_n12_release_settings_and_wilson_interval_are_locked() -> None:
    assert PRE_ROLL_CAP_S == 18.0
    assert REFERENCE_DENSIFY_STRIDE == 4
    assert TRACKER_LINK_RATE_Q_SCALE == 0.25
    assert SWITCH_TICK == 9700
    assert wilson95(24, 24) == (0.862, 1.0)


def test_explicit_gate_path_runs_all_three_release_seeds(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    class Completed:
        returncode = 0

    def fake_run(command, *, cwd, env, check):
        assert cwd == release.REPOSITORY
        assert check is False
        calls.append((command, env))
        return Completed()

    monkeypatch.setattr(release.subprocess, "run", fake_run)
    release._run_all_gates()

    assert [(call[0][3], call[0][5]) for call in calls] == [
        ("12345", "6"),
        ("777", "3"),
        ("2024", "3"),
    ]
    for _, environment in calls:
        assert environment["NLINKS"] == "12"
        assert environment["NOM_PATH"] == "runs/r2/nom_n12_4ms_fast.npz"
        assert environment["REFERENCE_DENSIFY_STRIDE"] == "4"
        assert environment["TRACKER_LINK_RATE_Q_SCALE"] == "0.25"
        assert environment["TRACKER_TO_HOLD_SWITCH_TICK"] == "9700"
        assert environment["PREROLL_TOL"] == "0"
        assert environment["PREROLL_VEL_Q_SCALE"] == "4"
