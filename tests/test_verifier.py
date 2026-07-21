from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from n14_cartpole import release, verifier
from n14_cartpole.release_audit import REPOSITORY

RETAINED_REPORT_PATH = REPOSITORY / "artifacts" / "verification.json"
IMMUTABLE_ARTIFACT_PATHS = (
    "artifacts/n14-witness.npz",
    "artifacts/expected-witness.json",
    "artifacts/provenance.json",
    "artifacts/verification.json",
)


def _retained_report() -> dict[str, object]:
    return json.loads(RETAINED_REPORT_PATH.read_text(encoding="utf-8"))


def _immutable_bytes() -> dict[str, bytes]:
    return {
        relative: (REPOSITORY / relative).read_bytes()
        for relative in IMMUTABLE_ARTIFACT_PATHS
    }


def _single_stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _authority_failure() -> dict[str, object]:
    return {
        "verdict": "FAIL",
        "artifact_count": 0,
        "source_count": 0,
        "artifacts": {},
        "sources": {},
        "failures": [{"path": "README.md", "reason": "sha256_mismatch"}],
    }


def test_locked_witness_replays_to_certifying_pass() -> None:
    result = verifier.run_verifier()
    if result["verdict"] != "PASS":
        pytest.fail(
            json.dumps(
                {
                    "failures": result["failures"],
                    "expected_witness": result["expected_witness"],
                    "metrics": result["metrics"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    assert result["failures"] == []
    assert result["metrics"]["control_count"] == 22009
    assert result["metrics"]["state_count"] == 22010
    assert result["metrics"]["longest_success_states"] >= verifier.REQUIRED_SUCCESS_STATES
    assert result["metrics"]["start_max_abs_from_exact_hanging"] == 0.0
    assert result["expected_witness"]["all_assertions_pass"] == (
        result["expected_witness"]["failures"] == []
    )
    assert result["runtime"] == {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if result["expected_witness"]["all_assertions_pass"]:
        retained = _retained_report()
        result_without_runtime = dict(result)
        retained_without_runtime = dict(retained)
        result_without_runtime.pop("runtime")
        retained_without_runtime.pop("runtime")
        assert result_without_runtime == retained_without_runtime


def test_historical_metric_drift_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_metrics = _retained_report()["metrics"]
    assert isinstance(retained_metrics, dict)
    monkeypatch.setattr(
        verifier,
        "replay_controls",
        lambda controls: {
            "metrics": retained_metrics,
            "success": np.ones(verifier.REQUIRED_SUCCESS_STATES, dtype=bool),
        },
    )
    monkeypatch.setattr(
        verifier,
        "_expected_checks",
        lambda metrics, expected: ["longest_success_first_tick"],
    )

    result = verifier.run_verifier()

    assert result["verdict"] == "PASS"
    assert result["failures"] == []
    assert result["expected_witness"] == {
        "all_assertions_pass": False,
        "failures": ["longest_success_first_tick"],
    }


def test_terminal_success_count_excludes_an_earlier_run() -> None:
    mask = np.asarray([True, True, True, False, True], dtype=bool)
    assert verifier.trailing_true_count(mask) == 1


def test_artifact_hash_is_frozen() -> None:
    assert verifier.sha256(verifier.ARTIFACT_PATH) == verifier.EXPECTED_ARTIFACT_SHA256


def test_authority_failure_prevents_loading_or_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_called = False
    load_called = False

    def unexpected_replay(controls: np.ndarray) -> dict[str, object]:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("authority failure reached replay")

    def unexpected_load(*args: object, **kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("authority failure reached artifact loading")

    monkeypatch.setattr(verifier, "audit_release", lambda: _authority_failure())
    monkeypatch.setattr(verifier, "replay_controls", unexpected_replay)
    monkeypatch.setattr(verifier.np, "load", unexpected_load)

    result = verifier.run_verifier()

    assert result["verdict"] == "FAIL"
    assert result["failures"] == ["release_authority"]
    assert result["release_authority"]["failures"] == _authority_failure()["failures"]
    assert load_called is False
    assert replay_called is False


def test_digest_mismatched_selected_artifact_never_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = tmp_path / "hostile.npz"
    hostile.write_bytes(b"not the frozen witness")

    def unexpected_replay(controls: np.ndarray) -> dict[str, object]:
        raise AssertionError("digest-mismatched artifact reached replay")

    monkeypatch.setattr(
        verifier,
        "audit_release",
        lambda: {
            "verdict": "PASS",
            "sources": dict(verifier.EXPECTED_SOURCE_SHA256),
        },
    )
    monkeypatch.setattr(verifier, "replay_controls", unexpected_replay)

    result = verifier.run_verifier(hostile)

    assert result["verdict"] == "FAIL"
    assert result["failures"] == ["artifact_sha256"]


def test_missing_selected_artifact_never_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_replay(controls: np.ndarray) -> dict[str, object]:
        raise AssertionError("missing artifact reached replay")

    monkeypatch.setattr(
        verifier,
        "audit_release",
        lambda: {
            "verdict": "PASS",
            "sources": dict(verifier.EXPECTED_SOURCE_SHA256),
        },
    )
    monkeypatch.setattr(verifier, "replay_controls", unexpected_replay)

    result = verifier.run_verifier(tmp_path / "missing.npz")

    assert result["verdict"] == "FAIL"
    assert result["failures"] == ["artifact_missing"]


def test_hostile_control_is_visible_to_low_level_replay() -> None:
    replay = verifier.replay_controls(np.asarray([verifier.FORCE_BOUND_N + 1.0]))
    assert replay["metrics"]["peak_force_n"] > verifier.FORCE_BOUND_N


def test_pass_only_outputs_match_retained_bytes_and_preserve_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _immutable_bytes()
    retained_report = _retained_report()
    output_path = tmp_path / "report.json"
    release_root = tmp_path / "release-root"
    release_output = release_root / "artifacts" / "verification.json"
    release_output.parent.mkdir(parents=True)
    monkeypatch.setattr(verifier, "run_verifier", lambda artifact: retained_report)

    assert verifier.cli(["--output", str(output_path)]) == 0
    assert capsys.readouterr().out == ""
    assert output_path.read_bytes() == before["artifacts/verification.json"]

    monkeypatch.setattr(release, "REPOSITORY", release_root)
    assert release.cli([]) == 0
    assert capsys.readouterr().out == ""
    assert release_output.read_bytes() == before["artifacts/verification.json"]
    assert _immutable_bytes() == before


def test_authority_failures_preserve_requested_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "verify-output.json"
    output_path.write_bytes(b"verify sentinel")
    release_root = tmp_path / "release-root"
    release_output = release_root / "artifacts" / "verification.json"
    release_output.parent.mkdir(parents=True)
    release_output.write_bytes(b"release sentinel")
    monkeypatch.setattr(verifier, "audit_release", lambda: _authority_failure())

    assert verifier.cli(["--output", str(output_path)]) == 1
    verify_result = _single_stdout_json(capsys)
    assert verify_result["verdict"] == "FAIL"
    assert verify_result["failures"] == ["release_authority"]
    assert output_path.read_bytes() == b"verify sentinel"

    monkeypatch.setattr(release, "REPOSITORY", release_root)
    assert release.cli([]) == 1
    release_result = _single_stdout_json(capsys)
    assert release_result["verdict"] == "FAIL"
    assert release_result["failures"] == ["release_authority"]
    assert release_output.read_bytes() == b"release sentinel"


def test_atomic_replace_error_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "report.json"
    output_path.write_bytes(b"sentinel")
    monkeypatch.setattr(verifier, "run_verifier", lambda artifact: _retained_report())

    def replace_error(source: Path, target: Path) -> None:
        raise OSError("replace denied")

    monkeypatch.setattr(verifier.os, "replace", replace_error)

    assert verifier.cli(["--output", str(output_path)]) == 2
    result = _single_stdout_json(capsys)
    assert result["verdict"] == "ERROR"
    assert output_path.read_bytes() == b"sentinel"


def test_invalid_command_syntax_uses_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as verify_exit:
        verifier.cli(["--unknown"])
    verify_captured = capsys.readouterr()
    assert verify_exit.value.code == 2
    assert verify_captured.out == ""
    assert "usage:" in verify_captured.err

    with pytest.raises(SystemExit) as release_exit:
        release.cli(["unexpected"])
    release_captured = capsys.readouterr()
    assert release_exit.value.code == 2
    assert release_captured.out == ""
    assert "usage:" in release_captured.err


def _run_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_installed_wheel_fails_closed_without_source_capsule(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    wheel_dir = tmp_path / "wheel"
    environment = tmp_path / "wheel-venv"
    working_directory = tmp_path / "wheel-cwd"
    wheel_dir.mkdir()
    working_directory.mkdir()

    _run_command([uv, "build", "--wheel", "--out-dir", str(wheel_dir)], REPOSITORY)
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = wheel.namelist()
    assert "pyproject.toml" not in members
    assert "src/n14_cartpole/verifier.py" not in members
    assert not any(member.startswith("artifacts/") for member in members)

    _run_command([uv, "venv", "--python", sys.executable, str(environment)], tmp_path)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run_command([uv, "pip", "install", "--python", str(python), str(wheels[0])], tmp_path)

    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    for command in ("n14-verify", "n14-release"):
        completed = subprocess.run(
            [str(scripts / f"{command}{suffix}")],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 1
        report = json.loads(completed.stdout)
        assert report["verdict"] == "FAIL"
        assert "source_capsule_required" in report["failures"]
        assert not (working_directory / "artifacts" / "verification.json").exists()
