from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from n12_cartpole import demo
from n12_cartpole.verifier import (
    ARTIFACT_PATH,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_SOURCE_SHA256,
    SOURCE_PATHS,
    sha256,
)


def test_frozen_source_and_artifact_hashes_match() -> None:
    assert sha256(ARTIFACT_PATH) == EXPECTED_ARTIFACT_SHA256
    assert {name: sha256(path) for name, path in SOURCE_PATHS.items()} == EXPECTED_SOURCE_SHA256


def test_frozen_nominal_has_the_locked_contract_shape() -> None:
    with np.load(ARTIFACT_PATH, allow_pickle=False) as nominal:
        assert nominal["x"].shape == (2501, 26)
        assert nominal["u"].shape == (2500,)
        assert float(nominal["force"]) == 150.0
        assert float(nominal["horizon"]) == 10.0
        assert int(nominal["n"]) == 12
        assert int(nominal["n_nodes"]) == 2500


def test_retained_verification_witness_is_passing() -> None:
    witness_path = Path("artifacts/verification.json")
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    assert witness["verdict"] == "PASS"
    assert witness["expected_witness"]["all_assertions_pass"] is True
    assert witness["first_current_target_contradiction"] is None


def test_expected_witness_pins_invariants_not_platform_sensitive_metrics() -> None:
    expected_path = Path("artifacts/expected-witness.json")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert set(expected["assertions"]["numeric"]) == {
        "numeric_witness.execution.duration_s",
        "numeric_witness.execution.start_max_abs_from_exact_hanging",
        "numeric_witness.forces.overall.max_raw_applied_abs_delta_n",
        "numeric_witness.success_set.continuous_in_success_set_duration_s",
        "numeric_witness.success_set.continuous_in_success_set_samples",
    }


def test_demo_cli_accepts_an_absolute_external_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output = (tmp_path / "demo.gif").resolve()
    monkeypatch.setattr(demo, "render", lambda path: path)

    assert demo.cli(["--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == output.as_posix()
