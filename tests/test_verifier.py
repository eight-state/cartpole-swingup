from __future__ import annotations

from pathlib import Path

import numpy as np

from n14_cartpole.verifier import (
    ARTIFACT_PATH,
    EXPECTED_ARTIFACT_SHA256,
    FORCE_BOUND_N,
    run_verifier,
    sha256,
)


def test_locked_witness_replays_to_pass() -> None:
    result = run_verifier()
    assert result["verdict"] == "PASS"
    assert result["failures"] == []
    assert result["metrics"]["longest_success_states"] == 13811
    assert result["metrics"]["state_count"] == 22010
    assert result["metrics"]["start_max_abs_from_exact_hanging"] == 0.0


def test_artifact_hash_is_frozen() -> None:
    assert sha256(ARTIFACT_PATH) == EXPECTED_ARTIFACT_SHA256


def test_over_limit_raw_control_is_rejected(tmp_path: Path) -> None:
    with np.load(ARTIFACT_PATH, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    controls = np.asarray(payload["u"], dtype=np.float64).copy()
    controls[0] = FORCE_BOUND_N + 1.0
    payload["u"] = controls
    hostile = tmp_path / "over-limit.npz"
    np.savez_compressed(hostile, **payload)

    result = run_verifier(hostile)
    assert result["verdict"] == "FAIL"
    assert "raw_force_bound" in result["failures"]
    assert "artifact_sha256" in result["failures"]
