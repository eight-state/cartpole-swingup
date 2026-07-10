"""Integrity and invariant checks for the banked N11 release artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "r2"
NOMINAL_DENSE = "nom_n11_dense1ms_capture025_smoke3t03.npz"
NOMINAL_PARENT = "nom_n11_4ms_capture025_smoke3t03.npz"
NOMINAL_REFERENCE = f"runs/r2/{NOMINAL_DENSE}"

EXPECTED_SHA256 = {
    NOMINAL_PARENT: "b190e1ff71fe5242c850e5eb817bf8401fc38f24f9e189e6b132e85471dcea86",
    NOMINAL_DENSE: "1b7458cefe5d91aeaa012e78c4edbf586cd0d989df8e8e6f7adb2000cbae290d",
    "gate_n11_preroll_seed12345.json": "fd7650b59ff15a41ecab6e83e5eab0b16e6331681533b39a4161055d59748f8c",
    "gate_n11_preroll_seed777.json": "a64dfcbce1cfa9ef8235ecd49645059094a613d2f5a29dc6e4d659630d63a756",
    "gate_n11_preroll_seed2024.json": "b7f263356415fc8bef7ed8a7c9f71e3d8bbe71b52beb11805384e30434080942",
}

GATE_SEEDS = (12345, 777, 2024)
N_LINKS = 11
SIGMA = 0.02
T_PRE_S = 9.0
PRE_ROLL_TOL = 0.0
PRE_ROLL_VEL_Q_SCALE = 4.0
HOLD_WINDOW_S = 10.0
HOLD_REQUIRED_S = 5.0


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson95(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the release gate's rounded Wilson 95 percent interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (round(center - half_width, 4), round(min(1.0, center + half_width), 4))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_close(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), label)


def _audit_nominal(filename: str, n_nodes: int) -> dict[str, Any]:
    path = RUNS / filename
    _require(path.exists(), f"missing artifact: {path.relative_to(REPO)}")
    digest = sha256_file(path)
    _require(digest == EXPECTED_SHA256[filename], f"SHA256 mismatch: {filename}")
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=float)
        u = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon = float(np.asarray(data["horizon"]).item())
        n_links = int(np.asarray(data["n"]).item())
        force = float(np.asarray(data["force"]).item())
        stored_nodes = int(np.asarray(data["n_nodes"]).item()) if "n_nodes" in data else len(u)
    _require(x.shape == (n_nodes + 1, 24), f"unexpected state shape: {filename}")
    _require(u.shape == (n_nodes,), f"unexpected control shape: {filename}")
    _require(n_links == N_LINKS, f"unexpected link count: {filename}")
    _require(stored_nodes == n_nodes, f"unexpected node count: {filename}")
    _require_close(horizon, 10.0, f"unexpected horizon: {filename}")
    _require_close(force, 150.0, f"unexpected force bound: {filename}")
    _require(bool(np.all(np.isfinite(x))), f"nonfinite state: {filename}")
    _require(bool(np.all(np.isfinite(u))), f"nonfinite control: {filename}")
    return {"sha256": digest, "n_nodes": n_nodes, "horizon_s": horizon}


def _audit_gate(seed: int) -> dict[str, Any]:
    filename = f"gate_n11_preroll_seed{seed}.json"
    path = RUNS / filename
    _require(path.exists(), f"missing artifact: {path.relative_to(REPO)}")
    digest = sha256_file(path)
    _require(digest == EXPECTED_SHA256[filename], f"SHA256 mismatch: {filename}")
    artifact = json.loads(path.read_text(encoding="utf-8"))

    expected_keys = {
        "controller", "n_links", "nominal", "sigma", "T_pre_s",
        "pre_roll_tol", "pre_roll_vel_q_scale", "hold_window_s",
        "n_success", "n_ic", "seed", "wilson95", "results",
    }
    _require(set(artifact) == expected_keys, f"unexpected gate keys: {filename}")
    _require(
        artifact["controller"] == "preroll_down_lqr+tvlqr_track+static_hold",
        f"unexpected controller: {filename}",
    )
    _require(artifact["n_links"] == N_LINKS, f"unexpected link count: {filename}")
    _require(artifact["nominal"] == NOMINAL_REFERENCE, f"unexpected nominal: {filename}")
    _require_close(float(artifact["sigma"]), SIGMA, f"unexpected sigma: {filename}")
    _require_close(float(artifact["T_pre_s"]), T_PRE_S, f"unexpected pre-roll cap: {filename}")
    _require_close(float(artifact["pre_roll_tol"]), PRE_ROLL_TOL, f"unexpected tolerance: {filename}")
    _require_close(
        float(artifact["pre_roll_vel_q_scale"]),
        PRE_ROLL_VEL_Q_SCALE,
        f"unexpected velocity scale: {filename}",
    )
    _require_close(
        float(artifact["hold_window_s"]), HOLD_WINDOW_S, f"unexpected hold window: {filename}"
    )
    _require(artifact["seed"] == seed, f"unexpected seed: {filename}")
    _require(artifact["n_ic"] == 24, f"unexpected trial count: {filename}")

    records = artifact["results"]
    _require(isinstance(records, list) and len(records) == 24, f"unexpected records: {filename}")
    record_keys = {
        "tag", "success", "handoff_deg", "hold_s", "peakF", "pert_deg",
        "resid", "t_pre", "track_ok", "fail",
    }
    _require([record["tag"] for record in records] == list(range(24)), f"tags: {filename}")
    for record in records:
        _require(set(record) == record_keys, f"unexpected record keys: {filename}")
        _require(record["success"] is True, f"unsuccessful trial: {filename}")
        _require(record["track_ok"] is True, f"track failure: {filename}")
        _require(record["fail"] is None, f"failure label: {filename}")
        for key in ("handoff_deg", "hold_s", "peakF", "pert_deg", "resid", "t_pre"):
            _require(math.isfinite(float(record[key])), f"nonfinite {key}: {filename}")
        _require(float(record["handoff_deg"]) <= 20.0, f"handoff limit: {filename}")
        _require(float(record["hold_s"]) >= HOLD_REQUIRED_S, f"hold predicate: {filename}")
        _require(float(record["peakF"]) <= 150.0, f"force bound: {filename}")
        _require_close(float(record["t_pre"]), T_PRE_S, f"pre-roll duration: {filename}")

    success_count = sum(record["success"] for record in records)
    _require(artifact["n_success"] == success_count == 24, f"success count: {filename}")
    _require(tuple(artifact["wilson95"]) == wilson95(success_count, len(records)), f"Wilson interval: {filename}")
    return {"sha256": digest, "seed": seed, "successes": success_count, "trials": len(records)}


def audit_release_artifacts() -> dict[str, Any]:
    """Verify hashes, nominal metadata, gate records, and the 72 trial total."""
    parent = _audit_nominal(NOMINAL_PARENT, 2_500)
    dense = _audit_nominal(NOMINAL_DENSE, 10_000)
    gates = [_audit_gate(seed) for seed in GATE_SEEDS]
    successes = sum(gate["successes"] for gate in gates)
    trials = sum(gate["trials"] for gate in gates)
    _require(successes == 72, f"aggregate successes: {successes}")
    _require(trials == 72, f"aggregate trials: {trials}")
    return {
        "parent": parent,
        "dense": dense,
        "gates": gates,
        "aggregate_successes": successes,
        "aggregate_trials": trials,
    }


if __name__ == "__main__":
    print(json.dumps(audit_release_artifacts(), indent=2, sort_keys=True))
