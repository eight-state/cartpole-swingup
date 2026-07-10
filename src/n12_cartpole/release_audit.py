"""Audit the banked N12 release artifacts and their locked configuration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
RUNS = REPOSITORY / "runs" / "r2"
NOMINAL_PATH = RUNS / "nom_n12_4ms_fast.npz"
MANIFEST_PATH = REPOSITORY / "artifacts" / "MANIFEST.json"
SOURCE_LEDGER_PATH = REPOSITORY / "artifacts" / "n12-release-source-sha256.txt"

N_LINKS = 12
SEEDS = (12345, 777, 2024)
TRIALS_PER_SEED = 24
SIGMA = 0.02
PRE_ROLL_CAP_S = 18.0
PRE_ROLL_TOL = 0.0
PRE_ROLL_VEL_Q_SCALE = 4.0
TRACKER_LINK_RATE_Q_SCALE = 0.25
REFERENCE_DENSIFY_STRIDE = 4
SWITCH_TICK = 9700
SWITCH_TIME_S = 9.7
HOLD_WINDOW_S = 10.0
HOLD_REQUIRED_S = 5.0
NOMINAL_REFERENCE = "runs/r2/nom_n12_4ms_fast.npz"

EXPECTED_SOURCE_SHA256 = {
    "src/cartpole_race/__init__.py": "92f02f32168d383b97f3bc2d853456427b14219a239609de480d5c400cc6b5a3",
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/cartpole_race/lqr.py": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
    "src/n12_cartpole/fast_pieces.py": "e49c94f4d763a89911fa6e55fd9a460f14748246c0096d49694429501e1e20a9",
    "scripts/gate_preroll.py": "7b98d48d469c64d4c6017631524daa63b54c1338ffc3118b5b268056ab55d663",
    "tests/test_gate_preroll_config.py": "2c8fa03cd587a81f9d1b196a6ac0a4264b3b9ff5c3ce230421de2cb8b23c3fa8",
}

EXPECTED_SHA256 = {
    "artifacts/n12-release-source-sha256.txt": "14bd927dc58a8b333af93e704d7d5c70478ee73ba1a7b9f1abddd215fbe98971",
    "artifacts/nom_n12_4ms_fast.npz": "bc49f597bc8235f391ff1deeb727a43c1264d10ced3ff5e961e09a1d92b6c2c0",
    "runs/r2/nom_n12_4ms_fast.npz": "bc49f597bc8235f391ff1deeb727a43c1264d10ced3ff5e961e09a1d92b6c2c0",
    "runs/r2/gate_n12_preroll_seed12345.json": "65a79f8d016cb163af59a34c64f102b776a6f7fe69a523a43542e857cda8aa16",
    "runs/r2/gate_n12_preroll_seed777.json": "369e5a07c706059556894d700eee2be1b004095c7e900fa8cd56a8fb33aa3ec3",
    "runs/r2/gate_n12_preroll_seed2024.json": "2cff94caf1d743c8884c02c752bc5e20ad2a11729aea46f49f8828d228f1a6b5",
    "runs/r2/demo_n12.gif": "d9257b2fa58c1cd7e8c51b600fae51dee7116b2f691754af3343408af01df7a5",
    **EXPECTED_SOURCE_SHA256,
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson95(successes: int, trials: int, z_score: float = 1.96) -> tuple[float, float]:
    """Return the rounded Wilson interval used by the release gate."""
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z_score * z_score / trials
    center = (proportion + z_score * z_score / (2 * trials)) / denominator
    half_width = z_score * math.sqrt(
        proportion * (1 - proportion) / trials
        + z_score * z_score / (4 * trials * trials)
    ) / denominator
    return (round(center - half_width, 4), round(min(1.0, center + half_width), 4))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_close(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), label)


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def _audit_hashes() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = REPOSITORY / relative
        _require(path.exists(), f"missing artifact: {relative}")
        digest = sha256_file(path)
        _require(digest == expected, f"SHA256 mismatch: {relative}")
        actual[relative] = digest
    return actual


def _audit_source_ledger() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        SOURCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        digest, separator, relative = line.partition("  ")
        _require(bool(separator and digest and relative), f"malformed source ledger row: {line_number}")
        _require(relative not in rows, f"duplicate source ledger path: {relative}")
        rows[relative] = digest
    _require(rows == EXPECTED_SOURCE_SHA256, "source ledger does not match the released source set")
    for relative, expected in rows.items():
        path = REPOSITORY / relative
        _require(path.is_file(), f"missing released source: {relative}")
        _require(sha256_file(path) == expected, f"source SHA256 mismatch: {relative}")
    return rows


def _audit_nominal() -> dict[str, Any]:
    with np.load(NOMINAL_PATH, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
        n_links = int(np.asarray(data["n"]).item())
        force_bound_n = float(np.asarray(data["force"]).item())
        n_nodes = int(np.asarray(data["n_nodes"]).item())
    _require(states.shape == (2501, 26), "unexpected N12 nominal state shape")
    _require(controls.shape == (2500,), "unexpected N12 nominal control shape")
    _require(n_links == N_LINKS, "unexpected N12 nominal link count")
    _require(n_nodes == 2500, "unexpected N12 nominal node count")
    _require_close(horizon_s, 10.0, "unexpected N12 nominal horizon")
    _require_close(force_bound_n, 150.0, "unexpected N12 nominal force bound")
    _require(bool(np.all(np.isfinite(states))), "N12 nominal states are nonfinite")
    _require(bool(np.all(np.isfinite(controls))), "N12 nominal controls are nonfinite")
    return {
        "controls": len(controls),
        "horizon_s": horizon_s,
        "state_shape": list(states.shape),
    }


def _audit_gate(seed: int) -> dict[str, Any]:
    path = RUNS / f"gate_n12_preroll_seed{seed}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "controller",
        "n_links",
        "nominal",
        "sigma",
        "T_pre_s",
        "pre_roll_tol",
        "pre_roll_vel_q_scale",
        "tracker_link_rate_q_scale",
        "reference_densified_from_coarse",
        "reference_densify_stride",
        "tracker_to_hold_switch_tick",
        "tracker_to_hold_switch_time_s",
        "hold_window_s",
        "n_success",
        "n_ic",
        "seed",
        "wilson95",
        "results",
    }
    _require(set(artifact) == expected_keys, f"unexpected gate keys: {_relative(path)}")
    _require(
        artifact["controller"] == "preroll_down_lqr+tvlqr_track+static_hold",
        f"unexpected controller: {_relative(path)}",
    )
    _require(artifact["n_links"] == N_LINKS, f"unexpected link count: {_relative(path)}")
    _require(artifact["nominal"] == NOMINAL_REFERENCE, f"unexpected nominal: {_relative(path)}")
    _require_close(float(artifact["sigma"]), SIGMA, f"unexpected sigma: {_relative(path)}")
    _require_close(float(artifact["T_pre_s"]), PRE_ROLL_CAP_S, f"unexpected pre-roll cap: {_relative(path)}")
    _require_close(float(artifact["pre_roll_tol"]), PRE_ROLL_TOL, f"unexpected pre-roll tolerance: {_relative(path)}")
    _require_close(
        float(artifact["pre_roll_vel_q_scale"]),
        PRE_ROLL_VEL_Q_SCALE,
        f"unexpected pre-roll velocity scale: {_relative(path)}",
    )
    _require_close(
        float(artifact["tracker_link_rate_q_scale"]),
        TRACKER_LINK_RATE_Q_SCALE,
        f"unexpected tracker velocity scale: {_relative(path)}",
    )
    _require(
        artifact["reference_densified_from_coarse"] is True,
        f"reference was not densified: {_relative(path)}",
    )
    _require(
        artifact["reference_densify_stride"] == REFERENCE_DENSIFY_STRIDE,
        f"unexpected densify stride: {_relative(path)}",
    )
    _require(
        artifact["tracker_to_hold_switch_tick"] == SWITCH_TICK,
        f"unexpected switch tick: {_relative(path)}",
    )
    _require_close(
        float(artifact["tracker_to_hold_switch_time_s"]),
        SWITCH_TIME_S,
        f"unexpected switch time: {_relative(path)}",
    )
    _require_close(
        float(artifact["hold_window_s"]),
        HOLD_WINDOW_S,
        f"unexpected hold window: {_relative(path)}",
    )
    _require(artifact["seed"] == seed, f"unexpected seed: {_relative(path)}")
    _require(artifact["n_ic"] == TRIALS_PER_SEED, f"unexpected trial count: {_relative(path)}")

    records = artifact["results"]
    _require(isinstance(records, list), f"records are not a list: {_relative(path)}")
    _require(len(records) == TRIALS_PER_SEED, f"unexpected record count: {_relative(path)}")
    _require([record["tag"] for record in records] == list(range(TRIALS_PER_SEED)), f"unexpected tags: {_relative(path)}")
    record_keys = {
        "tag",
        "success",
        "handoff_deg",
        "hold_s",
        "peakF",
        "pert_deg",
        "resid",
        "t_pre",
        "tracker_ticks",
        "track_ok",
        "fail",
    }
    for record in records:
        _require(set(record) == record_keys, f"unexpected record keys: {_relative(path)}")
        _require(record["success"] is True, f"unsuccessful trial: {_relative(path)}")
        _require(record["track_ok"] is True, f"track failure: {_relative(path)}")
        _require(record["fail"] is None, f"failure label: {_relative(path)}")
        _require(record["tracker_ticks"] == SWITCH_TICK, f"tracker ticks: {_relative(path)}")
        for key in ("handoff_deg", "hold_s", "peakF", "pert_deg", "resid", "t_pre"):
            _require(math.isfinite(float(record[key])), f"nonfinite {key}: {_relative(path)}")
        _require(float(record["handoff_deg"]) <= 20.0, f"handoff limit: {_relative(path)}")
        _require(float(record["hold_s"]) >= HOLD_REQUIRED_S, f"hold predicate: {_relative(path)}")
        _require(float(record["peakF"]) <= 150.0, f"applied force bound: {_relative(path)}")
        _require_close(float(record["t_pre"]), PRE_ROLL_CAP_S, f"pre-roll duration: {_relative(path)}")

    successes = sum(record["success"] for record in records)
    _require(artifact["n_success"] == successes == TRIALS_PER_SEED, f"success count: {_relative(path)}")
    _require(tuple(artifact["wilson95"]) == wilson95(successes, len(records)), f"Wilson interval: {_relative(path)}")
    return {
        "applied_peak_n": max(float(record["peakF"]) for record in records),
        "seed": seed,
        "successes": successes,
        "trials": len(records),
    }


def _audit_manifest(hashes: dict[str, str]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _require(manifest["schema_version"] == 1, "unexpected artifact manifest schema")
    _require(manifest["release"] == "N12", "unexpected artifact manifest release")
    _require(manifest["sha256"] == hashes, "artifact manifest does not match audited files")


def audit_release_artifacts() -> dict[str, Any]:
    """Verify hashes, metadata, N12 settings, and the 24 plus 24 plus 24 total."""
    hashes = _audit_hashes()
    source_hashes = _audit_source_ledger()
    nominal = _audit_nominal()
    gates = [_audit_gate(seed) for seed in SEEDS]
    successes = sum(gate["successes"] for gate in gates)
    trials = sum(gate["trials"] for gate in gates)
    _require(successes == 72, f"aggregate successes: {successes}")
    _require(trials == 72, f"aggregate trials: {trials}")
    _require(any(gate["applied_peak_n"] == 150.0 for gate in gates), "missing applied saturation record")
    _audit_manifest(hashes)
    return {
        "aggregate_successes": successes,
        "aggregate_trials": trials,
        "gates": gates,
        "hashes": hashes,
        "nominal": nominal,
        "source_hashes": source_hashes,
    }


if __name__ == "__main__":
    print(json.dumps(audit_release_artifacts(), indent=2, sort_keys=True))
