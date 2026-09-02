"""Immutable evidence inventory and structural audit for the n=8 release.

The files in ``results/`` are frozen release evidence. This module verifies
identity and internal provenance only; it never regenerates, overwrites, or
promotes a banked result. Fresh states and controls are recomputed separately
by the live simulator in :mod:`cartpole_race.n8`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_EVIDENCE: dict[str, dict[str, str]] = {
    "clvalidate_n8_composite_seed12345.json": {
        "sha256": "3f8db3d7eac50af0da774c08b2c4f2deb4edca2c3e5aa15d399504bd0548ad35",
        "role": "banked composite gate, seed 12345",
    },
    "clvalidate_n8_composite_seed777.json": {
        "sha256": "34dc24d37b1394bab43f0490bdec230917f271c02d98e5415c72445e6678c987",
        "role": "banked composite gate, seed 777",
    },
    "clvalidate_n8_fixed_seed12345.json": {
        "sha256": "3d77d53d964f5cefb12f41ee765dfa6b2b1b75e03e9786c5b8f83a3c369dad4e",
        "role": "banked fixed-nominal gate, seed 12345",
    },
    "clvalidate_n8_fixed_seed777.json": {
        "sha256": "1771ed02ddc430e44a94eb1b2b848b27aab1fdd12e39b8e87aefd71f715f8ab7",
        "role": "banked fixed-nominal gate, seed 777",
    },
    "nom_n8_4ms.npz": {
        "sha256": "cf0c6a23b4f344f0fe8da0714594bb4b0d05b72d07b0e936d8d1326cfbc4c61c",
        "role": "4 ms collocation parent nominal",
    },
    "nom_n8_dense1ms.npz": {
        "sha256": "dd3e64c87485649feb59d0cf2ded147732d70e955ce131911e41b96c22b20892",
        "role": "1 ms nominal used by the live reproduction controller",
    },
}

NOMINAL_FILE = "nom_n8_dense1ms.npz"
PARENT_FILE = "nom_n8_4ms.npz"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one frozen artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape(value: np.ndarray) -> list[int]:
    return list(np.asarray(value).shape)


def _audit_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {name: _shape(data[name]) for name in data.files}


def _audit_gate_json(path: Path, nominal_sha256: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{path.name}: missing results list")
    n_success = sum(bool(row.get("success")) for row in rows)
    if report.get("n_ic") != len(rows) or report.get("n_success") != n_success:
        raise ValueError(f"{path.name}: header count disagrees with result rows")
    if report.get("nominal_sha256") != nominal_sha256:
        raise ValueError(f"{path.name}: nominal SHA disagrees with dense nominal")
    if "n_saturated_ics" in report:
        n_saturated = sum(bool(row.get("saturated")) for row in rows)
        if report["n_saturated_ics"] != n_saturated:
            raise ValueError(f"{path.name}: saturation count disagrees with rows")
    demanded = [row.get("max_force_demanded") for row in rows]
    if all(value is not None for value in demanded):
        derived_max_demand = max(float(value) for value in demanded)
        header_max_demand = report.get("max_force_demanded_over_runs")
        if header_max_demand is not None and derived_max_demand != float(header_max_demand):
            raise ValueError(f"{path.name}: maximum demanded force disagrees with rows")
    else:
        derived_max_demand = None
    return {
        "n_success": n_success,
        "n_ic": len(rows),
        "seed": report.get("seed"),
        "banked_commit_sha": report.get("commit_sha"),
        "banked_git_dirty": report.get("git_dirty"),
        "max_force_demanded_over_runs": report.get("max_force_demanded_over_runs"),
        "derived_max_force_demanded": derived_max_demand,
    }


def audit_frozen_evidence(repo: Path) -> dict[str, Any]:
    """Verify every frozen artifact without changing it.

    The audit checks the release fingerprints, NPZ structure, and the
    self-consistency of each banked JSON.  It deliberately does not treat the
    historical JSON outcomes as newly recomputed results.
    """
    results = repo / "results"
    artifacts: dict[str, dict[str, Any]] = {}
    for name, expected in FROZEN_EVIDENCE.items():
        path = results / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen evidence: {path}")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(
                f"{name}: SHA-256 mismatch; expected {expected['sha256']}, got {actual}"
            )
        artifacts[name] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "role": expected["role"],
        }

    dense = results / NOMINAL_FILE
    parent = results / PARENT_FILE
    dense_shape = _audit_npz(dense)
    parent_shape = _audit_npz(parent)
    if dense_shape != {"x": [9001, 18], "u": [9000], "horizon": [], "n": [], "force": []}:
        raise ValueError(f"{NOMINAL_FILE}: unexpected structure {dense_shape}")
    if parent_shape != {
        "x": [2251, 18], "u": [2250], "horizon": [], "n": [], "force": [], "n_nodes": []
    }:
        raise ValueError(f"{PARENT_FILE}: unexpected structure {parent_shape}")
    nominal_sha = artifacts[NOMINAL_FILE]["sha256"]
    gates = {
        name: _audit_gate_json(results / name, nominal_sha)
        for name in sorted(name for name in FROZEN_EVIDENCE if name.startswith("clvalidate_"))
    }
    return {
        "artifacts": artifacts,
        "loaded_artifacts": {
            "dense_nominal": {"file": NOMINAL_FILE, "arrays": dense_shape},
            "parent_nominal": {"file": PARENT_FILE, "arrays": parent_shape},
        },
        "banked_gate_audit": gates,
    }
