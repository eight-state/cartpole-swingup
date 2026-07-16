"""Byte and arithmetic audit for immutable N5 nominal and historical ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

NOMINAL_SHA256 = "6a029c6892a5dcee8851537aabdb20fd4cc21dbabeed4a6d5a843f3f0ec189c1"
EVIDENCE_COMMIT = "fcb4759529cb25a54485588ec58ee0a924939e99"

# Filled from the final, byte-stable numerical runtime.  evidence.py itself is
# the auditor, so it does not attempt the impossible self-hash.
RUNTIME_SHA256 = {
    "src/cartpole_race/__init__.py": "9d465ec45c5f0192bf5664e65722fdfe290d05d0fdc095145026bfccb42f0324",
    "src/cartpole_race/dynamics.py": "6dfb65eae3128ea1a77ae0693ca91d6d3e0cf5fce48aa6d203282b8db22a5468",
    "src/cartpole_race/env_spec.py": "fed1475f165f480bdc50cf0c88d500d3220f4d091f7c81b8407dc97ff2da4e7d",
    "src/cartpole_race/lqr.py": "5e597d7cbce094e8b433024e18e2165acfc24559a97713d88b88bd01571a2772",
    "src/cartpole_race/predicate.py": "fbdd7263d9882395b329a5e52166a54d0a3a08fce62225c04d6e7ac4f534f31d",
    "src/cartpole_race/release.py": "dabf546ac519e3c2b55391b252057a02ffed6fe548fa556408faf2275f81b788",
    "src/cartpole_race/tvlqr.py": "889b815f6a32dccf8e471ebe20bde0349cf40359e698b0051b28f80371127a7c",
}

HISTORICAL = {
    "results/clvalidate_n5_F60_banked_seed12345.json": {
        "sha256": "ae1e06ee709331b4df33239b222608d72d1902a045f7209e4971825d59670513",
        "seed": 12345,
        "sigma": 0.02,
        "trials": 24,
        "successes": 24,
        "max_abs_force": 20.435720985310297,
        "max_abs_force_demanded": 20.435720985310297,
        "n_saturated_ics": 0,
        "max_abs_x": 3.7599322693665687,
    },
    "results/clvalidate_n5_F60_banked_seed999.json": {
        "sha256": "a2b9883a72f92b3845003f92f773d6a7f927d762fd26c08953543e22dda45fb7",
        "seed": 999,
        "sigma": 0.02,
        "trials": 40,
        "successes": 40,
        "max_abs_force": 28.40631097369357,
        "max_abs_force_demanded": 28.40631097369357,
        "n_saturated_ics": 0,
        "max_abs_x": 3.7832230699029474,
    },
    "results/clvalidate_n5_F60_fresh_seed7777.json": {
        "sha256": "60e20c89208dd83abc9be83361233e710e87cd5b7283789532982cc0067d0169",
        "seed": 7777,
        "sigma": 0.02,
        "trials": 24,
        "successes": 24,
        "max_abs_force": 27.79915926285554,
        "max_abs_force_demanded": 27.79915926285554,
        "n_saturated_ics": 0,
        "max_abs_x": 3.8193773370960002,
    },
    "results/clvalidate_n5_F60_stress_seed2024.json": {
        "sha256": "8697e3cd5398c3d68023a2b4a35cb0ea41c3b4eb379f0eb9170ea2e95314440a",
        "seed": 2024,
        "sigma": 0.10,
        "trials": 24,
        "successes": 24,
        "max_abs_force": 60.0,
        "max_abs_force_demanded": 139.89818484261536,
        "n_saturated_ics": 11,
        "max_abs_x": 4.075118855732107,
    },
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), label)


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float, float]:
    fraction = successes / trials
    denominator = 1.0 + z * z / trials
    center = (fraction + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        fraction * (1.0 - fraction) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return fraction, max(0.0, center - half_width), min(1.0, center + half_width)


def audit_authority_bytes(repo: Path) -> dict[str, str]:
    """Check the fixed numerical runtime, nominal, and all ledger bytes."""
    authorities = dict(RUNTIME_SHA256)
    authorities["results/nom_n5_gluck_cont.npz"] = NOMINAL_SHA256
    authorities.update({path: spec["sha256"] for path, spec in HISTORICAL.items()})
    actual: dict[str, str] = {}
    for relative, expected in authorities.items():
        _require(expected != "PENDING", f"unsealed authority hash: {relative}")
        path = repo / relative
        _require(path.is_file(), f"missing authority: {relative}")
        digest = sha256_file(path)
        _require(digest == expected, f"authority bytes changed: {relative}")
        actual[relative] = digest
    return actual


def _audit_predicate(report: dict[str, Any], relative: str) -> None:
    predicate = report.get("predicate")
    expected = {
        "version": "v1",
        "theta_tol_deg": 5.0,
        "thetadot_tol_rad_s": 0.5,
        "x_tol_m": 2.0,
        "xdot_tol_m_s": 0.5,
        "hold_time_s": 5.0,
        "rail_bound_m": 10.0,
        "note": "all n links within tolerances held continuously for the final 5 s; force and track respected over the WHOLE rollout",
    }
    _require(predicate == expected, f"{relative}: predicate drift")


def _audit_report(relative: str, expected: dict[str, Any], repo: Path) -> dict[str, Any]:
    report = json.loads((repo / relative).read_text(encoding="utf-8"))
    _require(report["commit_sha"] == EVIDENCE_COMMIT, f"{relative}: source commit")
    _require(report["seed"] == expected["seed"], f"{relative}: seed")
    _same(float(report["sigma"]), expected["sigma"], f"{relative}: sigma")
    _require(report["n_trials"] == expected["trials"], f"{relative}: trial count")
    _require(report["n_success"] == expected["successes"], f"{relative}: success count")
    _require(report["n_links"] == 5, f"{relative}: n_links")
    _require(report["nominal"] == "results/nom_n5_gluck_cont.npz", f"{relative}: nominal")
    _require(report["nominal_sha256"] == NOMINAL_SHA256, f"{relative}: nominal SHA")
    _same(float(report["nominal_horizon_s"]), 6.0, f"{relative}: horizon")
    _same(float(report["rollout_duration_s"]), 12.0, f"{relative}: duration")
    _require(report["nodes"] == 6000, f"{relative}: nodes")
    _same(float(report["force_limit"]), 60.0, f"{relative}: force limit")
    _same(float(report["monodromy_rho"]), 0.029767942980498303, f"{relative}: rho")
    _audit_predicate(report, relative)

    successes = int(report["n_success"])
    trials = int(report["n_trials"])
    fraction, wilson_lo, wilson_hi = _wilson(successes, trials)
    _same(float(report["frac"]), fraction, f"{relative}: fraction")
    _same(float(report["wilson_lo"]), wilson_lo, f"{relative}: Wilson low")
    _same(float(report["wilson_hi"]), wilson_hi, f"{relative}: Wilson high")

    applied_force = float(report["max_abs_force"])
    demanded_force = float(report["max_abs_force_demanded"])
    saturated_ics = int(report["n_saturated_ics"])
    peak_cart = float(report["max_abs_x"])
    for name, value in (
        ("max_abs_force", applied_force),
        ("max_abs_force_demanded", demanded_force),
        ("max_abs_x", peak_cart),
    ):
        _same(value, expected[name], f"{relative}: {name}")
    _require(saturated_ics == expected["n_saturated_ics"], f"{relative}: saturation count")
    _require(0 <= saturated_ics <= trials, f"{relative}: invalid saturation count")
    _require(applied_force <= float(report["force_limit"]), f"{relative}: applied force")
    _require(demanded_force >= applied_force, f"{relative}: raw force")
    _require(peak_cart <= float(report["predicate"]["rail_bound_m"]), f"{relative}: rail")
    if saturated_ics:
        _require(applied_force == float(report["force_limit"]), f"{relative}: saturation evidence")
        _require(demanded_force > applied_force, f"{relative}: demanded saturation evidence")

    return {
        "file": relative,
        "seed": expected["seed"],
        "sigma": expected["sigma"],
        "successes": successes,
        "trials": trials,
        "fraction": fraction,
        "wilson_95": [wilson_lo, wilson_hi],
        "peak_applied_force_n": applied_force,
        "peak_raw_force_n": demanded_force,
        "saturated_initial_conditions": saturated_ics,
        "peak_cart_m": peak_cart,
        "force_headroom_n": float(report["force_limit"]) - applied_force,
        "rail_margin_m": float(report["predicate"]["rail_bound_m"]) - peak_cart,
    }


def _source_provenance(repo: Path) -> str:
    """Report historical commit-object availability without claiming inspection."""
    try:
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{EVIDENCE_COMMIT}^{{commit}}"],
            cwd=repo,
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        return "unavailable: git is absent; historical source is unverified"
    if check.returncode:
        return (
            f"unavailable: embedded commit {EVIDENCE_COMMIT} is absent; "
            "historical source is unverified"
        )
    return (
        f"present: embedded commit {EVIDENCE_COMMIT} exists but was not inspected; "
        "historical source is unverified"
    )


def audit_historical_reports(repo: Path) -> dict[str, Any]:
    """Audit the four frozen ledgers without recreating any perturbation case."""
    rows = [_audit_report(path, expected, repo) for path, expected in HISTORICAL.items()]
    by_sigma: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_sigma.setdefault(f"{row['sigma']:.2f}", {"successes": 0, "trials": 0})
        bucket["successes"] += row["successes"]
        bucket["trials"] += row["trials"]
    return {
        "scope": "historical ledger arithmetic and bounds audited; perturbation reruns unsupported",
        "source_provenance": _source_provenance(repo),
        "reports": rows,
        "by_sigma": by_sigma,
    }
