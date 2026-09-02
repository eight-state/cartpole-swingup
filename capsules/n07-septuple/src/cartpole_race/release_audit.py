"""Audit frozen n=7 authorities and banked evidence, then rerun the live stack.

The audit verifies historical perturbation records as bytes and internally
consistent records. It does not support nominal synthesis or perturbation
reruns.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from cartpole_race.release import (
    NOMINAL_PATH,
    NOMINAL_SHA256,
    REPO,
    WORKING,
    ReleaseStack,
    build_release_stack,
    run_live,
    sha256,
    write_json,
)

AUTHORITY_SHA256 = {
    "configs/env-base.yaml": "b0f7a858159149db678ed7b4b5b53d37d2a7c37f6eb1da7eedafaf5d439b62c7",
    "results/nom_n7_dense1ms.npz": NOMINAL_SHA256,
    "src/cartpole_race/discrete_tvlqr.py": "afc41f0b323f3d337fc378eb8383a9ada7aa9d1ff0f1566fc0c1573c0a4bb3d2",
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/cartpole_race/lqr.py": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
}

BANKED_SHA256 = {
    "results/clvalidate_n7_composite_seed12345.json": "db119bb4cf16d07dda2b32467466da469b779cf63729bd586cdada713583105e",
    "results/clvalidate_n7_composite_seed777.json": "87bd6224b395f18288f96207cb728ccf199f58e3243c807b75666b648ae9b07f",
    "results/clvalidate_n7_fixed_seed12345.json": "856564c7ed85f221fd6a7fdb12760d6bb0a8b19900c905d3845d0a81bb6ff999",
    "results/gate_clean_seed12345.log": "e98ad02b4d7574fc712f2cc5e92d0725abf1540452dcfb10acc913623c80630f",
    "results/gate_clean_seed777.log": "69d32206416870e6fb32db03729ee98c4ff61f77e66107a6ff7a350675720815",
    "results/gate_iterbudget_seed12345.log": "5ca3be2bbfd005496242e2a3b50dc7eca84d31b4980bd6276e62d66c45d0acb4",
    "results/gate_iterbudget_seed777.log": "2a6a210f2fa5f97a5a8a9bf619f1af5c4755d89b231d4a3e605fc0e2c16a8568",
}

RECORD_EXPECTATIONS = (
    ("results/clvalidate_n7_composite_seed12345.json", 12345, 24),
    ("results/clvalidate_n7_composite_seed777.json", 777, 24),
    ("results/clvalidate_n7_fixed_seed12345.json", 12345, 18),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_authority_bytes() -> dict[str, str]:
    """Verify that every loaded frozen authority retains its baseline bytes."""
    actual: dict[str, str] = {}
    for relative, expected in AUTHORITY_SHA256.items():
        path = REPO / relative
        _require(path.is_file(), f"missing authority: {relative}")
        digest = sha256(path)
        _require(digest == expected, f"authority bytes changed: {relative}")
        actual[relative] = digest
    return actual


def _wilson_95(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _audit_record(relative: str, expected_seed: int, expected_successes: int) -> dict[str, Any]:
    record = json.loads((REPO / relative).read_text(encoding="utf-8"))
    rows = record["results"]
    successes = sum(bool(row["success"]) for row in rows)
    _require(record["seed"] == expected_seed, f"{relative}: unexpected seed")
    _require(record["n_ic"] == len(rows) == 24, f"{relative}: unexpected row count")
    _require(
        record["n_success"] == successes == expected_successes,
        f"{relative}: success summary disagrees with rows",
    )
    _require(
        record["nominal_sha256"] == NOMINAL_SHA256,
        f"{relative}: nominal provenance disagrees",
    )
    peak = max(float(row["max_force_demanded"]) for row in rows)
    if "wilson_95" in record:
        lower, upper = _wilson_95(successes, len(rows))
        reported_lower, reported_upper = record["wilson_95"]
        _require(
            math.isclose(reported_lower, lower, abs_tol=1e-12)
            and math.isclose(reported_upper, upper, abs_tol=1e-12),
            f"{relative}: Wilson interval disagrees with rows",
        )
    return {
        "file": relative,
        "sha256": BANKED_SHA256[relative],
        "seed": expected_seed,
        "successes": successes,
        "trials": len(rows),
        "peak_demanded_force_n": peak,
    }


def _audit_log(relative: str, expected_peak: float) -> dict[str, Any]:
    text = (REPO / relative).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.lstrip().startswith('{"tag"')]
    matches = re.findall(
        r"\] (\d+)/(\d+) success\s+max_force_demanded=([0-9.]+)N.*\((\d+)s\)", text
    )
    _require(matches, f"{relative}: missing gate footer")
    successes, trials, footer_peak, elapsed_s = matches[-1]
    _require(len(rows) == int(trials) == 24, f"{relative}: unexpected row count")
    _require(sum(bool(row["success"]) for row in rows) == int(successes) == 24, f"{relative}: success count")
    row_peak = max(float(row["max_force_demanded"]) for row in rows)
    _require(math.isclose(row_peak, expected_peak, abs_tol=1e-12), f"{relative}: JSON peak mismatch")
    _require(round(row_peak, 1) == float(footer_peak), f"{relative}: footer peak mismatch")
    return {
        "file": relative,
        "sha256": BANKED_SHA256[relative],
        "successes": int(successes),
        "trials": int(trials),
        "peak_demanded_force_n": row_peak,
        "elapsed_s": int(elapsed_s),
    }


def audit_banked_evidence() -> dict[str, Any]:
    """Verify immutable historical records without rerunning perturbations."""
    for relative, expected in BANKED_SHA256.items():
        _require(sha256(REPO / relative) == expected, f"banked evidence bytes changed: {relative}")

    records = [_audit_record(*expectation) for expectation in RECORD_EXPECTATIONS]
    composite_peaks = {record["seed"]: record["peak_demanded_force_n"] for record in records[:2]}
    logs = [
        _audit_log("results/gate_clean_seed12345.log", composite_peaks[12345]),
        _audit_log("results/gate_clean_seed777.log", composite_peaks[777]),
        _audit_log("results/gate_iterbudget_seed12345.log", composite_peaks[12345]),
        _audit_log("results/gate_iterbudget_seed777.log", composite_peaks[777]),
    ]
    return {
        "scope": "historical banked evidence audited; perturbation reruns are unsupported",
        "records": records,
        "logs": logs,
    }


def audit_nominal(stack: ReleaseStack) -> dict[str, float | int | str]:
    """Recompute the saved nominal's exact-ZOH residual against the live plant."""
    model = stack.model
    spec = model.spec
    substeps = max(1, int(np.ceil(spec.control_dt_s / spec.rk4_max_step_s)))
    substep_s = spec.control_dt_s / substeps
    worst_defect = 0.0
    for state, control, next_state in zip(
        stack.states[:-1], stack.controls, stack.states[1:], strict=True
    ):
        stepped = state.copy()
        for _ in range(substeps):
            stepped = model.rk4_step(stepped, float(control), substep_s)
        worst_defect = max(worst_defect, float(np.max(np.abs(stepped - next_state))))
    peak_force = float(np.max(np.abs(stack.controls)))
    _require(worst_defect < 2e-4, f"saved nominal ZOH defect too large: {worst_defect:.3e}")
    _require(peak_force < 30.0, f"saved nominal force too large: {peak_force:.3f}")
    return {
        "file": str(NOMINAL_PATH.relative_to(REPO)),
        "sha256": NOMINAL_SHA256,
        "control_ticks": len(stack.controls),
        "horizon_s": stack.horizon_s,
        "max_exact_zoh_defect": worst_defect,
        "peak_feedforward_force_n": peak_force,
    }


def main() -> int:
    """Run the full byte, evidence, nominal, and fresh-live audit."""
    authorities = audit_authority_bytes()
    stack = build_release_stack()
    nominal = audit_nominal(stack)
    evidence = audit_banked_evidence()
    live = run_live(stack)
    report = {
        "authority_bytes": authorities,
        "nominal": nominal,
        "banked_evidence": evidence,
        "fresh_live": live.metrics,
    }
    output = WORKING / "n7-verify" / "verification.json"
    write_json(output, report)
    print(f"[n7-verify] {len(authorities)} authority files byte-match")
    print(
        f"[n7-verify] nominal defect={nominal['max_exact_zoh_defect']:.3e} "
        f"rho={live.metrics['rho']:.4g} hold={live.metrics['final_hold_s']:.3f}s"
    )
    print("[n7-verify] banked evidence audited; perturbation reruns unsupported")
    return 0
