"""Audit disclosed N12 evidence, then recompute the same live rollout as the demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from n12_cartpole.simulator import (
    FORCE_BOUND_N,
    N_LINKS,
    NOMINAL_PATH,
    REPOSITORY,
    SWITCH_TICK,
    load_frozen_nominal,
    run_live_rollout,
)
from n12_cartpole.success import assess_rollout

EVIDENCE_PATH = REPOSITORY / "artifacts" / "n12-evidence.json"
DEFAULT_OUTPUT = REPOSITORY / ".working" / "n12-verify.json"


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one frozen artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson95(successes: int, trials: int, z_score: float = 1.96) -> tuple[float, float]:
    """Re-derive the rounded Wilson interval stored in the banked records."""
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (proportion + z_score**2 / (2.0 * trials)) / denominator
    half_width = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials + z_score**2 / (4.0 * trials**2)
    ) / denominator
    return (round(center - half_width, 4), round(min(1.0, center + half_width), 4))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_evidence() -> dict[str, Any]:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "evidence manifest must contain an object")
    return payload


def _audit_banked_gate(record: dict[str, Any]) -> dict[str, int | float]:
    required = {
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
    _require(set(record) == required, "banked gate has an unexpected schema")
    _require(record["controller"] == "preroll_down_lqr+tvlqr_track+static_hold", "unexpected banked controller")
    _require(record["n_links"] == N_LINKS, "banked gate is not N12")
    _require(record["nominal"] == "runs/r2/nom_n12_4ms_fast.npz", "unexpected historic nominal path")
    _require(record["sigma"] == 0.02, "unexpected banked perturbation sigma")
    _require(record["T_pre_s"] == 18.0, "unexpected banked pre-roll duration")
    _require(record["pre_roll_tol"] == 0.0, "unexpected banked pre-roll tolerance")
    _require(record["pre_roll_vel_q_scale"] == 4.0, "unexpected banked pre-roll velocity cost")
    _require(record["tracker_link_rate_q_scale"] == 0.25, "unexpected banked tracker cost")
    _require(record["reference_densified_from_coarse"] is True, "banked reference was not densified")
    _require(record["reference_densify_stride"] == 4, "unexpected banked densification")
    _require(record["tracker_to_hold_switch_tick"] == SWITCH_TICK, "unexpected banked switch tick")
    _require(math.isclose(record["tracker_to_hold_switch_time_s"], 9.7, abs_tol=1e-12), "unexpected banked switch time")
    _require(record["hold_window_s"] == 10.0, "unexpected banked hold window")
    results = record["results"]
    _require(isinstance(results, list) and len(results) == 24, "banked gate must contain 24 trials")
    _require(record["n_ic"] == len(results), "banked trial count disagrees with records")
    _require([result["tag"] for result in results] == list(range(24)), "banked tags are not ordered")
    record_keys = {
        "tag", "success", "handoff_deg", "hold_s", "peakF", "pert_deg", "resid",
        "t_pre", "tracker_ticks", "track_ok", "fail",
    }
    for result in results:
        _require(set(result) == record_keys, "banked trial has an unexpected schema")
        _require(result["success"] is True and result["track_ok"] is True, "banked failure record")
        _require(result["fail"] is None, "banked failure label")
        _require(result["tracker_ticks"] == SWITCH_TICK, "banked tracker schedule")
        _require(all(math.isfinite(float(result[key])) for key in ("handoff_deg", "hold_s", "peakF", "pert_deg", "resid", "t_pre")), "banked nonfinite measurement")
        _require(float(result["handoff_deg"]) <= 20.0, "banked handoff limit")
        _require(float(result["hold_s"]) >= 5.0, "banked hold predicate")
        _require(float(result["peakF"]) <= FORCE_BOUND_N, "banked applied force bound")
        _require(record["T_pre_s"] == float(result["t_pre"]), "banked pre-roll use")
    successes = sum(result["success"] is True for result in results)
    _require(successes == record["n_success"] == 24, "banked success count disagrees with records")
    _require(any(float(result["peakF"]) == FORCE_BOUND_N for result in results), "banked saturation evidence missing")
    _require(tuple(record["wilson95"]) == wilson95(successes, len(results)), "banked Wilson interval")
    return {"seed": int(record["seed"]), "successes": successes, "trials": len(results)}


def _historical_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    """Map immutable legacy evidence into a provenance-limited historical summary."""
    historic = evidence["unperturbed_achievement"]["historic_observation"]
    return {
        "classification": "immutable_historical_summary_not_live_evidence",
        "legacy_input_field_mapping": {"continuous_hold_s": "sampled_hold_s"},
        "provenance_limitations": {
            "historic_nominal_path": "runs/r2/nom_n12_4ms_fast.npz",
            "historic_nominal_path_present_in_checkout": False,
            "historic_nominal_digest_retained": False,
            "primary_trial_inputs_retained": False,
            "primary_trial_traces_retained": False,
        },
        "stored_observation": {
            "duration_s": historic["duration_s"],
            "raw_and_applied_force_peak_n": historic[
                "raw_and_applied_force_peak_n"
            ],
            "peak_abs_cart_m": historic["peak_abs_cart_m"],
            "switch_max_wrapped_link_angle_deg": historic[
                "switch_max_wrapped_link_angle_deg"
            ],
            "switch_max_link_rate_rad_s": historic["switch_max_link_rate_rad_s"],
            "sampled_hold_s": historic["continuous_hold_s"],
        },
    }


def audit_loaded_evidence() -> dict[str, Any]:
    """Audit frozen inputs and label historical data as a limited stored summary."""
    evidence = _read_evidence()
    _require(evidence.get("schema_version") == 1, "unexpected evidence schema")
    _require(evidence.get("release") == "N12", "unexpected evidence release")
    _require(evidence.get("capabilities", {}).get("nominal_synthesis") is False, "nominal synthesis must stay unsupported")
    _require(evidence.get("capabilities", {}).get("perturbed_gate_rerun") is False, "banked gates must stay loaded evidence")
    nominal_claim = evidence["frozen_nominal"]
    nominal = load_frozen_nominal()
    _require(nominal_claim["path"] == "artifacts/nom_n12_4ms_fast.npz", "unexpected nominal path")
    _require(sha256(NOMINAL_PATH) == nominal_claim["sha256"], "frozen nominal SHA256 mismatch")
    _require(nominal.states.shape == tuple(nominal_claim["state_shape"]), "frozen nominal state shape")
    _require(nominal.controls.shape == tuple(nominal_claim["control_shape"]), "frozen nominal control shape")
    _require(bool(np.all(np.isfinite(nominal.states))), "frozen nominal has nonfinite state")
    _require(bool(np.all(np.isfinite(nominal.controls))), "frozen nominal has nonfinite control")
    _require(nominal.metadata == nominal_claim["metadata"], "frozen nominal metadata")

    records = evidence["banked_gate"]["records"]
    _require(isinstance(records, list), "banked gate records must be a list")
    audited_gates = [_audit_banked_gate(record) for record in records]
    _require([gate["seed"] for gate in audited_gates] == [12345, 777, 2024], "unexpected banked seeds")
    successes = sum(int(gate["successes"]) for gate in audited_gates)
    trials = sum(int(gate["trials"]) for gate in audited_gates)
    _require(successes == 72 and trials == 72, "banked aggregate is not 72/72")
    _require(
        evidence["banked_gate"]["summary"] == {"successes": successes, "trials": trials},
        "banked summary disagrees with records",
    )
    historical_summary = _historical_summary(evidence)
    return {
        "evidence": evidence,
        "historical_summary": historical_summary,
        "frozen_nominal": {
            "sha256": nominal_claim["sha256"],
            "state_shape": list(nominal.states.shape),
            "control_shape": list(nominal.controls.shape),
            "metadata": nominal.metadata,
        },
        "banked_gate": {"seeds": [gate["seed"] for gate in audited_gates], "successes": successes, "trials": trials},
    }


def run_verifier() -> dict[str, Any]:
    """Return loaded-artifact audit and independently recomputed live evidence."""
    loaded = audit_loaded_evidence()
    rollout = run_live_rollout()
    live = assess_rollout(rollout)
    reference_ok = bool(
        rollout.dense_states.shape == (10001, 26)
        and rollout.dense_controls.shape == (10000,)
        and np.array_equal(
            rollout.dense_controls,
            np.repeat(rollout.nominal.controls, 4),
        )
    )
    historic = loaded["historical_summary"]["stored_observation"]
    switch_state = live["success_set"]["switch_state"]
    live_observation = {
        "duration_s": live["execution"]["duration_s"],
        "raw_and_applied_force_peak_n": live["forces"]["raw_peak_abs_n"],
        "peak_abs_cart_m": live["track"]["peak_abs_cart_m"],
        "switch_max_wrapped_link_angle_deg": switch_state[
            "max_wrapped_link_angle_deg"
        ],
        "switch_max_link_rate_rad_s": switch_state["max_link_rate_rad_s"],
        "sampled_hold_s": live["success_set"]["sampled_hold_s"],
    }
    observation_deltas = {
        key: float(live_observation[key] - historic[key])
        for key in live_observation
    }
    checks = {
        "frozen_nominal_and_banked_evidence": True,
        "no_nominal_synthesis_supported": loaded["evidence"]["capabilities"]["nominal_synthesis"] is False,
        "banked_gates_are_loaded_not_rerun": loaded["evidence"]["capabilities"]["perturbed_gate_rerun"] is False,
        "reference_is_reset_densified_from_loaded_nominal": reference_ok,
        "exact_hanging_start": bool(
            np.array_equal(rollout.states[0], rollout.model.x_equilibrium("down"))
        ),
        "live_policy_schedule": live["execution"]["phase_sequence_valid"]
        and live["execution"]["time_grid_exact_1khz"],
        "finite_live_values": all(live["finite"].values()),
        "no_raw_force_violation": live["forces"]["first_raw_over_force_bound"] is None,
        "no_simulator_clipping": live["forces"]["first_clipping"] is None,
        "track_bound": live["track"]["first_exceedance"] is None,
        "switch_in_locked_success_set": live["success_set"]["switch_state"]["in_success_set"],
        "sampled_locked_hold_at_least_5_s": live["success_set"]["sampled_hold_s"] >= 5.0,
        "full_static_window_has_every_1khz_sample_in_success_set": live[
            "success_set"
        ]["every_1khz_sample_from_switch_through_final_in_success_set"],
    }
    return {
        "schema_version": 1,
        "loaded": {
            "frozen_nominal": loaded["frozen_nominal"],
            "banked_gate": loaded["banked_gate"],
            "historical_summary": loaded["historical_summary"],
        },
        "recomputed": {
            "live_rollout": live,
            "banked_observation_deltas": observation_deltas,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def cli(argv: Sequence[str] | None = None) -> int:
    """Write the audit to ``.working/n12-verify.json`` and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPOSITORY / args.output
    payload = run_verifier()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        display_path = output.relative_to(REPOSITORY)
    except ValueError:
        display_path = output
    summary = {
        "banked_gate": payload["loaded"]["banked_gate"],
        "failed_checks": [
            name for name, passed in payload["checks"].items() if not passed
        ],
        "output": display_path.as_posix(),
        "recomputed_verdict": payload["verdict"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(cli())
