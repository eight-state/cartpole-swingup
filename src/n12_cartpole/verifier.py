"""Fresh-process verifier for the retained deterministic N12 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error
from n12_cartpole.fast_pieces import FastDTVLQR, make_densifier
from n12_cartpole.success import (
    MAX_ABS_CART_M,
    MAX_ABS_CART_RATE_M_S,
    MAX_LINK_RATE_RAD_S,
    MAX_WRAPPED_LINK_ANGLE_DEG,
    N_LINKS,
    in_success_set,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
JsonScalar: TypeAlias = str | int | float | bool | None
# JSON values come from NPZ metadata and the expected-witness file at runtime.
# Keep their dynamic boundary explicit while every numeric operation stays typed.
JsonValue: TypeAlias = Any
JsonObject: TypeAlias = dict[str, JsonValue]

REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPOSITORY / "artifacts" / "nom_n12_4ms_fast.npz"
EXPECTED_WITNESS_PATH = REPOSITORY / "artifacts" / "expected-witness.json"

CONTROL_DT_S = 0.001
RK4_SUBSTEP_S = 0.00025
FORCE_BOUND_N = 150.0
TRACK_BOUND_M = 10.0
SWITCH_TICK = 9700
TOTAL_TICKS = 21700
PROMOTION_ANGLE_DEG = 0.05
EXPECTED_ARTIFACT_SHA256 = "bc49f597bc8235f391ff1deeb727a43c1264d10ced3ff5e961e09a1d92b6c2c0"
EXPECTED_SOURCE_SHA256: dict[str, str] = {
    "package_init": "92f02f32168d383b97f3bc2d853456427b14219a239609de480d5c400cc6b5a3",
    "dynamics": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "env_spec": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "fast_pieces": "e49c94f4d763a89911fa6e55fd9a460f14748246c0096d49694429501e1e20a9",
    "lqr": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
}
SOURCE_PATHS: dict[str, Path] = {
    "package_init": REPOSITORY / "src" / "cartpole_race" / "__init__.py",
    "dynamics": REPOSITORY / "src" / "cartpole_race" / "dynamics.py",
    "env_spec": REPOSITORY / "src" / "cartpole_race" / "env_spec.py",
    "fast_pieces": REPOSITORY / "src" / "n12_cartpole" / "fast_pieces.py",
    "lqr": REPOSITORY / "src" / "cartpole_race" / "lqr.py",
}


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one repository file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: npt.ArrayLike) -> JsonScalar:
    """Convert a zero-dimensional NumPy value to a JSON scalar."""
    item = np.asarray(value).item()
    if isinstance(item, (str, int, float, bool)) or item is None:
        return item
    raise TypeError(f"NPZ metadata contains unsupported scalar type {type(item)!r}")


def first_event(
    mask: npt.NDArray[np.bool_],
    *,
    values: FloatArray | None = None,
    tick_offset: int = 0,
) -> JsonObject | None:
    """Return the first true entry in ``mask`` with its tick and value."""
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return None
    index = int(indexes[0])
    event: JsonObject = {
        "tick": tick_offset + index,
        "time_s": (tick_offset + index) * CONTROL_DT_S,
    }
    if values is not None:
        event["value"] = float(values[index])
    return event


def state_metrics(model: NLinkCartPole, state: FloatArray, upright: FloatArray) -> JsonObject:
    """Measure one state against the locked upright predicate."""
    error = wrap_state_error(state, upright, N_LINKS)
    return {
        "cart_position_m": float(state[0]),
        "cart_rate_m_s": float(state[N_LINKS + 1]),
        "finite": bool(np.all(np.isfinite(state))),
        "in_success_set": in_success_set(model, state),
        "max_link_rate_rad_s": float(np.max(np.abs(error[N_LINKS + 2 :]))),
        "max_wrapped_link_angle_deg": float(
            np.rad2deg(np.max(np.abs(error[1 : N_LINKS + 1])))
        ),
    }


def suffix_duration(in_set: npt.NDArray[np.bool_]) -> tuple[float, int]:
    """Return the final contiguous in-set duration and sample count."""
    samples = 0
    for value in in_set[::-1]:
        if not value:
            break
        samples += 1
    return max(0.0, (samples - 1) * CONTROL_DT_S), samples


def first_contradiction(
    checks: Sequence[tuple[str, JsonObject | None]],
) -> JsonObject | None:
    """Return the first invariant violation in execution order."""
    for name, event in checks:
        if event is not None:
            return {"invariant": name, **event}
    return None


def read_expected_witness() -> JsonObject:
    """Load the checked machine-readable expected witness."""
    data = json.loads(EXPECTED_WITNESS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("expected witness must contain a JSON object")
    return cast(JsonObject, data)


def nested_value(payload: Mapping[str, JsonValue], dotted_path: str) -> JsonValue:
    """Read a dot-separated path from a JSON object."""
    current: JsonValue = dict(payload)
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def expected_witness_checks(payload: JsonObject, expected: JsonObject) -> JsonObject:
    """Compare the recomputed payload with every expected witness assertion."""
    failures: list[JsonValue] = []
    integrity = cast(Mapping[str, JsonValue], expected["integrity"])
    expected_sources = cast(Mapping[str, JsonValue], integrity["source_sha256"])
    actual_sources = cast(Mapping[str, JsonValue], payload["integrity"])["source_sha256"]
    if actual_sources != dict(expected_sources):
        failures.append("source_sha256")
    if cast(Mapping[str, JsonValue], payload["integrity"])["artifact_sha256"] != integrity["artifact_sha256"]:
        failures.append("artifact_sha256")

    assertions = cast(Mapping[str, JsonValue], expected["assertions"])
    for dotted_path, target in cast(Mapping[str, JsonValue], assertions["equal"]).items():
        if nested_value(payload, dotted_path) != target:
            failures.append(dotted_path)
    for dotted_path, specification in cast(
        Mapping[str, JsonValue], assertions["numeric"]
    ).items():
        spec = cast(Mapping[str, JsonValue], specification)
        actual = nested_value(payload, dotted_path)
        value = spec["value"]
        tolerance = spec["atol"]
        if not isinstance(actual, (int, float)) or not isinstance(value, (int, float)):
            failures.append(dotted_path)
            continue
        if not isinstance(tolerance, (int, float)) or abs(actual - value) > tolerance:
            failures.append(dotted_path)
    return {"all_assertions_pass": not failures, "failures": failures}


def run_verifier() -> JsonObject:
    """Recompute the N12 result from the frozen nominal in one live rollout."""
    source_hashes = {name: sha256(path) for name, path in SOURCE_PATHS.items()}
    artifact_hash = sha256(ARTIFACT_PATH)
    with np.load(ARTIFACT_PATH, allow_pickle=False) as data:
        source_states = np.asarray(data["x"], dtype=float)
        source_controls = np.asarray(data["u"], dtype=float).reshape(-1)
        metadata = {
            key: scalar(data[key])
            for key in data.files
            if key not in {"x", "u"}
        }

    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1] * N_LINKS,
        link_lengths_m=[0.5] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND_N,
        track_half_length_m=TRACK_BOUND_M,
        control_rate_hz=1000.0,
        rk4_max_step_s=RK4_SUBSTEP_S,
    )
    model = NLinkCartPole(spec)
    hanging = model.x_equilibrium("down")
    upright = model.x_equilibrium("up")
    densify = make_densifier(model, CONTROL_DT_S, 4, 4, len(source_controls))
    dense_states, dense_controls = densify(source_states, source_controls)

    tracking_q = make_Q(N_LINKS)
    tracking_q[N_LINKS + 2 :, N_LINKS + 2 :] *= 0.25
    _, tracking_terminal_p = static_lqr(model, Q=tracking_q, R=make_R())
    tracker = FastDTVLQR(
        model,
        dense_states,
        dense_controls,
        CONTROL_DT_S,
        Qf=tracking_terminal_p,
        Q=tracking_q,
        R=make_R(),
    )
    static_gain, static_p = static_lqr(model)
    static_gain = np.asarray(static_gain).reshape(-1)

    raw_forces: list[float] = []
    phases: list[str] = []

    def live_policy(state: FloatArray, time_s: float) -> float:
        tick = int(round(time_s / CONTROL_DT_S))
        if tick < SWITCH_TICK:
            raw_force = float(tracker.policy(state, time_s))
            phase = "tvlqr"
        else:
            raw_force = -float(static_gain @ wrap_state_error(state, upright, N_LINKS))
            phase = "static_care"
        raw_forces.append(raw_force)
        phases.append(phase)
        return raw_force

    times, states, applied_forces = model.rollout_zoh(
        hanging,
        live_policy,
        TOTAL_TICKS * CONTROL_DT_S,
        CONTROL_DT_S,
        RK4_SUBSTEP_S,
    )
    raw_forces_array = np.asarray(raw_forces, dtype=float)
    phases_array = np.asarray(phases, dtype=str)
    prefix_states = states[: SWITCH_TICK + 1]
    prefix_raw = raw_forces_array[:SWITCH_TICK]
    prefix_applied = applied_forces[:SWITCH_TICK]
    hold_states = states[SWITCH_TICK:]
    hold_raw = raw_forces_array[SWITCH_TICK:]
    hold_applied = applied_forces[SWITCH_TICK:]
    all_in_set = np.asarray(
        [in_success_set(model, state) for state in states], dtype=bool
    )
    hold_in_set = all_in_set[SWITCH_TICK:]
    continuous_hold_s, continuous_hold_samples = suffix_duration(hold_in_set)

    reference_error = np.asarray(
        [
            wrap_state_error(state, reference, N_LINKS)
            for state, reference in zip(prefix_states, dense_states[: SWITCH_TICK + 1])
        ]
    )
    reference_max_angle_deg = np.rad2deg(
        np.max(np.abs(reference_error[:, 1 : N_LINKS + 1]), axis=1)
    )
    state_finite = np.all(np.isfinite(states), axis=1)
    raw_finite = np.isfinite(raw_forces_array)
    applied_finite = np.isfinite(applied_forces)
    force_delta = applied_forces - raw_forces_array
    clipping_mask = np.abs(force_delta) > 0.0
    raw_excess = np.abs(raw_forces_array) > FORCE_BOUND_N
    track_excess = np.abs(states[:, 0]) > TRACK_BOUND_M
    hold_out_of_set = ~hold_in_set

    input_compatibility: JsonObject = {
        "artifact_sha256_matches_frozen_value": artifact_hash == EXPECTED_ARTIFACT_SHA256,
        "dense_controls_are_repeated_source_controls": bool(
            np.array_equal(dense_controls, np.repeat(source_controls, 4))
        ),
        "dense_shapes_match": bool(
            dense_states.shape == (10001, 26) and dense_controls.shape == (10000,)
        ),
        "dense_start_max_abs_from_exact_hanging": float(
            np.max(np.abs(dense_states[0] - hanging))
        ),
        "force_bound_matches": spec.force_bound_n == FORCE_BOUND_N,
        "metadata": metadata,
        "metadata_matches": bool(
            metadata.get("n") == N_LINKS
            and metadata.get("force") == FORCE_BOUND_N
            and metadata.get("horizon") == 10.0
            and metadata.get("n_nodes") == 2500
        ),
        "plant_timing_matches": bool(
            spec.control_dt_s == CONTROL_DT_S
            and spec.rk4_max_step_s == RK4_SUBSTEP_S
        ),
        "rk4_substeps_per_control_tick": int(
            np.ceil(CONTROL_DT_S / RK4_SUBSTEP_S)
        ),
        "source_files_match_frozen_values": source_hashes == EXPECTED_SOURCE_SHA256,
        "source_finite": bool(
            np.all(np.isfinite(source_states)) and np.all(np.isfinite(source_controls))
        ),
        "source_shapes_match": bool(
            source_states.shape == (2501, 26) and source_controls.shape == (2500,)
        ),
        "source_start_max_abs_from_exact_hanging": float(
            np.max(np.abs(source_states[0] - hanging))
        ),
        "track_bound_matches": spec.track_half_length_m == TRACK_BOUND_M,
    }

    numeric_witness: JsonObject = {
        "execution": {
            "duration_s": TOTAL_TICKS * CONTROL_DT_S,
            "phase_sequence_valid": bool(
                np.all(phases_array[:SWITCH_TICK] == "tvlqr")
                and np.all(phases_array[SWITCH_TICK:] == "static_care")
            ),
            "single_rollout_zoh_call": True,
            "start_max_abs_from_exact_hanging": float(
                np.max(np.abs(states[0] - hanging))
            ),
            "start_state": "model.x_equilibrium('down')",
            "static_care_ticks": [SWITCH_TICK, TOTAL_TICKS - 1],
            "stored_controls_replayed": False,
            "stored_state_injected": False,
            "switch_tick": SWITCH_TICK,
            "switch_time_s": SWITCH_TICK * CONTROL_DT_S,
            "time_grid_exact_1khz": bool(
                np.array_equal(
                    times,
                    np.arange(TOTAL_TICKS + 1, dtype=float) * CONTROL_DT_S,
                )
            ),
            "total_ticks": TOTAL_TICKS,
            "tvlqr_ticks": [0, SWITCH_TICK - 1],
        },
        "finite": {
            "all_applied_forces": bool(np.all(applied_finite)),
            "all_raw_forces": bool(np.all(raw_finite)),
            "all_states": bool(np.all(state_finite)),
            "first_nonfinite_applied_force": first_event(~applied_finite),
            "first_nonfinite_raw_force": first_event(~raw_finite),
            "first_nonfinite_state": first_event(~state_finite),
        },
        "forces": {
            "overall": {
                "applied_peak_abs_n": float(np.max(np.abs(applied_forces))),
                "first_clipping": first_event(clipping_mask, values=force_delta),
                "first_raw_over_force_bound": first_event(
                    raw_excess, values=raw_forces_array
                ),
                "max_raw_applied_abs_delta_n": float(np.max(np.abs(force_delta))),
                "raw_peak_abs_n": float(np.max(np.abs(raw_forces_array))),
            },
            "static_hold": {
                "applied_peak_abs_n": float(np.max(np.abs(hold_applied))),
                "first_applied_force_n": float(hold_applied[0]),
                "first_clipping": first_event(
                    np.abs(hold_applied - hold_raw) > 0.0,
                    values=hold_applied - hold_raw,
                    tick_offset=SWITCH_TICK,
                ),
                "first_raw_force_n": float(hold_raw[0]),
                "first_raw_over_force_bound": first_event(
                    np.abs(hold_raw) > FORCE_BOUND_N,
                    values=hold_raw,
                    tick_offset=SWITCH_TICK,
                ),
                "max_raw_applied_abs_delta_n": float(
                    np.max(np.abs(hold_applied - hold_raw))
                ),
                "raw_peak_abs_n": float(np.max(np.abs(hold_raw))),
            },
            "tvlqr_prefix": {
                "applied_peak_abs_n": float(np.max(np.abs(prefix_applied))),
                "first_clipping": first_event(
                    np.abs(prefix_applied - prefix_raw) > 0.0,
                    values=prefix_applied - prefix_raw,
                ),
                "first_raw_over_force_bound": first_event(
                    np.abs(prefix_raw) > FORCE_BOUND_N, values=prefix_raw
                ),
                "max_raw_applied_abs_delta_n": float(
                    np.max(np.abs(prefix_applied - prefix_raw))
                ),
                "raw_peak_abs_n": float(np.max(np.abs(prefix_raw))),
            },
        },
        "promotion_screen_separate_from_current_target": {
            "does_not_change_current_target_verdict": True,
            "evaluated_only_on_live_tvlqr_prefix": [0, SWITCH_TICK],
            "first_error_over_0_05_deg": first_event(
                reference_max_angle_deg > PROMOTION_ANGLE_DEG,
                values=reference_max_angle_deg,
            ),
            "max_reference_error_deg_through_switch": float(
                np.max(reference_max_angle_deg)
            ),
            "part_of_current_target": False,
            "passes_promotion_screen": bool(
                np.all(reference_max_angle_deg <= PROMOTION_ANGLE_DEG)
            ),
            "screen": "reset-densified-reference maximum wrapped link-angle error <= 0.05 deg",
        },
        "static_controller": {
            "construction": "default static_lqr(model): continuous CARE with make_Q()/make_R()",
            "gain_peak_abs": float(np.max(np.abs(static_gain))),
            "p_condition_2": float(np.linalg.cond(static_p)),
            "q_diagonal": np.diag(make_Q(N_LINKS)).tolist(),
            "r": float(make_R()[0, 0]),
        },
        "success_set": {
            "continuous_in_success_set_duration_s": continuous_hold_s,
            "continuous_in_success_set_samples": continuous_hold_samples,
            "every_state_from_switch_through_final_in_success_set": bool(
                np.all(hold_in_set)
            ),
            "final_state": state_metrics(model, states[-1], upright),
            "first_hold_state_out_of_success_set": first_event(
                hold_out_of_set, tick_offset=SWITCH_TICK
            ),
            "locked_predicate": {
                "max_abs_cart_m": MAX_ABS_CART_M,
                "max_abs_cart_rate_m_s": MAX_ABS_CART_RATE_M_S,
                "max_link_rate_rad_s": MAX_LINK_RATE_RAD_S,
                "max_wrapped_link_angle_deg": MAX_WRAPPED_LINK_ANGLE_DEG,
            },
            "required_current_target_duration_s": 5.0,
            "stronger_full_hold_duration_s": 12.0,
            "switch_in_success_set": bool(hold_in_set[0]),
        },
        "switch_state": state_metrics(model, states[SWITCH_TICK], upright),
        "track_bound": {
            "bound_abs_m": TRACK_BOUND_M,
            "combined_peak_abs_cart_m": float(np.max(np.abs(states[:, 0]))),
            "first_exceedance": first_event(track_excess, values=np.abs(states[:, 0])),
            "hold_peak_abs_cart_m": float(np.max(np.abs(hold_states[:, 0]))),
            "prefix_peak_abs_cart_m": float(np.max(np.abs(prefix_states[:, 0]))),
        },
    }

    compatibility_ok = bool(
        input_compatibility["artifact_sha256_matches_frozen_value"]
        and input_compatibility["source_files_match_frozen_values"]
        and input_compatibility["metadata_matches"]
        and input_compatibility["source_shapes_match"]
        and input_compatibility["source_finite"]
        and input_compatibility["dense_shapes_match"]
        and float(input_compatibility["dense_start_max_abs_from_exact_hanging"])
        <= 1e-12
        and input_compatibility["dense_controls_are_repeated_source_controls"]
        and input_compatibility["plant_timing_matches"]
        and input_compatibility["rk4_substeps_per_control_tick"] == 4
        and input_compatibility["force_bound_matches"]
        and input_compatibility["track_bound_matches"]
    )
    checks = [
        (
            "input_compatibility",
            None if compatibility_ok else {"tick": None, "time_s": None},
        ),
        (
            "exact_hanging_start",
            None
            if numeric_witness["execution"]["start_max_abs_from_exact_hanging"] <= 1e-12
            else {
                "tick": 0,
                "time_s": 0.0,
                "value": numeric_witness["execution"]["start_max_abs_from_exact_hanging"],
            },
        ),
        (
            "live_policy_schedule",
            None
            if numeric_witness["execution"]["phase_sequence_valid"]
            and numeric_witness["execution"]["time_grid_exact_1khz"]
            else {"tick": None, "time_s": None},
        ),
        ("finite_state", numeric_witness["finite"]["first_nonfinite_state"]),
        ("finite_raw_force", numeric_witness["finite"]["first_nonfinite_raw_force"]),
        (
            "finite_applied_force",
            numeric_witness["finite"]["first_nonfinite_applied_force"],
        ),
        (
            "raw_force_bound",
            numeric_witness["forces"]["overall"]["first_raw_over_force_bound"],
        ),
        ("no_clipping", numeric_witness["forces"]["overall"]["first_clipping"]),
        ("track_bound", numeric_witness["track_bound"]["first_exceedance"]),
        (
            "switch_in_success_set",
            None
            if numeric_witness["success_set"]["switch_in_success_set"]
            else {"tick": SWITCH_TICK, "time_s": SWITCH_TICK * CONTROL_DT_S},
        ),
        (
            "first_static_raw_force_within_100n",
            None
            if abs(numeric_witness["forces"]["static_hold"]["first_raw_force_n"])
            <= 100.0
            else {
                "tick": SWITCH_TICK,
                "time_s": SWITCH_TICK * CONTROL_DT_S,
                "value": numeric_witness["forces"]["static_hold"]["first_raw_force_n"],
            },
        ),
        (
            "full_static_hold_in_success_set",
            numeric_witness["success_set"]["first_hold_state_out_of_success_set"],
        ),
        (
            "continuous_in_success_set_duration",
            None
            if numeric_witness["success_set"]["continuous_in_success_set_duration_s"]
            >= 5.0
            else {
                "tick": TOTAL_TICKS,
                "time_s": TOTAL_TICKS * CONTROL_DT_S,
                "value": numeric_witness["success_set"]["continuous_in_success_set_duration_s"],
            },
        ),
    ]
    contradiction = first_contradiction(cast(Sequence[tuple[str, JsonObject | None]], checks))
    payload: JsonObject = {
        "contract": {
            "control_dt_s": CONTROL_DT_S,
            "force_bound_n": FORCE_BOUND_N,
            "n_links": N_LINKS,
            "rk4_substeps_per_control_tick": 4,
            "rk4_substep_s": RK4_SUBSTEP_S,
            "switch_tick": SWITCH_TICK,
            "total_ticks": TOTAL_TICKS,
            "track_bound_m": TRACK_BOUND_M,
            "velocity_q_scale": 0.25,
        },
        "current_target": "one exact unperturbed 1 kHz live Qv x0.25 TVLQR swing-up from hanging, then default static CARE through a continuous >=5 s locked-success-set hold; no raw-force violation, clipping, non-finite value, or |cart| >10 m",
        "first_current_target_contradiction": contradiction,
        "input_compatibility": input_compatibility,
        "integrity": {
            "artifact_sha256": artifact_hash,
            "source_sha256": source_hashes,
        },
        "numeric_witness": numeric_witness,
        "runtime": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
        },
        "schema_version": 1,
        "verdict": "PASS" if contradiction is None else "FAIL",
    }
    expected_checks = expected_witness_checks(payload, read_expected_witness())
    payload["expected_witness"] = expected_checks
    if not expected_checks["all_assertions_pass"] and payload["verdict"] == "PASS":
        payload["verdict"] = "FAIL"
        payload["first_current_target_contradiction"] = {
            "invariant": "expected_witness_comparison",
            "failures": expected_checks["failures"],
        }
    return payload


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the verifier and optionally retain its complete JSON witness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the full verification JSON to this repository-relative path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero unless the target and expected witness pass.",
    )
    args = parser.parse_args(argv)
    payload = run_verifier()
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = REPOSITORY / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    witness = cast(Mapping[str, JsonValue], payload["numeric_witness"])
    summary = {
        "combined_cart_peak_m": cast(Mapping[str, JsonValue], witness["track_bound"])[
            "combined_peak_abs_cart_m"
        ],
        "continuous_hold_s": cast(Mapping[str, JsonValue], witness["success_set"])[
            "continuous_in_success_set_duration_s"
        ],
        "expected_witness_pass": cast(Mapping[str, JsonValue], payload["expected_witness"])[
            "all_assertions_pass"
        ],
        "first_current_target_contradiction": payload[
            "first_current_target_contradiction"
        ],
        "promotion_screen_pass": cast(Mapping[str, JsonValue], witness[
            "promotion_screen_separate_from_current_target"
        ])["passes_promotion_screen"],
        "static_first_raw_n": cast(Mapping[str, JsonValue], witness["forces"])[
            "static_hold"
        ]["first_raw_force_n"],
        "verdict": payload["verdict"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if args.check and payload["verdict"] != "PASS":
        return 1
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(cli())
