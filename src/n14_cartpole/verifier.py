"""Standalone exact-replay verifier for the locked N14 witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from n14_cartpole.success import (
    MAX_ABS_CART_M,
    MAX_ABS_CART_RATE_M_S,
    MAX_LINK_RATE_RAD_S,
    MAX_WRAPPED_LINK_ANGLE_DEG,
    N_LINKS,
    in_success_set,
    wrap_to_pi,
)

FloatArray = npt.NDArray[np.float64]
JsonObject = dict[str, Any]

REPOSITORY = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPOSITORY / "artifacts" / "n14-witness.npz"
EXPECTED_WITNESS_PATH = REPOSITORY / "artifacts" / "expected-witness.json"

CONTROL_DT_S = 0.001
RK4_SUBSTEP_S = 0.00025
FORCE_BOUND_N = 150.0
TRACK_BOUND_M = 10.0
SWITCH_TICK = 6009
TOTAL_CONTROLS = 22009
REQUIRED_SUCCESS_STATES = 5001
EXPECTED_ARTIFACT_SHA256 = "f10fc56e854050e6091f0ac7ce406772875ab70f38829d2a177d134e97ed0b29"
EXPECTED_SOURCE_SHA256 = {
    "src/cartpole_race/__init__.py": "338d2ee651a473ae73e2a876a23783a92c1f128976d2047567392a1a65641cd9",
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/n14_cartpole/success.py": "d976f237ae93d829bb52d1cf7c1d94335fc362130696ff7445042104fb032aca",
}
SOLE_PARENT_SHA256 = "fffb9466ee0be82646867ed6c8f13748827a2d157144eb0c81cfe642fc0a005b"
FULLMASS_SOURCE_SHA256 = "4d1c722e527a62c6989b69d9550e6e22b98211619515c3e476086bfc48a06799"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_locked_model() -> NLinkCartPole:
    """Construct the authoritative locked N14 plant."""
    return NLinkCartPole(
        CartPoleSpec(
            n_links=N_LINKS,
            cart_mass_kg=1.0,
            link_masses_kg=[0.1] * N_LINKS,
            link_lengths_m=[0.5] * N_LINKS,
            damping_cart_n_s_m=0.0,
            damping_links_n_m_s_rad=[0.0] * N_LINKS,
            force_bound_n=FORCE_BOUND_N,
            track_half_length_m=TRACK_BOUND_M,
            control_rate_hz=1000.0,
            rk4_max_step_s=RK4_SUBSTEP_S,
        )
    )


def longest_true_run(mask: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """Return the first index and sample count of the longest true run."""
    best_first = -1
    best_count = 0
    current_first = 0
    current_count = 0
    for index, value in enumerate(mask):
        if value:
            if current_count == 0:
                current_first = index
            current_count += 1
            if current_count > best_count:
                best_first = current_first
                best_count = current_count
        else:
            current_count = 0
    return best_first, best_count


def replay_controls(controls: npt.ArrayLike) -> JsonObject:
    """Replay raw controls under exact 1 kHz ZOH and quarter-step RK4."""
    raw = np.asarray(controls, dtype=np.float64).reshape(-1)
    model = make_locked_model()
    hanging = np.asarray(model.x_equilibrium("down"), dtype=np.float64)
    states = np.empty((raw.size + 1, model.nx), dtype=np.float64)
    states[0] = hanging
    quarter_cart_peak = abs(float(hanging[0]))
    nonfinite_tick: int | None = None
    for tick, control in enumerate(raw):
        state = states[tick].copy()
        for _ in range(4):
            state = np.asarray(
                model.rk4_step(state, float(control), RK4_SUBSTEP_S),
                dtype=np.float64,
            )
            quarter_cart_peak = max(quarter_cart_peak, abs(float(state[0])))
        states[tick + 1] = state
        if nonfinite_tick is None and not np.isfinite(state).all():
            nonfinite_tick = tick + 1

    success = np.asarray([in_success_set(state) for state in states], dtype=bool)
    success_first, success_count = longest_true_run(success)
    final = states[-1]
    final_angles = wrap_to_pi(final[1 : N_LINKS + 1])
    return {
        "controls": raw,
        "states": states,
        "success": success,
        "metrics": {
            "control_count": int(raw.size),
            "state_count": int(states.shape[0]),
            "duration_s": float(raw.size * CONTROL_DT_S),
            "start_max_abs_from_exact_hanging": float(np.max(np.abs(states[0] - hanging))),
            "peak_force_n": float(np.max(np.abs(raw))) if raw.size else 0.0,
            "quarter_cart_peak_m": quarter_cart_peak,
            "longest_success_first_tick": success_first,
            "longest_success_states": success_count,
            "longest_success_s": max(0, success_count - 1) * CONTROL_DT_S,
            "nonfinite_state_tick": nonfinite_tick,
            "final": {
                "cart_position_m": float(final[0]),
                "cart_rate_m_s": float(final[N_LINKS + 1]),
                "max_wrapped_link_angle_deg": float(np.rad2deg(np.max(np.abs(final_angles)))),
                "max_link_rate_rad_s": float(np.max(np.abs(final[N_LINKS + 2 :]))),
                "in_success_set": bool(success[-1]),
            },
        },
    }


def _metadata_scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    return np.asarray(data[key]).item()


def _expected_checks(metrics: JsonObject, expected: JsonObject) -> list[str]:
    failures: list[str] = []
    for key, target in expected["equal"].items():
        if metrics.get(key) != target:
            failures.append(key)
    for key, spec in expected["numeric"].items():
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or abs(float(actual) - float(spec["value"])) > float(spec["atol"]):
            failures.append(key)
    return failures


def run_verifier(artifact_path: Path = ARTIFACT_PATH) -> JsonObject:
    """Recompute all N14 witness claims from raw controls and locked physics."""
    artifact_hash = sha256(artifact_path)
    source_hashes = {
        relative: sha256(REPOSITORY / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    with np.load(artifact_path, allow_pickle=False) as data:
        controls = np.asarray(data["u"], dtype=np.float64).reshape(-1)
        metadata = {
            key: _metadata_scalar(data, key)
            for key in data.files
            if key != "u"
        }

    replay = replay_controls(controls)
    metrics = replay["metrics"]
    expected = json.loads(EXPECTED_WITNESS_PATH.read_text(encoding="utf-8"))
    expected_failures = _expected_checks(metrics, expected["metrics"])
    metadata_expected = {
        "control_dt_s": CONTROL_DT_S,
        "force_bound_n": FORCE_BOUND_N,
        "n": N_LINKS,
        "rk4_substep_s": RK4_SUBSTEP_S,
        "sole_parent_sha256": SOLE_PARENT_SHA256,
        "source_fullmass_sha256": FULLMASS_SOURCE_SHA256,
        "switch_tick": SWITCH_TICK,
        "track_bound_m": TRACK_BOUND_M,
    }
    metadata_failures = [
        key for key, value in metadata_expected.items() if metadata.get(key) != value
    ]
    physical_failures: list[str] = []
    if controls.shape != (TOTAL_CONTROLS,):
        physical_failures.append("control_shape")
    if not np.isfinite(controls).all():
        physical_failures.append("nonfinite_control")
    if metrics["nonfinite_state_tick"] is not None:
        physical_failures.append("nonfinite_state")
    if metrics["peak_force_n"] > FORCE_BOUND_N:
        physical_failures.append("raw_force_bound")
    if metrics["quarter_cart_peak_m"] > TRACK_BOUND_M:
        physical_failures.append("quarter_step_rail_bound")
    if metrics["start_max_abs_from_exact_hanging"] != 0.0:
        physical_failures.append("exact_hanging_start")
    if metrics["longest_success_states"] < REQUIRED_SUCCESS_STATES:
        physical_failures.append("success_duration")

    integrity_failures: list[str] = []
    if artifact_hash != EXPECTED_ARTIFACT_SHA256:
        integrity_failures.append("artifact_sha256")
    if source_hashes != EXPECTED_SOURCE_SHA256:
        integrity_failures.append("source_sha256")
    all_failures = integrity_failures + metadata_failures + physical_failures + expected_failures
    return {
        "schema_version": 1,
        "release": "N14",
        "verdict": "PASS" if not all_failures else "FAIL",
        "failures": all_failures,
        "integrity": {
            "artifact_sha256": artifact_hash,
            "artifact_sha256_matches": artifact_hash == EXPECTED_ARTIFACT_SHA256,
            "source_sha256": source_hashes,
            "source_sha256_matches": source_hashes == EXPECTED_SOURCE_SHA256,
        },
        "locked_contract": {
            "n_links": N_LINKS,
            "link_mass_kg": 0.1,
            "link_length_m": 0.5,
            "cart_mass_kg": 1.0,
            "force_bound_n": FORCE_BOUND_N,
            "track_bound_m": TRACK_BOUND_M,
            "control_dt_s": CONTROL_DT_S,
            "rk4_substep_s": RK4_SUBSTEP_S,
            "required_success_states": REQUIRED_SUCCESS_STATES,
            "success_bounds": {
                "max_abs_cart_m": MAX_ABS_CART_M,
                "max_abs_cart_rate_m_s": MAX_ABS_CART_RATE_M_S,
                "max_link_rate_rad_s": MAX_LINK_RATE_RAD_S,
                "max_wrapped_link_angle_deg": MAX_WRAPPED_LINK_ANGLE_DEG,
            },
        },
        "metadata": metadata,
        "metrics": metrics,
        "expected_witness": {
            "all_assertions_pass": not expected_failures,
            "failures": expected_failures,
        },
        "runtime": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the verifier and print or persist its JSON result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_verifier(args.artifact.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.resolve().write_text(rendered, encoding="utf-8")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(cli())
