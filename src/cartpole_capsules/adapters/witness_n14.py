"""n14 witness adapter: pre-replay rejection plus exact controls-only replay.

Ports capsules/n14-quattuordecuple/src/n14_cartpole/{verifier.py,success.py}.
The witness stores controls only (22009 entries); states are recomputed by
exact 1 kHz ZOH replay with four 0.25 ms RK4 substeps per tick, starting from
the exact hanging equilibrium. Over-limit controls are rejected BEFORE replay
(a fail-closed physical gate, never a clip), the success set rejects
non-finite states, and the expected-witness protocol compares metrics with
per-key absolute tolerances from the retained artifact JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import base
from .base import RolloutRecord, RungConfig, RunResult


@dataclass(frozen=True)
class WitnessStack:
    """Hash-checked controls artifact plus the locked plant."""

    cfg: RungConfig
    model: Any
    controls: np.ndarray
    metadata: dict[str, Any]
    expected: dict[str, Any]

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:  # pragma: no cover
        raise NotImplementedError("n14 has no live policy; use replay_controls")

    def phase(self, tick: int) -> str:  # pragma: no cover
        raise NotImplementedError("n14 has no live policy; use replay_controls")


def audit_authority(cfg: RungConfig) -> dict[str, Any]:
    """Check retained artifacts and source hashes against the legacy release tag."""
    from cartpole_capsules.legacy import (
        audit_legacy_source,
        legacy_file_sha256,
    )

    repository = base.repository_root()
    root = base.capsule_root(cfg)
    legacy = audit_legacy_source(repository, cfg)
    for relative in (
        "artifacts/MANIFEST.json",
        "artifacts/source-sha256.json",
        "artifacts/verification.json",
    ):
        current = base.sha256_file(root / relative)
        expected = legacy_file_sha256(repository, cfg, relative)
        if current != expected:
            raise ValueError(f"retained n=14 authority changed: {relative}")
    manifest = base.load_json(root / "artifacts/MANIFEST.json")
    if manifest.get("schema_version") != 1 or manifest.get("release") != "N14":
        raise ValueError("n=14 artifact manifest schema changed")
    for relative, expected in manifest.get("sha256", {}).items():
        if base.sha256_file(root / relative) != expected:
            raise ValueError(f"n=14 artifact changed: {relative}")
    source_lock = base.load_json(root / "artifacts/source-sha256.json")
    if source_lock.get("schema_version") != 1 or source_lock.get("release") != "N14":
        raise ValueError("n=14 source manifest schema changed")
    for relative, expected in source_lock.get("sha256", {}).items():
        if legacy_file_sha256(repository, cfg, relative) != expected:
            raise ValueError(f"n=14 legacy source changed: {relative}")
    return {
        "legacy": legacy,
        "artifact_count": len(manifest["sha256"]),
        "source_count": len(source_lock["sha256"]),
    }


def load(cfg: RungConfig) -> WitnessStack:
    """Audit authority, then load the controls-only witness."""
    audit_authority(cfg)
    active = next(n for n in cfg.nominals if n.role == "active")
    path = base.capsule_root(cfg) / active.path
    if base.sha256_file(path) != active.sha256:
        raise ValueError(f"witness bytes do not match the released authority: {active.path}")
    with np.load(path, allow_pickle=False) as data:
        controls = np.asarray(data["u"], dtype=np.float64).reshape(-1)
        metadata = {key: np.asarray(data[key]).item() for key in data.files if key != "u"}
    expected_path = base.capsule_root(cfg) / cfg.extras["expected_witness"]
    return WitnessStack(
        cfg=cfg,
        model=base.build_model(cfg),
        controls=controls,
        metadata=metadata,
        expected=base.load_json(expected_path),
    )


def replay_controls(cfg: RungConfig, stack: WitnessStack) -> RolloutRecord:
    """Exact replay: no policy, no clipping; raw == applied == stored controls."""
    from cartpole_capsules.core.rollout import replay_controls as core_replay

    return core_replay(
        stack.model,
        stack.controls,
        cfg.control_dt_s,
        cfg.rk4_max_step_s,
        quarter_metrics=True,
    )


def replay_and_check(cfg: RungConfig, stack: WitnessStack) -> RunResult:
    """Reject over-limit controls pre-replay, then run the full gate set."""
    raw = stack.controls
    failures: list[str] = []
    physical: list[str] = []

    if raw.shape != (int(cfg.total_ticks),):
        physical.append("control_shape")
    if not np.isfinite(raw).all():
        physical.append("nonfinite_control")
    pre_replay = bool(cfg.extras.get("reject_over_limit_before_replay", True))
    peak_demand = float(np.max(np.abs(raw))) if raw.size else 0.0
    if pre_replay and peak_demand > cfg.force_bound_n:
        physical.append("raw_force_bound")

    if physical:
        # Fail closed before simulating: a tampered witness never replays.
        return RunResult(
            cfg.rung, _empty_record(), {"physical_failures": physical}, False, tuple(physical)
        )

    record = replay_controls(cfg, stack)
    metrics: dict[str, Any] = _replay_metrics(cfg, record)
    physical.extend(_physical_gates(cfg, record, metrics))
    metadata_failures = _metadata_failures(cfg, stack)
    expected_failures = expected_witness_failures(metrics, stack.expected["metrics"])
    failures = physical + metadata_failures + expected_failures
    return RunResult(cfg.rung, record, metrics, not failures, tuple(failures))


def _replay_metrics(cfg: RungConfig, record: RolloutRecord) -> dict[str, Any]:
    n = cfg.n_links
    hanging = _hanging_state(cfg)
    mask = success_mask_stateless(cfg, record.states)
    first, count = base.longest_run_span(mask)
    final = record.states[-1]
    final_angles = _wrap_to_pi(final[1 : n + 1])
    quarter_peak = (
        record.quarter_cart_peak_m
        if record.quarter_cart_peak_m is not None
        else (float(np.max(np.abs(record.states[:, 0]))))
    )
    return {
        "control_count": int(record.raw.size),
        "state_count": int(record.states.shape[0]),
        "duration_s": float(record.raw.size * cfg.control_dt_s),
        "start_max_abs_from_exact_hanging": float(np.max(np.abs(record.states[0] - hanging))),
        "peak_force_n": float(np.max(np.abs(record.raw))) if record.raw.size else 0.0,
        "quarter_cart_peak_m": quarter_peak,
        "longest_success_first_tick": first,
        "longest_success_states": count,
        "longest_success_s": max(0, count - 1) * cfg.control_dt_s,
        "nonfinite_state_tick": (
            None if record.first_nonfinite is None else int(record.first_nonfinite["tick"])
        ),
        "final": {
            "cart_position_m": float(final[0]),
            "cart_rate_m_s": float(final[n + 1]),
            "max_wrapped_link_angle_deg": float(np.rad2deg(np.max(np.abs(final_angles)))),
            "max_link_rate_rad_s": float(np.max(np.abs(final[n + 2 :]))),
            "in_success_set": bool(mask[-1]),
        },
    }


def _physical_gates(cfg: RungConfig, record: RolloutRecord, metrics: dict[str, Any]) -> list[str]:
    required_states = int(cfg.extras["required_success_states"])
    gates: list[str] = []
    if metrics["nonfinite_state_tick"] is not None:
        gates.append("nonfinite_state")
    if metrics["peak_force_n"] > cfg.force_bound_n:
        gates.append("raw_force_bound")
    if metrics["quarter_cart_peak_m"] > cfg.track_half_length_m:
        gates.append("quarter_step_rail_bound")
    if metrics["start_max_abs_from_exact_hanging"] != 0.0:
        gates.append("exact_hanging_start")
    trailing = _trailing_success_states(record, cfg)
    if trailing < required_states:
        gates.append("success_duration")
    return gates


def success_mask_stateless(cfg: RungConfig, states: np.ndarray) -> np.ndarray:
    """n14 predicate without a model: upright equilibrium is all zeros."""
    n = cfg.n_links
    theta_tol = np.deg2rad(cfg.theta_tol_deg)
    mask = np.empty(len(states), dtype=bool)
    for index, state in enumerate(states):
        if cfg.reject_nonfinite and not np.isfinite(state).all():
            mask[index] = False
            continue
        angles = _wrap_to_pi(state[1 : n + 1])
        mask[index] = bool(
            abs(float(state[0])) <= cfg.cart_tol_m
            and abs(float(state[n + 1])) <= cfg.cart_rate_tol_m_s
            and np.max(np.abs(angles)) <= theta_tol
            and np.max(np.abs(state[n + 2 :])) <= cfg.theta_rate_tol_rad_s
        )
    return mask


def _trailing_success_states(record: RolloutRecord, cfg: RungConfig) -> int:
    return base.trailing_run_length(success_mask_stateless(cfg, record.states))


def expected_witness_failures(metrics: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Port of the n14 two-section protocol: exact 'equal' + atol 'numeric'."""
    failures: list[str] = []
    for key, target in expected.get("equal", {}).items():
        if metrics.get(key) != target:
            failures.append(key)
    for key, spec in expected.get("numeric", {}).items():
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or abs(
            float(actual) - float(spec["value"])
        ) > float(spec["atol"]):
            failures.append(key)
    return failures


def _metadata_failures(cfg: RungConfig, stack: WitnessStack) -> list[str]:
    expected_meta = cfg.extras.get("expected_metadata") or {}
    return [key for key, value in expected_meta.items() if stack.metadata.get(key) != value]


def _hanging_state(cfg: RungConfig) -> np.ndarray:
    state = np.zeros(2 * (cfg.n_links + 1))
    state[1 : 1 + cfg.n_links] = np.pi
    return state


def _upright_state(cfg: RungConfig) -> np.ndarray:
    return np.zeros(2 * (cfg.n_links + 1))


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _empty_record() -> RolloutRecord:
    empty = np.empty(0)
    return RolloutRecord(
        times=empty,
        states=np.empty((0, 0)),
        raw=empty,
        applied=empty,
        phases=(),
    )
