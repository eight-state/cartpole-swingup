"""n13 proof adapter: B2 source-closure authority plus composed replay.

Ports the authority half of
capsules/n13-tredecuple/src/n13_proof/capsule.py (validate_b2_authority,
prepare_run, fresh_composed_rollout). The tracker is NOT rebuilt from the
base nominal: the released affine defect tracker (feedback_k, feedforward,
static_default_sda_k) is loaded as hash-pinned B2 evidence and composed
directly. Rail and non-finite watches run at quarter-step resolution, and
"raw equals applied" is a gate (no clipping on this rung).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import base
from .base import RungConfig, RunResult


@dataclass(frozen=True)
class ProofStack:
    """Hash-checked B2 controller payload plus the locked plant."""

    cfg: RungConfig
    model: Any
    x_ref: np.ndarray
    u_ref: np.ndarray
    feedback: np.ndarray  # (TRACKER_TICKS, 1, nx)
    feedforward: np.ndarray  # (TRACKER_TICKS,)
    static_k: np.ndarray  # (nx,)
    switch_tick: int
    authority_inputs: tuple[dict[str, str], ...]

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
        from cartpole_capsules.core.lqr import wrap_state_error

        if tick < self.switch_tick:
            error = wrap_state_error(state, self.x_ref[tick], self.model.n)
            return float(
                self.u_ref[tick]
                - (self.feedback[tick] @ error).reshape(-1)[0]
                - self.feedforward[tick]
            )
        return float(
            -self.static_k @ wrap_state_error(state, self.model.x_equilibrium("up"), self.model.n)
        )

    def phase(self, tick: int) -> str:
        return "affine_defect_tracker" if tick < self.switch_tick else "static_default_sda"


class AuthorityError(RuntimeError):
    """An immutable input is absent, malformed, or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _sha256_bundled(root: Any, relative: str) -> str:
    return base.sha256_file(root / relative)


def audit_authority(cfg: RungConfig, root: Any) -> dict[str, Any]:
    """Fail closed unless every pinned B2 source and bundle byte is intact.

    Pinned tables come from the registry (extras.source_sha256,
    extras.fixed_file_sha256, extras.controller_payload_sha256). The B2
    manifest at runtime must agree with them, exactly as capsule.py required.
    """
    source_hashes = cfg.extras.get("source_sha256") or {}
    fixed_hashes = cfg.extras.get("fixed_file_sha256") or {}
    payload_hashes = cfg.extras.get("controller_payload_sha256") or {}
    bundle_rel = cfg.extras["bundle_dir"]
    from cartpole_capsules.legacy import legacy_file_sha256

    file_checks: dict[str, Any] = {}
    for relative, expected in source_hashes.items():
        actual = legacy_file_sha256(base.repository_root(), cfg, relative)
        _require(actual == expected, f"legacy B2 source hash mismatch: {relative}")
        file_checks[relative] = {"expected": expected, "actual": actual, "passed": True}
    for relative, expected in fixed_hashes.items():
        actual = _sha256_bundled(root, relative) if (root / relative).is_file() else "missing"
        _require(actual == expected, f"fixed B2 evidence hash mismatch: {relative}")
        file_checks[relative] = {"expected": expected, "actual": actual, "passed": True}

    manifest = base.load_json(root / bundle_rel / "00-source-manifest.json")
    sources = manifest.get("source_files")
    _require(isinstance(sources, dict), "B2 manifest source_files is absent")
    _require(set(sources) == set(source_hashes), "B2 manifest source set drift")
    for relative, expected in source_hashes.items():
        record = sources.get(relative)
        _require(isinstance(record, dict), f"B2 manifest source record drift: {relative}")
        _require(record.get("sha256") == expected, f"B2 manifest source hash drift: {relative}")

    controller_path = root / bundle_rel / "01-affine-controller.npz"
    payload_checks: dict[str, Any] = {}
    try:
        with np.load(controller_path, allow_pickle=False) as archive:
            payloads = {name: np.asarray(archive[name]) for name in payload_hashes}
    except (KeyError, OSError, ValueError) as error:
        raise AuthorityError(f"cannot validate fixed B2 controller: {error}") from error
    tracker_ticks = int(cfg.extras["tracker_ticks"])
    nx = 2 * (cfg.n_links + 1)
    shapes = {"feedback_k": (tracker_ticks, 1, nx), "feedforward": (tracker_ticks,)}
    for name, expected in payload_hashes.items():
        value = payloads[name]
        actual = _array_sha256(value)
        passed = value.shape == shapes[name] and np.all(np.isfinite(value)) and actual == expected
        _require(passed, f"fixed B2 controller payload mismatch: {name}")
        payload_checks[name] = {"expected": expected, "actual": actual, "passed": True}

    classification = base.load_json(root / bundle_rel / "05-b2-classification.json")
    _require(
        classification.get("classification") == cfg.extras["classification"],
        "B2 classification claim is not N13_ONE_RUN_PASS",
    )
    with np.load(root / bundle_rel / "05-b2-classification.npz", allow_pickle=False) as archive:
        _require(set(archive.files) == {"selected_switch_tick"}, "classification NPZ key set drift")
        switch_tick = int(np.asarray(archive["selected_switch_tick"]))
    _require(
        switch_tick == cfg.switch_tick,
        f"B2 switch tick is {switch_tick}, expected {cfg.switch_tick}",
    )

    base_rel = cfg.extras["base_nominal"]
    try:
        with np.load(root / base_rel, allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"x", "u", "horizon", "n", "force", "n_nodes"},
                "B0 base archive key set drift",
            )
            coarse_x = np.asarray(archive["x"])
            coarse_u = np.asarray(archive["u"])
            _require(
                coarse_x.dtype == np.dtype(np.float64)
                and coarse_x.shape == (cfg.extras["base_nodes"] + 1, nx),
                "B0 base state shape drift",
            )
            _require(
                coarse_u.dtype == np.dtype(np.float64)
                and coarse_u.shape == (cfg.extras["base_nodes"],),
                "B0 base control shape drift",
            )
    except (OSError, ValueError) as error:
        raise AuthorityError(f"cannot load B0 base archive: {error}") from error

    roles = cfg.extras.get("authority_roles") or {}
    authority_inputs = tuple(
        {
            "tier": roles.get(relative, "B2"),
            "role": role,
            "path": relative,
            "sha256": _sha256_bundled(root, relative),
        }
        for relative, role in roles.items()
    )
    return {
        "passed": True,
        "files": file_checks,
        "controller_payloads": payload_checks,
        "authority_inputs": authority_inputs,
        "classification": classification.get("classification"),
        "switch_tick": switch_tick,
    }


def _array_sha256(value: np.ndarray) -> str:
    import hashlib

    array = np.asarray(value)
    _require(array.dtype == np.dtype(np.float64), f"array is not float64: {array.dtype}")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load(cfg: RungConfig, root: Any | None = None) -> ProofStack:
    """Audit authority, cross-check the densified reference, return the stack."""
    from cartpole_capsules.core.fast_pieces import make_densifier

    root = root or base.capsule_root(cfg)
    audit = audit_authority(cfg, root)
    bundle_rel = cfg.extras["bundle_dir"]
    base_rel = cfg.extras["base_nominal"]

    try:
        with np.load(root / bundle_rel / "01-affine-controller.npz", allow_pickle=False) as archive:
            controller = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise AuthorityError(f"cannot load B2 controller: {error}") from error
    expected_keys = {
        "x_ref",
        "u_ref",
        "defect_raw",
        "defect_wrapped",
        "feedback_k",
        "feedforward",
        "static_default_sda_k",
        "static_default_sda_p",
        "q",
        "r",
    }
    _require(set(controller) == expected_keys, "controller contract failed: key set drift")

    model = base.build_model(cfg)
    with np.load(root / base_rel, allow_pickle=False) as archive:
        coarse_x = np.asarray(archive["x"])
        coarse_u = np.asarray(archive["u"])
    stride = int(cfg.extras.get("densify_stride", 4))
    dense_x, dense_u = make_densifier(
        model, cfg.control_dt_s, stride, stride, cfg.extras["base_nodes"]
    )(coarse_x, coarse_u)
    _require(
        base.max_abs_delta(dense_x, controller["x_ref"]) <= 1e-12,
        "B0 dense reference states differ from B2 beyond the portable tolerance",
    )
    _require(
        base.max_abs_delta(dense_u, controller["u_ref"]) <= 1e-12,
        "B0 dense reference controls differ from B2 beyond the portable tolerance",
    )

    arm_a_rel = cfg.extras["arm_a"]
    try:
        with np.load(root / arm_a_rel, allow_pickle=False) as archive:
            arm_k = np.asarray(archive["static_default_sda_k"])
            arm_p = np.asarray(archive["tracker_terminal_p"])
    except (OSError, KeyError, ValueError) as error:
        raise AuthorityError(f"cannot load B0 Arm-A archive: {error}") from error
    _require(
        base.byte_equal(arm_k, controller["static_default_sda_k"]), "B0 Arm-A K differs from B2"
    )
    _require(
        base.byte_equal(arm_p, controller["static_default_sda_p"]), "B0 Arm-A P differs from B2"
    )

    return ProofStack(
        cfg=cfg,
        model=model,
        x_ref=np.asarray(controller["x_ref"], dtype=np.float64),
        u_ref=np.asarray(controller["u_ref"], dtype=np.float64),
        feedback=np.asarray(controller["feedback_k"], dtype=np.float64),
        feedforward=np.asarray(controller["feedforward"], dtype=np.float64),
        static_k=np.asarray(arm_k, dtype=np.float64),
        switch_tick=int(cfg.switch_tick),
        authority_inputs=audit["authority_inputs"],
    )


def run(cfg: RungConfig, stack: ProofStack) -> RunResult:
    """The fresh composed rollout with n13's quarter-step gates ported."""
    from cartpole_capsules.core.rollout import run_policy

    switch = int(cfg.switch_tick)
    hold_ticks = int(cfg.extras["hold_ticks"])
    ticks = switch + hold_ticks
    record = run_policy(
        stack.model,
        stack,
        ticks,
        cfg.control_dt_s,
        cfg.rk4_max_step_s,
        stack.model.x_equilibrium("down"),
        return_raw=True,
        phase_label=stack.phase,
        quarter_metrics=True,
    )
    mask = base.success_mask(stack.model, record.states[switch:], cfg)
    trailing_s = base.trailing_hold_s(mask, cfg.control_dt_s)
    raw = record.raw
    applied = record.applied
    gates: dict[str, bool] = {
        "fresh_exact_hanging_start": base.max_abs_delta(
            record.states[0], stack.model.x_equilibrium("down")
        )
        <= 1e-12,
        "all_values_finite": bool(np.all(np.isfinite(record.states)) and np.all(np.isfinite(raw))),
        "raw_equals_applied": base.byte_equal(raw, applied),
        "raw_peak_le_150_n": bool(np.max(np.abs(raw)) <= cfg.force_bound_n),
        "node_cart_le_10_m": bool(np.max(np.abs(record.states[:, 0])) <= cfg.track_half_length_m),
        "quarter_cart_le_10_m": record.first_nonfinite is None
        and record.first_rail_violation is None,
        "switch_in_instantaneous_success_set": bool(mask[0]),
        "trailing_in_set_ge_5_s": trailing_s >= cfg.hold_required_s,
    }
    metrics: dict[str, Any] = {
        "switch_tick": switch,
        "switch_time_s": switch * cfg.control_dt_s,
        "raw_peak_n": float(np.max(np.abs(raw))),
        "quarter_cart_peak_m": record.quarter_cart_peak_m,
        "first_nonfinite": record.first_nonfinite,
        "first_rail_violation": record.first_rail_violation,
        "trailing_success_s": trailing_s,
        "trailing_success_samples": base.trailing_run_length(mask),
        "gates": gates,
        "authority_inputs": [dict(item) for item in stack.authority_inputs],
        "saved_tracking_reference_loaded": True,
        "saved_b2_rollout_trace_loaded": False,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return RunResult(cfg.rung, record, metrics, not failures, tuple(failures))
