"""n7..n11 adapter: dense-nominal exact-ZOH tracker plus CARE static hold.

Ports the release stacks of capsules/n07..n11 (release.py / n10_release.py).
One shared flow with registry-driven deltas: start state (hanging vs the
nominal's first state, n11), hold metric (suffix for n7/n8, longest run for
n9..n11, n11 measured from the handoff tick), policy-level saturation of the
static stage (n7/n8 only), strict nominal metadata (n9/n10/n11), and
per-rung nominal defect audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import base
from .base import RolloutRecord, RungConfig, RunResult, force_stats, track_stats


@dataclass(frozen=True)
class DiscreteStack:
    """Dense nominal plus the two rebuilt feedback stages."""

    cfg: RungConfig
    model: Any
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    horizon_s: float
    tracker: Any  # core.discrete_tvlqr.DiscreteTVLQR (n10 API)
    static_gain: np.ndarray
    static_saturated: bool

    @property
    def horizon_tick(self) -> int:
        return len(self.nominal_controls)

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
        if time_s < self.horizon_s:
            return float(self.tracker.policy(state, time_s))
        if self.static_saturated:
            return float(self._static_policy()(state, time_s))
        from cartpole_capsules.core.lqr import wrap_state_error

        upright = self.model.x_equilibrium("up")
        return float(-(self.static_gain @ wrap_state_error(state, upright, self.model.n)).item())

    def _static_policy(self):
        from cartpole_capsules.core.lqr import StaticLQRPolicy

        return StaticLQRPolicy(self.model, self.static_gain)

    def phase(self, tick: int) -> str:
        return "tvlqr_track" if tick < self.horizon_tick else "static_care"


def _load_dense_nominal(cfg: RungConfig, model: Any) -> tuple[np.ndarray, np.ndarray, float]:
    active = next(n for n in cfg.nominals if n.role == "active")
    path = base.rung_root(cfg) / active.path
    if base.sha256_file(path) != active.sha256:
        raise ValueError(f"dense nominal bytes do not match the released authority: {active.path}")
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(archive["horizon"]).item())
        if active.strict_meta:
            if int(np.asarray(archive["n"]).item()) != cfg.n_links:
                raise ValueError(f"nominal n is not {cfg.n_links}: {active.path}")
            if float(np.asarray(archive["force"]).item()) != cfg.force_bound_n:
                raise ValueError(
                    f"nominal force metadata is not {cfg.force_bound_n}: {active.path}"
                )
    if states.shape != (active.n_nodes + 1, model.nx):
        raise ValueError(f"dense nominal state shape is not the released grid: {active.path}")
    if controls.shape != (active.n_nodes,):
        raise ValueError(f"dense nominal control shape is not the released grid: {active.path}")
    if abs(horizon_s - active.horizon_s) > 1e-12:
        raise ValueError(f"dense nominal horizon is not {active.horizon_s} s: {active.path}")
    if not (np.all(np.isfinite(states)) and np.all(np.isfinite(controls))):
        raise ValueError(f"dense nominal contains non-finite values: {active.path}")
    return states, controls, horizon_s


def _load_parent_nominal(cfg: RungConfig) -> tuple[np.ndarray, np.ndarray, float]:
    parent = next(n for n in cfg.nominals if n.role == "parent")
    path = base.rung_root(cfg) / parent.path
    if base.sha256_file(path) != parent.sha256:
        raise ValueError(f"parent nominal bytes do not match authority: {parent.path}")
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(archive["horizon"]).item())
    if states.shape != (parent.n_nodes + 1, 2 * (cfg.n_links + 1)):
        raise ValueError(f"parent nominal state shape mismatch: {parent.path}")
    if controls.shape != (parent.n_nodes,):
        raise ValueError(f"parent nominal control shape mismatch: {parent.path}")
    return states, controls, horizon_s


def load(cfg: RungConfig) -> DiscreteStack:
    """Load the dense nominal and rebuild both feedback controllers locally."""
    from cartpole_capsules.core.discrete_tvlqr import DiscreteTVLQR
    from cartpole_capsules.core.lqr import static_lqr

    model = base.build_model(cfg)
    states, controls, horizon_s = _load_dense_nominal(cfg, model)
    saturated = bool(cfg.extras.get("static_saturated", False))
    static_gain, _ = static_lqr(model)
    tracker = DiscreteTVLQR(model, states, controls, cfg.control_dt_s)
    return DiscreteStack(
        cfg=cfg,
        model=model,
        nominal_states=states,
        nominal_controls=controls,
        horizon_s=horizon_s,
        tracker=tracker,
        static_gain=static_gain,
        static_saturated=saturated,
    )


def audit_nominals(cfg: RungConfig, stack: DiscreteStack) -> dict[str, Any]:
    """Recompute parent/dense transcription defects against per-rung limits."""
    limits = cfg.extras.get("defect_limits") or {}
    if not cfg.extras.get("audit_dense_defects", True):
        return {}
    report: dict[str, Any] = {}
    if any(n.role == "parent" for n in cfg.nominals):
        parent_states, parent_controls, parent_horizon_s = _load_parent_nominal(cfg)
        step_s = parent_horizon_s / len(parent_controls)
        parent_defect = 0.0
        for state, control, nxt in zip(
            parent_states[:-1], parent_controls, parent_states[1:], strict=True
        ):
            stepped = stack.model.rk4_step(state, float(control), step_s)
            parent_defect = max(parent_defect, float(np.max(np.abs(stepped - nxt))))
        limit = float(limits.get("parent_rk4_4ms", 5e-7))
        if parent_defect >= limit:
            raise ValueError("parent nominal no longer satisfies its release bound")
        report["parent_rk4_4ms_defect"] = parent_defect
    intra, seam = _dense_defects(cfg, stack)
    if intra >= float(limits.get("dense_intra_segment", 1e-10)):
        raise ValueError("dense intra-segment defect exceeds bound")
    if seam >= float(limits.get("dense_seam", 5e-5)):
        raise ValueError("dense 4 ms seam exceeds bound")
    report["dense_intra_segment_defect"] = intra
    report["dense_4ms_seam"] = seam
    report["peak_feedforward_n"] = float(np.max(np.abs(stack.nominal_controls)))
    return report


def _dense_defects(cfg: RungConfig, stack: DiscreteStack) -> tuple[float, float]:
    """Max intra-segment and 4 ms-boundary residuals of the dense nominal."""
    substeps = max(1, int(np.ceil(cfg.control_dt_s / cfg.rk4_max_step_s)))
    substep_s = cfg.control_dt_s / substeps
    intra = seam = 0.0
    for tick, (state, control, nxt) in enumerate(
        zip(
            stack.nominal_states[:-1],
            stack.nominal_controls,
            stack.nominal_states[1:],
            strict=True,
        )
    ):
        stepped = state.copy()
        for _ in range(substeps):
            stepped = stack.model.rk4_step(stepped, float(control), substep_s)
        defect = float(np.max(np.abs(stepped - nxt)))
        if tick % 4 == 0:
            seam = max(seam, defect)
        else:
            intra = max(intra, defect)
    return intra, seam


def run(cfg: RungConfig, stack: DiscreteStack) -> RunResult:
    """One fresh replay with the rung's hold metric and gate set."""
    from cartpole_capsules.core.rollout import run_policy

    if cfg.start_state == "nominal_first":
        start = stack.nominal_states[0]
    else:
        start = stack.model.x_equilibrium("down")
    total_ticks = cfg.total_ticks
    if total_ticks is None:
        total_ticks = stack.horizon_tick + int(
            round((cfg.post_horizon_s + cfg.hold_required_s) * cfg.control_rate_hz)
        )
    record = run_policy(
        stack.model,
        stack,
        total_ticks,
        cfg.control_dt_s,
        cfg.rk4_max_step_s,
        start,
        return_raw=True,
        phase_label=stack.phase,
    )
    metrics: dict[str, Any] = {
        "nominal": {
            "sha256": next(n for n in cfg.nominals if n.role == "active").sha256,
            "horizon_s": stack.horizon_s,
            "control_ticks": stack.horizon_tick,
            "role": "controller reference only; never rendered",
        },
        "controller": {"monodromy_rho": stack.tracker.monodromy()},
        "handoff_max_angle_error_deg": base.handoff_angle_error_deg(
            stack.model, record.states[stack.horizon_tick]
        ),
        **force_stats(record, cfg),
        **track_stats(record, cfg),
    }
    failures: list[str] = []
    _assess(cfg, stack, record, metrics, failures)
    return RunResult(cfg.rung, record, metrics, not failures, tuple(failures))


def _assess(
    cfg: RungConfig,
    stack: DiscreteStack,
    record: RolloutRecord,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    mask = base.success_mask(stack.model, record.states, cfg)
    if cfg.hold_scope == "from_switch":
        hold_mask = mask[stack.horizon_tick :]
    else:
        hold_mask = mask
    hold_s = (
        base.trailing_hold_s(hold_mask, cfg.control_dt_s)
        if cfg.hold_metric == "suffix"
        else base.longest_run_hold_s(hold_mask, cfg.control_dt_s)
    )
    metrics["hold_metric"] = cfg.hold_metric
    metrics["hold_s"] = hold_s
    gates = set(cfg.extras.get("gates", ()))
    if "finite" in gates and not (
        np.all(np.isfinite(record.states)) and np.all(np.isfinite(record.raw))
    ):
        failures.append("nonfinite_state_or_control")
    if "monodromy" in gates and metrics["controller"]["monodromy_rho"] >= 1.0:
        failures.append("monodromy_does_not_contract")
    passed = bool(
        hold_s >= cfg.hold_required_s - 1e-9
        and metrics["peak_abs_cart_m"] <= cfg.track_half_length_m
        and metrics["applied_peak_abs_n"] <= cfg.force_bound_n + cfg.force_eps
    )
    metrics["passed"] = passed
    if not passed:
        failures.append("fresh closed loop failed the released predicate")
