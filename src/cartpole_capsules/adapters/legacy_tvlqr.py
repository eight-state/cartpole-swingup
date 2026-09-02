"""n5/n6 legacy adapter: continuous TVLQR catch plus saturated CARE hold.

Ports capsules/n05-quintuple/src/cartpole_race/release.py and
capsules/n06-sextuple/src/cartpole_race/n6.py replay logic. The nominal is a
sha-pinned 1 ms continuous trajectory; TVLQR is rebuilt locally with
``Qf = P_static`` (the validated scale-1 path, never the 25x M1 helper), and
the static stage saturates at the policy level exactly as StaticLQRPolicy did.
Raw demanded forces are logged every tick (dynamics Gen-A ``u_raw_log``
behavior, delivered by ``core.rollout.run_policy(..., return_raw=True)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import base
from .base import RolloutRecord, RungConfig, RunResult, force_stats, track_stats


@dataclass(frozen=True)
class LegacyStack:
    """Sha-pinned nominal plus both feedback stages rebuilt from it."""

    cfg: RungConfig
    model: Any
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    horizon_s: float
    tvlqr: Any
    static_policy: Any

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
        if time_s < self.horizon_s:
            return float(self.tvlqr.policy(state, time_s))
        return float(self.static_policy(state, time_s))

    def phase(self, tick: int) -> str:
        horizon_tick = len(self.nominal_controls)
        return "tvlqr" if tick < horizon_tick else "static_care"


def _load_active_nominal(cfg: RungConfig) -> tuple[np.ndarray, np.ndarray, float]:
    active = next(n for n in cfg.nominals if n.role == "active")
    path = base.rung_root(cfg) / active.path
    if base.sha256_file(path) != active.sha256:
        raise ValueError(f"nominal bytes do not match the released authority: {active.path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "u", "horizon"}
        if not required.issubset(archive.files):
            raise ValueError(f"nominal lacks required arrays: {active.path}")
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(archive["horizon"])
        if active.alias_dialect:
            # n5/n6 files carry byte-identical alias keys; assert, never rewrite.
            if not np.array_equal(states, np.asarray(archive["states"], dtype=float)):
                raise ValueError("nominal x and states alias arrays differ")
            if not np.array_equal(controls, np.asarray(archive["forces"], dtype=float).reshape(-1)):
                raise ValueError("nominal u and forces alias arrays differ")
        if active.strict_meta:
            if int(archive["n"]) != cfg.n_links:
                raise ValueError(f"nominal n is not {cfg.n_links}: {active.path}")
            if float(archive["control_dt"]) != cfg.control_dt_s:
                raise ValueError(f"nominal control_dt is not {cfg.control_dt_s}: {active.path}")
            if not np.array_equal(
                np.asarray(archive["t"], dtype=float),
                np.linspace(0.0, horizon_s, active.n_nodes + 1),
            ):
                raise ValueError(f"nominal time grid is not the fixed 1 ms grid: {active.path}")
    if states.shape != (active.n_nodes + 1, 2 * (cfg.n_links + 1)):
        raise ValueError(f"nominal state shape is not the frozen grid: {active.path}")
    if controls.shape != (active.n_nodes,):
        raise ValueError(f"nominal control shape is not the frozen grid: {active.path}")
    if not (np.all(np.isfinite(states)) and np.all(np.isfinite(controls))):
        raise ValueError(f"nominal contains non-finite values: {active.path}")
    return states, controls, horizon_s


def load(cfg: RungConfig) -> LegacyStack:
    """Load the fixed nominal and rebuild TVLQR plus static LQR from it."""
    from cartpole_capsules.core.lqr import StaticLQRPolicy, static_lqr
    from cartpole_capsules.core.tvlqr import TVLQR

    states, controls, horizon_s = _load_active_nominal(cfg)
    model = base.build_model(cfg)
    static_gain, static_p = static_lqr(model)
    padded = np.append(controls, controls[-1])
    times = np.linspace(0.0, horizon_s, len(states))
    tvlqr = TVLQR(model, times, states, padded, Qf=static_p, n_eval=400)
    return LegacyStack(
        cfg=cfg,
        model=model,
        nominal_states=states,
        nominal_controls=controls,
        horizon_s=horizon_s,
        tvlqr=tvlqr,
        static_policy=StaticLQRPolicy(model, static_gain),
    )


def audit_nominal(cfg: RungConfig, stack: LegacyStack) -> dict[str, Any]:
    """n5-only arithmetic audit: the nominal satisfies the simulator bounds."""
    if cfg.rung != 5:
        return {}
    model = stack.model
    substeps = int(np.ceil(cfg.control_dt_s / cfg.rk4_max_step_s))
    substep_s = cfg.control_dt_s / substeps
    defect = 0.0
    for state, force, nxt in zip(
        stack.nominal_states[:-1], stack.nominal_controls, stack.nominal_states[1:], strict=True
    ):
        stepped = state.copy()
        for _ in range(substeps):
            stepped = model.rk4_step(stepped, float(force), substep_s)
        defect = max(defect, float(np.max(np.abs(stepped - nxt))))
    peak_force_n = float(np.max(np.abs(stack.nominal_controls)))
    peak_cart_m = float(np.max(np.abs(stack.nominal_states[:, 0])))
    expected_peak = float(cfg.extras["expected_peak_force_n"])
    if defect >= 1e-10 or peak_force_n != expected_peak or peak_cart_m >= 10.0:
        raise ValueError("nominal no longer satisfies the released simulator bounds")
    return {
        "max_zoh_defect": defect,
        "peak_feedforward_force_n": peak_force_n,
        "peak_cart_m": peak_cart_m,
    }


def closed_loop_monodromy(cfg: RungConfig, stack: LegacyStack) -> float:
    """Exact-ZOH closed-loop spectral radius over the n5 nominal (n5 gate)."""
    import scipy.linalg

    model = stack.model
    nx = model.nx
    transition = np.eye(nx)
    for tick in range(len(stack.nominal_controls)):
        time_s = tick * cfg.control_dt_s
        state, force = stack.tvlqr._nom_at(time_s)
        a_matrix, b_matrix = model.linearize(state, force)
        block = np.zeros((nx + 1, nx + 1))
        block[:nx, :nx] = a_matrix * cfg.control_dt_s
        block[:nx, nx:] = b_matrix.reshape(nx, 1) * cfg.control_dt_s
        lifted = scipy.linalg.expm(block)
        transition = (lifted[:nx, :nx] - lifted[:nx, nx:] @ stack.tvlqr.K_at(time_s)) @ transition
    return float(np.max(np.abs(np.linalg.eigvals(transition))))


def run(cfg: RungConfig, stack: LegacyStack) -> RunResult:
    """One fresh hanging-start replay with the n5/n6 verdict semantics."""
    from cartpole_capsules.core.rollout import run_policy

    total_ticks = cfg.total_ticks
    if total_ticks is None:
        total_ticks = len(stack.nominal_controls) + int(
            round(cfg.post_horizon_s * cfg.control_rate_hz)
        )
    record = run_policy(
        stack.model,
        stack,
        total_ticks,
        cfg.control_dt_s,
        cfg.rk4_max_step_s,
        stack.model.x_equilibrium("down"),
        return_raw=True,
        phase_label=stack.phase,
    )
    metrics: dict[str, Any] = {
        "nominal": {
            "sha256": next(n for n in cfg.nominals if n.role == "active").sha256,
            "horizon_s": stack.horizon_s,
            "control_ticks": len(stack.nominal_controls),
            "role": "controller reference only; never rendered",
        },
        **force_stats(record, cfg),
        **track_stats(record, cfg),
    }
    failures: list[str] = []
    if cfg.rung == 5:
        metrics["closed_loop_monodromy_rho"] = closed_loop_monodromy(cfg, stack)
        metrics["nominal_audit"] = audit_nominal(cfg, stack)
    _assess(cfg, stack, record, metrics, failures)
    return RunResult(cfg.rung, record, metrics, not failures, tuple(failures))


def _assess(
    cfg: RungConfig,
    stack: LegacyStack,
    record: RolloutRecord,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    mask = base.success_mask(stack.model, record.states, cfg)
    hold_s = base.trailing_hold_s(mask, cfg.control_dt_s)
    metrics["final_hold_s"] = hold_s
    metrics["final_hold_samples"] = base.trailing_run_length(mask)
    rail_ok = metrics["peak_abs_cart_m"] <= cfg.track_half_length_m
    force_ok = metrics["applied_peak_abs_n"] <= cfg.force_bound_n + cfg.force_eps
    finite_ok = bool(np.all(np.isfinite(record.states)) and np.all(np.isfinite(record.raw)))
    metrics["track_ok"] = rail_ok
    metrics["force_ok"] = force_ok
    metrics["finite_ok"] = finite_ok
    metrics["success"] = bool(
        rail_ok and force_ok and finite_ok and hold_s >= cfg.hold_required_s - 1e-9
    )
    if not metrics["success"]:
        failures.append("fresh trajectory failed the sampled success predicate")
    pinned = cfg.extras.get("expected_metrics") or {}
    failures.extend(
        base.expected_metric_failures(
            metrics,
            {
                key: base.ExpectedMetric(value=float(spec["value"]), atol=float(spec["atol"]))
                for key, spec in pinned.items()
            },
        )
    )
