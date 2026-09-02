"""n12 composite adapter: densified frozen nominal, FastDTVLQR, tick switch.

Ports capsules/n12-duodecuple/src/n12_cartpole/{simulator.py,success.py}. The
released nominal is a 4 ms-grid artifact (2500 nodes); the reference used by
the tracker is the stride-4 reset densification of exactly those bytes, so the
tracker grid is 1 ms (10000 ticks) while the switch to static CARE happens at
tick 9700. The tracker Q multiplies the angular-velocity block by 0.25
(``tracker_link_rate_q_scale`` in the banked evidence) with ``Qf`` the static
CARE solution recomputed at that modified Q. All forces are raw demands; the
simulator clips, and "no clipping observed" is itself a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import base
from .base import RungConfig, RunResult, force_stats, track_stats


@dataclass(frozen=True)
class CompositeStack:
    """Frozen nominal, its densified reference, and both controller stages."""

    cfg: RungConfig
    model: Any
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    metadata: dict[str, Any]
    dense_states: np.ndarray
    dense_controls: np.ndarray
    tracker: Any  # core.fast_pieces.FastDTVLQR
    static_gain: np.ndarray

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
        if tick < (self.cfg.switch_tick or 0):
            return float(self.tracker.policy(state, time_s))
        from cartpole_capsules.core.lqr import wrap_state_error

        upright = self.model.x_equilibrium("up")
        return float(-(self.static_gain @ wrap_state_error(state, upright, self.model.n)).item())

    def phase(self, tick: int) -> str:
        return "tvlqr" if tick < (self.cfg.switch_tick or 0) else "static_care"


def _load_frozen_nominal(cfg: RungConfig) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    active = next(n for n in cfg.nominals if n.role == "active")
    path = base.capsule_root(cfg) / active.path
    if base.sha256_file(path) != active.sha256:
        raise ValueError(f"frozen nominal bytes do not match authority: {active.path}")
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
        metadata = {
            key: np.asarray(data[key]).item() for key in data.files if key not in {"x", "u"}
        }
    expected = cfg.extras.get("nominal_meta") or {}
    if states.shape != tuple(expected.get("state_shape", ())):
        raise ValueError(f"frozen nominal state shape mismatch: {active.path}")
    if controls.shape != tuple(expected.get("control_shape", ())):
        raise ValueError(f"frozen nominal control shape mismatch: {active.path}")
    if not (np.all(np.isfinite(states)) and np.all(np.isfinite(controls))):
        raise ValueError(f"frozen nominal has nonfinite values: {active.path}")
    return states, controls, float(metadata.get("horizon", active.horizon_s)), metadata


def _tracker_cost(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Locked n12 tracking cost: ang-vel block x0.25, Qf = CARE at that Q."""
    from cartpole_capsules.core.lqr import make_Q, make_R, static_lqr

    tracking_q = make_Q(model.n)
    tracking_q[model.n + 2 :, model.n + 2 :] *= 0.25
    _, tracking_terminal_p = static_lqr(model, Q=tracking_q, R=make_R())
    return tracking_q, tracking_terminal_p


def load(cfg: RungConfig) -> CompositeStack:
    """Load the frozen nominal, densify the reference, rebuild the tracker."""
    from cartpole_capsules.core.fast_pieces import FastDTVLQR, make_densifier
    from cartpole_capsules.core.lqr import make_R, static_lqr

    model = base.build_model(cfg)
    states, controls, horizon_s, metadata = _load_frozen_nominal(cfg)
    stride = int(cfg.extras.get("densify_stride", 4))
    n_sub = int(cfg.extras.get("densify_n_sub", 4))
    densify = make_densifier(model, cfg.control_dt_s, n_sub, stride, len(controls))
    dense_states, dense_controls = densify(states, controls)
    if dense_states.shape != (len(controls) * stride + 1, model.nx):
        raise ValueError("densified reference shape mismatch")
    if not np.array_equal(dense_controls, np.repeat(controls, stride)):
        raise ValueError("densified reference controls are not stride-repeated")
    tracking_q, tracking_terminal_p = _tracker_cost(model)
    tracker = FastDTVLQR(
        model,
        dense_states,
        dense_controls,
        cfg.control_dt_s,
        Qf=tracking_terminal_p,
        Q=tracking_q,
        R=make_R(),
    )
    static_gain, _ = static_lqr(model)
    static_gain = np.asarray(static_gain).reshape(-1)
    return CompositeStack(
        cfg=cfg,
        model=model,
        nominal_states=states,
        nominal_controls=controls,
        metadata={**metadata, "horizon": horizon_s},
        dense_states=dense_states,
        dense_controls=dense_controls,
        tracker=tracker,
        static_gain=static_gain,
    )


def run(cfg: RungConfig, stack: CompositeStack) -> RunResult:
    """The one live 21700-tick rollout with the n12 checks ported verbatim."""
    from cartpole_capsules.core.rollout import run_policy

    total_ticks = cfg.total_ticks
    if total_ticks is None:
        raise ValueError("n12 registry must pin total_ticks")
    switch = cfg.switch_tick
    if switch is None:
        raise ValueError("n12 registry must pin switch_tick")
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
    mask = base.success_mask(stack.model, record.states, cfg)
    hold_mask = mask[switch:]
    force = force_stats(record, cfg)
    track = track_stats(record, cfg)
    metrics: dict[str, Any] = {
        "nominal": {
            "sha256": next(n for n in cfg.nominals if n.role == "active").sha256,
            "metadata": stack.metadata,
            "densify_stride": int(cfg.extras["densify_stride"]),
        },
        "execution": {
            "duration_s": total_ticks * cfg.control_dt_s,
            "phase_sequence_valid": base.phase_schedule(record, cfg),
            "start_state": "exact_hanging_equilibrium",
            "switch_tick": switch,
            "total_ticks": total_ticks,
            "time_grid_exact_1khz": bool(
                np.array_equal(record.times, np.arange(total_ticks + 1) * cfg.control_dt_s)
            ),
        },
        "forces": force,
        "track": track,
        "success_set": {
            "sampled_hold_s": base.trailing_hold_s(hold_mask, cfg.control_dt_s),
            "sampled_hold_samples": base.trailing_run_length(hold_mask),
            "every_1khz_sample_from_switch_through_final_in_success_set": bool(np.all(hold_mask)),
            "first_hold_state_out_of_success_set": base.first_event(
                ~hold_mask, tick_offset=switch, control_dt_s=cfg.control_dt_s
            ),
            "switch_state_in_success_set": bool(mask[switch]),
        },
        "finite": {
            "all_applied_forces": bool(np.all(np.isfinite(record.applied))),
            "all_raw_forces": bool(np.all(np.isfinite(record.raw))),
            "all_states": bool(np.all(np.isfinite(record.states))),
        },
    }
    checks: dict[str, bool] = {
        "exact_hanging_start": bool(
            np.array_equal(record.states[0], stack.model.x_equilibrium("down"))
        ),
        "live_policy_schedule": metrics["execution"]["phase_sequence_valid"]
        and metrics["execution"]["time_grid_exact_1khz"],
        "finite_live_values": all(metrics["finite"].values()),
        "no_raw_force_violation": force["first_raw_over_force_bound"] is None,
        "no_simulator_clipping": force["first_clipping"] is None,
        "track_bound": track["first_exceedance"] is None,
        "switch_in_locked_success_set": metrics["success_set"]["switch_state_in_success_set"],
        "sampled_locked_hold_at_least_5_s": metrics["success_set"]["sampled_hold_s"] >= 5.0,
        "full_static_window_has_every_1khz_sample_in_success_set": bool(np.all(hold_mask)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return RunResult(cfg.rung, record, {**metrics, "checks": checks}, not failures, tuple(failures))
