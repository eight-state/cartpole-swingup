"""Shared closed-loop and controls-only rollout helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from cartpole_capsules.core.dynamics import NLinkCartPole


class Strategy(Protocol):
    """Controller interface consumed by :func:`run_policy`."""

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float: ...


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout with state, raw-force, applied-force, and event records."""

    times: np.ndarray
    states: np.ndarray
    raw: np.ndarray
    applied: np.ndarray
    phases: tuple[str, ...]
    first_nonfinite: dict[str, Any] | None = None
    first_rail_violation: dict[str, Any] | None = None
    quarter_cart_peak_m: float | None = None


def run_policy(
    model: NLinkCartPole,
    strategy: Strategy,
    total_ticks: int,
    control_dt_s: float,
    rk4_max_step_s: float,
    initial_state: np.ndarray,
    *,
    return_raw: bool = True,
    phase_label: Callable[[int], str] | None = None,
    quarter_metrics: bool = False,
) -> RolloutRecord:
    """Run a zero-order-hold policy with clipping at the simulator boundary."""
    if total_ticks <= 0:
        raise ValueError("total_ticks must be positive")
    n_substeps = max(1, int(np.ceil(control_dt_s / rk4_max_step_s)))
    substep_s = control_dt_s / n_substeps
    state = np.asarray(initial_state, dtype=float).reshape(model.nx).copy()
    states = np.empty((total_ticks + 1, model.nx), dtype=float)
    raw = np.empty(total_ticks, dtype=float)
    applied = np.empty(total_ticks, dtype=float)
    phases: list[str] = []
    states[0] = state
    first_nonfinite: dict[str, Any] | None = None
    first_rail: dict[str, Any] | None = None
    quarter_peak = abs(float(state[0]))

    for tick in range(total_ticks):
        time_s = tick * control_dt_s
        raw_force = float(strategy.control(tick, state.copy(), time_s))
        applied_force = float(
            np.clip(raw_force, -model.spec.force_bound_n, model.spec.force_bound_n)
        )
        raw[tick] = raw_force
        applied[tick] = applied_force
        phases.append(phase_label(tick) if phase_label is not None else "")
        for quarter in range(1, n_substeps + 1):
            state = model.rk4_step(state, applied_force, substep_s)
            if quarter_metrics:
                quarter_peak = max(quarter_peak, abs(float(state[0])))
                event_time_s = time_s + quarter * substep_s
                if first_nonfinite is None and not np.all(np.isfinite(state)):
                    first_nonfinite = {
                        "tick": tick,
                        "quarter": quarter,
                        "time_s": event_time_s,
                    }
                if first_rail is None and abs(float(state[0])) > model.spec.track_half_length_m:
                    first_rail = {
                        "tick": tick,
                        "quarter": quarter,
                        "time_s": event_time_s,
                        "value": abs(float(state[0])),
                    }
        states[tick + 1] = state

    return RolloutRecord(
        times=np.arange(total_ticks + 1, dtype=float) * control_dt_s,
        states=states,
        raw=raw if return_raw else applied.copy(),
        applied=applied,
        phases=tuple(phases),
        first_nonfinite=first_nonfinite,
        first_rail_violation=first_rail,
        quarter_cart_peak_m=quarter_peak if quarter_metrics else None,
    )


def replay_controls(
    model: NLinkCartPole,
    controls: np.ndarray,
    control_dt_s: float,
    rk4_max_step_s: float,
    *,
    quarter_metrics: bool = False,
) -> RolloutRecord:
    """Replay stored controls without clipping after the caller checks bounds."""
    controls = np.asarray(controls, dtype=float).reshape(-1)

    class StoredControls:
        def control(self, tick: int, state: np.ndarray, time_s: float) -> float:
            return float(controls[tick])

    record = run_policy(
        model,
        StoredControls(),
        len(controls),
        control_dt_s,
        rk4_max_step_s,
        model.x_equilibrium("down"),
        return_raw=True,
        quarter_metrics=quarter_metrics,
    )
    if not np.array_equal(record.raw, record.applied):
        raise ValueError("controls-only replay received an over-limit force")
    return record
