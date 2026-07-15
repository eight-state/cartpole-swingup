"""The locked N12 upright success predicate and rollout measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error
from n12_cartpole.simulator import (
    CONTROL_DT_S,
    FORCE_BOUND_N,
    N_LINKS,
    SWITCH_TICK,
    TOTAL_TICKS,
    TRACK_HALF_LENGTH_M,
)

if TYPE_CHECKING:
    from n12_cartpole.simulator import LiveRollout

MAX_WRAPPED_LINK_ANGLE_DEG = 5.0
MAX_LINK_RATE_RAD_S = 0.5
MAX_ABS_CART_M = 2.0
MAX_ABS_CART_RATE_M_S = 0.5


def in_success_set(model: NLinkCartPole, state: np.ndarray) -> bool:
    """Return whether ``state`` satisfies the locked upright hold predicate."""
    up = model.x_equilibrium("up")
    error = wrap_state_error(state, up, N_LINKS)
    angle_error = error[1 : N_LINKS + 1]
    link_rates = state[N_LINKS + 2 :]
    return bool(
        np.all(np.abs(angle_error) <= np.deg2rad(MAX_WRAPPED_LINK_ANGLE_DEG))
        and np.all(np.abs(link_rates) <= MAX_LINK_RATE_RAD_S)
        and abs(state[0]) <= MAX_ABS_CART_M
        and abs(state[N_LINKS + 1]) <= MAX_ABS_CART_RATE_M_S
    )


def state_metrics(model: NLinkCartPole, state: np.ndarray) -> dict[str, float | bool]:
    """Measure one state against the predicate."""
    error = wrap_state_error(state, model.x_equilibrium("up"), N_LINKS)
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


def _first_event(
    mask: np.ndarray, values: np.ndarray | None = None, tick_offset: int = 0
) -> dict[str, float | int] | None:
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return None
    index = int(indexes[0])
    tick = tick_offset + index
    event: dict[str, float | int] = {
        "tick": tick,
        "time_s": tick * CONTROL_DT_S,
    }
    if values is not None:
        event["value"] = float(values[index])
    return event


def _trailing_duration(mask: np.ndarray) -> tuple[float, int]:
    samples = 0
    for value in mask[::-1]:
        if not value:
            break
        samples += 1
    return max(0.0, (samples - 1) * CONTROL_DT_S), samples


def assess_rollout(rollout: LiveRollout) -> dict[str, Any]:
    """Recompute the release measurements from the live trajectory."""
    in_set = np.asarray(
        [in_success_set(rollout.model, state) for state in rollout.states],
        dtype=bool,
    )
    hold_in_set = in_set[SWITCH_TICK:]
    hold_duration_s, hold_samples = _trailing_duration(hold_in_set)
    force_delta = rollout.applied_forces - rollout.raw_forces
    raw_over_bound = np.abs(rollout.raw_forces) > FORCE_BOUND_N
    track_exceeded = np.abs(rollout.states[:, 0]) > TRACK_HALF_LENGTH_M
    return {
        "execution": {
            "duration_s": TOTAL_TICKS * CONTROL_DT_S,
            "phase_sequence_valid": bool(
                rollout.phases[:SWITCH_TICK] == ("tvlqr",) * SWITCH_TICK
                and rollout.phases[SWITCH_TICK:]
                == ("static_care",) * (TOTAL_TICKS - SWITCH_TICK)
            ),
            "single_live_rollout": True,
            "start_state": "exact_hanging_equilibrium",
            "switch_tick": SWITCH_TICK,
            "total_ticks": TOTAL_TICKS,
            "time_grid_exact_1khz": bool(
                np.array_equal(
                    rollout.times,
                    np.arange(TOTAL_TICKS + 1) * CONTROL_DT_S,
                )
            ),
        },
        "finite": {
            "all_applied_forces": bool(
                np.all(np.isfinite(rollout.applied_forces))
            ),
            "all_raw_forces": bool(np.all(np.isfinite(rollout.raw_forces))),
            "all_states": bool(np.all(np.isfinite(rollout.states))),
        },
        "forces": {
            "applied_peak_abs_n": float(
                np.max(np.abs(rollout.applied_forces))
            ),
            "first_clipping": _first_event(
                np.abs(force_delta) > 0.0, force_delta
            ),
            "first_raw_over_force_bound": _first_event(
                raw_over_bound, rollout.raw_forces
            ),
            "max_raw_applied_abs_delta_n": float(
                np.max(np.abs(force_delta))
            ),
            "raw_peak_abs_n": float(np.max(np.abs(rollout.raw_forces))),
        },
        "success_set": {
            "continuous_hold_s": hold_duration_s,
            "continuous_hold_samples": hold_samples,
            "every_state_from_switch_through_final_in_success_set": bool(
                np.all(hold_in_set)
            ),
            "first_hold_state_out_of_success_set": _first_event(
                ~hold_in_set, tick_offset=SWITCH_TICK
            ),
            "switch_state": state_metrics(
                rollout.model, rollout.states[SWITCH_TICK]
            ),
        },
        "track": {
            "bound_abs_m": TRACK_HALF_LENGTH_M,
            "first_exceedance": _first_event(
                track_exceeded, np.abs(rollout.states[:, 0])
            ),
            "peak_abs_cart_m": float(
                np.max(np.abs(rollout.states[:, 0]))
            ),
        },
    }
