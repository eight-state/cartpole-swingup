"""The single live N12 rollout used by both the demo and verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error
from n12_cartpole.fast_pieces import FastDTVLQR, make_densifier

REPOSITORY = Path(__file__).resolve().parents[2]
NOMINAL_PATH = REPOSITORY / "artifacts" / "nom_n12_4ms_fast.npz"
N_LINKS = 12
CONTROL_DT_S = 0.001
RK4_SUBSTEP_S = 0.00025
FORCE_BOUND_N = 150.0
TRACK_HALF_LENGTH_M = 10.0
SWITCH_TICK = 9700
TOTAL_TICKS = 21700


@dataclass(frozen=True)
class FrozenNominal:
    """Data loaded from the disclosed, precomputed 4 ms nominal artifact."""

    states: np.ndarray
    controls: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LiveRollout:
    """One recomputed closed-loop trajectory and its loaded reference."""

    model: NLinkCartPole
    nominal: FrozenNominal
    dense_states: np.ndarray
    dense_controls: np.ndarray
    times: np.ndarray
    states: np.ndarray
    raw_forces: np.ndarray
    applied_forces: np.ndarray
    phases: tuple[str, ...]


def load_frozen_nominal(path: Path = NOMINAL_PATH) -> FrozenNominal:
    """Load the released nominal without synthesising or altering it."""
    with np.load(path, allow_pickle=False) as data:
        metadata = {
            key: np.asarray(data[key]).item()
            for key in data.files
            if key not in {"x", "u"}
        }
        return FrozenNominal(
            states=np.asarray(data["x"], dtype=float),
            controls=np.asarray(data["u"], dtype=float).reshape(-1),
            metadata=metadata,
        )


def run_live_rollout() -> LiveRollout:
    """Recompute the released unperturbed rollout from exact hanging."""
    nominal = load_frozen_nominal()
    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.1] * N_LINKS,
        link_lengths_m=[0.5] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND_N,
        track_half_length_m=TRACK_HALF_LENGTH_M,
        control_rate_hz=1000.0,
        rk4_max_step_s=RK4_SUBSTEP_S,
    )
    model = NLinkCartPole(spec)
    dense_states, dense_controls = make_densifier(
        model, CONTROL_DT_S, 4, 4, len(nominal.controls)
    )(nominal.states, nominal.controls)
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
    static_gain, _ = static_lqr(model)
    static_gain = np.asarray(static_gain).reshape(-1)
    upright = model.x_equilibrium("up")
    raw_forces: list[float] = []
    phases: list[str] = []

    def live_policy(state: np.ndarray, time_s: float) -> float:
        tick = int(round(time_s / CONTROL_DT_S))
        if tick < SWITCH_TICK:
            force_n = float(tracker.policy(state, time_s))
            phases.append("tvlqr")
        else:
            force_n = -float(
                static_gain @ wrap_state_error(state, upright, N_LINKS)
            )
            phases.append("static_care")
        raw_forces.append(force_n)
        return force_n

    times, states, applied_forces = model.rollout_zoh(
        model.x_equilibrium("down"),
        live_policy,
        TOTAL_TICKS * CONTROL_DT_S,
        CONTROL_DT_S,
        RK4_SUBSTEP_S,
    )
    return LiveRollout(
        model=model,
        nominal=nominal,
        dense_states=dense_states,
        dense_controls=dense_controls,
        times=times,
        states=states,
        raw_forces=np.asarray(raw_forces, dtype=float),
        applied_forces=applied_forces,
        phases=tuple(phases),
    )
