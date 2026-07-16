"""Exact-ZOH discrete TVLQR used by the authoritative n=8 rollout."""
from __future__ import annotations

import numpy as np
import scipy.linalg as scipy_linalg

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error


def zoh_ab(
    model: NLinkCartPole, state: np.ndarray, force: float, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize one linearization with an exact zero-order hold."""
    a_continuous, b_continuous = model.linearize(state, force)
    state_size = a_continuous.shape[0]
    augmented = np.zeros((state_size + 1, state_size + 1))
    augmented[:state_size, :state_size] = a_continuous * dt_s
    augmented[:state_size, state_size] = b_continuous.reshape(-1) * dt_s
    exponential = scipy_linalg.expm(augmented)
    return exponential[:state_size, :state_size], exponential[:state_size, state_size]


class DiscreteTVLQR:
    """Per-tick feedback ``u[k] - K[k] @ (x - x_nom[k])``."""

    def __init__(
        self,
        model: NLinkCartPole,
        states: np.ndarray,
        controls: np.ndarray,
        dt_s: float,
    ) -> None:
        self.model = model
        self.n = model.n
        self.states = np.asarray(states, dtype=float)
        self.controls = np.asarray(controls, dtype=float).reshape(-1)
        self.dt_s = dt_s
        self.n_controls = len(self.controls)
        if len(self.states) != self.n_controls + 1:
            raise ValueError("states must have one more row than controls")

        q = make_Q(self.n)
        r = float(make_R()[0, 0])
        _, terminal_cost = static_lqr(model)
        state_size = model.nx
        ad = np.empty((self.n_controls, state_size, state_size))
        bd = np.empty((self.n_controls, state_size))
        for tick in range(self.n_controls):
            ad[tick], bd[tick] = zoh_ab(
                model, self.states[tick], self.controls[tick], dt_s
            )

        gains = np.empty((self.n_controls, state_size))
        cost_to_go = terminal_cost.copy()
        for tick in range(self.n_controls - 1, -1, -1):
            a_discrete, b_discrete = ad[tick], bd[tick]
            cost_b = cost_to_go @ b_discrete
            gain = (a_discrete.T @ cost_b) / (r + b_discrete @ cost_b)
            gains[tick] = gain
            closed_loop = a_discrete - np.outer(b_discrete, gain)
            cost_to_go = q + r * np.outer(gain, gain) + closed_loop.T @ cost_to_go @ closed_loop
            cost_to_go = 0.5 * (cost_to_go + cost_to_go.T)

        self.ad = ad
        self.bd = bd
        self.gains = gains

    def policy(self, state: np.ndarray, time_s: float) -> float:
        """Return the nominal-plus-feedback force demand at ``time_s``."""
        tick = min(max(int(round(time_s / self.dt_s)), 0), self.n_controls - 1)
        error = wrap_state_error(state, self.states[tick], self.n)
        return float(self.controls[tick] - self.gains[tick] @ error)

    def monodromy_spectral_radius(self) -> float:
        """Return the spectral radius of the complete discrete closed loop."""
        transition = np.eye(self.ad.shape[1])
        for a_discrete, b_discrete, gain in zip(self.ad, self.bd, self.gains, strict=True):
            transition = (a_discrete - np.outer(b_discrete, gain)) @ transition
        return float(np.max(np.abs(np.linalg.eigvals(transition))))
