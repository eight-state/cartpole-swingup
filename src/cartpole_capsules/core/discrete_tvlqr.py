"""Exact-ZOH discrete-time TVLQR tracker about a nominal control sequence."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from cartpole_capsules.core.lqr import make_Q, make_R, static_lqr, wrap_state_error


def zoh_ab(model, x: np.ndarray, u: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretize one linearization with a zero-order-held input exactly."""
    a, b = model.linearize(x, float(u))
    nx = a.shape[0]
    augmented = np.zeros((nx + 1, nx + 1))
    augmented[:nx, :nx] = a * dt_s
    augmented[:nx, nx] = np.asarray(b).reshape(-1) * dt_s
    exponential = sla.expm(augmented)
    return exponential[:nx, :nx], exponential[:nx, nx]


class DiscreteTVLQR:
    """Per-tick tracker: ``u_k = u_nom[k] - K_k (x - x_nom[k])``.

    Block-expm ZOH discretization per tick, backward scalar-control Riccati
    ``S = Q + Rv kk' + Acl' S Acl``, monodromy as the spectral radius of the
    closed-loop product. Accepts an explicit per-tick-uniform ``q`` override
    (per-rung glue may scale blocks, e.g. the n12 ang-vel block by 0.25).
    """

    def __init__(
        self,
        model,
        states: np.ndarray,
        controls: np.ndarray,
        dt_s: float,
        qf: np.ndarray | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
    ) -> None:
        n = model.n
        nx = model.nx
        n_ticks = len(controls)
        if len(states) != n_ticks + 1:
            raise ValueError("the nominal needs one more state than controls")
        q = make_Q(n) if q is None else q
        r = make_R() if r is None else r
        r_scalar = float(np.asarray(r).reshape(-1)[0])
        if qf is None:
            _, qf = static_lqr(model)

        self.model = model
        self.n = n
        self.states = np.asarray(states, dtype=float)
        self.controls = np.asarray(controls, dtype=float).reshape(-1)
        self.dt_s = float(dt_s)
        self.n_ticks = n_ticks

        ad = np.empty((n_ticks, nx, nx))
        bd = np.empty((n_ticks, nx))
        for tick in range(n_ticks):
            ad[tick], bd[tick] = zoh_ab(model, self.states[tick], self.controls[tick], self.dt_s)

        gains = np.empty((n_ticks, nx))
        riccati = qf.copy()
        for tick in range(n_ticks - 1, -1, -1):
            a_tick, b_tick = ad[tick], bd[tick]
            sb = riccati @ b_tick
            gain = (a_tick.T @ sb) / (r_scalar + b_tick @ sb)
            gains[tick] = gain
            closed_loop = a_tick - np.outer(b_tick, gain)
            riccati = q + r_scalar * np.outer(gain, gain) + closed_loop.T @ riccati @ closed_loop
            riccati = 0.5 * (riccati + riccati.T)

        self.gains = gains
        self.ad = ad
        self.bd = bd
        self.initial_cost = riccati

    def policy(self, state: np.ndarray, time_s: float) -> float:
        """Return the unsaturated controller demand at a simulator tick."""
        tick = min(max(int(round(time_s / self.dt_s)), 0), self.n_ticks - 1)
        error = wrap_state_error(state, self.states[tick], self.n)
        return float(self.controls[tick] - self.gains[tick] @ error)

    def monodromy(self) -> float:
        """Return the spectral radius of the full discrete closed-loop product."""
        product = np.eye(self.ad.shape[1])
        for a_tick, b_tick, gain in zip(self.ad, self.bd, self.gains, strict=True):
            product = (a_tick - np.outer(b_tick, gain)) @ product
        return float(np.max(np.abs(np.linalg.eigvals(product))))
