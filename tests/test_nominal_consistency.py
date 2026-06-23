"""Committed check that the saved n=5 nominal is 1 ms ZOH-consistent.

The headline README/METHOD claim is that ``results/nom_n5_gluck_cont.npz`` is
dynamically consistent under the simulator's own zero-order-hold step (4 RK4
substeps per 1 ms control tick, matching ``rollout_zoh``), with a per-tick
defect on the order of 1e-13. This test turns that claim into a committed gate:
integrate every saved tick forward with the simulator's n_sub-substep ZOH step
and require the worst single-step defect to be < 1e-10 state units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent
NOM_PATH = REPO / "results" / "nom_n5_gluck_cont.npz"


def _zoh_step(model: NLinkCartPole, x: np.ndarray, u: float,
              control_dt: float, rk4_max_step: float) -> np.ndarray:
    """One control tick via the simulator's fixed n_sub-substep ZOH RK4 step.

    Mirrors the substep schedule inside ``NLinkCartPole.rollout_zoh`` exactly
    (``n_sub = ceil(control_dt / rk4_max_step)`` substeps of equal size, force
    held constant across the tick).
    """
    n_sub = max(1, int(np.ceil(control_dt / rk4_max_step)))
    dt_sub = control_dt / n_sub
    xx = np.asarray(x, dtype=float).reshape(-1).copy()
    for _ in range(n_sub):
        xx = model.rk4_step(xx, u, dt_sub)
    return xx


def test_nominal_zoh_one_step_defect_below_1e_10() -> None:
    """Each saved tick reproduces the next under the simulator's ZOH step."""
    assert NOM_PATH.exists(), f"missing nominal: {NOM_PATH}"
    d = np.load(NOM_PATH)
    x_nom = np.asarray(d["x"], dtype=float)  # (N+1, nx)
    u_nom = np.asarray(d["u"], dtype=float).reshape(-1)  # (N,)
    horizon = float(d["horizon"])

    n = (x_nom.shape[1] // 2) - 1
    assert n == 5, f"expected n=5 nominal, got n={n}"

    spec = CartPoleSpec().with_n_links(n)
    model = NLinkCartPole(spec)
    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s

    n_ticks = len(u_nom)
    # The nominal control tick spacing must match the 1 ms control_dt.
    dt_grid = horizon / (len(x_nom) - 1)
    assert abs(dt_grid - control_dt) < 1e-9, (
        f"nominal grid spacing {dt_grid} != control_dt {control_dt}")

    max_defect = 0.0
    for k in range(n_ticks):
        x_next = _zoh_step(model, x_nom[k], float(u_nom[k]), control_dt, rk4_max)
        defect = float(np.max(np.abs(x_next - x_nom[k + 1])))
        if defect > max_defect:
            max_defect = defect

    assert max_defect < 1e-10, (
        f"nominal not 1 ms ZOH-consistent: max one-step defect {max_defect:.3e}"
    )
