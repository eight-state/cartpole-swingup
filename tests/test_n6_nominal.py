"""The fixed nominal is the public replay input and must fit the plant tick."""

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.n6 import (
    CONTROL_DT_S,
    NOMINAL_HORIZON_S,
    NOMINAL_NODES,
    _fixed_spec,
    _load_nominal,
)


def test_fixed_nominal_is_zoh_rk4_consistent() -> None:
    """Every saved control tick advances to its saved successor."""
    x_nom, u_nom, t_nom = _load_nominal()
    model = NLinkCartPole(_fixed_spec())
    assert x_nom.shape == (NOMINAL_NODES + 1, model.nx)
    assert u_nom.shape == (NOMINAL_NODES,)
    assert np.isclose(t_nom[-1], NOMINAL_HORIZON_S)

    n_substeps = int(np.ceil(CONTROL_DT_S / model.spec.rk4_max_step_s))
    dt_substep = CONTROL_DT_S / n_substeps
    max_defect = 0.0
    for index, force in enumerate(u_nom):
        state = x_nom[index].copy()
        for _ in range(n_substeps):
            state = model.rk4_step(state, float(force), dt_substep)
        max_defect = max(max_defect, float(np.max(np.abs(state - x_nom[index + 1]))))
    assert max_defect < 1e-10
