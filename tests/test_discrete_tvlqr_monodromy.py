"""The banked N11 discrete TVLQR must contract along its dense nominal."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _dtvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent
DENSE = REPO / "runs" / "r2" / "nom_n11_dense1ms_capture025_smoke3t03.npz"


def test_discrete_tvlqr_monodromy_contracts() -> None:
    with np.load(DENSE, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
    spec = CartPoleSpec().with_n_links(11)
    model = NLinkCartPole(spec)
    rho = DiscreteTVLQR(model, states, controls, spec.control_dt_s).monodromy()
    assert rho < 1.0, f"closed loop does not contract: rho={rho:.4g}"
    assert rho < 0.5, f"rho exceeds the release rigor bound: rho={rho:.4g}"
