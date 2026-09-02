"""Committed gate: the exact-ZOH discrete TVLQR closed loop CONTRACTS along
the shipped n=10 nominal (monodromy spectral radius < 1).

This is the controller fact behind the replay: the discrete-time design uses
per-tick exact-ZOH discretization and a backward discrete Riccati recursion,
which gives rho ~= 0.1042 for the shipped nominal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent


def test_discrete_tvlqr_monodromy_contracts() -> None:
    d = np.load(REPO / "results" / "nom_n10_dense1ms_wv1en3t.npz")
    spec = CartPoleSpec().with_n_links(10)
    m = NLinkCartPole(spec)
    tv = DiscreteTVLQR(m, np.asarray(d["x"], float),
                       np.asarray(d["u"], float), spec.control_dt_s)
    rho = tv.monodromy()
    assert rho < 1.0, f"closed loop does not contract: rho={rho:.4g}"
    assert rho < 0.5, f"rho unexpectedly large vs banked 0.1042: {rho:.4g}"
