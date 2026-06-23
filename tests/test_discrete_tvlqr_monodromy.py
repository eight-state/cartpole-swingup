"""Committed gate: the exact-ZOH discrete TVLQR closed loop CONTRACTS along
the shipped n=8 nominal (monodromy spectral radius < 1).

This is the controller fact that unlocked n=8: the repo's continuous-Riccati
TVLQR with interpolated gains is closed-loop UNSTABLE along this nominal
(rho = 47.85 at n=7), while the discrete-time design (per-tick exact ZOH
discretization + backward discrete Riccati, scripts/_dtvlqr.py) gives
rho ~= 0.156. See docs/METHOD.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from _dtvlqr import DiscreteTVLQR  # noqa: E402


def test_discrete_tvlqr_monodromy_contracts() -> None:
    d = np.load(REPO / "results" / "nom_n8_dense1ms.npz")
    spec = CartPoleSpec().with_n_links(8)
    m = NLinkCartPole(spec)
    tv = DiscreteTVLQR(m, np.asarray(d["x"], float),
                       np.asarray(d["u"], float), spec.control_dt_s)
    rho = tv.monodromy()
    assert rho < 1.0, f"closed loop does not contract: rho={rho:.4g}"
    assert rho < 0.5, f"rho unexpectedly large vs banked 0.156: {rho:.4g}"
