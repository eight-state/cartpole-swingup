"""Shared deterministic dynamics and evidence checks for the n=8 release."""
from __future__ import annotations

import os as _os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_name, "1")

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402

__all__ = ["CartPoleSpec", "NLinkCartPole"]
