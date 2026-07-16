"""Focused, reproducible N12 cart-pole comparison."""

from __future__ import annotations

import os

# CasADi, NumPy, and SciPy are initialized after this package import. Keeping
# native pools single-threaded makes the disclosed numerical environment less
# surprising without changing the model or controller.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
