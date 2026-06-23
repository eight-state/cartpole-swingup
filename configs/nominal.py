"""Single source of truth for the n=6 nominal trajectory and its grid.

This file is the one place that pins which n=6 nominal the repo loads and the
grid (per-control-tick spacing) that nominal was solved on. Every entry point
(``reproduce_n6.py``, ``scripts/demo_sextuple.py``,
``scripts/gen_validation_reports.py``) imports ``NOMINAL`` from here. Nothing
else hard-codes the filename or the grid.

The shipped n=6 nominal is on the native 1 ms grid (7000 control intervals,
7.0 s, 14-state), 1-step ZOH defect 0.0 (bit-exact 1 ms-consistent), at parity
with the n=4 / n=5 nominals. It is closed-loop validated in the 1 ms saturated
simulator (TVLQR tracks the native 1 ms reference at the 1 ms control rate).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent
REPO = CONFIGS_DIR.parent
RESULTS = REPO / "results"


@dataclass(frozen=True)
class NominalSpec:
    """A nominal file plus the grid it was solved on.

    ``grid_dt_s`` is the node spacing of the saved nominal. The closed-loop
    validator runs the plant at ``control_dt_s`` (1 ms) and interpolates the
    nominal, so ``grid_dt_s`` is a rigor fact about the reference, not the
    simulation rate. For the shipped nominal the two coincide at 1 ms.
    """

    file: str                 # filename inside results/
    grid_dt_s: float          # node spacing of the saved nominal
    n_nodes: int              # number of control intervals (len(u))
    horizon_s: float          # trajectory duration
    is_native_1ms: bool       # True when grid == the 1 ms control tick
    label: str                # short human label for the rigor status

    @property
    def path(self) -> Path:
        return RESULTS / self.file

    @property
    def grid_ms(self) -> float:
        return self.grid_dt_s * 1e3

    @property
    def substeps_per_node(self) -> float:
        """Control ticks per nominal node (1.0 when the nominal is native 1 ms)."""
        return self.grid_dt_s / 0.001


# Shipped n=6 nominal: native 1 ms grid (7000 control intervals, 7.0 s,
# 14-state), 1-step ZOH defect 0.0 (bit-exact), terminal max link angle
# 0.246 deg, peak feedforward 21.56 N.
NOMINAL = NominalSpec(
    file="nom_n6_gluck_cont.npz",
    grid_dt_s=0.001,
    n_nodes=7000,
    horizon_s=7.0,
    is_native_1ms=True,
    label="1 ms native grid (parity with n=4 / n=5); 1-step ZOH defect 0.0, bit-exact",
)

GRID = NOMINAL
FILE = NOMINAL.file
