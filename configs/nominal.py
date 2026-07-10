"""Locations and grid contracts for the banked N11 nominal trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "r2"


@dataclass(frozen=True)
class NominalSpec:
    file: str
    grid_dt_s: float
    n_nodes: int
    horizon_s: float

    @property
    def path(self) -> Path:
        return RUNS / self.file


NOMINAL = NominalSpec(
    file="nom_n11_dense1ms_capture025_smoke3t03.npz",
    grid_dt_s=0.001,
    n_nodes=10_000,
    horizon_s=10.0,
)

NOMINAL_4MS = NominalSpec(
    file="nom_n11_4ms_capture025_smoke3t03.npz",
    grid_dt_s=0.004,
    n_nodes=2_500,
    horizon_s=10.0,
)
