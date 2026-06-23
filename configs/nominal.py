"""Single source of truth for the n=8 nominal trajectory and its grid.

The shipped n=8 dense nominal is the 1 ms DENSIFICATION of a 4 ms
collocation solve (`nom_n8_4ms.npz`, transcription defect 2.1e-12, peak
feedforward 23.2 N, terminal 0.0115 deg, 804 IPOPT iterations from the
Glueck MS continuation seed `nom_n8_gluck.npz`): each node's constant force
integrated through the simulator's exact ZOH stepping (4x RK4 0.25 ms
substeps per 1 ms tick; max node-boundary seam 4.233e-3, larger than
n=7 because the n=8 trajectory is stiffer — absorbed by feedback, peak
closed-loop demand 23.2 N). Closed-loop validation runs the REAL saturated
plant at 1 ms with exact-ZOH discrete TVLQR (monodromy rho = 0.156).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent
REPO = CONFIGS_DIR.parent
RESULTS = REPO / "results"


@dataclass(frozen=True)
class NominalSpec:
    file: str
    grid_dt_s: float
    n_nodes: int
    horizon_s: float
    is_native_1ms: bool
    label: str

    @property
    def path(self) -> Path:
        return RESULTS / self.file


NOMINAL = NominalSpec(
    file="nom_n8_dense1ms.npz",
    grid_dt_s=0.001,
    n_nodes=9000,
    horizon_s=9.0,
    is_native_1ms=True,
    label="densified 4 ms collocation nominal (1 ms grid)",
)

NOMINAL_4MS = NominalSpec(
    file="nom_n8_4ms.npz",
    grid_dt_s=0.004,
    n_nodes=2250,
    horizon_s=9.0,
    is_native_1ms=False,
    label="4 ms collocation parent solve",
)
