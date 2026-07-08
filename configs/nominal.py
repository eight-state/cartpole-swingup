"""Single source of truth for the n=10 nominal trajectory and its grid.

The shipped n=10 dense nominal is the 1 ms DENSIFICATION of a 4 ms collocation
solve (`nom_n10_4ms_wv1en3t.npz`, RK4-4ms transcription defect 1.361e-07, peak
feedforward 35.97 N, terminal 0.0115 deg, from the Glueck MS continuation seed
`nom_n10_gluck.npz`): each node's constant force integrated through the
simulator's exact ZOH stepping (4x RK4 0.25 ms substeps per 1 ms tick; max
node-boundary seam 8.195e-06). Closed-loop validation runs the REAL saturated
plant at 1 ms with exact-ZOH discrete TVLQR (monodromy rho = 0.1042).

The velocity penalty (`w_v = 1e-3`, a running Sum vel^2 cost in the collocation
objective) carries over the idea from n=9 (which used `w_v = 6e-4`); the ``t``
(tight) suffix marks the re-solve
whose exit rule required defect < 1e-6 and dual < 5e-4. An earlier
acceptable_tol=1e-4 shortcut left defect 5.6e-6 and the track blew through the
stiff t~2.23 s window; the tightened re-solve fixed it. See docs/METHOD.md.
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
    file="nom_n10_dense1ms_wv1en3t.npz",
    grid_dt_s=0.001,
    n_nodes=10000,
    horizon_s=10.0,
    is_native_1ms=True,
    label="densified 4 ms w_v=1e-3 (tight) collocation nominal (1 ms grid)",
)

NOMINAL_4MS = NominalSpec(
    file="nom_n10_4ms_wv1en3t.npz",
    grid_dt_s=0.004,
    n_nodes=2500,
    horizon_s=10.0,
    is_native_1ms=False,
    label="4 ms w_v=1e-3 (tight) collocation parent solve",
)
