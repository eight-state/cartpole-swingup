"""Rigor checks computed from the banked N11 parent and dense nominals."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "r2"
DENSE = RUNS / "nom_n11_dense1ms_capture025_smoke3t03.npz"
PARENT = RUNS / "nom_n11_4ms_capture025_smoke3t03.npz"
N_LINKS = 11


def _model() -> tuple[NLinkCartPole, CartPoleSpec]:
    spec = CartPoleSpec().with_n_links(N_LINKS)
    return NLinkCartPole(spec), spec


def _zoh_step(
    model: NLinkCartPole,
    state: np.ndarray,
    control: float,
    control_dt_s: float,
    rk4_max_step_s: float,
) -> np.ndarray:
    n_substeps = max(1, int(np.ceil(control_dt_s / rk4_max_step_s)))
    substep_s = control_dt_s / n_substeps
    advanced = np.asarray(state, dtype=float).reshape(-1).copy()
    for _ in range(n_substeps):
        advanced = model.rk4_step(advanced, control, substep_s)
    return advanced


def test_nominal_metadata_and_feedforward_margin() -> None:
    with np.load(DENSE, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
        n_links = int(np.asarray(data["n"]).item())
        force = float(np.asarray(data["force"]).item())
    assert states.shape == (10_001, 24)
    assert controls.shape == (10_000,)
    assert horizon_s == 10.0
    assert n_links == N_LINKS
    assert force == 150.0
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(controls))
    assert float(np.max(np.abs(controls))) < 40.0


def test_dense_nominal_matches_fixed_step_simulator() -> None:
    """Every dense tick agrees with the saturated simulator to a tight bound."""
    with np.load(DENSE, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
    model, spec = _model()
    worst = 0.0
    for index, control in enumerate(controls):
        advanced = _zoh_step(
            model,
            states[index],
            float(control),
            spec.control_dt_s,
            spec.rk4_max_step_s,
        )
        worst = max(worst, float(np.max(np.abs(advanced - states[index + 1]))))
    assert worst < 1e-6, f"dense simulator mismatch: {worst:.3e}"


def test_dense_nodes_match_parent_nodes() -> None:
    with np.load(DENSE, allow_pickle=False) as dense_data:
        dense_states = np.asarray(dense_data["x"], dtype=float)
    with np.load(PARENT, allow_pickle=False) as parent_data:
        parent_states = np.asarray(parent_data["x"], dtype=float)
    seam = float(np.max(np.abs(dense_states[4::4] - parent_states[1:])))
    assert seam < 1e-6, f"dense parent seam: {seam:.3e}"
