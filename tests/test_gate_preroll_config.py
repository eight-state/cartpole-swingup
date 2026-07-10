"""Focused coverage for the opt-in release-gate schedule controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec


_GATE_PATH = Path("scripts/gate_preroll.py")
_SPEC = importlib.util.spec_from_file_location("gate_preroll_under_test", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


def test_n9_to_n11_gate_controls_remain_opt_in_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """N12 controls are absent unless explicitly configured."""
    for name in (
        "TRACKER_LINK_RATE_Q_SCALE",
        "REFERENCE_DENSIFY_STRIDE",
        "TRACKER_TO_HOLD_SWITCH_TICK",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("gate_preroll_defaults", _GATE_PATH)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    assert gate.TRACKER_LINK_RATE_Q_SCALE == 1.0
    assert gate.REFERENCE_DENSIFY_STRIDE == 0
    assert gate.TRACKER_TO_HOLD_SWITCH_TICK is None
    assert gate.tracker_to_hold_switch_tick(None, 10_000) == 10_000


@pytest.mark.parametrize("value", ["", "9700.0", "-1", " 9700", "9700 "])
def test_tracker_to_hold_switch_rejects_non_integer_ticks(value: str) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        _GATE.tracker_to_hold_switch_tick(value, 10_000)


def test_tracker_to_hold_switch_accepts_exact_n12_tick_and_rejects_overrun() -> None:
    assert _GATE.tracker_to_hold_switch_tick("9700", 10_000) == 9700
    with pytest.raises(ValueError, match="exceeds nominal ticks"):
        _GATE.tracker_to_hold_switch_tick("10001", 10_000)


def test_opted_in_reference_densification_repeats_source_controls(tmp_path: Path) -> None:
    """A 4 ms source produces exactly four 1 ms ZOH controls per source control."""
    spec = CartPoleSpec(
        n_links=1,
        link_masses_kg=[0.1],
        link_lengths_m=[0.5],
        damping_links_n_m_s_rad=[0.0],
    )
    model = NLinkCartPole(spec)
    source = tmp_path / "coarse.npz"
    x0 = model.x_equilibrium("down")
    np.savez(source, x=np.vstack([x0, x0]), u=np.array([12.5]), horizon=0.004)

    old_nom = _GATE.NOM
    old_stride = _GATE.REFERENCE_DENSIFY_STRIDE
    try:
        _GATE.NOM = str(source)
        _GATE.REFERENCE_DENSIFY_STRIDE = 4
        dense_x, dense_u, horizon, densified = _GATE._load_reference(model, spec)
    finally:
        _GATE.NOM = old_nom
        _GATE.REFERENCE_DENSIFY_STRIDE = old_stride

    assert densified is True
    assert horizon == pytest.approx(0.004)
    assert dense_x.shape == (5, model.nx)
    assert np.array_equal(dense_u, np.full(4, 12.5))
    assert np.array_equal(dense_x[0], x0)
