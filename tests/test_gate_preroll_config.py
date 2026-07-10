"""Release-setting coverage for the fixed N11 pre-roll gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GATE_PATH = Path("scripts/gate_preroll.py")
_SPEC = importlib.util.spec_from_file_location("gate_preroll_under_test", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


def test_n11_release_schedule_is_fixed() -> None:
    assert _GATE.N_LINKS == 11
    assert _GATE.NOMINAL_PATH == "runs/r2/nom_n11_dense1ms_capture025_smoke3t03.npz"
    assert _GATE.T_PRE_S == 9.0
    assert _GATE.PRE_ROLL_TOL == 0.0
    assert _GATE.PRE_ROLL_VEL_Q_SCALE == 4.0
    assert _GATE.HOLD_WINDOW_S == 10.0
    assert _GATE.DEFAULT_WORKERS == 6


def test_wilson_interval_recomputes_banked_24_of_24_result() -> None:
    assert _GATE.wilson(24, 24) == (0.862, 1.0)


@pytest.mark.parametrize("n_trials", [0, -1])
def test_gate_rejects_nonpositive_trial_count(monkeypatch: pytest.MonkeyPatch, n_trials: int) -> None:
    monkeypatch.setattr(_GATE.sys, "argv", ["gate_preroll.py", str(n_trials)])
    with pytest.raises(ValueError, match="positive"):
        _GATE.main()
