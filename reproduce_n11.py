"""Reproduce the released N11 result from committed artifacts.

    uv sync --locked
    uv run python reproduce_n11.py
    uv run python reproduce_n11.py --gate

The default command checks every artifact hash and gate invariant, recomputes
nominal facts, builds the discrete TVLQR monodromy, and reruns the unperturbed
closed loop. ``--gate`` additionally reruns the 24-trial gate for seeds 12345,
777, and 2024 with the release settings. It rewrites the gate JSONs and then
requires byte-identical hashes, so it fails if the rerun differs from the banked
release artifacts.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO / "configs")]

from _dtvlqr import DiscreteTVLQR  # noqa: E402
from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from cartpole_race.funnels import in_success_set  # noqa: E402
from cartpole_race.lqr import static_lqr, wrap_state_error  # noqa: E402
from fast_pieces import FastDTVLQR  # noqa: E402
from nominal import NOMINAL, NOMINAL_4MS  # noqa: E402
from release_audit import audit_release_artifacts  # noqa: E402

N_LINKS = 11
HOLD_S = 5.0
HOLD_WINDOW_S = 12.0


def _single_step_parent_mismatch(model: NLinkCartPole) -> float:
    with np.load(NOMINAL_4MS.path, allow_pickle=False) as data:
        states = np.asarray(data["x"], dtype=float)
        controls = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
    step_s = horizon_s / len(controls)
    worst = 0.0
    for index, control in enumerate(controls):
        state = states[index]
        k1 = model.f(state, float(control))
        k2 = model.f(state + 0.5 * step_s * np.asarray(k1).reshape(-1), float(control))
        k3 = model.f(state + 0.5 * step_s * np.asarray(k2).reshape(-1), float(control))
        k4 = model.f(state + step_s * np.asarray(k3).reshape(-1), float(control))
        advanced = state + (step_s / 6.0) * (
            np.asarray(k1).reshape(-1)
            + 2 * np.asarray(k2).reshape(-1)
            + 2 * np.asarray(k3).reshape(-1)
            + np.asarray(k4).reshape(-1)
        )
        worst = max(worst, float(np.max(np.abs(advanced - states[index + 1]))))
    return worst


def _dense_parent_seam() -> float:
    with np.load(NOMINAL.path, allow_pickle=False) as dense_data:
        dense_x = np.asarray(dense_data["x"], dtype=float)
    with np.load(NOMINAL_4MS.path, allow_pickle=False) as parent_data:
        parent_x = np.asarray(parent_data["x"], dtype=float)
    return float(np.max(np.abs(dense_x[4::4] - parent_x[1:])))


def _trailing_success_seconds(model: NLinkCartPole, states: np.ndarray) -> float:
    run = 0
    for state in states:
        run = run + 1 if in_success_set(model, state) else 0
    return max(0, run - 1) * model.spec.control_dt_s


def _unperturbed(model: NLinkCartPole, spec: CartPoleSpec) -> dict[str, Any]:
    with np.load(NOMINAL.path, allow_pickle=False) as data:
        nominal_x = np.asarray(data["x"], dtype=float)
        nominal_u = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
    tracker = FastDTVLQR(model, nominal_x, nominal_u, spec.control_dt_s)
    force_bound = spec.force_bound_n

    def tracking_policy(state: np.ndarray, elapsed_s: float) -> float:
        return float(np.clip(tracker.policy(state, elapsed_s), -force_bound, force_bound))

    _, tracked_x, tracked_u = model.rollout_zoh(
        nominal_x[0],
        tracking_policy,
        horizon_s,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    x_up = model.x_equilibrium("up")
    handoff = tracked_x[-1]
    handoff_deg = float(
        np.rad2deg(
            np.max(np.abs(wrap_state_error(handoff, x_up, N_LINKS)[1 : 1 + N_LINKS]))
        )
    )
    hold_gain, _ = static_lqr(model)
    hold_gain = np.asarray(hold_gain).reshape(-1)

    def hold_policy(state: np.ndarray, _: float) -> float:
        return float(
            np.clip(
                -float(hold_gain @ wrap_state_error(state, x_up, N_LINKS)),
                -force_bound,
                force_bound,
            )
        )

    _, held_x, held_u = model.rollout_zoh(
        handoff,
        hold_policy,
        HOLD_WINDOW_S,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    hold_s = _trailing_success_seconds(model, held_x)
    max_cart_abs_m = float(
        max(np.max(np.abs(tracked_x[:, 0])), np.max(np.abs(held_x[:, 0])))
    )
    success = bool(
        handoff_deg < 0.05
        and hold_s >= HOLD_S
        and max_cart_abs_m <= spec.track_half_length_m
    )
    return {
        "handoff_deg": handoff_deg,
        "hold_trailing_s": hold_s,
        "track_peak_force_n": float(np.max(np.abs(tracked_u))),
        "hold_peak_force_n": float(np.max(np.abs(held_u))),
        "max_cart_abs_m": max_cart_abs_m,
        "success": success,
    }


def _run_gate() -> int:
    return_code = 0
    for seed in (12345, 777, 2024):
        print(f"[4] gate seed {seed}: 24 trials, 9 s pre-roll cap, six workers")
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "gate_preroll.py"),
                "24",
                str(seed),
                "9.0",
                "6",
            ],
            cwd=REPO,
            check=False,
        )
        return_code = return_code or completed.returncode
    return return_code


def main() -> int:
    audit = audit_release_artifacts()
    print(
        "[0] artifact audit: "
        f"{audit['aggregate_successes']}/{audit['aggregate_trials']} banked gate successes; "
        "five SHA256 digests match"
    )

    spec = CartPoleSpec().with_n_links(N_LINKS)
    model = NLinkCartPole(spec)
    with np.load(NOMINAL.path, allow_pickle=False) as data:
        dense_x = np.asarray(data["x"], dtype=float)
        dense_u = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
    peak_feedforward = float(np.max(np.abs(dense_u)))
    single_step_parent_mismatch = _single_step_parent_mismatch(model)
    dense_parent_seam = _dense_parent_seam()
    print(
        f"[1] nominal: {NOMINAL.file}, {len(dense_u)} ticks, {horizon_s:.1f} s, "
        f"peak feedforward {peak_feedforward:.6f} N"
    )
    print(f"    diagnostic one-step 4 ms integration mismatch {single_step_parent_mismatch:.3e}")
    print(f"    release-schedule dense to parent seam {dense_parent_seam:.3e}")

    print("[2] building exact-ZOH discrete TVLQR")
    discrete_tracker = DiscreteTVLQR(model, dense_x, dense_u, spec.control_dt_s)
    rho = discrete_tracker.monodromy()
    print(f"    closed-loop monodromy rho {rho:.7g}")

    result = _unperturbed(model, spec)
    print("[3] unperturbed saturated closed loop")
    print(
        f"    handoff {result['handoff_deg']:.7f} deg, "
        f"track peak {result['track_peak_force_n']:.7f} N, "
        f"hold {result['hold_trailing_s']:.1f} s, "
        f"hold peak {result['hold_peak_force_n']:.7f} N, "
        f"max cart {result['max_cart_abs_m']:.7f} m"
    )
    if not result["success"] or rho >= 1.0:
        print("REPRODUCTION FAILED")
        return 1
    print("N11 UNPERTURBED CLOSED-LOOP PASS")

    if "--gate" not in sys.argv:
        return 0
    return_code = _run_gate()
    if return_code:
        return return_code
    rerun_audit = audit_release_artifacts()
    print(
        "[5] gate artifact audit: "
        f"{rerun_audit['aggregate_successes']}/{rerun_audit['aggregate_trials']} and hashes match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
