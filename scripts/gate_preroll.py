"""Run the fixed N11 perturbed-initial-condition release gate.

Usage: uv run python scripts/gate_preroll.py [n_ic] [seed] [T_pre_s] [workers]

The released command uses 24 trials, each released seed, and six workers. The
controller uses the fixed 9 s pre-roll, PREROLL_TOL=0, and
PREROLL_VEL_Q_SCALE=4 settings that produced the banked artifacts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "r2"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

N_LINKS = 11
NOMINAL_PATH = "runs/r2/nom_n11_dense1ms_capture025_smoke3t03.npz"
SIGMA = 0.02
T_PRE_S = 9.0
PRE_ROLL_TOL = 0.0
PRE_ROLL_VEL_Q_SCALE = 4.0
HOLD_S = 5.0
HOLD_WINDOW_S = 10.0
DEFAULT_WORKERS = 6

_GLOBALS: dict[str, Any] = {}


def _initialise() -> None:
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.funnels import in_success_set
    from cartpole_race.lqr import make_R, static_lqr, wrap_state_error
    from fast_pieces import FastDTVLQR

    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.10] * N_LINKS,
        link_lengths_m=[0.50] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=150.0,
    )
    model = NLinkCartPole(spec)
    with np.load(REPO / NOMINAL_PATH, allow_pickle=False) as data:
        nominal_x = np.asarray(data["x"], dtype=float)
        nominal_u = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(data["horizon"]).item())
    tracker = FastDTVLQR(model, nominal_x, nominal_u, spec.control_dt_s)
    hold_gain, _ = static_lqr(model)
    x_down = model.x_equilibrium("down")
    ad, bd = model.linearize(x_down, 0.0)
    down_q = np.concatenate(
        [
            [200.0],
            80.0 * np.ones(N_LINKS),
            [50.0 * PRE_ROLL_VEL_Q_SCALE],
            80.0 * PRE_ROLL_VEL_Q_SCALE * np.ones(N_LINKS),
        ]
    )
    down_p = sla.solve_continuous_are(ad, bd, np.diag(down_q), make_R())
    down_gain = np.linalg.solve(make_R(), bd.T @ down_p).reshape(-1)
    _GLOBALS.update(
        model=model,
        spec=spec,
        nominal_x=nominal_x,
        horizon_s=horizon_s,
        tracker=tracker,
        hold_gain=np.asarray(hold_gain).reshape(-1),
        x_up=model.x_equilibrium("up"),
        x_down=x_down,
        down_gain=down_gain,
        wrap=wrap_state_error,
        in_success_set=in_success_set,
    )


def _wrap_down(state: np.ndarray) -> np.ndarray:
    error = np.asarray(state, dtype=float).reshape(-1) - _GLOBALS["x_down"]
    error[1 : 1 + N_LINKS] = (
        error[1 : 1 + N_LINKS] + np.pi
    ) % (2 * np.pi) - np.pi
    return error


def _trailing_success_seconds(states: np.ndarray) -> float:
    model = _GLOBALS["model"]
    in_success_set = _GLOBALS["in_success_set"]
    run = 0
    for state in states:
        run = run + 1 if in_success_set(model, state) else 0
    return max(0, run - 1) * model.spec.control_dt_s


def run_ic(task: tuple[int, int]) -> dict[str, Any]:
    """Evaluate one fixed-seed perturbation with the released controller."""
    tag, seed = task
    model = _GLOBALS["model"]
    spec = _GLOBALS["spec"]
    nominal_x = _GLOBALS["nominal_x"]
    tracker = _GLOBALS["tracker"]
    hold_gain = _GLOBALS["hold_gain"]
    x_up = _GLOBALS["x_up"]
    down_gain = _GLOBALS["down_gain"]
    wrap = _GLOBALS["wrap"]
    force_bound = spec.force_bound_n

    rng = np.random.default_rng((seed, tag))
    perturbation = np.zeros(model.nx)
    perturbation[0] = rng.normal(0, SIGMA)
    perturbation[1 : 1 + N_LINKS] = rng.normal(0, SIGMA, N_LINKS)
    perturbation[1 + N_LINKS] = rng.normal(0, SIGMA)
    perturbation[2 + N_LINKS :] = rng.normal(0, SIGMA, N_LINKS)
    state = (nominal_x[0] + perturbation).copy()
    perturbation_deg = float(
        np.rad2deg(np.max(np.abs(perturbation[1 : 1 + N_LINKS])))
    )

    def pre_roll_policy(x: np.ndarray, _: float) -> float:
        return float(np.clip(-float(down_gain @ _wrap_down(x)), -force_bound, force_bound))

    elapsed_s = 0.0
    pre_roll_peak_force = 0.0
    pre_roll_peak_cart = 0.0
    while elapsed_s < T_PRE_S - 1e-9:
        _, pre_roll_x, pre_roll_u = model.rollout_zoh(
            state,
            pre_roll_policy,
            0.5,
            spec.control_dt_s,
            spec.rk4_max_step_s,
        )
        state = pre_roll_x[-1]
        elapsed_s += 0.5
        pre_roll_peak_force = max(pre_roll_peak_force, float(np.max(np.abs(pre_roll_u))))
        pre_roll_peak_cart = max(pre_roll_peak_cart, float(np.max(np.abs(pre_roll_x[:, 0]))))
        metric = _wrap_down(state)
        radius = max(
            float(np.max(np.abs(metric[1 : 1 + N_LINKS]))),
            float(np.max(np.abs(metric[2 + N_LINKS :]))),
        )
        if radius < PRE_ROLL_TOL:
            break

    residual = float(np.max(np.abs(_wrap_down(state))))

    def tracking_policy(x: np.ndarray, t: float) -> float:
        return float(np.clip(tracker.policy(x, t), -force_bound, force_bound))

    _, tracked_x, tracked_u = model.rollout_zoh(
        state,
        tracking_policy,
        _GLOBALS["horizon_s"],
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    handoff = tracked_x[-1]
    if np.any(np.isnan(handoff)):
        return {
            "tag": tag,
            "success": False,
            "fail": "track_nan",
            "pert_deg": round(perturbation_deg, 3),
            "resid": round(residual, 5),
        }

    handoff_deg = float(
        np.rad2deg(np.max(np.abs(wrap(handoff, x_up, N_LINKS)[1 : 1 + N_LINKS])))
    )
    if handoff_deg > 20.0:
        return {
            "tag": tag,
            "success": False,
            "fail": "track_diverged",
            "handoff_deg": round(handoff_deg, 3),
            "pert_deg": round(perturbation_deg, 3),
            "resid": round(residual, 5),
        }

    def hold_policy(x: np.ndarray, _: float) -> float:
        return float(np.clip(-float(hold_gain @ wrap(x, x_up, N_LINKS)), -force_bound, force_bound))

    _, held_x, held_u = model.rollout_zoh(
        handoff,
        hold_policy,
        HOLD_WINDOW_S,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    hold_s = _trailing_success_seconds(held_x)
    track_ok = bool(
        max(
            np.max(np.abs(tracked_x[:, 0])),
            np.max(np.abs(held_x[:, 0])),
            pre_roll_peak_cart,
        )
        <= spec.track_half_length_m
    )
    peak_force = float(
        max(
            np.max(np.abs(tracked_u)),
            np.max(np.abs(held_u)),
            pre_roll_peak_force,
        )
    )
    success = bool(hold_s >= HOLD_S - 1e-9 and track_ok)
    return {
        "tag": tag,
        "success": success,
        "handoff_deg": round(handoff_deg, 4),
        "hold_s": round(hold_s, 2),
        "peakF": round(peak_force, 1),
        "pert_deg": round(perturbation_deg, 3),
        "resid": round(residual, 5),
        "t_pre": round(elapsed_s, 2),
        "track_ok": track_ok,
        "fail": None if success else "hold",
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (round(float(center - half_width), 4), round(float(min(1.0, center + half_width)), 4))


def main() -> int:
    n_ic = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    requested_pre_roll_s = float(sys.argv[3]) if len(sys.argv) > 3 else T_PRE_S
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_WORKERS
    if n_ic <= 0 or workers <= 0:
        raise ValueError("n_ic and workers must be positive")
    if not np.isclose(requested_pre_roll_s, T_PRE_S, rtol=0.0, atol=1e-12):
        raise ValueError(f"this release requires T_pre_s={T_PRE_S}")

    tasks = [(tag, seed) for tag in range(n_ic)]
    started = time.time()
    results: list[dict[str, Any]] = []
    if workers == 1:
        _initialise()
        for task in tasks:
            result = run_ic(task)
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialise) as executor:
            for result in executor.map(run_ic, tasks):
                results.append(result)
                print(json.dumps(result), flush=True)
    results.sort(key=lambda result: result["tag"])
    successes = sum(result["success"] for result in results)
    low, high = wilson(successes, n_ic)
    print(
        f"[GATE-n11-PREROLL] {successes}/{n_ic} success sigma={SIGMA} "
        f"T_pre={T_PRE_S}s pre_tol={PRE_ROLL_TOL!r} "
        f"vel_q_scale={PRE_ROLL_VEL_Q_SCALE!r} Wilson95=[{low},{high}] "
        f"({time.time() - started:.0f}s, {workers}w)",
        flush=True,
    )
    output = RUNS / f"gate_n11_preroll_seed{seed}.json"
    output.write_text(
        json.dumps(
            {
                "controller": "preroll_down_lqr+tvlqr_track+static_hold",
                "n_links": N_LINKS,
                "nominal": NOMINAL_PATH,
                "sigma": SIGMA,
                "T_pre_s": T_PRE_S,
                "pre_roll_tol": PRE_ROLL_TOL,
                "pre_roll_vel_q_scale": PRE_ROLL_VEL_Q_SCALE,
                "hold_window_s": HOLD_WINDOW_S,
                "n_success": successes,
                "n_ic": n_ic,
                "seed": seed,
                "wilson95": [low, high],
                "results": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"saved {output.relative_to(REPO)}", flush=True)
    return 0 if successes == n_ic else 1


if __name__ == "__main__":
    raise SystemExit(main())
