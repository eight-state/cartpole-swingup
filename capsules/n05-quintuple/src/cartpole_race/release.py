"""Fresh N5 replay commands built from one fixed nominal reference.

The live path rebuilds both feedback controllers and generates a new saturated
ZOH/RK4 trajectory from the hanging equilibrium. Historical perturbation ledgers
are audited separately; no command recreates them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.predicate import evaluate_success_predicate
from cartpole_race.tvlqr import TVLQR

REPO = Path(__file__).resolve().parents[2]
WORKING = REPO / ".working"
NOMINAL_PATH = REPO / "results" / "nom_n5_gluck_cont.npz"
NOMINAL_SHA256 = "6a029c6892a5dcee8851537aabdb20fd4cc21dbabeed4a6d5a843f3f0ec189c1"
N_LINKS = 5
HOLD_TIME_S = 5.0
EXPECTED_FINAL_HOLD_S = 6.381
EXPECTED_PEAK_FORCE_N = 20.175585913426993
EXPECTED_RHO = 0.029767942980498303


@dataclass(frozen=True)
class ReleaseStack:
    """One loaded nominal and controllers recomputed from it locally."""

    model: NLinkCartPole
    times: np.ndarray
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    horizon_s: float
    tvlqr: TVLQR
    static_policy: StaticLQRPolicy


@dataclass(frozen=True)
class LiveRun:
    """A fresh hanging-start trajectory and values derived from it."""

    times: np.ndarray
    states: np.ndarray
    applied_controls: np.ndarray
    raw_controls: np.ndarray
    metrics: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of raw file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_spec() -> CartPoleSpec:
    """Return the fixed five-link, 60 N release plant specification."""
    return CartPoleSpec(force_bound_n=60.0)


def _load_nominal() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and validate the sole fixed reference artifact."""
    if sha256_file(NOMINAL_PATH) != NOMINAL_SHA256:
        raise ValueError("nominal bytes do not match the released authority")
    with np.load(NOMINAL_PATH, allow_pickle=False) as archive:
        required = {"x", "u", "states", "forces", "t", "horizon", "n", "control_dt"}
        if not required.issubset(archive.files):
            raise ValueError("nominal lacks required release arrays")
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        times = np.asarray(archive["t"], dtype=float).reshape(-1)
        horizon_s = float(archive["horizon"])
        if not np.array_equal(states, archive["states"]):
            raise ValueError("nominal x and states arrays differ")
        if not np.array_equal(controls, np.asarray(archive["forces"]).reshape(-1)):
            raise ValueError("nominal u and forces arrays differ")
        if int(archive["n"]) != N_LINKS or float(archive["control_dt"]) != 0.001:
            raise ValueError("nominal metadata is not the fixed N5 release")
    if states.shape != (6001, 12) or controls.shape != (6000,) or times.shape != (6001,):
        raise ValueError("nominal shape is not the fixed 6,000-tick N5 release")
    if horizon_s != 6.0 or not np.array_equal(times, np.linspace(0.0, horizon_s, 6001)):
        raise ValueError("nominal time grid is not the fixed 1 ms release grid")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError("nominal contains non-finite values")
    return times, states, controls, horizon_s


def build_release_stack() -> ReleaseStack:
    """Load the nominal and locally rebuild TVLQR plus static LQR."""
    times, states, controls, horizon_s = _load_nominal()
    spec = release_spec()
    model = NLinkCartPole(spec)
    static_gain, static_p = static_lqr(model)
    padded_controls = np.append(controls, controls[-1])
    tvlqr = TVLQR(model, times, states, padded_controls, Qf=static_p, n_eval=400)
    return ReleaseStack(
        model=model,
        times=times,
        nominal_states=states,
        nominal_controls=controls,
        horizon_s=horizon_s,
        tvlqr=tvlqr,
        static_policy=StaticLQRPolicy(model, static_gain),
    )


def closed_loop_monodromy(stack: ReleaseStack) -> float:
    """Recompute the exact-ZOH closed-loop monodromy spectral radius."""
    model = stack.model
    nx = model.nx
    transition = np.eye(nx)
    control_dt_s = model.spec.control_dt_s
    for tick in range(len(stack.nominal_controls)):
        time_s = tick * control_dt_s
        state, force = stack.tvlqr._nom_at(time_s)
        a_matrix, b_matrix = model.linearize(state, force)
        block = np.zeros((nx + 1, nx + 1))
        block[:nx, :nx] = a_matrix * control_dt_s
        block[:nx, nx:] = b_matrix.reshape(nx, 1) * control_dt_s
        lifted = scipy.linalg.expm(block)
        phi = lifted[:nx, :nx]
        gamma = lifted[:nx, nx:]
        transition = (phi - gamma @ stack.tvlqr.K_at(time_s)) @ transition
    return float(np.max(np.abs(np.linalg.eigvals(transition))))


def run_live(stack: ReleaseStack | None = None) -> LiveRun:
    """Generate exactly one fresh hanging-start closed-loop trajectory."""
    stack = build_release_stack() if stack is None else stack
    model = stack.model
    raw_controls: list[float] = []

    def policy(state: np.ndarray, time_s: float) -> float:
        raw = (
            stack.tvlqr.policy(state, time_s)
            if time_s < stack.horizon_s
            else stack.static_policy(state, time_s)
        )
        raw_controls.append(raw)
        return raw

    times, states, applied_controls, raw_from_simulator = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        stack.horizon_s + HOLD_TIME_S,
        model.spec.control_dt_s,
        model.spec.rk4_max_step_s,
    )
    raw_controls_array = np.asarray(raw_controls, dtype=float)
    if not np.array_equal(raw_controls_array, raw_from_simulator):
        raise RuntimeError("simulator raw-force log disagrees with the live policy")

    predicate = evaluate_success_predicate(model, states, applied_controls, HOLD_TIME_S)
    rho = closed_loop_monodromy(stack)
    metrics: dict[str, Any] = {
        "nominal": {
            "path": str(NOMINAL_PATH.relative_to(REPO)),
            "sha256": NOMINAL_SHA256,
            "stored_controller": "none",
            "saved_nominal_state_trace_loaded": True,
            "saved_nominal_role": "controller reference only; never rendered",
            "saved_rollout_states_rendered": False,
        },
        "recomputed": {
            "tvlqr": "whole-trajectory CARE-terminal TVLQR",
            "static_lqr": "upright continuous CARE",
            "states": "fresh force-saturated ZOH/RK4 rollout from hanging",
            "controls": "fresh raw demands and simulator-applied clipped forces",
        },
        "closed_loop_monodromy_rho": rho,
        "predicate": predicate,
        "peak_raw_force_n": float(np.max(np.abs(raw_controls_array))),
    }
    _require_live_baseline(metrics)
    return LiveRun(times, states, applied_controls, raw_controls_array, metrics)


def _require_live_baseline(metrics: dict[str, Any]) -> None:
    predicate = metrics["predicate"]
    checks = (
        (predicate["success"], "fresh trajectory failed the sampled success predicate"),
        (
            math.isclose(
                predicate["final_hold_s"],
                EXPECTED_FINAL_HOLD_S,
                rel_tol=0.0,
                abs_tol=0.001,
            ),
            "final hold drifted",
        ),
        (
            math.isclose(
                predicate["peak_applied_force_n"],
                EXPECTED_PEAK_FORCE_N,
                rel_tol=0.0,
                abs_tol=1e-5,
            ),
            "peak force drifted",
        ),
        (
            math.isclose(
                metrics["closed_loop_monodromy_rho"],
                EXPECTED_RHO,
                rel_tol=0.0,
                abs_tol=1e-8,
            ),
            "monodromy rho drifted",
        ),
    )
    for condition, message in checks:
        if not condition:
            raise RuntimeError(message)


def audit_nominal_consistency() -> dict[str, float | int]:
    """Recompute the fixed nominal's simulator defect and scalar limits."""
    times, states, controls, horizon_s = _load_nominal()
    del times
    model = NLinkCartPole(release_spec())
    control_dt_s = model.spec.control_dt_s
    substeps = int(math.ceil(control_dt_s / model.spec.rk4_max_step_s))
    substep_s = control_dt_s / substeps
    defect = 0.0
    for state, force, next_state in zip(states[:-1], controls, states[1:], strict=True):
        stepped = state.copy()
        for _ in range(substeps):
            stepped = model.rk4_step(stepped, float(force), substep_s)
        defect = max(defect, float(np.max(np.abs(stepped - next_state))))
    peak_force_n = float(np.max(np.abs(controls)))
    peak_cart_m = float(np.max(np.abs(states[:, 0])))
    if defect >= 1e-10 or peak_force_n != EXPECTED_PEAK_FORCE_N or peak_cart_m >= 10.0:
        raise ValueError("nominal no longer satisfies the released simulator bounds")
    return {
        "horizon_s": horizon_s,
        "control_ticks": len(controls),
        "max_zoh_defect": defect,
        "peak_feedforward_force_n": peak_force_n,
        "peak_cart_m": peak_cart_m,
    }


def _working_path(path: Path) -> Path:
    output = path.resolve()
    try:
        output.relative_to(WORKING.resolve())
    except ValueError as error:
        raise ValueError("generated output must stay below .working/") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = _working_path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _render_gif(run: LiveRun, model: NLinkCartPole, horizon_s: float, path: Path) -> Path:
    """Render the supplied fresh trace; this function never runs a simulation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = _working_path(path)
    fps = 25
    step = max(1, int(round(1.0 / (fps * model.spec.control_dt_s))))
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-5.0, 5.0)
    axis.set_ylim(-3.0, 3.2)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        xs = [float(state[0])]
        ys = [0.0]
        for link in range(model.n):
            xs.append(xs[-1] + model.spec.link_lengths_m[link] * np.sin(state[1 + link]))
            ys.append(ys[-1] + model.spec.link_lengths_m[link] * np.cos(state[1 + link]))
        return xs, ys

    def update(frame: int):
        xs, ys = points(run.states[frame])
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        phase = "swing-up" if run.times[frame] < horizon_s else "balance"
        title.set_text(f"N5 cart-pole — {phase}  t={run.times[frame]:5.2f} s")
        control_index = min(frame, len(run.applied_controls) - 1)
        force_text.set_text(
            f"applied force {run.applied_controls[control_index]:+6.2f} N "
            f"(|u| <= {model.spec.force_bound_n:.0f} N)"
        )
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=range(0, len(run.states), step), blit=False)
    animation.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path


def demo_main() -> int:
    """Run and render the one authoritative fresh N5 trajectory."""
    stack = build_release_stack()
    run = run_live(stack)
    output_dir = WORKING / "n5-demo"
    metrics_path = _write_json(output_dir / "live-metrics.json", run.metrics)
    gif_path = _render_gif(run, stack.model, stack.horizon_s, output_dir / "n5-demo.gif")
    predicate = run.metrics["predicate"]
    print(
        f"[n5-demo] PASS hold={predicate['final_hold_s']:.2f}s "
        f"peak={predicate['peak_applied_force_n']:.2f}N "
        f"rho={run.metrics['closed_loop_monodromy_rho']:.7f}"
    )
    print(f"[n5-demo] GIF {gif_path.relative_to(REPO)}")
    print(f"[n5-demo] metrics {metrics_path.relative_to(REPO)}")
    return 0


def verify_main() -> int:
    """Audit frozen authorities and historical ledgers, then rerun one baseline."""
    from cartpole_race.evidence import audit_authority_bytes, audit_historical_reports

    authority = audit_authority_bytes(REPO)
    nominal = audit_nominal_consistency()
    historical = audit_historical_reports(REPO)
    live = run_live()
    report = {
        "authority_bytes": authority,
        "nominal": nominal,
        "historical_ledgers": historical,
        "fresh_baseline": live.metrics,
    }
    report_path = _write_json(WORKING / "n5-verify" / "verification.json", report)
    predicate = live.metrics["predicate"]
    print(f"[n5-verify] {len(authority)} authority bytes match")
    print("[n5-verify] four historical ledgers audited; perturbation reruns unsupported")
    print(f"[n5-verify] source provenance: {historical['source_provenance']}")
    print(
        f"[n5-verify] baseline hold={predicate['final_hold_s']:.2f}s "
        f"peak={predicate['peak_applied_force_n']:.2f}N "
        f"rho={live.metrics['closed_loop_monodromy_rho']:.7f}"
    )
    print(f"[n5-verify] report {report_path.relative_to(REPO)}")
    return 0
