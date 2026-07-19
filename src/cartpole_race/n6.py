"""The only N6 replay path: fixed nominal, rebuilt feedback, fresh plant run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec, load_spec
from cartpole_race.lqr import StaticLQRPolicy, static_lqr, wrap_state_error
from cartpole_race.tvlqr import TVLQR

REPO = Path(__file__).resolve().parents[2]
NOMINAL_PATH = REPO / "results" / "nom_n6_gluck_cont.npz"
NOMINAL_SHA256 = "b4309874c531ef38f7a8aa612bf0e45ad8494acc73e81df00e02a53860b8d047"
EVIDENCE_SHA256 = {
    "clvalidate_n6_F60_banked_seed12345.json": "93800c5ee5237483abd11779e5c94cb0433d92a607137e5f44ce938ecc8a4e26",
    "clvalidate_n6_F60_banked_seed999.json": "c974dec49e0b040c988b59ac6ecbd095d2604f4b313d0b045dbb436d92c61348",
}
EVIDENCE_PATHS = tuple(REPO / "results" / name for name in EVIDENCE_SHA256)
EVIDENCE_COMMIT = "6f5237819203bc4d9cd30037f06aff8a486e1ff5"
CONFIG_PATH = REPO / "configs" / "env-base.yaml"
WORKING = REPO / ".working"
DEMO_OUTPUT_DIR = WORKING / "n6-demo"
VERIFY_OUTPUT_DIR = WORKING / "n6-verify"

N_LINKS = 6
NOMINAL_NODES = 7000
NOMINAL_HORIZON_S = 7.0
CONTROL_DT_S = 0.001
FORCE_BOUND_N = 60.0
HOLD_S = 5.0
SETTLE_S = 1.0


@dataclass(frozen=True)
class ReplayResult:
    """Fresh replay metrics and logs; no replay data are loaded from disk."""

    success: bool
    hold_s: float
    peak_force_n: float
    peak_cart_m: float
    peak_raw_force_n: float
    nominal_sha256: str
    t_log: np.ndarray
    x_log: np.ndarray
    u_log: np.ndarray
    u_raw_log: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "hold_s": self.hold_s,
            "peak_force_n": self.peak_force_n,
            "peak_cart_m": self.peak_cart_m,
            "peak_raw_force_n": self.peak_raw_force_n,
            "nominal_sha256": self.nominal_sha256,
        }


@dataclass(frozen=True)
class HistoricalLeg:
    """Aggregate data retained in one historical gate JSON."""

    file: str
    seed: int
    successes: int
    trials: int


@dataclass(frozen=True)
class EvidenceAudit:
    """Integrity findings for historical evidence, separate from fresh proof."""

    legs: tuple[HistoricalLeg, ...]
    total_successes: int
    total_trials: int
    errors: tuple[str, ...]
    provenance: str
    row_records_available: bool

    def summary(self) -> dict[str, Any]:
        return {
            "legs": [asdict(leg) for leg in self.legs],
            "total_successes": self.total_successes,
            "total_trials": self.total_trials,
            "errors": list(self.errors),
            "provenance": self.provenance,
            "row_records_available": self.row_records_available,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _working_dir(path: Path) -> Path:
    """Accept generated output directories only below ignored ``.working``."""
    resolved = path.resolve()
    try:
        resolved.relative_to(WORKING.resolve())
    except ValueError as error:
        raise ValueError("generated output must stay below .working") from error
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _fixed_spec() -> CartPoleSpec:
    """Load the fixed N6 plant and set the replay's saturated force bound."""
    return load_spec(CONFIG_PATH).model_copy(update={"force_bound_n": FORCE_BOUND_N})


def _load_nominal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load exactly the released nominal and reject any substituted artifact."""
    if not NOMINAL_PATH.exists():
        raise FileNotFoundError(f"missing fixed nominal: {NOMINAL_PATH}")
    nominal_sha256 = _sha256(NOMINAL_PATH)
    if nominal_sha256 != NOMINAL_SHA256:
        raise ValueError(
            "fixed nominal SHA-256 mismatch: "
            f"expected {NOMINAL_SHA256}, got {nominal_sha256}"
        )

    with np.load(NOMINAL_PATH) as data:
        x_nom = np.asarray(data["x"], dtype=float)
        u_nom = np.asarray(data["u"], dtype=float).reshape(-1)
        horizon_s = float(data["horizon"])

    expected_shape = (NOMINAL_NODES + 1, 2 * (N_LINKS + 1))
    if x_nom.shape != expected_shape or u_nom.shape != (NOMINAL_NODES,):
        raise ValueError(
            f"fixed nominal shape mismatch: x={x_nom.shape}, u={u_nom.shape}"
        )
    if not math.isclose(horizon_s, NOMINAL_HORIZON_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"fixed nominal horizon mismatch: {horizon_s}")

    return x_nom, u_nom, np.linspace(0.0, horizon_s, len(x_nom))


def in_success_set(model: NLinkCartPole, state: np.ndarray) -> bool:
    """Per-sample N6 upright acceptance predicate for logged 1 kHz states."""
    error = wrap_state_error(state, model.x_equilibrium("up"), model.n)
    angles = error[1 : 1 + model.n]
    angular_rates = state[model.nq + 1 :]
    return bool(
        np.all(np.abs(angles) <= np.deg2rad(5.0))
        and np.all(np.abs(angular_rates) <= 0.5)
        and abs(state[0]) <= 2.0
        and abs(state[model.nq]) <= 0.5
    )


def continuous_hold_s(in_set: np.ndarray, control_dt_s: float) -> float:
    """Return the elapsed time spanned by the final sampled in-set suffix."""
    suffix_samples = 0
    for sample in np.asarray(in_set, dtype=bool)[::-1]:
        if not sample:
            break
        suffix_samples += 1
    return max(0, suffix_samples - 1) * control_dt_s


def _build_feedback(
    model: NLinkCartPole, t_nom: np.ndarray, x_nom: np.ndarray, u_nom: np.ndarray
) -> tuple[TVLQR, StaticLQRPolicy]:
    """Recompute both feedback stages from the fixed plant and nominal."""
    static_gain, static_cost = static_lqr(model)
    padded_u = np.append(u_nom, u_nom[-1])
    tvlqr = TVLQR(model, t_nom, x_nom, padded_u, Qf=static_cost, n_eval=400)
    return tvlqr, StaticLQRPolicy(model, static_gain)


def run_demo(render: bool = True, output_dir: Path | None = None) -> ReplayResult:
    """Run one fresh saturated ZOH/RK4 N6 replay from the hanging state."""
    x_nom, u_nom, t_nom = _load_nominal()
    model = NLinkCartPole(_fixed_spec())
    tvlqr, static_policy = _build_feedback(model, t_nom, x_nom, u_nom)
    horizon_s = float(t_nom[-1])

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < horizon_s:
            return tvlqr.policy(state, time_s)
        return static_policy(state, time_s)

    total_s = horizon_s + SETTLE_S + HOLD_S
    t_log, x_log, u_log, u_raw_log = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        total_s,
        model.spec.control_dt_s,
        model.spec.rk4_max_step_s,
    )
    hold_s = continuous_hold_s(
        np.array([in_success_set(model, state) for state in x_log]),
        model.spec.control_dt_s,
    )
    peak_force_n = float(np.max(np.abs(u_log)))
    peak_cart_m = float(np.max(np.abs(x_log[:, 0])))
    peak_raw_force_n = float(np.max(np.abs(u_raw_log)))
    rail_ok = bool(np.all(np.abs(x_log[:, 0]) <= model.spec.track_half_length_m))
    force_ok = bool(np.all(np.abs(u_log) <= FORCE_BOUND_N + 1e-12))
    finite_ok = bool(np.all(np.isfinite(x_log)) and np.all(np.isfinite(u_log)))
    success = bool(
        finite_ok and rail_ok and force_ok and hold_s >= HOLD_S - 1e-12
    )
    result = ReplayResult(
        success=success,
        hold_s=hold_s,
        peak_force_n=peak_force_n,
        peak_cart_m=peak_cart_m,
        peak_raw_force_n=peak_raw_force_n,
        nominal_sha256=NOMINAL_SHA256,
        t_log=t_log,
        x_log=x_log,
        u_log=u_log,
        u_raw_log=u_raw_log,
    )
    if not result.success:
        raise RuntimeError(
            "fresh N6 replay failed: "
            f"hold={hold_s:.12g}s rail_ok={rail_ok} force_ok={force_ok} "
            f"finite_ok={finite_ok}"
        )
    if render:
        render_demo(result, output_dir or DEMO_OUTPUT_DIR)
    return result


def render_demo(result: ReplayResult, output_dir: Path) -> Path:
    """Render only the freshly integrated states to an ignored GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_dir = _working_dir(output_dir)
    path = output_dir / "n6-demo.gif"
    spec = _fixed_spec()
    fps = 25
    frame_step = int(round(1.0 / (fps * CONTROL_DT_S)))
    frames = range(0, len(result.x_log), frame_step)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.5, 6.5)
    axis.set_ylim(-3.5, 3.7)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        x_coordinates = [float(state[0])]
        y_coordinates = [0.0]
        for index in range(N_LINKS):
            x_coordinates.append(
                x_coordinates[-1] + spec.link_lengths_m[index] * np.sin(state[1 + index])
            )
            y_coordinates.append(
                y_coordinates[-1] + spec.link_lengths_m[index] * np.cos(state[1 + index])
            )
        return x_coordinates, y_coordinates

    def update(frame_index: int):
        state = result.x_log[frame_index]
        x_coordinates, y_coordinates = points(state)
        cart.set_data([x_coordinates[0]], [0.0])
        chain.set_data(x_coordinates, y_coordinates)
        elapsed_s = result.t_log[frame_index]
        phase = "swing-up" if elapsed_s < NOMINAL_HORIZON_S else "balance"
        title.set_text(f"N6 cart-pole: {phase}, t={elapsed_s:5.2f} s")
        control_index = min(frame_index, len(result.u_log) - 1)
        force_text.set_text(
            f"applied force {result.u_log[control_index]:+6.1f} N (|u| <= 60)"
        )
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path


def _source_provenance() -> str:
    """Report object availability only; this does not verify historical source."""
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{EVIDENCE_COMMIT}^{{commit}}"],
        cwd=REPO,
        check=False,
        capture_output=True,
    )
    if check.returncode:
        return (
            f"unavailable: embedded commit {EVIDENCE_COMMIT} is absent locally; "
            "historical source is not verified"
        )
    return (
        f"present: embedded commit {EVIDENCE_COMMIT} exists locally but was not "
        "inspected; historical source is not verified"
    )


def audit_historical_evidence() -> EvidenceAudit:
    """Audit retained aggregate gates without rerunning perturbation studies."""
    errors: list[str] = []
    legs: list[HistoricalLeg] = []
    row_records_available = True
    expected_seeds = {12345, 999}
    for path in EVIDENCE_PATHS:
        if not path.exists():
            errors.append(f"missing historical gate: {path.relative_to(REPO)}")
            row_records_available = False
            continue
        try:
            if _sha256(path) != EVIDENCE_SHA256[path.name]:
                errors.append(f"{path.name}: canonical evidence bytes changed")
                row_records_available = False
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{path.name}: unreadable or malformed historical gate")
            row_records_available = False
            continue
        if not isinstance(report, dict):
            errors.append(f"{path.name}: malformed historical gate")
            row_records_available = False
            continue
        rows = report.get("rows")
        row_records_available = (
            row_records_available and isinstance(rows, list) and bool(rows)
        )
        try:
            seed = int(report["seed"])
            successes = int(report["n_success"])
            trials = int(report["n_trials"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path.name}: invalid seed or count fields")
            row_records_available = False
            continue
        legs.append(HistoricalLeg(path.name, seed, successes, trials))
        if report.get("commit_sha") != EVIDENCE_COMMIT:
            errors.append(f"{path.name}: embedded source commit mismatch")
        if report.get("nominal") != "results/nom_n6_gluck_cont.npz":
            errors.append(f"{path.name}: fixed nominal path mismatch")
        if report.get("nominal_sha256") != NOMINAL_SHA256:
            errors.append(f"{path.name}: fixed nominal SHA-256 mismatch")
        if report.get("n_links") != N_LINKS or report.get("force_limit") != FORCE_BOUND_N:
            errors.append(f"{path.name}: N6 force/model metadata mismatch")
        if seed not in expected_seeds or trials != 24 or successes != 24:
            errors.append(f"{path.name}: expected historical 24/24 gate")
        if trials <= 0 or not 0 <= successes <= trials:
            errors.append(f"{path.name}: invalid historical success count")
            continue
        fraction = report.get("frac")
        if not isinstance(fraction, (int, float)) or not math.isclose(
            float(fraction), successes / trials, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"{path.name}: fraction does not equal count ratio")

    legs.sort(key=lambda leg: leg.seed)
    if {leg.seed for leg in legs} != expected_seeds:
        errors.append("historical gate seeds are incomplete or duplicated")
    total_successes = sum(leg.successes for leg in legs)
    total_trials = sum(leg.trials for leg in legs)
    if total_successes != 48 or total_trials != 48:
        errors.append("historical two-seed total is not 48/48")
    return EvidenceAudit(
        legs=tuple(legs),
        total_successes=total_successes,
        total_trials=total_trials,
        errors=tuple(errors),
        provenance=_source_provenance(),
        row_records_available=row_records_available,
    )


def demo_main(argv: list[str] | None = None) -> int:
    """Console entry point for the one rendered fresh replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEMO_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_demo(render=True, output_dir=args.output_dir)
    output_dir = _working_dir(args.output_dir)
    metrics_path = output_dir / "live-metrics.json"
    metrics_path.write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[n6-demo] hold={result.hold_s:.12g} s")
    print(f"[n6-demo] peak force={result.peak_force_n:.14g} N")
    print(f"[n6-demo] peak cart={result.peak_cart_m:.15g} m")
    print(f"[n6-demo] render={(output_dir / 'n6-demo.gif').resolve()}")
    print(f"[n6-demo] metrics={metrics_path.resolve()}")
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    """Console entry point for the historical audit plus an unrendered fresh proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=VERIFY_OUTPUT_DIR)
    args = parser.parse_args(argv)

    audit = audit_historical_evidence()
    print(
        "[n6-verify] historical aggregate gates: "
        + ", ".join(f"seed {leg.seed} {leg.successes}/{leg.trials}" for leg in audit.legs)
        + f"; total {audit.total_successes}/{audit.total_trials}"
    )
    if audit.row_records_available:
        print("[n6-verify] historical rows are present")
    else:
        print(
            "[n6-verify] historical rows are absent; retained counts are aggregate "
            "evidence, not independently row-derived"
        )
    print(f"[n6-verify] source provenance: {audit.provenance}")
    for error in audit.errors:
        print(f"[n6-verify] evidence error: {error}")

    fresh: ReplayResult | None = None
    fresh_error: str | None = None
    try:
        fresh = run_demo(render=False)
    except Exception as exc:  # pragma: no cover - command diagnostic path
        fresh_error = f"{type(exc).__name__}: {exc}"
        print(f"[n6-verify] fresh proof: FAIL ({fresh_error})")
    else:
        print(
            "[n6-verify] fresh proof: PASS "
            f"hold={fresh.hold_s:.12g}s peak-force={fresh.peak_force_n:.14g}N "
            f"peak-cart={fresh.peak_cart_m:.15g}m"
        )

    output_dir = _working_dir(args.output_dir)
    summary = {
        "historical_evidence": audit.summary(),
        "fresh_proof": fresh.summary() if fresh else None,
        "fresh_error": fresh_error,
    }
    output = output_dir / "n6-verify.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[n6-verify] summary={output.resolve()}")
    return 0 if fresh is not None and not audit.errors else 1
