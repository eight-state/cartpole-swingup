"""The supported N11 replay and immutable-evidence audit.

The release has one executable path: load the YAML plant and frozen dense
nominal, rebuild exact-ZOH TVLQR plus upright LQR, run the saturated simulator,
apply the elapsed-time predicate, then render or report that fresh record.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec, load_spec
from cartpole_race.lqr import static_lqr, wrap_state_error
from cartpole_race.predicate import longest_continuous_hold_s

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "r2"
WORKING = REPO / ".working"
N11_SPEC_PATH = REPO / "configs" / "env-n11.yaml"
N_LINKS = 11
HOLD_REQUIRED_S = 5.0
HOLD_WINDOW_S = 12.0


@dataclass(frozen=True)
class NominalArtifact:
    """Identity and expected grid of one frozen nominal input."""

    file: str
    sha256: str
    n_ticks: int
    horizon_s: float

    @property
    def path(self) -> Path:
        return RUNS / self.file

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(REPO).as_posix()


NOMINAL_PARENT = NominalArtifact(
    file="nom_n11_4ms_capture025_smoke3t03.npz",
    sha256="b190e1ff71fe5242c850e5eb817bf8401fc38f24f9e189e6b132e85471dcea86",
    n_ticks=2_500,
    horizon_s=10.0,
)
NOMINAL_DENSE = NominalArtifact(
    file="nom_n11_dense1ms_capture025_smoke3t03.npz",
    sha256="1b7458cefe5d91aeaa012e78c4edbf586cd0d989df8e8e6f7adb2000cbae290d",
    n_ticks=10_000,
    horizon_s=10.0,
)
GATE_ARTIFACTS = (
    (
        12345,
        "gate_n11_preroll_seed12345.json",
        "1e73bb12234ade44c455f75eb001f41d02402ca5eec1041275642cdac617ea6f",
    ),
    (
        777,
        "gate_n11_preroll_seed777.json",
        "8164df430f2eda7a88995e928847e5e2da7f1c4973392f10741833853ec10edc",
    ),
    (
        2024,
        "gate_n11_preroll_seed2024.json",
        "bb62a57ee05cff69b3285f3faedfb189b484f293d6bea89730d65108152a5d25",
    ),
)
SOURCE_ARTIFACTS = {
    "configs/env-n11.yaml": "0cdbc0ebe17c814191e4b7c4ae2fa83a874668cc40bddcd339eb1b8e6dbeff65",
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/cartpole_race/lqr.py": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
    "src/cartpole_race/discrete_tvlqr.py": "afc41f0b323f3d337fc378eb8383a9ada7aa9d1ff0f1566fc0c1573c0a4bb3d2",
}


@dataclass(frozen=True)
class ReleaseStack:
    """Frozen nominal and feedback gains recomputed from the YAML plant."""

    model: NLinkCartPole
    states: np.ndarray
    controls: np.ndarray
    horizon_s: float
    tracker: DiscreteTVLQR
    static_gain: np.ndarray


@dataclass(frozen=True)
class LiveRun:
    """One fresh saturated simulation and its release metrics."""

    times: np.ndarray
    states: np.ndarray
    applied_controls: np.ndarray
    raw_controls: np.ndarray
    metrics: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_close(actual: float, expected: float, message: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading arbitrary code."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source_identity() -> list[dict[str, str]]:
    """Rehash the frozen plant, controller, and YAML authority sources."""
    records = []
    for relative_path, expected_sha256 in SOURCE_ARTIFACTS.items():
        path = REPO / relative_path
        _require(path.is_file(), f"missing frozen source: {relative_path}")
        actual_sha256 = file_sha256(path)
        _require(actual_sha256 == expected_sha256, f"unexpected digest for {relative_path}")
        records.append({"file": relative_path, "sha256": actual_sha256})
    return records


def load_release_spec() -> CartPoleSpec:
    """Load and validate the one YAML authority for the N11 plant and timing."""
    spec = load_spec(N11_SPEC_PATH)
    _require(spec.n_links == N_LINKS, f"YAML links={spec.n_links}, expected {N_LINKS}")
    _require_close(spec.control_dt_s, 0.001, "YAML control period is not 1 ms")
    _require_close(spec.rk4_max_step_s, 0.00025, "YAML RK4 step is not 0.25 ms")
    return spec


def _require_asset_identity(asset: NominalArtifact) -> None:
    _require(asset.path.is_file(), f"missing frozen nominal: {asset.relative_path}")
    actual = file_sha256(asset.path)
    _require(actual == asset.sha256, f"unexpected digest for {asset.relative_path}")


def _load_nominal(
    asset: NominalArtifact, spec: CartPoleSpec
) -> tuple[np.ndarray, np.ndarray, float]:
    _require_asset_identity(asset)
    with np.load(asset.path, allow_pickle=False) as archive:
        required = {"x", "u", "horizon", "n", "force"}
        _require(required <= set(archive.files), f"incomplete nominal: {asset.file}")
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(archive["horizon"]).item())
        n_links = int(np.asarray(archive["n"]).item())
        force_bound_n = float(np.asarray(archive["force"]).item())
        stored_ticks = (
            int(np.asarray(archive["n_nodes"]).item())
            if "n_nodes" in archive.files
            else len(controls)
        )
    _require(states.shape == (asset.n_ticks + 1, spec.nx), f"state shape: {asset.file}")
    _require(controls.shape == (asset.n_ticks,), f"control shape: {asset.file}")
    _require(n_links == spec.n_links, f"link count: {asset.file}")
    _require(stored_ticks == asset.n_ticks, f"tick count: {asset.file}")
    _require_close(horizon_s, asset.horizon_s, f"horizon: {asset.file}")
    _require_close(force_bound_n, spec.force_bound_n, f"force bound: {asset.file}")
    _require(bool(np.all(np.isfinite(states))), f"nonfinite state: {asset.file}")
    _require(bool(np.all(np.isfinite(controls))), f"nonfinite control: {asset.file}")
    return states, controls, horizon_s


def build_release_stack() -> ReleaseStack:
    """Load the dense nominal and rebuild both feedback controllers locally."""
    audit_source_identity()
    spec = load_release_spec()
    model = NLinkCartPole(spec)
    states, controls, horizon_s = _load_nominal(NOMINAL_DENSE, spec)
    tracker = DiscreteTVLQR(model, states, controls, spec.control_dt_s)
    static_gain, _ = static_lqr(model)
    return ReleaseStack(model, states, controls, horizon_s, tracker, static_gain)


def _parent_rk4_defect(
    model: NLinkCartPole, states: np.ndarray, controls: np.ndarray, horizon_s: float
) -> float:
    """Recompute the stored 4 ms parent transcription residuals."""
    step_s = horizon_s / len(controls)
    worst = 0.0
    for state, control, next_state in zip(states[:-1], controls, states[1:], strict=True):
        stepped = model.rk4_step(state, float(control), step_s)
        worst = max(worst, float(np.max(np.abs(stepped - next_state))))
    return worst


def _dense_simulator_defects(stack: ReleaseStack) -> tuple[float, float]:
    """Return max dense-tick and dense-parent-boundary simulation residuals."""
    spec = stack.model.spec
    n_substeps = max(1, int(np.ceil(spec.control_dt_s / spec.rk4_max_step_s)))
    substep_s = spec.control_dt_s / n_substeps
    tick_worst = boundary_worst = 0.0
    for tick, (state, control, next_state) in enumerate(
        zip(stack.states[:-1], stack.controls, stack.states[1:], strict=True)
    ):
        stepped = state.copy()
        for _ in range(n_substeps):
            stepped = stack.model.rk4_step(stepped, float(control), substep_s)
        defect = float(np.max(np.abs(stepped - next_state)))
        tick_worst = max(tick_worst, defect)
        if tick % 4 == 0:
            boundary_worst = max(boundary_worst, defect)
    return tick_worst, boundary_worst


def audit_nominal_artifacts(stack: ReleaseStack | None = None) -> dict[str, Any]:
    """Audit both frozen nominal inputs without synthesizing a trajectory."""
    stack = build_release_stack() if stack is None else stack
    parent_states, parent_controls, parent_horizon_s = _load_nominal(
        NOMINAL_PARENT, stack.model.spec
    )
    parent_defect = _parent_rk4_defect(
        stack.model, parent_states, parent_controls, parent_horizon_s
    )
    dense_defect, dense_boundary_defect = _dense_simulator_defects(stack)
    upright = stack.model.x_equilibrium("up")
    terminal_error = wrap_state_error(stack.states[-1], upright, stack.model.n)
    terminal_angle_error_deg = float(
        np.rad2deg(np.max(np.abs(terminal_error[1 : 1 + stack.model.n])))
    )
    _require(parent_defect < 2e-5, "parent nominal no longer meets its release bound")
    _require(dense_defect < 1e-6, "dense nominal no longer matches the simulator")
    return {
        "parent": {
            "file": NOMINAL_PARENT.relative_path,
            "sha256": NOMINAL_PARENT.sha256,
            "n_ticks": NOMINAL_PARENT.n_ticks,
            "horizon_s": parent_horizon_s,
        },
        "dense": {
            "file": NOMINAL_DENSE.relative_path,
            "sha256": NOMINAL_DENSE.sha256,
            "n_ticks": NOMINAL_DENSE.n_ticks,
            "horizon_s": stack.horizon_s,
        },
        "n_ticks": len(stack.controls),
        "peak_feedforward_n": float(np.max(np.abs(stack.controls))),
        "terminal_max_angle_error_deg": terminal_angle_error_deg,
        "parent_rk4_4ms_defect": parent_defect,
        "dense_simulator_defect": dense_defect,
        "dense_4ms_boundary_defect": dense_boundary_defect,
    }


def wilson95(successes: int, trials: int, z: float = 1.96) -> list[float]:
    """Recompute the release gate's two-sided Wilson 95 percent interval."""
    _require(trials > 0, "Wilson interval needs at least one trial")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
    ) / denominator
    return [round(center - half_width, 4), round(min(1.0, center + half_width), 4)]


def _audit_gate_artifact(
    seed: int, filename: str, expected_sha256: str, spec: CartPoleSpec
) -> dict[str, Any]:
    path = RUNS / filename
    _require(path.is_file(), f"missing banked gate record: {filename}")
    actual_sha256 = file_sha256(path)
    _require(actual_sha256 == expected_sha256, f"unexpected digest for {filename}")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "controller",
        "n_links",
        "nominal",
        "sigma",
        "T_pre_s",
        "pre_roll_tol",
        "pre_roll_vel_q_scale",
        "hold_window_s",
        "n_success",
        "n_ic",
        "seed",
        "wilson95",
        "results",
    }
    _require(set(record) == expected_keys, f"unexpected gate fields: {filename}")
    _require(
        record["controller"] == "preroll_down_lqr+tvlqr_track+static_hold",
        f"controller provenance: {filename}",
    )
    _require(record["n_links"] == spec.n_links, f"link count: {filename}")
    _require(record["nominal"] == NOMINAL_DENSE.relative_path, f"nominal: {filename}")
    _require_close(float(record["sigma"]), 0.02, f"sigma: {filename}")
    _require_close(float(record["T_pre_s"]), 9.0, f"pre-roll duration: {filename}")
    _require_close(float(record["pre_roll_tol"]), 0.0, f"pre-roll tolerance: {filename}")
    _require_close(
        float(record["pre_roll_vel_q_scale"]), 4.0, f"pre-roll velocity scale: {filename}"
    )
    _require_close(float(record["hold_window_s"]), 10.0, f"hold window: {filename}")
    _require(record["seed"] == seed, f"seed: {filename}")
    _require(record["n_ic"] == 24, f"trial count: {filename}")

    rows = record["results"]
    row_keys = {
        "tag",
        "success",
        "handoff_deg",
        "hold_s",
        "peakF",
        "pert_deg",
        "resid",
        "t_pre",
        "track_ok",
        "fail",
    }
    _require(isinstance(rows, list) and len(rows) == record["n_ic"], f"rows: {filename}")
    _require([row.get("tag") for row in rows] == list(range(24)), f"tags: {filename}")
    for row in rows:
        _require(set(row) == row_keys, f"row fields: {filename}")
        _require(row["success"] is True, f"failed historical row: {filename}")
        _require(row["track_ok"] is True, f"track row: {filename}")
        _require(row["fail"] is None, f"failure label: {filename}")
        for key in ("handoff_deg", "hold_s", "peakF", "pert_deg", "resid", "t_pre"):
            _require(math.isfinite(float(row[key])), f"nonfinite {key}: {filename}")
        _require(float(row["handoff_deg"]) <= 20.0, f"handoff limit: {filename}")
        _require(float(row["hold_s"]) >= HOLD_REQUIRED_S, f"hold predicate: {filename}")
        _require(float(row["peakF"]) <= spec.force_bound_n, f"force bound: {filename}")
        _require_close(float(row["t_pre"]), 9.0, f"pre-roll row: {filename}")

    successes = sum(row["success"] for row in rows)
    trials = len(rows)
    interval = wilson95(successes, trials)
    _require(record["n_success"] == successes == 24, f"success count: {filename}")
    _require(record["wilson95"] == interval, f"Wilson interval: {filename}")
    return {
        "file": f"runs/r2/{filename}",
        "sha256": actual_sha256,
        "seed": seed,
        "successes": successes,
        "trials": trials,
        "wilson95": interval,
    }


def audit_banked_gate_evidence() -> dict[str, Any]:
    """Audit immutable historical gate records; never rerun perturbations."""
    spec = load_release_spec()
    files = [
        _audit_gate_artifact(seed, filename, expected_sha256, spec)
        for seed, filename, expected_sha256 in GATE_ARTIFACTS
    ]
    successes = sum(record["successes"] for record in files)
    trials = sum(record["trials"] for record in files)
    _require(successes == 72 and trials == 72, "banked gate aggregate is not 72/72")
    return {
        "status": "banked evidence audited; no perturbed cases were rerun",
        "files": files,
        "total_successes": successes,
        "total_trials": trials,
    }


def run_live(stack: ReleaseStack | None = None) -> LiveRun:
    """Run one fresh replay through the saturated simulator boundary."""
    stack = build_release_stack() if stack is None else stack
    model = stack.model
    spec = model.spec
    upright = model.x_equilibrium("up")
    raw_controls: list[float] = []

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < stack.horizon_s:
            raw = stack.tracker.policy(state, time_s)
        else:
            error = wrap_state_error(state, upright, model.n)
            raw = -float(np.asarray(stack.static_gain).reshape(-1) @ error)
        raw_controls.append(raw)
        return raw

    times, states, applied_controls = model.rollout_zoh(
        stack.states[0],
        policy,
        stack.horizon_s + HOLD_WINDOW_S,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    raw = np.asarray(raw_controls, dtype=float)
    handoff_tick = int(round(stack.horizon_s / spec.control_dt_s))
    handoff = states[handoff_tick]
    handoff_error = wrap_state_error(handoff, upright, model.n)
    handoff_angle_error_deg = float(
        np.rad2deg(np.max(np.abs(handoff_error[1 : 1 + model.n])))
    )
    hold_s = longest_continuous_hold_s(
        model, states[handoff_tick:], spec.control_dt_s
    )
    max_cart_abs_m = float(np.max(np.abs(states[:, 0])))
    static_raw = raw[handoff_tick:]
    clip_ticks = int(np.count_nonzero(np.abs(raw) > spec.force_bound_n + 1e-9))
    passed = bool(
        hold_s >= HOLD_REQUIRED_S - 1e-9
        and max_cart_abs_m <= spec.track_half_length_m
    )
    metrics = {
        "loaded_artifacts": {
            "plant_yaml": N11_SPEC_PATH.relative_to(REPO).as_posix(),
            "dense_nominal": NOMINAL_DENSE.relative_path,
            "dense_nominal_sha256": NOMINAL_DENSE.sha256,
            "parent_nominal_audited_separately": NOMINAL_PARENT.relative_path,
            "stored_controller": "none",
            "stored_rollout": "none",
        },
        "recomputed": {
            "discrete_tvlqr": "exact-ZOH linearizations and backward Riccati gains",
            "static_lqr": "upright continuous Riccati gain",
            "states": "fresh rollout_zoh states",
            "controls": "fresh raw demands and simulator-applied clipped controls",
        },
        "controller": {
            "monodromy_rho": stack.tracker.monodromy(),
            "force_bound_n": spec.force_bound_n,
        },
        "live_closed_loop": {
            "handoff_max_angle_error_deg": handoff_angle_error_deg,
            "longest_continuous_hold_s": hold_s,
            "track_peak_abs_m": max_cart_abs_m,
            "applied_peak_force_n": float(np.max(np.abs(applied_controls))),
            "raw_peak_force_n": float(np.max(np.abs(raw))),
            "static_raw_peak_force_n": float(np.max(np.abs(static_raw))),
            "clip_ticks": clip_ticks,
            "passed": passed,
        },
    }
    return LiveRun(times, states, applied_controls, raw, metrics)


def render_live_gif(run: LiveRun, model: NLinkCartPole, horizon_s: float) -> Path:
    """Render an existing fresh record below ignored ``.working/`` only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = WORKING / "n11" / "demo.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    frame_step = int(round(1.0 / (fps * model.spec.control_dt_s)))
    frames = range(0, len(run.states), frame_step)
    link_length = model.spec.link_lengths_m[0]

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.5, 6.5)
    axis.set_ylim(-6.4, 6.6)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        x_coordinates = [float(state[0])]
        y_coordinates = [0.0]
        for index in range(model.n):
            x_coordinates.append(
                x_coordinates[-1] + link_length * np.sin(state[1 + index])
            )
            y_coordinates.append(
                y_coordinates[-1] + link_length * np.cos(state[1 + index])
            )
        return x_coordinates, y_coordinates

    def update(frame_index: int):
        state = run.states[frame_index]
        x_coordinates, y_coordinates = points(state)
        cart.set_data([x_coordinates[0]], [0.0])
        chain.set_data(x_coordinates, y_coordinates)
        elapsed_s = run.times[frame_index]
        phase = "swing-up" if elapsed_s < horizon_s else "balance"
        title.set_text(f"n=11 cart-pole: {phase}, t={elapsed_s:5.2f} s")
        control_index = min(frame_index, len(run.applied_controls) - 1)
        force_text.set_text(
            f"applied force {run.applied_controls[control_index]:+6.1f} N (|u| <= 150)"
        )
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write generated output only under the ignored workspace directory."""
    working_root = WORKING.resolve()
    resolved = path.resolve()
    _require(resolved.is_relative_to(working_root), "generated output must stay in .working")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def _report_live(live: LiveRun) -> None:
    closed_loop = live.metrics["live_closed_loop"]
    controller = live.metrics["controller"]
    print(
        f"[live] handoff {closed_loop['handoff_max_angle_error_deg']:.7f} deg; "
        f"hold {closed_loop['longest_continuous_hold_s']:.1f} s; "
        f"applied/raw peak {closed_loop['applied_peak_force_n']:.7f}/"
        f"{closed_loop['raw_peak_force_n']:.7f} N; "
        f"max cart {closed_loop['track_peak_abs_m']:.7f} m; "
        f"clips {closed_loop['clip_ticks']}; rho {controller['monodromy_rho']:.7g} -> "
        f"{'PASS' if closed_loop['passed'] else 'FAIL'}",
        flush=True,
    )


def demo_main(argv: Sequence[str] | None = None) -> int:
    """Run the first supported live command and render its fresh record."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    stack = build_release_stack()
    live = run_live(stack)
    metrics_path = _write_json(WORKING / "n11" / "live-metrics.json", live.metrics)
    gif_path = render_live_gif(live, stack.model, stack.horizon_s)
    _report_live(live)
    print(f"[render] {gif_path.relative_to(REPO).as_posix()}", flush=True)
    print(f"[metrics] {metrics_path.relative_to(REPO).as_posix()}", flush=True)
    return 0 if live.metrics["live_closed_loop"]["passed"] else 1


def verify_main(argv: Sequence[str] | None = None) -> int:
    """Audit frozen evidence and run the same fresh live stack without rendering."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    stack = build_release_stack()
    nominal = audit_nominal_artifacts(stack)
    gate = audit_banked_gate_evidence()
    live = run_live(stack)
    report = {
        "frozen_source_identity": audit_source_identity(),
        "nominal_artifacts": nominal,
        "banked_gate_evidence": gate,
        "fresh_live": live.metrics,
    }
    report_path = _write_json(WORKING / "n11-verify" / "verification.json", report)
    print(
        f"[nominal] {nominal['parent']['sha256']} {nominal['parent']['file']}; "
        f"{nominal['dense']['sha256']} {nominal['dense']['file']}",
        flush=True,
    )
    for record in gate["files"]:
        print(
            f"[gate] seed {record['seed']}: {record['successes']}/{record['trials']}; "
            f"sha256 {record['sha256']}",
            flush=True,
        )
    print(
        f"[gate] aggregate {gate['total_successes']}/{gate['total_trials']}; "
        "historical records only, no perturbations rerun",
        flush=True,
    )
    print(
        f"[nominal] peak feedforward {nominal['peak_feedforward_n']:.7f} N; "
        f"terminal {nominal['terminal_max_angle_error_deg']:.7f} deg; "
        f"parent/dense defects {nominal['parent_rk4_4ms_defect']:.3e}/"
        f"{nominal['dense_simulator_defect']:.3e}",
        flush=True,
    )
    _report_live(live)
    print(f"[report] {report_path.relative_to(REPO).as_posix()}", flush=True)
    return 0 if live.metrics["live_closed_loop"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(demo_main())
