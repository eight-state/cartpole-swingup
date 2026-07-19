"""Shared live replay and evidence audit for the public n=10 release.

The dense nominal is a banked input.  This module never solves the historical
NLP: it verifies the input identity, rebuilds both feedback controllers, and
runs the saturated simulator afresh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import load_spec
from cartpole_race.lqr import static_lqr, wrap_state_error
from cartpole_race.predicate import longest_continuous_hold_s

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO = PROJECT_ROOT
RESULTS = REPO / "results"
N10_SPEC_PATH = REPO / "configs" / "env-base.yaml"
N_LINKS = 10
HOLD_S = 5.0
HOLD_WINDOW_S = 10.0


@dataclass(frozen=True)
class NominalArtifact:
    """Identity of a fixed NPZ input; this release does not regenerate it."""

    file: str
    sha256: str
    grid_dt_s: float
    n_nodes: int
    horizon_s: float
    is_densified: bool
    label: str

    @property
    def path(self) -> Path:
        return RESULTS / self.file


NOMINAL = NominalArtifact(
    file="nom_n10_dense1ms_wv1en3t.npz",
    sha256="df40be618e0f39a0c38243a23d239978518e33976a7814bd7b6a68f50dfe59e6",
    grid_dt_s=0.001,
    n_nodes=10_000,
    horizon_s=10.0,
    is_densified=True,
    label="banked 1 ms dense trajectory from the tight 4 ms collocation solve",
)
NOMINAL_4MS = NominalArtifact(
    file="nom_n10_4ms_wv1en3t.npz",
    sha256="3889f22a8d66935181e648477c24f3d6689f5359d9f0f0cbc07e899a539e77ae",
    grid_dt_s=0.004,
    n_nodes=2_500,
    horizon_s=10.0,
    is_densified=False,
    label="banked 4 ms tight collocation parent trajectory",
)
BANKED_GATE_SHA256 = {
    "gate_n10_preroll_seed12345.json": "ccb791e3201124547b0bb55e8ca5f75afabbc1fc8dcfde775a93bba1497d7922",
    "gate_n10_preroll_seed777.json": "a33c4015e7905d566622c54244ad42fbbfe9370bd05e5c1d35630c79bb615f2e",
    "gate_n10_preroll_seed2024.json": "5a8608d47eba81a19de909f97a81c16e947f394882322f1ac81e0ad35e84a917",
}


@dataclass(frozen=True)
class ReleaseStack:
    """Loaded nominal plus controllers recomputed from the release source."""

    model: NLinkCartPole
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    horizon_s: float
    tracker: DiscreteTVLQR
    static_gain: np.ndarray


@dataclass(frozen=True)
class LiveRun:
    """A fresh simulator record and its scalar release metrics."""

    t_log: np.ndarray
    x_log: np.ndarray
    applied_controls: np.ndarray
    raw_controls: np.ndarray
    metrics: dict[str, Any]


def file_sha256(path: Path) -> str:
    """Return a file digest without loading arbitrary code from an artifact."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_asset_identity(asset) -> None:
    if not asset.path.is_file():
        raise FileNotFoundError(f"missing banked artifact: {asset.path}")
    actual = file_sha256(asset.path)
    if actual != asset.sha256:
        raise ValueError(f"unexpected digest for {asset.file}: {actual}")


def build_release_stack() -> ReleaseStack:
    """Load the fixed nominal and recompute its live feedback controllers."""
    _require_asset_identity(NOMINAL)
    spec = load_spec(N10_SPEC_PATH)
    if spec.n_links != N_LINKS:
        raise ValueError(f"release spec has {spec.n_links} links, expected {N_LINKS}")
    model = NLinkCartPole(spec)
    with np.load(NOMINAL.path, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(archive["horizon"])
        artifact_links = int(archive["n"])
        force_bound_n = float(archive["force"])
    if artifact_links != N_LINKS or force_bound_n != spec.force_bound_n:
        raise ValueError("dense nominal metadata does not match the frozen release spec")
    if states.shape != (NOMINAL.n_nodes + 1, model.nx) or controls.shape != (NOMINAL.n_nodes,):
        raise ValueError("dense nominal shape does not match the release identity")
    if abs(horizon_s - NOMINAL.horizon_s) > 1e-12:
        raise ValueError("dense nominal horizon does not match the release identity")

    tracker = DiscreteTVLQR(model, states, controls, spec.control_dt_s)
    static_gain, _ = static_lqr(model)
    return ReleaseStack(model, states, controls, horizon_s, tracker, static_gain)


def _parent_defect(model: NLinkCartPole) -> float:
    """Recompute every RK4-4ms parent transcription residual."""
    _require_asset_identity(NOMINAL_4MS)
    with np.load(NOMINAL_4MS.path, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(archive["horizon"])
        artifact_links = int(archive["n"])
        n_nodes = int(archive["n_nodes"])
    if artifact_links != N_LINKS or n_nodes != NOMINAL_4MS.n_nodes:
        raise ValueError("parent nominal metadata does not match the release identity")
    if states.shape != (n_nodes + 1, model.nx) or controls.shape != (n_nodes,):
        raise ValueError("parent nominal shape does not match the release identity")

    step_s = horizon_s / len(controls)
    worst = 0.0
    for state, control, next_state in zip(states[:-1], controls, states[1:], strict=True):
        k1 = np.asarray(model.f(state, float(control))).reshape(-1)
        k2 = np.asarray(model.f(state + 0.5 * step_s * k1, float(control))).reshape(-1)
        k3 = np.asarray(model.f(state + 0.5 * step_s * k2, float(control))).reshape(-1)
        k4 = np.asarray(model.f(state + step_s * k3, float(control))).reshape(-1)
        stepped = state + (step_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        worst = max(worst, float(np.max(np.abs(stepped - next_state))))
    return worst


def _dense_step_defects(stack: ReleaseStack) -> tuple[float, float]:
    """Return dense intra-segment and 4ms-boundary residuals independently."""
    spec = stack.model.spec
    n_substeps = max(1, int(np.ceil(spec.control_dt_s / spec.rk4_max_step_s)))
    substep_s = spec.control_dt_s / n_substeps
    intra_segment = seam = 0.0
    for tick, (state, control, next_state) in enumerate(
        zip(stack.nominal_states[:-1], stack.nominal_controls, stack.nominal_states[1:], strict=True)
    ):
        stepped = state.copy()
        for _ in range(n_substeps):
            stepped = stack.model.rk4_step(stepped, float(control), substep_s)
        defect = float(np.max(np.abs(stepped - next_state)))
        if tick % 4 == 0:
            seam = max(seam, defect)
        else:
            intra_segment = max(intra_segment, defect)
    return intra_segment, seam


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Recompute the two-sided Wilson interval used in the banked gate rows."""
    if trials <= 0:
        raise ValueError("Wilson interval needs at least one trial")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(round(center - half_width, 4)), float(round(min(1.0, center + half_width), 4))


def audit_nominal_artifacts(stack: ReleaseStack | None = None) -> dict[str, float | int | str]:
    """Audit the two shipped nominal files without generating a new nominal."""
    stack = build_release_stack() if stack is None else stack
    intra_segment, seam = _dense_step_defects(stack)
    parent_defect = _parent_defect(stack.model)
    if intra_segment >= 1e-10 or seam >= 5e-5 or parent_defect >= 5e-7:
        raise ValueError("the banked nominal no longer satisfies its release bounds")
    return {
        "dense_file": NOMINAL.file,
        "dense_sha256": NOMINAL.sha256,
        "parent_file": NOMINAL_4MS.file,
        "parent_sha256": NOMINAL_4MS.sha256,
        "n_ticks": len(stack.nominal_controls),
        "horizon_s": stack.horizon_s,
        "peak_feedforward_n": float(np.max(np.abs(stack.nominal_controls))),
        "dense_intra_segment_defect": intra_segment,
        "dense_4ms_seam": seam,
        "parent_rk4_4ms_defect": parent_defect,
    }


def run_unperturbed(stack: ReleaseStack | None = None) -> LiveRun:
    """Run a new hanging-to-upright closed loop in the saturated simulator."""
    stack = build_release_stack() if stack is None else stack
    model = stack.model
    spec = model.spec
    upright = model.x_equilibrium("up")
    raw_controls: list[float] = []

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < stack.horizon_s:
            raw = stack.tracker.policy(state, time_s)
        else:
            raw = -(stack.static_gain @ wrap_state_error(state, upright, model.n)).item()
        raw_controls.append(raw)
        return raw

    t_log, x_log, applied_controls = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        stack.horizon_s + HOLD_WINDOW_S + 1.0,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    raw = np.asarray(raw_controls, dtype=float)
    handoff_tick = int(round(stack.horizon_s / spec.control_dt_s))
    handoff = x_log[min(handoff_tick, len(x_log) - 1)]
    handoff_error = wrap_state_error(handoff, upright, model.n)
    handoff_deg = float(np.rad2deg(np.max(np.abs(handoff_error[1 : 1 + model.n]))))
    hold_s = longest_continuous_hold_s(model, x_log, spec.control_dt_s)
    track_peak = float(np.max(np.abs(x_log[:, 0])))
    static_raw = raw[handoff_tick:]
    clip_ticks = int(np.count_nonzero(np.abs(raw) > spec.force_bound_n + 1e-9))
    passed = bool(hold_s >= HOLD_S - 1e-9 and track_peak <= spec.track_half_length_m)
    metrics = {
        "loaded_artifacts": {
            "dense_nominal": f"results/{NOMINAL.file}",
            "dense_nominal_sha256": NOMINAL.sha256,
            "parent_nominal_audited_separately": f"results/{NOMINAL_4MS.file}",
            "stored_controller": "none",
            "stored_rollout": "none",
        },
        "recomputed": {
            "discrete_tvlqr": "exact-ZOH linearizations and backward Riccati gains",
            "static_lqr": "upright continuous Riccati gain",
            "states": "fresh rollout_zoh states from the exact hanging equilibrium",
            "controls": "fresh raw demands and simulator-applied clipped controls",
        },
        "nominal": {
            "n_ticks": len(stack.nominal_controls),
            "horizon_s": stack.horizon_s,
            "peak_feedforward_n": float(np.max(np.abs(stack.nominal_controls))),
            "parent_rk4_4ms_defect": _parent_defect(model),
        },
        "controller": {
            "monodromy_rho": stack.tracker.monodromy(),
            "force_bound_n": spec.force_bound_n,
        },
        "live_closed_loop": {
            "handoff_max_angle_error_deg": handoff_deg,
            "longest_sampled_hold_s": hold_s,
            "track_peak_abs_m": track_peak,
            "applied_peak_force_n": float(np.max(np.abs(applied_controls))),
            "raw_peak_force_n": float(np.max(np.abs(raw))),
            "static_raw_peak_force_n": float(np.max(np.abs(static_raw))),
            "clip_ticks": clip_ticks,
            "passed": passed,
        },
    }
    if not passed or metrics["controller"]["monodromy_rho"] >= 1.0:
        raise RuntimeError("fresh n=10 closed-loop release check failed")
    return LiveRun(t_log, x_log, applied_controls, raw, metrics)


def render_live_gif(run: LiveRun, model: NLinkCartPole, horizon_s: float, output_path: Path) -> Path:
    """Render an existing live record; this function never runs a simulator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_links = model.n
    link_length = model.spec.link_lengths_m[0]
    fps = 25
    step = int(round(1.0 / (fps * model.spec.control_dt_s)))
    frames = range(0, len(run.x_log), step)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.2, 6.2)
    axis.set_ylim(-5.9, 6.1)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        xs = [float(state[0])]
        ys = [0.0]
        for link in range(n_links):
            xs.append(xs[-1] + link_length * np.sin(state[1 + link]))
            ys.append(ys[-1] + link_length * np.cos(state[1 + link]))
        return xs, ys

    def update(frame: int):
        state = run.x_log[frame]
        xs, ys = points(state)
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        time_s = run.t_log[frame]
        phase = "swing-up" if time_s < horizon_s else "balance"
        title.set_text(f"n={n_links} cart-pole — {phase}  t={time_s:5.2f} s")
        control_index = min(frame, len(run.applied_controls) - 1)
        force_text.set_text(f"applied force {run.applied_controls[control_index]:+6.1f} N (|u|<=150)")
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def audit_banked_gate_evidence() -> dict[str, Any]:
    """Hash stored records, validate metadata, count flags, and rederive Wilson intervals."""
    rows: list[dict[str, Any]] = []
    total_successes = total_trials = 0
    for filename, expected_sha in BANKED_GATE_SHA256.items():
        path = RESULTS / filename
        if file_sha256(path) != expected_sha:
            raise ValueError(f"unexpected digest for banked evidence: {filename}")
        record = json.loads(path.read_text(encoding="utf-8"))
        result_rows = record.get("results", [])
        successes = sum(bool(row.get("success")) for row in result_rows)
        trials = int(record.get("n_ic", -1))
        if record.get("controller") != "preroll_down_lqr+tvlqr_track+static_hold":
            raise ValueError(f"unexpected controller label in {filename}")
        if record.get("n_links") != N_LINKS or trials != len(result_rows):
            raise ValueError(f"malformed trial count in {filename}")
        if successes != int(record.get("n_success", -1)):
            raise ValueError(f"row success count disagrees with summary in {filename}")
        if Path(record.get("nominal", "")).name != NOMINAL.file:
            raise ValueError(f"unexpected nominal provenance label in {filename}")
        computed_wilson = list(wilson_interval(successes, trials))
        if computed_wilson != record.get("wilson95"):
            raise ValueError(f"Wilson interval disagrees with rows in {filename}")
        rows.append(
            {
                "file": filename,
                "sha256": expected_sha,
                "seed": int(record["seed"]),
                "successes": successes,
                "trials": trials,
                "wilson95": computed_wilson,
                "recorded_nominal_label": record["nominal"],
            }
        )
        total_successes += successes
        total_trials += trials
    return {
        "status": (
            "stored historical records hashed; metadata validated; stored success flags "
            "counted; Wilson intervals recomputed; historical outcomes not re-evaluated; "
            "perturbations not rerun"
        ),
        "files": rows,
        "total_successes": total_successes,
        "total_trials": total_trials,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a reproducibility artifact only to the caller-selected path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
