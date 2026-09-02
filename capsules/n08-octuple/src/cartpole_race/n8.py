"""Fresh authoritative n=8 verification and demo commands."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.evidence import (
    NOMINAL_FILE,
    PARENT_FILE,
    audit_frozen_evidence,
    sha256_file,
)
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.render import render_cartpole_gif
from cartpole_race.verification import evaluate_success_predicate

REPO = Path(__file__).resolve().parents[2]
WORKING = REPO / ".working"
HOLD_TIME_S = 5.0
SETTLE_TIME_S = 1.0
BANKED_SOURCE_PROVENANCE = "unverified"


def _model() -> tuple[NLinkCartPole, CartPoleSpec]:
    spec = CartPoleSpec().with_n_links(8)
    return NLinkCartPole(spec), spec


def _zoh_tick(
    model: NLinkCartPole,
    state: np.ndarray,
    force: float,
    control_dt_s: float,
    rk4_max_step_s: float,
) -> np.ndarray:
    """Integrate a frozen nominal state through one live simulator tick."""
    substeps = max(1, int(np.ceil(control_dt_s / rk4_max_step_s)))
    next_state = np.asarray(state, dtype=float).copy()
    for _ in range(substeps):
        next_state = model.rk4_step(next_state, force, control_dt_s / substeps)
    return next_state


def _nominal_metrics(
    model: NLinkCartPole,
    spec: CartPoleSpec,
    dense: Any,
    parent: Any,
) -> dict[str, float]:
    """Recompute baseline nominal metrics from frozen arrays and the live EOM."""
    dense_states = np.asarray(dense["x"], dtype=float)
    dense_controls = np.asarray(dense["u"], dtype=float).reshape(-1)
    parent_states = np.asarray(parent["x"], dtype=float)
    parent_controls = np.asarray(parent["u"], dtype=float).reshape(-1)

    intra_segment_defect = 0.0
    parent_boundary_seam = 0.0
    for tick, force in enumerate(dense_controls):
        predicted = _zoh_tick(
            model,
            dense_states[tick],
            float(force),
            spec.control_dt_s,
            spec.rk4_max_step_s,
        )
        defect = float(np.max(np.abs(predicted - dense_states[tick + 1])))
        if tick % 4 == 0:
            parent_boundary_seam = max(parent_boundary_seam, defect)
        else:
            intra_segment_defect = max(intra_segment_defect, defect)

    parent_dt_s = float(parent["horizon"]) / len(parent_controls)
    parent_rk4_defect = 0.0
    for tick, force in enumerate(parent_controls):
        predicted = model.rk4_step(parent_states[tick], float(force), parent_dt_s)
        parent_rk4_defect = max(
            parent_rk4_defect,
            float(np.max(np.abs(predicted - parent_states[tick + 1]))),
        )

    return {
        "dense_intra_segment_zoh_defect_max": intra_segment_defect,
        "dense_parent_boundary_seam_max": parent_boundary_seam,
        "parent_rk4_4ms_defect_max": parent_rk4_defect,
        "dense_peak_feedforward_n": float(np.max(np.abs(dense_controls))),
    }


def _require_baseline(metrics: dict[str, float]) -> None:
    limits = {
        "dense_intra_segment_zoh_defect_max": 1e-10,
        "dense_parent_boundary_seam_max": 5e-3,
        "parent_rk4_4ms_defect_max": 1e-8,
        "dense_peak_feedforward_n": 30.0,
    }
    for name, limit in limits.items():
        if metrics[name] >= limit:
            raise RuntimeError(f"{name}={metrics[name]:.6g} exceeds {limit:.6g}")


def _git_identity() -> dict[str, str | None]:
    """Record the checkout separately from immutable banked-evidence hashes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_source_dirty": None}
    return {"commit": commit, "tracked_source_dirty": str(bool(dirty)).lower()}


def _banked_commit_identities(audit: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(gate["banked_commit_sha"])
            for gate in audit["banked_gate_audit"].values()
            if gate["banked_commit_sha"]
        }
    )


def _banked_evidence_authority(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_hashes": {name: item["sha256"] for name, item in audit["artifacts"].items()},
        "banked_commit_identities": _banked_commit_identities(audit),
        "banked_source_provenance": BANKED_SOURCE_PROVENANCE,
    }


def _working_output(path: Path) -> Path:
    output = path.resolve()
    try:
        output.relative_to(WORKING.resolve())
    except ValueError as error:
        raise ValueError(f"output must be under {WORKING}") from error
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_authoritative_rollout(out_dir: Path, *, render: bool) -> dict[str, Any]:
    """Audit frozen evidence and produce one fresh unperturbed n=8 rollout."""
    out_dir = _working_output(out_dir)
    audit = audit_frozen_evidence(REPO)
    _write_json(out_dir / "evidence-audit.json", audit)
    print(f"[audit] {len(audit['artifacts'])} frozen artifact hashes match")

    model, spec = _model()
    results = REPO / "results"
    with np.load(results / NOMINAL_FILE, allow_pickle=False) as dense, np.load(
        results / PARENT_FILE, allow_pickle=False
    ) as parent:
        dense_states = np.asarray(dense["x"], dtype=float)
        dense_controls = np.asarray(dense["u"], dtype=float).reshape(-1)
        horizon_s = float(dense["horizon"])
        if dense_states.shape != (9001, 18) or len(dense_controls) != 9000 or horizon_s != 9.0:
            raise ValueError("dense nominal identity is not the authoritative n=8 baseline")
        metrics = _nominal_metrics(model, spec, dense, parent)

    _require_baseline(metrics)
    print(
        "[baseline] "
        f"peak={metrics['dense_peak_feedforward_n']:.4f} N, "
        f"seam={metrics['dense_parent_boundary_seam_max']:.3e}, "
        f"parent-defect={metrics['parent_rk4_4ms_defect_max']:.3e}"
    )

    controller = DiscreteTVLQR(model, dense_states, dense_controls, spec.control_dt_s)
    monodromy_rho = controller.monodromy_spectral_radius()
    if monodromy_rho >= 1.0:
        raise RuntimeError(f"closed-loop monodromy does not contract: rho={monodromy_rho}")

    static_gain, _ = static_lqr(model)
    static_policy = StaticLQRPolicy(model, static_gain)

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < horizon_s:
            return controller.policy(state, time_s)
        return static_policy(state, time_s)

    times, states, applied_controls = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        horizon_s + HOLD_TIME_S + SETTLE_TIME_S,
        spec.control_dt_s,
        spec.rk4_max_step_s,
    )
    predicate = evaluate_success_predicate(model, states, applied_controls, HOLD_TIME_S)
    if not predicate["success"]:
        raise RuntimeError(f"fresh unperturbed rollout failed: {predicate}")

    handoff_tick = int(round(horizon_s / spec.control_dt_s))
    upright = model.x_equilibrium("up")
    handoff_error = states[handoff_tick, 1 : 1 + model.n] - upright[1 : 1 + model.n]
    handoff_deviation_deg = float(
        np.rad2deg(
            np.max(np.abs(np.arctan2(np.sin(handoff_error), np.cos(handoff_error))))
        )
    )
    swing_peak_n = float(np.max(np.abs(applied_controls[:handoff_tick])))
    hold_peak_n = float(np.max(np.abs(applied_controls[handoff_tick:])))

    rollout_path = out_dir / "fresh-unperturbed-rollout.npz"
    np.savez(
        rollout_path,
        t=times,
        x=states,
        u_applied=applied_controls,
        horizon_s=horizon_s,
        hold_s=HOLD_TIME_S,
        settle_s=SETTLE_TIME_S,
    )
    summary: dict[str, Any] = {
        "authority": {
            **_banked_evidence_authority(audit),
            "checkout": _git_identity(),
        },
        "baseline": metrics,
        "fresh_unperturbed_rollout": {
            "closed_loop_monodromy_rho": monodromy_rho,
            "handoff_deviation_deg": handoff_deviation_deg,
            "swing_peak_applied_force_n": swing_peak_n,
            "hold_peak_applied_force_n": hold_peak_n,
            "predicate": predicate,
        },
        "outputs": {
            "evidence_audit": "evidence-audit.json",
            "fresh_states_and_applied_controls": rollout_path.name,
            "fresh_states_and_applied_controls_sha256": sha256_file(rollout_path),
        },
    }
    print(
        f"[live] rho={monodromy_rho:.4f}; handoff={handoff_deviation_deg:.4f} deg; "
        f"swing={swing_peak_n:.1f} N; hold={hold_peak_n:.1f} N; "
        f"predicate=PASS ({predicate['tail_hold_s']:.3f} s)"
    )

    if render:
        render_path = out_dir / "fresh-unperturbed-rollout.gif"
        render_cartpole_gif(
            render_path,
            times,
            states,
            applied_controls,
            n_links=model.n,
            link_length_m=spec.link_lengths_m[0],
            swingup_horizon_s=horizon_s,
            force_bound_n=spec.force_bound_n,
        )
        summary["outputs"]["fresh_render"] = render_path.name
        summary["outputs"]["fresh_render_sha256"] = sha256_file(render_path)
        print(f"[render] {render_path.relative_to(REPO).as_posix()}")

    _write_json(out_dir / "summary.json", summary)
    return summary


def _parser(default_out_dir: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="local output directory under .working/",
    )
    return parser


def verify_main() -> int:
    args = _parser(WORKING / "n8-verify", __doc__).parse_args()
    run_authoritative_rollout(args.out_dir, render=False)
    return 0


def demo_main() -> int:
    args = _parser(WORKING / "n8-demo", __doc__).parse_args()
    run_authoritative_rollout(args.out_dir, render=True)
    return 0
