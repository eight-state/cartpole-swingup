"""Console commands for the n=10 replay and its evidence audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from cartpole_race.n10_release import (
    REPO,
    audit_banked_gate_evidence,
    audit_nominal_artifacts,
    build_release_stack,
    render_live_gif,
    run_unperturbed,
    write_json,
)


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one fresh n=10 closed-loop replay.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / ".working" / "n10-live",
        help="ignored directory for fresh metrics and render",
    )
    parser.add_argument("--no-render", action="store_true", help="skip the GIF")
    return parser


def demo_main() -> int:
    """Run the fresh replay and optionally render its exact live record."""
    args = _demo_parser().parse_args()
    output_dir = args.output_dir.resolve()
    stack = build_release_stack()
    live = run_unperturbed(stack)
    metrics_path = write_json(output_dir / "live-metrics.json", live.metrics)
    nominal = live.metrics["nominal"]
    controller = live.metrics["controller"]
    closed_loop = live.metrics["live_closed_loop"]
    print(f"[loaded] {live.metrics['loaded_artifacts']['dense_nominal']} (banked nominal; no stored gains)")
    print(
        f"[recomputed] parent defect {nominal['parent_rk4_4ms_defect']:.3e}; "
        f"rho {controller['monodromy_rho']:.4g}"
    )
    print(
        f"[live] handoff {closed_loop['handoff_max_angle_error_deg']:.4f} deg; "
        f"hold {closed_loop['longest_continuous_hold_s']:.1f} s; "
        f"applied/raw peak {closed_loop['applied_peak_force_n']:.1f}/"
        f"{closed_loop['raw_peak_force_n']:.1f} N; clips {closed_loop['clip_ticks']} -> "
        f"{'PASS' if closed_loop['passed'] else 'FAIL'}"
    )
    if not args.no_render:
        gif_path = render_live_gif(live, stack.model, stack.horizon_s, output_dir / "demo.gif")
        print(f"[render] {gif_path}")
    print(f"[metrics] {metrics_path}")
    return 0


def _verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit banked n=10 evidence and rerun the live stack.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / ".working" / "n10-verification",
        help="ignored directory for this audit and fresh live metrics",
    )
    return parser


def verify_main() -> int:
    """Audit fixed evidence and run the same unperturbed live stack."""
    args = _verify_parser().parse_args()
    output_dir = args.output_dir.resolve()
    stack = build_release_stack()
    nominal = audit_nominal_artifacts(stack)
    banked_gate = audit_banked_gate_evidence()
    live = run_unperturbed(stack)
    report = {"nominal_artifacts": nominal, "banked_gate_evidence": banked_gate, "fresh_live": live.metrics}
    report_path = write_json(output_dir / "verification.json", report)

    print(
        f"[nominal audit] parent defect {nominal['parent_rk4_4ms_defect']:.3e}; "
        f"dense seam {nominal['dense_4ms_seam']:.3e}; "
        f"peak ff {nominal['peak_feedforward_n']:.1f} N"
    )
    print(
        f"[banked gate audit] {banked_gate['total_successes']}/{banked_gate['total_trials']} rows; "
        "historical records only, not a fresh perturbation rerun"
    )
    fresh = live.metrics["live_closed_loop"]
    print(
        f"[same live stack] hold {fresh['longest_continuous_hold_s']:.1f} s; "
        f"applied/raw peak {fresh['applied_peak_force_n']:.1f}/{fresh['raw_peak_force_n']:.1f} N; "
        f"clips {fresh['clip_ticks']} -> {'PASS' if fresh['passed'] else 'FAIL'}"
    )
    print(f"[report] {report_path}")
    return 0
