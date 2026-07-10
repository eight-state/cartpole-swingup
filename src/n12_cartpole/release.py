"""Run the default N12 release verification or all three exact gate reruns."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from n12_cartpole.release_audit import audit_release_artifacts
from n12_cartpole.verifier import run_verifier

REPOSITORY = Path(__file__).resolve().parents[2]
GATE_SEEDS_AND_WORKERS = ((12345, 6), (777, 3), (2024, 3))
GATE_ENV = {
    "NLINKS": "12",
    "NOM_PATH": "runs/r2/nom_n12_4ms_fast.npz",
    "REFERENCE_DENSIFY_STRIDE": "4",
    "TRACKER_LINK_RATE_Q_SCALE": "0.25",
    "TRACKER_TO_HOLD_SWITCH_TICK": "9700",
    "PREROLL_TOL": "0",
    "PREROLL_VEL_Q_SCALE": "4",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _summary(audit: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    return {
        "banked_gate": f"{audit['aggregate_successes']}/{audit['aggregate_trials']}",
        "gate_seeds": [seed for seed, _ in GATE_SEEDS_AND_WORKERS],
        "unperturbed_expected_witness": verifier["expected_witness"]["all_assertions_pass"],
        "unperturbed_verdict": verifier["verdict"],
    }


def _run_all_gates() -> None:
    for seed, workers in GATE_SEEDS_AND_WORKERS:
        command = [
            sys.executable,
            str(REPOSITORY / "scripts" / "gate_preroll.py"),
            "24",
            str(seed),
            "18.0",
            str(workers),
        ]
        environment = os.environ.copy()
        environment.update(GATE_ENV)
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"gate seed {seed} failed with exit code {completed.returncode}")


def cli(argv: Sequence[str] | None = None) -> int:
    """Run banked verification, with optional exact reruns for all three seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Rerun 24 trials for seeds 12345, 777, and 2024 before the final audit.",
    )
    args = parser.parse_args(argv)

    initial_audit = audit_release_artifacts()
    verifier = run_verifier()
    if verifier["verdict"] != "PASS" or not verifier["expected_witness"]["all_assertions_pass"]:
        print(
            json.dumps(
                {
                    "release_summary": _summary(initial_audit, verifier),
                    "verifier": verifier,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    if not args.gate:
        print(json.dumps(_summary(initial_audit, verifier), indent=2, sort_keys=True))
        return 0

    _run_all_gates()
    final_audit = audit_release_artifacts()
    print(json.dumps(_summary(final_audit, verifier), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
