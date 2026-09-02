"""Shared evidence audits and replay orchestration for every rung."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cartpole_capsules.adapters import base
from cartpole_capsules.adapters.base import GateSpec, RungConfig


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _wilson_raw(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval needs at least one trial")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return center - half_width, min(1.0, center + half_width)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the stored four-decimal Wilson interval used by row gates."""
    lower, upper = _wilson_raw(successes, trials)
    return round(lower, 4), round(upper, 4)


def _checked_path(gate: GateSpec, root: Path) -> Path:
    path = root / gate.file
    if not path.is_file():
        raise ValueError(f"missing banked evidence: {gate.file}")
    actual = sha256_file(path)
    if actual != gate.sha256:
        raise ValueError(f"unexpected digest for banked evidence: {gate.file}")
    return path


def audit_hash_gate(gate: GateSpec, root: Path) -> dict[str, Any]:
    """Check one opaque historical record by digest only."""
    _checked_path(gate, root)
    return {"file": gate.file, "sha256": gate.sha256, "kind": "hash-only"}


def audit_aggregate_gate(gate: GateSpec, cfg: RungConfig, root: Path) -> dict[str, Any]:
    """Audit an n=5 or n=6 aggregate record with no per-trial rows."""
    record = load_json(_checked_path(gate, root))
    trials = int(record.get("n_trials", -1))
    successes = int(record.get("n_success", -1))
    if trials != gate.expected_trials or successes != gate.expected_successes:
        raise ValueError(f"aggregate count drift: {gate.file}")
    if gate.seed is not None and int(record.get("seed", -1)) != gate.seed:
        raise ValueError(f"aggregate seed drift: {gate.file}")
    if int(record.get("n_links", -1)) != cfg.n_links:
        raise ValueError(f"aggregate link count drift: {gate.file}")
    if float(record.get("force_limit", math.inf)) != gate.force_bound_n:
        raise ValueError(f"aggregate force boundary drift: {gate.file}")
    active = next(nominal for nominal in cfg.nominals if nominal.role == "active")
    if record.get("nominal_sha256") != active.sha256:
        raise ValueError(f"aggregate nominal digest drift: {gate.file}")
    lower, upper = _wilson_raw(successes, trials)
    if not math.isclose(float(record.get("wilson_lo", math.nan)), lower, abs_tol=1e-12):
        raise ValueError(f"aggregate Wilson lower bound drift: {gate.file}")
    if not math.isclose(float(record.get("wilson_hi", math.nan)), upper, abs_tol=1e-12):
        raise ValueError(f"aggregate Wilson upper bound drift: {gate.file}")
    return {
        "file": gate.file,
        "sha256": gate.sha256,
        "seed": gate.seed,
        "successes": successes,
        "trials": trials,
        "wilson95": [lower, upper],
    }


def audit_composite_gate(gate: GateSpec, cfg: RungConfig, root: Path) -> dict[str, Any]:
    """Audit n=7 or n=8 composite and retained fixed-failure records."""
    record = load_json(_checked_path(gate, root))
    rows = record.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"composite rows missing: {gate.file}")
    trials = int(record.get("n_ic", -1))
    successes = sum(row.get("success") is True for row in rows if isinstance(row, dict))
    if len(rows) != trials or trials != gate.expected_trials:
        raise ValueError(f"composite trial count drift: {gate.file}")
    if successes != int(record.get("n_success", -1)) or successes != gate.expected_successes:
        raise ValueError(f"composite success count drift: {gate.file}")
    if gate.seed is not None and int(record.get("seed", -1)) != gate.seed:
        raise ValueError(f"composite seed drift: {gate.file}")
    if [row.get("tag") for row in rows] != list(range(trials)):
        raise ValueError(f"composite row tags drift: {gate.file}")
    active = next(nominal for nominal in cfg.nominals if nominal.role == "active")
    if record.get("nominal_sha256") != active.sha256:
        raise ValueError(f"composite nominal digest drift: {gate.file}")
    passing_leg = successes == trials
    for row in rows:
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("track_ok"), bool):
            raise ValueError(f"composite boolean fields drift: {gate.file}")
        if passing_leg and row.get("track_ok") is not True:
            raise ValueError(f"composite passing-leg track record drift: {gate.file}")
        for key in ("peakF", "max_force_demanded"):
            if key in row and not math.isfinite(float(row[key])):
                raise ValueError(f"nonfinite composite {key}: {gate.file}")
    return {
        "file": gate.file,
        "sha256": gate.sha256,
        "seed": gate.seed,
        "successes": successes,
        "trials": trials,
        "status": "passing leg" if successes == trials else "retained failure leg",
    }


def audit_row_gate(gate: GateSpec, cfg: RungConfig, root: Path) -> dict[str, Any]:
    """Audit one n=9 through n=11 per-trial gate without rerunning it."""
    record = load_json(_checked_path(gate, root))
    if gate.header_fields is not None and set(record) != set(gate.header_fields):
        raise ValueError(f"unexpected gate fields: {gate.file}")
    if gate.controller is not None and record.get("controller") != gate.controller:
        raise ValueError(f"unexpected controller label: {gate.file}")
    if gate.nominal is not None:
        stored = str(record.get("nominal", ""))
        matched = (
            stored == gate.nominal
            if gate.nominal_match == "exact"
            else (Path(stored).name == Path(gate.nominal).name)
        )
        if not matched:
            raise ValueError(f"unexpected nominal provenance: {gate.file}")
    rows = record.get("results")
    if not isinstance(rows, list) or int(record.get("n_ic", -1)) != len(rows):
        raise ValueError(f"trial count disagrees with rows: {gate.file}")
    if [row.get("tag") for row in rows] != list(range(len(rows))):
        raise ValueError(f"tags are not ordered: {gate.file}")
    if gate.row_fields is not None and any(set(row) != set(gate.row_fields) for row in rows):
        raise ValueError(f"row fields drift: {gate.file}")
    for row in rows:
        if row.get("success") is not True or row.get("track_ok") is not True:
            raise ValueError(f"failed historical row: {gate.file}")
        if row.get("fail") is not None:
            raise ValueError(f"failure label present: {gate.file}")
        for key in ("handoff_deg", "hold_s", "peakF", "pert_deg", "resid", "t_pre"):
            if key in row and not math.isfinite(float(row[key])):
                raise ValueError(f"nonfinite measurement {key}: {gate.file}")
        if (
            gate.handoff_limit_deg is not None
            and float(row["handoff_deg"]) > gate.handoff_limit_deg
        ):
            raise ValueError(f"handoff limit exceeded: {gate.file}")
        if float(row.get("hold_s", 0.0)) < gate.hold_required_s:
            raise ValueError(f"hold predicate failed: {gate.file}")
        if float(row.get("peakF", math.inf)) > gate.force_bound_n:
            raise ValueError(f"force bound exceeded: {gate.file}")
        if gate.check_t_pre and float(row["t_pre"]) != float(record["T_pre_s"]):
            raise ValueError(f"pre-roll disagreement: {gate.file}")
        if gate.check_tracker_ticks and int(row.get("tracker_ticks", -1)) != cfg.switch_tick:
            raise ValueError(f"tracker schedule mismatch: {gate.file}")
    if gate.require_saturation and not any(
        float(row.get("peakF", 0.0)) == gate.force_bound_n for row in rows
    ):
        raise ValueError(f"banked saturation evidence missing: {gate.file}")
    successes = sum(row["success"] is True for row in rows)
    if successes != int(record.get("n_success", -1)):
        raise ValueError(f"success count disagrees with rows: {gate.file}")
    if gate.seed is not None and int(record.get("seed", -1)) != gate.seed:
        raise ValueError(f"seed mismatch: {gate.file}")
    if "wilson95" in record and tuple(record["wilson95"]) != wilson_interval(successes, len(rows)):
        raise ValueError(f"Wilson interval disagrees with rows: {gate.file}")
    return {
        "file": gate.file,
        "sha256": gate.sha256,
        "seed": record.get("seed"),
        "successes": successes,
        "trials": len(rows),
        "wilson95": list(record.get("wilson95", ())),
    }


def audit_n12_evidence(gate: GateSpec, cfg: RungConfig, root: Path) -> dict[str, Any]:
    """Audit n=12 embedded banked rows and its frozen-nominal declaration."""
    evidence = load_json(_checked_path(gate, root))
    if evidence.get("schema_version") != 1 or evidence.get("release") != "N12":
        raise ValueError("unexpected n=12 evidence schema")
    capabilities = evidence.get("capabilities", {})
    if capabilities != {"nominal_synthesis": False, "perturbed_gate_rerun": False}:
        raise ValueError("n=12 capability boundary drift")
    nominal = next(item for item in cfg.nominals if item.role == "active")
    frozen = evidence.get("frozen_nominal", {})
    if frozen.get("path") != nominal.path or frozen.get("sha256") != nominal.sha256:
        raise ValueError("n=12 frozen nominal identity drift")
    if tuple(frozen.get("state_shape", ())) != (nominal.n_nodes + 1, 2 * (cfg.n_links + 1)):
        raise ValueError("n=12 frozen nominal state shape drift")

    audited: list[dict[str, Any]] = []
    records = evidence.get("banked_gate", {}).get("records")
    if not isinstance(records, list):
        raise ValueError("n=12 banked records missing")
    for record in records:
        rows = record.get("results")
        if not isinstance(rows, list) or len(rows) != int(record.get("n_ic", -1)):
            raise ValueError("n=12 banked row count drift")
        if [row.get("tag") for row in rows] != list(range(len(rows))):
            raise ValueError("n=12 banked row tags drift")
        for row in rows:
            if row.get("success") is not True or row.get("track_ok") is not True:
                raise ValueError("n=12 contains a failed banked row")
            if row.get("fail") is not None:
                raise ValueError("n=12 contains a banked failure label")
            if float(row["hold_s"]) < gate.hold_required_s:
                raise ValueError("n=12 banked hold drift")
            if float(row["peakF"]) > gate.force_bound_n:
                raise ValueError("n=12 banked force drift")
            if int(row["tracker_ticks"]) != cfg.switch_tick:
                raise ValueError("n=12 banked switch drift")
        successes = sum(row["success"] is True for row in rows)
        if successes != int(record.get("n_success", -1)):
            raise ValueError("n=12 banked success count drift")
        interval = wilson_interval(successes, len(rows))
        if tuple(record.get("wilson95", ())) != interval:
            raise ValueError("n=12 banked Wilson interval drift")
        audited.append(
            {
                "seed": int(record["seed"]),
                "successes": successes,
                "trials": len(rows),
                "wilson95": list(interval),
            }
        )
    if [item["seed"] for item in audited] != [12345, 777, 2024]:
        raise ValueError("n=12 banked seed order drift")
    successes = sum(item["successes"] for item in audited)
    trials = sum(item["trials"] for item in audited)
    if evidence["banked_gate"].get("summary") != {"successes": successes, "trials": trials}:
        raise ValueError("n=12 banked summary drift")
    return {
        "file": gate.file,
        "sha256": gate.sha256,
        "seeds": [item["seed"] for item in audited],
        "successes": successes,
        "trials": trials,
    }


def audit_gates(cfg: RungConfig, root: Path) -> dict[str, Any]:
    """Audit every historical record with its declared dialect."""
    auditors = {
        "aggregate": lambda gate: audit_aggregate_gate(gate, cfg, root),
        "composite": lambda gate: audit_composite_gate(gate, cfg, root),
        "hash": lambda gate: audit_hash_gate(gate, root),
        "rows": lambda gate: audit_row_gate(gate, cfg, root),
        "n12-evidence": lambda gate: audit_n12_evidence(gate, cfg, root),
    }
    audited = []
    for gate in cfg.gates:
        try:
            auditor = auditors[gate.dialect]
        except KeyError as exc:
            raise ValueError(f"unknown gate dialect {gate.dialect!r}") from exc
        audited.append(auditor(gate))
    return {
        "status": "historical evidence audited; no perturbations rerun",
        "files": audited,
        "total_successes": sum(int(item.get("successes", 0)) for item in audited),
        "total_trials": sum(int(item.get("trials", 0)) for item in audited),
    }


def _runtime_block() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def check_rung_authority(cfg: RungConfig) -> dict[str, Any]:
    """Check one rung's tag, retained artifacts, and historical records."""
    from cartpole_capsules.adapters import proof_n13, witness_n14
    from cartpole_capsules.legacy import audit_legacy_source

    root = base.capsule_root(cfg)
    nominals = []
    for nominal in cfg.nominals:
        actual = sha256_file(root / nominal.path)
        if actual != nominal.sha256:
            raise ValueError(f"nominal bytes changed: {nominal.path}")
        nominals.append({"role": nominal.role, "path": nominal.path, "sha256": actual})
    special: dict[str, Any] = {}
    if cfg.adapter == "proof_n13":
        special = proof_n13.audit_authority(cfg, root)
    elif cfg.adapter == "witness_n14":
        special = witness_n14.audit_authority(cfg)
    return {
        "legacy_source": audit_legacy_source(base.repository_root(), cfg),
        "nominals": nominals,
        "banked_gates": audit_gates(cfg, root),
        "special": special,
    }


def execute_rung(cfg: RungConfig) -> Any:
    """Build and run one shared adapter."""
    adapter = base.build_adapter(cfg)
    stack = adapter.load(cfg)
    return (
        adapter.replay_and_check(cfg, stack)
        if cfg.adapter == "witness_n14"
        else adapter.run(cfg, stack)
    )


def verify_rung(cfg: RungConfig, no_replay: bool = False) -> dict[str, Any]:
    """Run authority checks, then the rung's fresh replay unless disabled."""
    authority = check_rung_authority(cfg)
    report: dict[str, Any] = {
        "schema_version": 1,
        "rung": cfg.rung,
        "slug": cfg.slug,
        "adapter": cfg.adapter,
        "evidence_class": cfg.evidence_class,
        "host_sensitive": cfg.host_sensitive,
        "runtime": _runtime_block(),
        "authority": authority,
    }
    if no_replay:
        report["replay"] = {"skipped": True}
        report["verdict"] = "AUDITED"
        return report
    result = execute_rung(cfg)
    report["replay"] = {
        "passed": result.passed,
        "failures": list(result.failures),
        "metrics": base.to_jsonable(result.metrics),
    }
    report["verdict"] = "PASS" if result.passed else "FAIL"
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Verify one rung from the shared registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rung", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.root is not None:
        base.ROOT_OVERRIDE = arguments.root.resolve()
    registry = base.load_registry()
    if arguments.rung not in registry:
        parser.error(f"unknown rung {arguments.rung}; registered: {sorted(registry)}")
    report = verify_rung(registry[arguments.rung], no_replay=arguments.no_replay)
    output = (
        arguments.out
        or base.repository_root() / ".working" / "verify" / f"n{arguments.rung:02d}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"rung": arguments.rung, "verdict": report["verdict"], "output": str(output)}))
    return 0 if report["verdict"] in {"PASS", "AUDITED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
