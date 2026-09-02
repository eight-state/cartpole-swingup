"""One command-line interface for every CartPole rung."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cartpole_capsules.adapters import base
from cartpole_capsules.adapters.verification import (
    check_rung_authority,
    execute_rung,
    verify_rung,
)
from cartpole_capsules.core.render import render_cartpole_gif


def build_parser() -> argparse.ArgumentParser:
    """Build the repository CLI."""
    parser = argparse.ArgumentParser(prog="cartpole-capsule")
    parser.add_argument("--root", type=Path, help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list rungs")
    commands.add_parser("check", help="audit all tags and retained evidence")
    verify = commands.add_parser("verify", help="verify one rung or all rungs")
    verify.add_argument("target", help="rung number or all")
    verify.add_argument("--audit-only", action="store_true")
    demo = commands.add_parser("demo", help="verify and render one rung")
    demo.add_argument("rung", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository CLI."""
    arguments = build_parser().parse_args(argv)
    if arguments.root is not None:
        base.ROOT_OVERRIDE = arguments.root.resolve()
    try:
        registry = base.load_registry()
        if arguments.command == "list":
            return _list(registry)
        if arguments.command == "check":
            return _check(registry)
        if arguments.command == "verify":
            return _verify(registry, arguments.target, arguments.audit_only)
        if arguments.command == "demo":
            return _demo(registry, arguments.rung)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    raise AssertionError(f"unhandled command: {arguments.command}")


def _list(registry: dict[int, base.RungConfig]) -> int:
    print("RUNG  ADAPTER           RUNNER          EVIDENCE")
    for rung, config in registry.items():
        print(f"n={rung:<2}  {config.adapter:<17} {config.runner:<15} {config.evidence_class}")
    return 0


def _check(registry: dict[int, base.RungConfig]) -> int:
    for rung, config in registry.items():
        authority = check_rung_authority(config)
        files = len(authority["banked_gates"]["files"])
        print(f"n={rung}: authority PASS, {files} historical record(s)")
    return 0


def _targets(registry: dict[int, base.RungConfig], target: str) -> list[int]:
    if target == "all":
        return list(registry)
    try:
        rung = int(target)
    except ValueError as exc:
        raise ValueError("target must be a rung number or all") from exc
    if rung not in registry:
        raise ValueError(f"unknown rung {rung}; registered: {sorted(registry)}")
    return [rung]


def _verify(registry: dict[int, base.RungConfig], target: str, audit_only: bool) -> int:
    passed = True
    for rung in _targets(registry, target):
        report = verify_rung(registry[rung], no_replay=audit_only)
        output = base.repository_root() / ".working" / "verify" / f"n{rung:02d}.json"
        _write_json(output, report)
        print(f"n={rung}: {report['verdict']}")
        passed = passed and report["verdict"] in {"PASS", "AUDITED"}
    return 0 if passed else 1


def _demo(registry: dict[int, base.RungConfig], rung: int) -> int:
    if rung not in registry:
        raise ValueError(f"unknown rung {rung}; registered: {sorted(registry)}")
    config = registry[rung]
    check_rung_authority(config)
    result = execute_rung(config)
    if not result.passed:
        raise RuntimeError(f"n={rung} replay failed: {', '.join(result.failures)}")
    output = base.repository_root() / ".working" / "demo" / f"n{rung:02d}.gif"
    active = next((item for item in config.nominals if item.role == "active"), None)
    horizon = (
        active.horizon_s if active is not None else (config.switch_tick or 0) * config.control_dt_s
    )
    render_cartpole_gif(
        output,
        result.record.times,
        result.record.states,
        result.record.applied,
        n_links=config.n_links,
        link_length_m=config.link_length_m,
        swingup_horizon_s=horizon,
        force_bound_n=config.force_bound_n,
    )
    _write_json(output.with_suffix(".json"), base.to_jsonable(result.metrics))
    print(f"n={rung}: PASS, rendered {output.relative_to(base.repository_root())}")
    return 0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
