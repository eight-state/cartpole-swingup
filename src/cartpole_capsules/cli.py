"""Command-line interface for the capsule registry."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from cartpole_capsules.layout import find_repo_root
from cartpole_capsules.manifest import CapsuleEntry, ManifestError, capsule_directory, load_manifest
from cartpole_capsules.registry import check_registry


def verification_steps(entry: CapsuleEntry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the two commands used to verify a capsule."""
    command = tuple(shlex.split(entry.verification_command, posix=True))
    if not command or command[:2] != ("uv", "run"):
        raise ManifestError(f"invalid verification command for {entry.slug}")
    return (("uv", "sync", "--locked"), command)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="cartpole-capsule")
    parser.add_argument("--root", type=Path, help="repository root; auto-detected by default")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered capsules")
    commands.add_parser("check", help="verify manifest and imported capsule bytes")
    verify = commands.add_parser("verify", help="run one capsule's own locked verifier")
    verify.add_argument("rung", type=int, help="link count, for example 5")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capsule CLI."""
    arguments = build_parser().parse_args(argv)
    root = arguments.root.resolve() if arguments.root else find_repo_root()
    try:
        if arguments.command == "list":
            return _list(root)
        if arguments.command == "check":
            checks = check_registry(root)
            print(f"Verified {len(checks)} registered capsule(s).")
            return 0
        if arguments.command == "verify":
            return _verify(root, arguments.rung)
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    raise AssertionError(f"unhandled command: {arguments.command}")


def _list(root: Path) -> int:
    entries = load_manifest(root)
    if not entries:
        print("No capsules registered.")
        return 0
    print("RUNG  CAPSULE              RUNNER          EVIDENCE")
    for entry in entries:
        print(f"n={entry.rung:<2}  {entry.slug:<20} {entry.runner:<15} {entry.evidence_class}")
    return 0


def _verify(root: Path, rung: int) -> int:
    entries = load_manifest(root)
    entry = next((candidate for candidate in entries if candidate.rung == rung), None)
    if entry is None:
        raise ManifestError(f"no capsule registered for n={rung}")
    directory = capsule_directory(root, entry)
    for command in verification_steps(entry):
        result = subprocess.run(command, cwd=directory, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
