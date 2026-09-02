"""Register one newly imported capsule in capsules/imports.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from cartpole_capsules.manifest import ManifestError
from cartpole_capsules.registry import register_capsule


def build_parser() -> argparse.ArgumentParser:
    """Build the registration parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", required=True, type=int)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--verification-command", required=True)
    parser.add_argument("--demo-command")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--evidence-class", required=True)
    return parser


def main() -> int:
    """Register one capsule and print its authority summary."""
    arguments = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        entry = register_capsule(
            root,
            rung=arguments.rung,
            slug=arguments.slug,
            source_url=arguments.source_url,
            source_commit=arguments.source_commit,
            source_tree=arguments.source_tree,
            verification_command=arguments.verification_command,
            demo_command=arguments.demo_command,
            runner=arguments.runner,
            evidence_class=arguments.evidence_class,
        )
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"Registered n={entry.rung} at {entry.capsule_path}: "
        f"{entry.file_count} files, sha256 {entry.content_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
