"""Generate the retained N14 verification report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from n14_cartpole.release_audit import REPOSITORY
from n14_cartpole.verifier import cli as verifier_cli


def cli(argv: Sequence[str] | None = None) -> int:
    """Verify the release and atomically replace its report only on PASS."""
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    return verifier_cli(["--output", str(REPOSITORY / "artifacts" / "verification.json")])


if __name__ == "__main__":
    raise SystemExit(cli())
