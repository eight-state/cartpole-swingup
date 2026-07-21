"""Generate the retained N14 verification report."""

from __future__ import annotations

from collections.abc import Sequence

from n14_cartpole.verifier import REPOSITORY, cli as verifier_cli


def cli(argv: Sequence[str] | None = None) -> int:
    """Verify the release and write ``artifacts/verification.json``."""
    if argv:
        raise ValueError("n14-release accepts no arguments")
    return verifier_cli(["--output", str(REPOSITORY / "artifacts" / "verification.json")])


if __name__ == "__main__":
    raise SystemExit(cli())
