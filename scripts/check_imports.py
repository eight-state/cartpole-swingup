"""Check every registered capsule against the import manifest."""

from __future__ import annotations

from pathlib import Path

from cartpole_capsules.cli import main

if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    raise SystemExit(main(("--root", str(repository), "check")))
