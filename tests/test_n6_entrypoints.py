"""The capsule exposes only the two public N6 commands."""

from cartpole_race.n6 import demo_main, verify_main


def test_entry_points_are_callable() -> None:
    """Console handlers parse help without starting a replay."""
    for entry_point in (demo_main, verify_main):
        try:
            entry_point(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:  # pragma: no cover - argparse always exits for --help
            raise AssertionError("entry point did not handle --help")
