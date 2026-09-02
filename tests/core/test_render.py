"""Tests for cartpole_capsules.core.render. Tiny 2-frame GIF only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _matplotlib_available(), reason="matplotlib or pillow not available")
def test_render_cartpole_gif_writes_file(tmp_path: Path):
    from cartpole_capsules.core.render import render_cartpole_gif

    n_links = 1
    t = np.linspace(0.0, 0.04 * 4, 5)  # 4 ms apart, 25 fps picks 2 frames
    nx = 2 * (n_links + 1)
    x = np.zeros((len(t), nx))
    x[:, 0] = np.linspace(0.0, 0.1, len(t))
    u = np.zeros(len(t) - 1)
    output = tmp_path / "out" / "tiny.gif"
    render_cartpole_gif(
        output,
        t,
        x,
        u,
        n_links=n_links,
        link_length_m=0.5,
        swingup_horizon_s=0.08,
        force_bound_n=150.0,
    )
    assert output.exists()
    assert output.stat().st_size > 0
