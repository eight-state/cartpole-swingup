# N10 cart-pole replay

`n10-demo` loads the fixed 1 ms dense nominal as its tracking reference, audits the fixed 4 ms parent nominal, rebuilds exact-ZOH discrete TVLQR and upright LQR, and runs one fresh force-saturated simulation from hanging. It writes `.working/n10-live/live-metrics.json` with controller and predicate metrics and `.working/n10-live/demo.gif` from the fresh states.

```sh
uv sync --locked
uv run n10-demo
uv run n10-verify
uv run ruff check .
uv run python -m pytest -q
```

The model is ten 0.5 m, 0.1 kg links on a 1 kg cart: 1 kHz ZOH, four 0.25 ms RK4 substeps, 150 N force-magnitude clipping, and full-state feedback. Success requires every wrapped link angle within 5°, link rate within 0.5 rad/s, cart position within 2 m, and cart velocity within 0.5 m/s for a continuous 5 s.

`n10-verify` checks SHA-256 authority bytes for both nominals and the three banked gate records, recomputes nominal defects and each gate row: 24/24 for seeds 12345, 777, and 2024, or 72/72 total. The gate records provide historical evidence only. This repository does not support nominal synthesis or perturbation reruns. This repository limits its scope to deterministic simulation and excludes hardware and perturbation-robustness claims.
