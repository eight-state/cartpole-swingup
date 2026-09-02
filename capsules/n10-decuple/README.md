# N10 cart-pole replay

`n10-demo` loads the fixed 1 ms dense nominal as its tracking reference, audits the fixed 4 ms parent nominal, rebuilds exact-ZOH discrete TVLQR and upright LQR, and runs one fresh force-saturated simulation from hanging. It writes `.working/n10-live/live-metrics.json` with controller and predicate metrics and `.working/n10-live/demo.gif` from the fresh states.

```sh
uv sync --locked
uv run n10-demo
uv run n10-verify
uv run ruff check .
uv run python -m pytest -q
```

> Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

The model is ten 0.5 m, 0.1 kg links on a 1 kg cart: 1 kHz ZOH, four 0.25 ms RK4 substeps, 150 N force-magnitude clipping, and full-state feedback. Acceptance is evaluated at 1 kHz control-boundary samples: a five-second hold is 5,001 consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

`n10-verify` checks SHA-256 authority bytes for both nominals and the three banked gate records, recomputes nominal defects, hashes records, validates stored metadata, counts stored `success` flags, and recomputes Wilson intervals: 24/24 for seeds 12345, 777, and 2024, or 72/72 total. It neither re-evaluates historical outcomes nor reruns perturbations. This repository does not support nominal synthesis or perturbation reruns. This repository limits its scope to deterministic simulation and excludes hardware and perturbation-robustness claims.
