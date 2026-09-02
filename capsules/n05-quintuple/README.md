# N5 cart-pole replay

`n5-demo` loads the fixed n=5 nominal as a controller reference, rebuilds TVLQR and static LQR locally, then runs one new hanging-start closed loop. `.working/n5-demo/live-metrics.json` records its recomputed controller and run metrics; `.working/n5-demo/n5-demo.gif` renders only its freshly integrated states.

```sh
uv sync --locked
uv run --locked n5-demo
uv run --locked n5-verify
uv run --locked ruff check .
uv run --locked python -m pytest -q
```

Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

`n5-demo` exposes one supported path for inspection: the `release.py` nominal loader → `tvlqr.py` and `lqr.py` → `dynamics.py:rollout_zoh` → `predicate.py` → the `release.py` renderer.

Acceptance is evaluated at 1 kHz control-boundary samples. A five-second hold is `5,001` consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

The run succeeds when every wrapped link angle stays within 5 degrees, each link rate within 0.5 rad/s, the cart within 2 m, and its speed within 0.5 m/s for the final sampled five-second hold.

`n5-verify` byte-checks the frozen numerical runtime, nominal, and four seed/stress JSON ledgers. It recomputes their counts, Wilson intervals, force limits, and cart checks. It identifies the ledgers as historical evidence, not perturbation reruns. Legacy ledger wording that says “held continuously” is immutable historical wording, not the live contract. This checkout lacks their embedded source commit `fcb4759529cb25a54485588ec58ee0a924939e99`, so `n5-verify` reports historical source provenance as unverified. N5 does not support nominal synthesis or perturbation reruns. N5 provides simulation evidence, not hardware or robustness evidence.
