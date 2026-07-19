# Duodecuple Cart-Pole

This repository runs the N12 controller in a fresh simulation from exact hanging.

![N12 cart-pole](runs/r2/demo_n12.gif)

## Run

```text
uv sync --locked
uv run n12-demo
uv run n12-verify
```

> Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

`n12-demo` writes `.working/n12-demo.gif`; `n12-verify` writes `.working/n12-verify.json`.

## What runs locally

`n12-demo` loads only the 4 ms nominal from `artifacts/nom_n12_4ms_fast.npz`; it generates the dense reference, gains, states, forces, measurements, and GIF locally. Acceptance is evaluated at 1 kHz control-boundary samples: a five-second hold is 5,001 consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

This repository does not support nominal synthesis or perturbation reruns. `artifacts/n12-evidence.json` is immutable historical content: its legacy continuous wording is not the present contract. Its 72 stored successful rows are internally consistent, but they name the absent historic nominal path `runs/r2/nom_n12_4ms_fast.npz` and retain neither a historic nominal digest nor primary trial inputs or traces.

## Read the code

The deterministic simulation path contains [`env_spec.py`](src/cartpole_race/env_spec.py), [`dynamics.py`](src/cartpole_race/dynamics.py), [`lqr.py`](src/cartpole_race/lqr.py), [`fast_pieces.py`](src/n12_cartpole/fast_pieces.py), [`simulator.py`](src/n12_cartpole/simulator.py), and [`success.py`](src/n12_cartpole/success.py). [`demo.py`](src/n12_cartpole/demo.py) renders the rollout; [`verifier.py`](src/n12_cartpole/verifier.py) audits it.

The claim covers deterministic simulation with full-state feedback. It excludes hardware, model mismatch, and formal guarantees.

This repository uses the [MIT License](LICENSE).
