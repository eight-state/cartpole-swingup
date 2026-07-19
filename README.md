# N7 cart-pole fresh closed-loop replay

`n7-demo` tracks the fixed seven-link nominal from exact hanging, applies the saturated simulator, checks the five-second hold, and writes a fresh GIF and metrics under `.working/n7-demo/`.

```sh
uv sync --locked
uv run n7-demo
uv run n7-verify
```

Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

Acceptance is evaluated at 1 kHz control-boundary samples. A five-second hold is `5,001` consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

The nominal supplies reference states and feedforward controls. `n7-demo` writes locally integrated states and locally rebuilt controller outputs. It never uses a saved rollout for displayed motion or changes tracked files.

`n7-verify` writes `.working/n7-verify/verification.json`. The report contains six authority hashes, the nominal exact-ZOH residual, and the live predicate result. `n7-verify` derives row counts from three banked JSON records and four historical logs and classifies those records as historical evidence.

The success predicate requires every link within 5 degrees and 0.5 rad/s of upright, cart position within 2 m, cart velocity within 0.5 m/s, and the final sampled five-second hold. The 10 m cart-travel bound is a sampled acceptance envelope.

## Scope

`n7` supports deterministic simulation replay. It excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness claims.

This repository publishes its MIT license in [LICENSE](LICENSE).
