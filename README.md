# N8 cart-pole fresh closed-loop replay

`n8-demo` runs one fresh, unperturbed eight-link swing-up and balance simulation and writes its state trace, controls, metrics, evidence audit, and GIF under `.working/n8-demo/`.

```bash
uv sync --locked
uv run --locked n8-demo
uv run --locked n8-verify
```

Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

Acceptance is evaluated at 1 kHz control-boundary samples. A five-second hold is `5,001` consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

The controller loads the fixed 1 ms dense nominal as its reference. `n8-demo` writes locally integrated states and locally rebuilt controller outputs from the saturated simulator. The verifier separately reports the fixed 4 ms parent nominal's residual. Generated outputs never change tracked files.

`n8-verify` writes `.working/n8-verify/summary.json`. The report contains six authority hashes, nominal defects, live controller metrics, and the sampled success predicate. It derives row counts from four banked gate JSONs and classifies them as historical evidence. The four banked records declare commit `5d3c2a7e7386e9b1477960436d0fc0e1a9794ad4`, absent from the reviewed public checkout. They are byte-preserved, internal-consistency evidence, not source-traceable or rerunnable runs.

## Scope

`n8` supports deterministic simulation replay. It excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness claims.
