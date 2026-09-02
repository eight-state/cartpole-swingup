# N11 cart-pole replay

`n11-demo` produces one fresh force-saturated simulation and locally recomputes its two feedback gains. `.working/n11/live-metrics.json` identifies the banked nominal as its loaded reference; `.working/n11/demo.gif` renders its fresh states.

```sh
uv sync --locked
uv run n11-demo
uv run n11-verify
```

> Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

The YAML authority defines eleven 0.5 m, 0.1 kg links on a 1 kg cart: 1 kHz ZOH, four 0.25 ms RK4 substeps, 150 N force-magnitude clipping, and full-state feedback. Acceptance is evaluated at 1 kHz control-boundary samples: a five-second hold is 5,001 consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

`n11-verify` checks SHA-256 identity and metadata for the frozen sources, both banked nominals, and three banked gate records. It recomputes each record’s 24/24 count and the 72/72 aggregate, then reruns the unperturbed live stack. The gate records are historical stored summaries from earlier runs, not rerunnable trials. N11 supports one unperturbed live-stack simulation and audits of those records; it excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness evidence.
