# N6 cart-pole replay

`n6-demo` loads the fixed six-link nominal as its tracking reference, rebuilds TVLQR and static LQR locally, then integrates one fresh hanging-start closed loop. `.working/n6-demo/live-metrics.json` records its recomputed result; `.working/n6-demo/n6-demo.gif` renders only its fresh states.

```bash
uv sync --locked
uv run n6-demo
uv run n6-verify
```

Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

Acceptance is evaluated at 1 kHz control-boundary samples. A five-second hold is `5,001` consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

The simulator applies 1 kHz zero-order hold with four 0.25 ms RK4 substeps and a 60 N force-magnitude limit. The fresh baseline reaches a 5.684 s final sampled hold, 43.92637560427462 N peak applied force, and 4.582039973868613 m peak cart displacement.

`n6-verify` checks canonical bytes for the nominal and two historical gate JSONs, validates declared 24/24 counts and their declared 48/48 total, then runs the same fresh stack. The historical JSONs authenticate aggregate declarations and their arithmetic, not trial outcomes; they contain aggregate counts and omit trial rows. This checkout lacks embedded commit `6f5237819203bc4d9cd30037f06aff8a486e1ff5`, so the command reports historical source provenance as unverified.

N6 does not support nominal synthesis or perturbation reruns. N6 provides simulation evidence from an unperturbed live replay. N6 excludes hardware and robustness evidence.
