# N8 cart-pole fresh closed-loop replay

`n8-demo` runs one fresh, unperturbed eight-link swing-up and balance simulation and writes its state trace, controls, metrics, evidence audit, and GIF under `.working/n8-demo/`.

```bash
uv sync --locked
uv run --locked n8-demo
uv run --locked n8-verify
```

The controller loads the fixed 1 ms dense nominal as its reference. `n8-demo` writes locally integrated states and locally rebuilt controller outputs from the saturated simulator. The verifier separately reports the fixed 4 ms parent nominal's residual. Generated outputs never change tracked files.

`n8-verify` writes `.working/n8-verify/summary.json`. The report contains six authority hashes, nominal defects, live controller metrics, and the elapsed-time success predicate. It derives row counts from four banked gate JSONs and classifies them as historical evidence.

## Scope

`n8` supports deterministic simulation replay. It excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness claims.
