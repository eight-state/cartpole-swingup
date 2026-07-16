# N11 cart-pole replay

`n11-demo` produces one fresh force-saturated simulation and locally recomputes its two feedback gains. `.working/n11/live-metrics.json` identifies the banked nominal as its loaded reference; `.working/n11/demo.gif` renders its fresh states.

```sh
uv sync --locked
uv run n11-demo
uv run n11-verify
```

The YAML authority defines eleven 0.5 m, 0.1 kg links on a 1 kg cart: 1 kHz ZOH, four 0.25 ms RK4 substeps, 150 N force-magnitude clipping, and full-state feedback. The run succeeds when every wrapped link angle stays within 5°, each link rate within 0.5 rad/s, the cart within 2 m, and its rate within 0.5 m/s for 5 continuous seconds.

`n11-verify` checks SHA-256 identity and metadata for the frozen sources, both banked nominals, and three banked gate records. It recomputes each record’s 24/24 count and the 72/72 aggregate, then reruns the unperturbed live stack. Gate records contain historical simulation evidence from earlier runs. N11 supports one unperturbed live-stack simulation and audits of those records; it excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness evidence.
