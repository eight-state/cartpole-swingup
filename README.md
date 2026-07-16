# N6 cart-pole replay

`n6-demo` loads the fixed six-link nominal as its tracking reference, rebuilds TVLQR and static LQR locally, then integrates one fresh hanging-start closed loop. `.working/n6-demo/live-metrics.json` records its recomputed result; `.working/n6-demo/n6-demo.gif` renders only its fresh states.

```bash
uv sync --locked
uv run n6-demo
uv run n6-verify
```

The simulator applies 1 kHz zero-order hold with four 0.25 ms RK4 substeps and a 60 N force-magnitude limit. The fresh baseline reaches 5.684 s final continuous hold, 43.92637560427462 N peak applied force, and 4.582039973868613 m peak cart displacement.

`n6-verify` checks canonical bytes for the nominal and two historical gate JSONs, recomputes each declared 24/24 count and their 48/48 total, then runs the same fresh stack. The JSONs contain aggregate counts and omit trial rows. The aggregate-only JSONs prevent independent derivation of the 48/48 total. This checkout lacks embedded commit `6f5237819203bc4d9cd30037f06aff8a486e1ff5`, so the command reports historical source provenance as unverified.

N6 does not support nominal synthesis or perturbation reruns. N6 provides simulation evidence from an unperturbed live replay. N6 excludes hardware and robustness evidence.
