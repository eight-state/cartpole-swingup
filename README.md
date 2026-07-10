# Duodecuple Cart-Pole

This repository reproduces the N12 cart-pole release from frozen artifacts in deterministic, force-saturated simulation.

![N12 cart-pole swing-up and balance](runs/r2/demo_n12.gif)

## Reproduce

```text
uv sync --locked
uv run n12-release
```

`n12-release` runs the artifact audit and the deterministic live verifier. The command exits nonzero when a release assertion fails.

```text
uv run n12-release --gate
```

`--gate` reruns the three fixed-seed perturbation gates and requires each regenerated JSON file to match its banked SHA256 value.

```text
uv run n12-demo
uv run ruff check .
uv run mypy
uv run pytest
```

`n12-demo` regenerates `runs/r2/demo_n12.gif` from the frozen nominal and live controller schedule.

## Repository map

- [`docs/METHOD.md`](docs/METHOD.md): plant, integrator, controller schedule, perturbation distribution, and success predicate.
- [`docs/RELEASE_EVIDENCE.md`](docs/RELEASE_EVIDENCE.md): gate totals, unperturbed witness, force boundary, promotion result, and statistical scope.
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): related Eight State releases and comparison requirements.
- [`PROVENANCE.md`](PROVENANCE.md): source origin, digest ledger, and license.
- [`artifacts/MANIFEST.json`](artifacts/MANIFEST.json): audited release paths and SHA256 values.

## License

MIT. See [LICENSE](LICENSE).
