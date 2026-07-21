# Quattuordecuple cart-pole

A deterministic swing-up and balance witness for fourteen undamped links on one cart.

The checked witness starts from exact hanging, applies 22,009 raw 1 kHz controls, stays inside the ±150 N force and ±10 m rail limits, and finishes with 13,811 consecutive states inside the locked success set. Every 1 ms transition is integrated as four recursive 0.25 ms RK4 steps.

## Verify

```bash
uv sync --locked
uv run n14-verify
```

A passing run prints `"verdict": "PASS"`. The verifier reads only the frozen raw controls, reconstructs the locked plant, replays from exact hanging, audits force and quarter-step rail limits, and recomputes the success run.

Run the complete local gate with:

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv run n14-release
```

`n14-release` writes the recomputed report to `artifacts/verification.json`.

## Locked result

- Fourteen links: 0.10 kg and 0.50 m each
- Cart: 1.0 kg
- Damping: zero
- Force bound: ±150 N
- Rail bound: ±10 m
- Control: 1 kHz zero-order hold
- Integration: four recursive 0.25 ms RK4 steps per control
- Switch tick: 6,009
- Peak raw force: 87.54201341513544 N
- Quarter-step cart peak: 7.098485449991383 m
- Longest success run: 13,811 states / 13.81 s

This release proves one deterministic trajectory. It does not claim robustness to perturbations or a global region of attraction.

See [METHOD](docs/METHOD.md), [RELEASE_EVIDENCE](docs/RELEASE_EVIDENCE.md), and [PROVENANCE](PROVENANCE.md).
