# Quattuordecuple cart-pole

A deterministic swing-up and balance witness for fourteen undamped links on one cart.

The checked witness starts from exact hanging, applies 22,009 raw 1 kHz controls, stays inside the ±150 N force and ±10 m rail limits, and finishes with 13,811 consecutive states inside the locked success set. Every 1 ms transition is integrated as four recursive 0.25 ms RK4 steps.

## Verify

Verification is supported only from a complete source capsule containing:

```text
pyproject.toml
artifacts/MANIFEST.json
artifacts/source-sha256.json
artifacts/n14-witness.npz
artifacts/expected-witness.json
artifacts/provenance.json
src/n14_cartpole/verifier.py
```

```bash
uv sync --locked
uv run n14-verify
```

Before loading the witness or replaying any control, both `n14-verify` and `n14-release` require that capsule and audit the exact three immutable artifact rows and 21 source-lock rows. The executing verifier must be the capsule's `src/n14_cartpole/verifier.py`.

A PASS on `n14-verify` prints one JSON report and exits 0. `n14-verify --output PATH` writes that same report atomically only on PASS; a FAIL prints one JSON report to stdout, exits 1, and leaves `PATH` untouched. `n14-release` applies the same rule to `artifacts/verification.json`. Invalid command syntax uses argparse usage on stderr and exits 2; an unexpected post-parse error emits one JSON `ERROR` report and exits 2.

An installed wheel intentionally fails closed: both console commands emit a `FAIL` report containing `source_capsule_required` and exit 1. Wheels are not a supported verification environment.

Run the complete local gate with:

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv run n14-release
```

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

A PASS is a deterministic replay of the frozen witness under the audited local source capsule. It is not a robustness, region-of-attraction, hardware, or independent publication-identity claim.

Local Git history and in-tree locks establish reproducible current-tree consistency. They do not establish identity for an external reviewer until a commit, signed tag, or archive digest is published through a separately trusted channel, and they cannot prevent deliberate co-modification of executable code with mutable in-tree locks. This repair deliberately adds neither signing nor archive-authority machinery.

See [METHOD](docs/METHOD.md), [RELEASE_EVIDENCE](docs/RELEASE_EVIDENCE.md), and [PROVENANCE](PROVENANCE.md).
