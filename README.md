# N-link CartPole swing-up

One shared Python implementation reproduces the CartPole swing-up results for n=5 through n=14. The repository has one package, one lockfile, one rung registry, and one verification command. No n=15 result exists yet.

The numerical model, integration, LQR, continuous TVLQR, discrete TVLQR, predicates, rendering, and evidence audits are shared. Five small adapters preserve the controller differences that affect results. Every other rung difference is data in `rungs.toml`.

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run cartpole list
uv run cartpole check
uv run cartpole verify 5
uv run cartpole verify all
uv run cartpole demo 14
```

`check` verifies the original release tags and retained evidence without running a simulation. `verify` adds a fresh shared-code replay. `demo` verifies one rung and writes a GIF under the ignored root `.working/` directory.

## Repository layout

```text
src/cartpole_capsules/core/       shared plant and controller math
src/cartpole_capsules/adapters/   five release-specific strategy families
rungs.toml                        all per-rung constants and evidence identities
rungs/n05-* ... n14-*             numerical evidence only
tests/                            shared and parameterized tests
```

No rung has its own Python package, dependency lock, workflow, or copy of the dynamics.

## Controller progression

| Rungs | Shared strategy | Main difference |
|---|---|---|
| n=5 to n=6 | Continuous TVLQR | 60 N force bound |
| n=7 to n=11 | Discrete TVLQR | Hold and handoff settings |
| n=12 | Densified discrete TVLQR | Tick-9700 switch |
| n=13 | Affine tracker and static hold | Preserved proof payload |
| n=14 | Controls-only replay | Pre-replay force rejection |

A new rung reuses an existing adapter unless its controller math is genuinely different. The planned n=15 rung therefore starts as configuration and evidence, not another repository.

## Evidence boundaries

The shared verifier keeps three checks separate:

1. Original release identity from the `legacy/nXX-*` Git tags.
2. Historical evidence hashes and internal consistency.
3. Fresh numerical replay through the shared implementation.

Historical perturbation records are audited, not rerun. n=13 and n=14 remain host-sensitive numerical results. n=14 CI checks its retained authority on Windows and does not turn that result into a cross-platform guarantee.

The force policy is data, not a global assumption. n=5 and n=6 use a 60 N bound. n=7 through n=14 use 150 N. n=14 rejects over-limit controls before replay instead of clipping them.

## Claim language

- **Success** means the rung's committed sampled predicate passed.
- **Proof** means one deterministic run passed a fixed verifier. It is not a mathematical proof.
- **Historical evidence** means frozen records passed hash and consistency checks.
- **Reproducible** remains bounded by the documented numerical host and dependencies.

These releases do not claim hardware validation, global robustness, a region of attraction, or formal stability guarantees.

## Preserved history

The exact standalone releases remain reachable at `legacy/n05-quintuple` through `legacy/n14-quattuordecuple`. Their full source, tests, lockfiles, documentation, and Git histories stay in this repository even though duplicate project files are absent from `main`.

The old GitHub repositories remain unchanged until this monorepo is public and independently checked. Repository deletion is a separate irreversible action.

## License

MIT. Historical tags retain their original copyright notices.
