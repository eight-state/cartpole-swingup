# CartPole capsules

This repository preserves the ten public CartPole swing-up releases for n=5 through n=14 without changing their evidence. No n=15 release exists yet.

Each release is imported as a frozen capsule. Its source, lockfile, numerical artifacts, verifier, tests, and historical prose stay together at their original revision. The root tools check those bytes and invoke each capsule's verifier, but they do not replace its verdict.

## What the claims mean

- **Success** means the capsule's committed sampled predicate passed. The exact predicate and hold measurement remain capsule-specific.
- **Proof** means one deterministic run passed a fixed verifier. It does not mean a mathematical proof of stability, robustness, or reachability.
- **Historical evidence** means committed records are checked for integrity and internal consistency. A verifier does not rerun historical trials unless its own documentation says it does.
- **Exact** can mean byte-identical replay, exact zero-order-hold discretization, or an exact equilibrium start. Each use must name which meaning applies.
- **Reproducible** is bounded by the capsule's documented numerical host and dependencies. Trailing floating-point digits are not assumed portable.

The releases do not claim hardware validation, a region of attraction, global robustness, or formal guarantees.

## Preserved releases

| Rung | Source release | Evidence boundary |
|---|---|---|
| n=5 | [quintuple](https://github.com/eight-state/quintuple-cartpole) | Historical ledgers and fresh replay |
| n=6 | [sextuple](https://github.com/eight-state/sextuple-cartpole) | Historical ledgers and fresh replay |
| n=7 | [septuple](https://github.com/eight-state/septuple-cartpole) | Historical ledgers, logs, and fresh replay |
| n=8 | [octuple](https://github.com/eight-state/octuple-cartpole) | Historical ledgers and fresh replay |
| n=9 | [nonuple](https://github.com/eight-state/nonuple-cartpole) | Historical ledgers and fresh replay |
| n=10 | [decuple](https://github.com/eight-state/decuple-cartpole) | Historical ledgers and fresh replay |
| n=11 | [undecuple](https://github.com/eight-state/undecuple-cartpole) | Historical ledgers and fresh replay |
| n=12 | [duodecuple](https://github.com/eight-state/duodecuple-cartpole) | Historical aggregate and fresh replay |
| n=13 | [tredecuple](https://github.com/eight-state/tredecuple-cartpole) | Source closure and archived verifier run |
| n=14 | [quattuordecuple](https://github.com/eight-state/quattuordecuple-cartpole) | Artifact hashes and exact replay gate |

n=5 and n=6 use a 60 N release boundary. Later releases use their own committed bounds and saturation rules. n=14 rejects over-limit controls before replay and its retained witness does not use clipping. The consolidated tools never replace these rules with one global force policy.

## Use the registry

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run cartpole-capsule list
uv run cartpole-capsule check
uv run cartpole-capsule verify 5
```

`check` compares every registered capsule with the import manifest. `verify` enters the selected capsule, runs its locked setup, then runs its own verification command. Run a capsule's shipped command directly when reviewing a release claim.

Complete source checkouts are the supported distribution. This project does not publish the capsules as wheels, source distributions, or package-index releases because their evidence depends on repository-relative files.

## Adding a future capsule

A future n=15 result starts as a new capsule with its own immutable evidence and verifier. It does not inherit a claim from n=14 or from the clean root tooling. The acceptance requirements are documented in `docs/future-capsules.txt`.

## Evidence safety

Files inside registered capsule directories are immutable imports. Corrections ship as a new capsule generation rather than edits to historical bytes. This rule preserves n=13 paths under its tracked `.working/` directory and n=14 files covered by its source lock.

The root `/.working/` directory is disposable and ignored. Nested capsule `.working/` directories are evidence and remain tracked.

## License

The registry tooling is MIT licensed. Every imported capsule retains its original license and copyright notice.
