# Release evidence

The retained control sequence has 22,009 controls and starts from exact hanging.

| Check | Result |
|---|---:|
| Replay state delta against the retained composed run | 0.0 |
| Peak raw force | 87.54201341513544 N |
| Quarter-step cart peak | 7.098485449991383 m |
| Longest success run | 13,811 states |
| Longest success duration | 13.81 s |
| First tick of longest run | 8,199 |
| Force clipping | none |
| State injection or repaired nodes | none |

The saved-control replay regenerated every state exactly. A second source-only construction recomputed the dense reference, affine tracker, feedforward terms, feedback terms, controls, states, success flags, weighted hold gain, and hold rollout. Every compared array was bit-for-bit identical to the retained candidate.

Machine-readable hashes and reproduction facts are in `artifacts/provenance.json`. `artifacts/verification.json` records the original release host and its exact replay metrics.

## Acceptance boundary

The live verifier requires the frozen artifact and source hashes, exact hanging start, finite controls and states, the force and quarter-step rail limits, and at least 5,001 trailing success states. It reports exact comparisons with the retained host metrics as diagnostics.

The raw-control replay is numerically host-sensitive. A compatible host returns PASS. Other hosts can diverge enough to fail the live physical gate. `artifacts/verification.json` records the release host and does not establish cross-platform replay equivalence. GitHub Actions checks the artifact and source authority boundary plus non-replay behavior; it does not certify the numerical trajectory.
