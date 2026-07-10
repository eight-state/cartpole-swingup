# Prior art and comparison boundary

This repository publishes one reproducible N11 simulation artifact with defined plant parameters, controller code, banked trajectories, gate records, hashes, and tests.

## Related Eight State releases

Eight State publishes the preceding cart-pole results in [quintuple-cartpole](https://github.com/eight-state/quintuple-cartpole), [sextuple-cartpole](https://github.com/eight-state/sextuple-cartpole), [septuple-cartpole](https://github.com/eight-state/septuple-cartpole), [octuple-cartpole](https://github.com/eight-state/octuple-cartpole), [nonuple-cartpole](https://github.com/eight-state/nonuple-cartpole), and [decuple-cartpole](https://github.com/eight-state/decuple-cartpole). The N11 package retains the decuple repository's shared runtime core and adds its own nominal and gate artifacts.

## Comparison requirements

Researchers comparing another result with N11 must match the plant, task, simulator, actuator bound, initial-condition distribution, feedback information, success predicate, trial count, seeds, and released code or data. A record that omits one of these fields does not support an ordered comparison.

The N11 evidence covers one cart-actuated eleven-link plant with full-state feedback in deterministic simulation. This repository makes no hardware, model-mismatch, or formal-guarantee claim.

## Comparison inputs

- `configs/env-n11.yaml`: plant and integrator parameters.
- `scripts/gate_preroll.py`: perturbed-initial-condition controller.
- `runs/r2/`: parent nominal, dense nominal, three gate records, and rendered demo.
- `scripts/release_audit.py`: hash, record, count, and interval checks.
- `docs/METHOD.md`: controller phases and predicate.
