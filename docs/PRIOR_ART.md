# Prior art and comparison boundary

## Related Eight State releases

Eight State publishes the preceding cart-pole results in [quintuple-cartpole](https://github.com/eight-state/quintuple-cartpole), [sextuple-cartpole](https://github.com/eight-state/sextuple-cartpole), [septuple-cartpole](https://github.com/eight-state/septuple-cartpole), [octuple-cartpole](https://github.com/eight-state/octuple-cartpole), [nonuple-cartpole](https://github.com/eight-state/nonuple-cartpole), [decuple-cartpole](https://github.com/eight-state/decuple-cartpole), and [undecuple-cartpole](https://github.com/eight-state/undecuple-cartpole).

## Comparison requirements

Researchers comparing another result with N12 must match the plant parameters, task definition, integrator, actuator bound, initial-condition distribution, feedback inputs, success predicate, trial count, seeds, and released code or data. A record that omits one of these fields does not support an ordered comparison.

This repository reports a deterministic simulation with full-state feedback. It makes no hardware, model-mismatch, or formal-guarantee claim.
