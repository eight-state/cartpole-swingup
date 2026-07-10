# Method

## Plant and simulator

`configs/env-n11.yaml` defines a 1.0 kg cart, eleven 0.1 kg links of length 0.5 m, zero damping, a 150 N force bound, and a 10 m track half-length. `NLinkCartPole` advances the state at 1 kHz with zero-order-held control. Each control tick contains four 0.25 ms RK4 steps.

The coordinates are cart position, eleven link angles, cart velocity, and eleven link rates. An angle of zero represents upright. An angle of pi represents hanging.

## Nominal trajectory

The release stores a 2,500-node, 4 ms parent nominal in `runs/r2/nom_n11_4ms_capture025_smoke3t03.npz`. It stores the corresponding 10,000-tick dense nominal in `runs/r2/nom_n11_dense1ms_capture025_smoke3t03.npz`. The dense artifact carries the parent control as four 1 ms zero-order-held ticks per parent interval.

`tests/test_nominal_consistency.py` checks every 1 ms dense step with the release simulator and compares every fourth dense state with its parent node. Those two checks use the released schedule of sixteen 0.25 ms RK4 substeps per 4 ms parent interval. The test also enforces the 40 N feedforward margin.

`reproduce_n11.py` reports one direct 4 ms RK4 step as a diagnostic. That single-step calculation uses a different integration schedule, so the release does not treat its difference as a transcription defect or an open-loop fidelity claim.

## Controller

The unperturbed reproduction constructs `FastDTVLQR` on the fixed dense nominal. The controller clips each demand to the 150 N bound. At 10.0 s, `static_lqr` takes over around the upright equilibrium. The reproduction watches a 12.0 s hold rollout and requires a continuous 5.0 s predicate run.

`reproduce_n11.py` separately constructs `DiscreteTVLQR` with exact zero-order-hold discretization and reports its monodromy spectral radius. `tests/test_discrete_tvlqr_monodromy.py` requires that radius to remain below one.

## Perturbed-initial-condition gate

`scripts/gate_preroll.py` samples each state coordinate from N(0, 0.02) around the hanging start. The gate uses the following fixed settings:

| Setting | Value |
|---|---:|
| Link count | 11 |
| Pre-roll cap | 9.0 s |
| `PREROLL_TOL` | 0 |
| `PREROLL_VEL_Q_SCALE` | 4 |
| Gate trials per seed | 24 |
| Seeds | 12345, 777, 2024 |
| Hold window | 10.0 s |
| Required continuous hold | 5.0 s |
| Gate workers | 6 |

The controller first applies an LQR gain around hanging for the full cap. It then applies the fixed dense-nominal `FastDTVLQR` tracker for 10.0 s and the static upright LQR for the hold window. `PREROLL_TOL=0` keeps the gate in the pre-roll loop until the 9.0 s cap.

## Success predicate and audit

`cartpole_race.funnels.in_success_set` evaluates the state predicate: each link angle has magnitude at most 5 degrees, each link rate has magnitude at most 0.5 rad/s, cart position has magnitude at most 2 m, and cart velocity has magnitude at most 0.5 m/s. A successful trial holds this set continuously for at least 5.0 s and keeps the cart within the 10 m track bound through pre-roll, tracking, and hold.

`scripts/release_audit.py` re-derives the 24 of 24 count from each JSON record, sums the three gates to 72 of 72, and re-derives the Wilson interval. It also checks the five committed SHA256 digests before accepting a result.
