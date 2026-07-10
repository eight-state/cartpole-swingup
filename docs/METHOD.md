# Method

## Plant and simulator

The release simulates a 1.0 kg cart with twelve 0.10 kg links of length 0.50 m, 9.81 m/s² gravity, zero damping, a 150 N applied-force bound, and a 10 m track half-length. `NLinkCartPole` advances the state at exactly 1 kHz under zero-order-held control. Each tick contains four 0.25 ms RK4 substeps.

The state contains cart position, twelve link angles, cart velocity, and twelve link rates. Zero link angle is upright. Pi radians is hanging.

## Nominal and controllers

`runs/r2/nom_n12_4ms_fast.npz` stores the 2,500-control, 4 ms N12 nominal. The release gate regenerates its 10,000-tick 1 ms reference through `make_densifier` with `REFERENCE_DENSIFY_STRIDE=4`. Each source control repeats for four 1 ms zero-order-held ticks.

The deterministic verifier starts at the exact hanging equilibrium. It runs Qv x0.25 TVLQR on ticks 0 through 9699, then the default static CARE controller on ticks 9700 through 21699. The verifier runs one 21.7 s live rollout and records raw and applied force separately.

## Perturbed gate

Each banked gate draws 24 initial conditions by sampling every state coordinate from N(0, 0.02) around hanging. The controller applies an LQR about hanging for the full 18.0 s cap, tracks the reset-densified reference for 9.7 s, then applies the static CARE controller for a 10.0 s hold window. `PREROLL_TOL=0` keeps every trial in pre-roll through the cap. `PREROLL_VEL_Q_SCALE=4` and `TRACKER_LINK_RATE_Q_SCALE=0.25` define the released gains.

| Setting | Value |
|---|---:|
| Links | 12 |
| Gate seeds | 12345, 777, 2024 |
| Trials per seed | 24 |
| Pre-roll cap | 18.0 s |
| Tracker handoff | Tick 9700, 9.700 s |
| Reference densification | Four 1 ms ticks per 4 ms source control |
| Hold window | 10.0 s |
| Required continuous hold | 5.0 s |

A successful trial keeps the cart inside the 10 m track bound and ends with a continuous 5.0 s run inside the locked upright set. The set requires each wrapped link angle at or below 5 degrees, each link rate at or below 0.5 rad/s, cart position at or below 2 m, and cart rate at or below 0.5 m/s.
