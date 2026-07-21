# Method

The released artifact contains the raw control sequence only. The verifier constructs the locked fourteen-link plant and starts from its exact hanging equilibrium. It rejects malformed, nonfinite, or over-limit controls before certifying the result.

For each 1 ms zero-order-hold control, the verifier performs four recursive 0.25 ms RK4 steps. Cart position is audited after every quarter step, not only at control boundaries. No force clipping, state injection, repaired replay node, or alternate simulator is used.

A state is successful when all four conditions hold:

- every wrapped link angle is within 5° of upright;
- every link rate is at most 0.5 rad/s in magnitude;
- cart position is at most 2 m in magnitude;
- cart rate is at most 0.5 m/s in magnitude.

Passing requires at least 5,001 consecutive 1 ms states. The retained witness has 13,811 consecutive successful states.

## Controller used to discover the witness

The swing-up reference was restored by hard-terminal direct collocation with exact dynamics constraints. A finite-horizon affine tracker used 1.6 seconds of lookahead with 100 ms overlapping strides. At tick 6,009, control transferred to a high-precision discrete LQR with state weights `Q_cart_pos=100`, `Q_cart_vel=1`, `Q_angle=80`, `Q_ang_vel=5`, and `R=0.02`.

The release verifier does not trust those construction steps. It certifies the retained raw controls directly against the locked physics and success contract.
