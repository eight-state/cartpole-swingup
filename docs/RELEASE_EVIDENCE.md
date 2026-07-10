# N12 release evidence

## Perturbed gate

The banked gate records contain 24 successful trials for each fixed seed: 12345, 777, and 2024. The individual records sum to 24 + 24 + 24 = 72 successes from 72 trials. Direct Wilson arithmetic gives the recorded 95 percent interval `[0.862, 1.000]` for each 24 of 24 gate.

Successful perturbed trials reach the 150 N applied-force boundary. The records store applied force after saturation, so they do not establish a 150 N bound on every raw perturbed controller demand.

## Unperturbed witness

The deterministic verifier executes one 21.700 s live rollout from exact hanging. At the controller switch, it measures a 0.0745564116 degree maximum wrapped link angle and a 0.0184142239 rad/s maximum link rate. Across the rollout, it measures a 36.3255632667 N raw and applied peak, no clipping, a 6.6084840785 m cart peak, and a 12.0 s continuous hold in the locked success set.

## Promotion result

The separate 0.05 degree promotion screen fails. It measures a 1.6396136559 degree maximum reset-reference wrapped-link error through the tracker prefix, first crossing the screen at tick 3534.

## Scope

The perturbation evidence covers the three named seeds, 24 draws per seed, the distribution and controller contract in `METHOD.md`, and this numerical environment. It does not establish broader robustness.

GitHub Actions evaluates source identity, schedule, boundary events, and success predicates on Windows Server. `artifacts/verification.json` retains the Windows 11 floating-point measurements quoted above.
