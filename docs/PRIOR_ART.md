# Prior art: multi-link inverted pendulums

This table is the accounting behind the claim that this repo is, to our
knowledge, the first open-source, code-reproducible, non-RL n=6 cart-pole
swing-up-and-balance artifact. It is explicitly not the first 6-link pendulum
solve and not the first public 6-pendulum cartpole. Where we could not verify a
claim (a code-less video on a blocked platform, or an RL post with no released
code), the row says so.

"Cart" below means a single horizontally-actuated cart carrying a serial chain of
passive links, the classic underactuated cart-pole. "Bottom-pivot torque" means a
base joint is actuated directly, a different plant. "RL stack" means a
reinforcement-learning training pipeline (MuJoCo + PPO) rather than an
inversion-based trajectory design.

| Work | System | Links | Task | Medium | Public code? | Why distinct from this repo |
|---|---|---|---|---|---|---|
| **yacine (@yacineMTB), 2026-06-09 public post** | Cart (MuJoCo cartpole) | **n=6** | Swing-up + balance (RL policy) | Simulation (MuJoCo, pufferlib (its standard PPO)) | No reproducible code artifact seen | Posted a 6-pendulum cartpole solve publicly first; we concede this. It is an RL/learned policy, not an inversion-based design, and we found no released reproducible code. This repo's distinction is the open-source, code-reproducible, non-RL artifact, not a priority claim. |
| Glück, Eder & Kugi 2013 (*Automatica* 49(3):801 to 808) | Cart | n=3 (triple) | Swing-up **and** balance | Hardware (real rig, experimental validation) | No controller code published | Peer-reviewed cart **hardware** swing-up, the strongest peer-reviewed cart swing-up benchmark we found. Stops at the triple; no public reproducible artifact. This repo is sim-only but reaches n=6 with a one-command repro. |
| Lam & Davison 2006 (*ACC*) | Bottom-pivot torque (base-actuated chain, **not a cart**) | up to n=7 | **Balance / stabilization only** | Simulation | No | Different plant (base torque, not a cart force) and a different task (local stabilizability radius, never a swing-up). Establishes that n≥5 balance-only sim exists, on a non-cart system. |
| "The quintuple inverted pendulum" YouTube video, id `WwR92kx6tcA` | Cart (Matlab/Simulink simulation) | n=5 | Closed-loop swing-up **and** stabilization | Video (Matlab/Simulink) | No | Stepan Ozana's 2022 quintuple simulation video. Cart-mounted (his lab's pendulum rigs are all cart-driven), swing-up + stabilization per the description; ships no code and is not reproducible. |
| Kotelovych et al. 2024 (Springer, "Stabilization of a Quintuple Inverted Pendulum System in Isaac Sim") | Cart (modelled with explicit cart mass and horizontal force), simulated in Isaac Sim | n=5 | **Balance / stabilization only**: near-upright LQR stabilization, not swing-up (stabilization-only per the abstract/title; full text not obtained) | Simulation (Isaac Sim) | Not a self-contained cart-pole swing-up artifact | n=5 stabilization (near-upright LQR) in Isaac Sim. An n=5 quintuple cart-pole sim exists in the literature; it is balance, not swing-up, and not a standalone reproducible cart swing-up artifact. |
| **This repo (sextuple-cartpole)** | Cart (single underactuated cart) | **n=6** | **Swing-up and balance** | Simulation (1 kHz ODE sim, force-saturated) | **Yes** (saved nominal + one-command closed-loop validation + strict predicate + tests) | Open-source, code-reproducible, non-RL n=6 cart swing-up-and-balance with a committed nominal, saturated closed-loop validation against a strict committed predicate, and one-command repro. To our knowledge no prior open-source, reproducible, non-RL artifact combines cart + n=6 + swing-up + released code. |

## Where each result sits

yacine (@yacineMTB) posted a 6-pendulum cartpole RL solve publicly first on
2026-06-09; that is conceded above, and we make no priority claim. yacine's is an
RL policy with no reproducible code artifact seen; this one is a saved nominal
plus released code plus a strict gate. The distinct contribution is the artifact's
nature, not its date: an open-source, code-reproducible, non-RL, inversion-based
n=6 cart swing-up and balance that a saturated closed-loop validation reproduces
against a strict committed predicate, runnable from a clean clone in one command.

The strongest peer-reviewed cart swing-up on a real rig is the triple (Glück et
al. 2013); more links in sim is not the same achievement as fewer links on
hardware. Robustness here is empirical (script-verified counts under a documented
perturbation distribution and predicate), not a theorem. The n=6 nominal is on the
native 1 ms grid and self-consistent by construction (cc-iLQG integrates the exact
1 ms 4-substep ZOH tick, 1-step ZOH defect 0.0; this closes the
node-spacing-vs-control-rate gap, not a continuous-dynamics accuracy claim), at
parity with n=4 and n=5; see the README "The n=6 delta" section.

## Sources

- yacine (@yacineMTB), public post, 2026-06-09: a 6-pendulum cartpole swing-up/
  balance trained with RL (pufferlib, its standard PPO) in MuJoCo. Posted
  publicly first; no reproducible code artifact seen.
- Glück, T., Eder, A., Kugi, A. "Swing-up control of a triple pendulum on a cart
  with experimental validation." *Automatica* 49(3):801 to 808, 2013.
- Lam, J., Davison, E. J. "The real stabilizability radius of the multi-link
  inverted pendulum." *Proc. American Control Conference (ACC)*, 2006.
  (Bottom-pivot torque chain, stabilization only, up to n=7, simulation.)
- "The quintuple inverted pendulum." YouTube video id `WwR92kx6tcA`. (Matlab/
  Simulink per the description; no public code. Cart-vs-rotary and
  swing-up-vs-balance rest on the title and description only.)
- Kotelovych et al. "Stabilization of a Quintuple Inverted Pendulum System in
  Isaac Sim." Springer, 2024. (n=5 stabilization in Isaac Sim: near-upright LQR
  balance, not swing-up; stabilization-only per the abstract/title, full text not
  obtained.)
