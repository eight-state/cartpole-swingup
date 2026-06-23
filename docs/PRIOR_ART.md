# Prior art: multi-link inverted pendulums

This table is the accounting behind the novelty claim: this repo is the **first
public, code-reproducible n=5 cart-pole swing-up-and-balance artifact**. It lists
the prior n=5 and multi-link results and shows how each one differs. Every row
has been verified against primary sources (full text, video frames, author code
profiles) as of 2026-06-11.

"Cart" below means a single horizontally-actuated cart carrying a serial chain of
passive links (the classic underactuated cart-pole). "Bottom-pivot torque" means
a base joint is actuated directly (a different plant, not a cart). "Isaac stack"
means an Isaac Sim / Isaac Gym RL training stack rather than a hand-built
ODE/EOM simulator.

| Work | System | Links | Task | Medium | Public code? | Why distinct from this repo |
|---|---|---|---|---|---|---|
| Glück, Eder & Kugi 2013 (*Automatica* 49(3):801 to 808) | Cart | n=3 (triple) | Swing-up **and** balance | Hardware (real rig, experimental validation) | No controller code published | Peer-reviewed cart **hardware** swing-up, the strongest cart swing-up benchmark in this list. Stops at the triple; no public reproducible artifact. This repo is sim-only but reaches n=5 with a one-command repro. |
| Lam & Davison 2006 (*ACC*, "The real stabilizability radius of the multi-link inverted pendulum") | Bottom-pivot torque (base-actuated chain, **not a cart**) | up to n=7 | **Balance / stabilization only** (no swing-up) | Simulation | No | Different plant (base torque, not a cart force) and a different task (local stabilizability radius, never a swing-up). Establishes that n≥5 *balance-only* sim exists, on a non-cart system. |
| "The quintuple inverted pendulum" YouTube video, id `WwR92kx6tcA` (Stepan Ozana, VSB-TU Ostrava, ~2022) | **Cart** (verified from video frames: five-link chain on a translating cart; joint traces sweep from hanging to upright). The description's "two-degree-of-freedom structure" is the 2-DOF *control* architecture (feedforward + time-varying LQR tracking), not the mechanism. | n=5 (quintuple) | Closed-loop swing-up **and** stabilization (per the description, consistent with the frames) | Video of a Matlab/Simulink simulation | **No**: nothing on his GitHub (single NMPC repo), smartcontrols.cz (downloads dead), or MATLAB File Exchange; his 2024 ResearchSquare manuscript (DOI 10.21203/rs.3.rs-4319123/v1) covers IPEN3/IPEN4 only, with zero IPEN5 mentions and zero code links. | The feat was **demonstrated** here first, in simulation, code-less. It is **not reproducible**, which is the gap this repo fills: the claim is the artifact, not the feat. |
| Kotelovych et al. 2024 (Springer, "Stabilization of a Quintuple Inverted Pendulum System in Isaac Sim") | Cart (full text verified): single horizontal cart force (`F_ext(t)` on the cart coordinate in its Eq. 15), simulated in Isaac Sim | n=5 (quintuple) | **Balance / stabilization only** (full text verified): no swing-up phase anywhere in the text; near-upright LQR | Simulation (Isaac Sim only, no hardware) | **No**: no supplementary code, data, or repository links in the chapter or author profiles | n=5 **stabilization** (near-upright LQR) in Isaac Sim. An n=5 quintuple cart-pole sim exists in the literature. It is **balance**, not swing-up, and not a standalone reproducible cart swing-up artifact. |
| **This repo (quintuple-cartpole)** | Cart (single underactuated cart) | **n=5 (quintuple)** | **Swing-up and balance** | Simulation (1 kHz ODE sim, force-saturated) | **Yes** (saved nominal + one-command closed-loop validation + strict predicate + tests) | Public, code-reproducible n=5 cart swing-up-and-balance with a committed nominal, saturated closed-loop validation against a strict committed predicate, and one-command repro. No prior public artifact known to the author combines all of: cart + n=5 + swing-up + released reproducible code. |

## Where each prior result sits

Every prior n=5 result sits on the wrong side of at least one axis this artifact
gets right. The Ozana video demonstrates cart-driven n=5 swing-up in simulation
(the feat itself predates this repo), but it ships no code and no quintuple-specific
paper exists. Kotelovych is balance rather than swing-up. Lam & Davison reach n=7 in sim, on a base-torque chain rather than a
cart, and balance only. The strongest peer-reviewed **cart** swing-up on a real
rig is the triple (Glück et al. 2013); more links in sim is a different
achievement from fewer links on hardware. The robustness here is empirical:
script-verified counts under a documented perturbation distribution and predicate,
not a theorem.

What survives all of that is the **artifact**: a saved n=5 cart swing-up nominal
that a saturated closed-loop validation reproduces against a strict
committed predicate, runnable from a clean clone in one command.

## Sources

- Glück, T., Eder, A., Kugi, A. "Swing-up control of a triple pendulum on a cart
  with experimental validation." *Automatica* 49(3):801 to 808, 2013.
- Lam, J., Davison, E. J. "The real stabilizability radius of the multi-link
  inverted pendulum." *Proc. American Control Conference (ACC)*, 2006.
  (Bottom-pivot torque chain, stabilization only, up to n=7, simulation.)
- Ozana, S. "The quintuple inverted pendulum." YouTube video id `WwR92kx6tcA`,
  ~2022. (Matlab/Simulink simulation; cart-driven swing-up and stabilization,
  verified from the description and video frames; no public code. His 2024
  ResearchSquare manuscript, DOI 10.21203/rs.3.rs-4319123/v1, covers IPEN3/IPEN4
  only and links no code.)
- Kotelovych et al. "Stabilization of a Quintuple Inverted Pendulum System in
  Isaac Sim." Springer, 2024. DOI 10.1007/978-3-031-70959-3_9. (Full text
  verified: n=5 near-upright LQR stabilization on a horizontally actuated cart,
  Isaac Sim only, no swing-up phase, no supplementary code.)
