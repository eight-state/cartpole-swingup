# Prior art: multi-link inverted pendulums (n=10)

This is the accounting behind the claim that this repo is, to our knowledge,
the **first public n=10 cart-pole swing-up-and-balance artifact by any method**
(unperturbed closed-loop leg + a banked perturbed-IC pre-roll gate, 24/24 on
three seeds), and certainly the first open-source, code-reproducible one. The
full prior-art tables for n=5 and n=6 live in the sibling releases:

- n=5: https://github.com/eight-state/quintuple-cartpole
- n=6: https://github.com/eight-state/sextuple-cartpole (docs/PRIOR_ART.md)

The table applies with one more row; what changes at n=10:

| Work | System | Links | Task | Why distinct from this repo |
|---|---|---|---|---|
| Oh et al. 2025 (RL) | Cart | n=4 | Swing-up + balance | The standing **published** frontier by any method. RL; six links short. |
| Lam & Davison 2006 | Bottom-pivot torque chain (**not a cart**) | up to n=7 | **Balance only** | Different plant (base torque), different task (local stabilization, never swing-up). |
| yacine (@yacineMTB), 2026 | Cart (MuJoCo, pufferlib PPO) | n=6 | Swing-up + balance (RL) | Public-first at n=6 (conceded in the n=6 repo). No released code artifact at higher n seen. |
| Our n=5..8 releases | Cart | 5, 6, 7, 8 | Swing-up + balance | Predecessors; n=8 used a per-IC-replanning composite gate (24/24 × 2 seeds). |
| Our n=9 release (nonuple-cartpole) | Cart | n=9 | Swing-up + balance | The immediate predecessor: introduced the **pre-roll** gate (24/24 × 3 seeds, no per-IC NLP). |
| **This repo (decuple-cartpole)** | Cart (single 150 N force) | **n=10** | **Swing-up + balance** | To our knowledge no public n=10 cart-pole swing-up claim exists by ANY method. Reproducible from a clean clone in one command. Banked perturbed-IC pre-roll gate: 24/24 × 3 seeds, **same architecture as n=9 with zero re-tuning**, and a *gentler* (un-saturated, 98.6 N) catch than n=9. |

## Honest scope

Same boundary as the siblings: simulation only (1 kHz saturated ODE sim, not
hardware), full-state feedback, exact model, deterministic; robustness is
empirical (script-verified counts under a documented perturbation distribution
and committed predicate v1), not a theorem. The n=10 result deliberately reuses
the n=9 pre-roll controller unchanged; the only new engineering was a tighter
NLP exit rule on the nominal solve (see METHOD §2). The "first" claim is "first
public artifact we could find," dated 2026-07-06 (first n=10 σ=0.02 gate pass);
it is falsifiable by counter-example and we will concede priority exactly as the
n=6 repo did if a prior public n=10 claim surfaces.
