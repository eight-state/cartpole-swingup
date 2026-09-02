"""Shared numerical core for the cart-pole capsules.

Single home of the dynamics, spec, LQR/TVLQR controllers, success predicate,
renderer, and fast gate pieces. Modules import each other by the
``cartpole_capsules.core`` name; per-rung glue builds on this package.
"""

from cartpole_capsules.core.discrete_tvlqr import DiscreteTVLQR, zoh_ab
from cartpole_capsules.core.dynamics import NLinkCartPole, Policy
from cartpole_capsules.core.env_spec import CartPoleSpec, load_spec
from cartpole_capsules.core.fast_pieces import FastDTVLQR, check_bitexact_densify, make_densifier
from cartpole_capsules.core.lqr import (
    Q_ANG_VEL,
    Q_ANGLE,
    Q_CART_POS,
    Q_CART_VEL,
    R_STATIC,
    StaticLQRPolicy,
    make_Q,
    make_R,
    static_lqr,
    wrap_state_error,
    wrap_to_pi,
)
from cartpole_capsules.core.predicate import (
    ANGLE_TOLERANCE_RAD,
    ANGULAR_RATE_TOLERANCE_RAD_S,
    CART_POSITION_TOLERANCE_M,
    CART_SPEED_TOLERANCE_M_S,
    evaluate_success_predicate,
    in_success_set,
    longest_hold_s,
    trailing_hold_s,
)
from cartpole_capsules.core.render import render_cartpole_gif
from cartpole_capsules.core.rollout import RolloutRecord, replay_controls, run_policy
from cartpole_capsules.core.tvlqr import TVLQR, build_upright_tvlqr

__all__ = [
    "ANGLE_TOLERANCE_RAD",
    "ANGULAR_RATE_TOLERANCE_RAD_S",
    "CART_POSITION_TOLERANCE_M",
    "CART_SPEED_TOLERANCE_M_S",
    "DiscreteTVLQR",
    "FastDTVLQR",
    "NLinkCartPole",
    "Policy",
    "Q_ANGLE",
    "Q_ANG_VEL",
    "Q_CART_POS",
    "Q_CART_VEL",
    "R_STATIC",
    "StaticLQRPolicy",
    "TVLQR",
    "CartPoleSpec",
    "build_upright_tvlqr",
    "check_bitexact_densify",
    "evaluate_success_predicate",
    "in_success_set",
    "load_spec",
    "longest_hold_s",
    "make_densifier",
    "make_Q",
    "make_R",
    "render_cartpole_gif",
    "RolloutRecord",
    "replay_controls",
    "run_policy",
    "static_lqr",
    "trailing_hold_s",
    "wrap_state_error",
    "wrap_to_pi",
    "zoh_ab",
]
