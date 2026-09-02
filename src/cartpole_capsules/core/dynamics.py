"""Single source-of-truth n-link cart-pole dynamics (CasADi Lagrangian).

The only place equations of motion exist. Collocation, linearization,
LQR/TVLQR, RK4 simulation, and rendering consume the :class:`ca.Function`
objects and methods built here. No handwritten NumPy dynamics, no second
simulator model.

Coordinate convention: ``q = [x_cart, theta_1..theta_n]``, ``state = [q, qdot]``,
length ``2 * (n + 1)``. ``theta_i = 0`` means link ``i`` points vertically up,
``theta_i = pi`` means down. Angles are absolute world angles.

Control: scalar horizontal cart force ``u`` (Newtons), applied only to the cart
coordinate. ``u`` is never clipped inside the symbolic dynamics; force
saturation happens only at controller and simulator boundaries so
linearization stays honest.
"""

from __future__ import annotations

from collections.abc import Callable

import casadi as ca
import numpy as np

from cartpole_capsules.core.env_spec import CartPoleSpec

# A control policy maps (state, time) -> scalar force. Numpy in, float out.
Policy = Callable[[np.ndarray, float], float]


class NLinkCartPole:
    """n-link cart-pole dynamics built once as a CasADi symbolic graph.

    The constructor assembles the Lagrangian equations of motion symbolically
    and compiles reusable :class:`ca.Function` objects for the continuous
    dynamics ``f(x, u)`` and its Jacobians. All numeric methods (``f``,
    ``linearize``, ``rk4_step``, ``rollout_zoh``) route through these compiled
    functions so the EOM is defined exactly once.
    """

    def __init__(self, spec: CartPoleSpec) -> None:
        """Build the symbolic dynamics graph for ``spec``."""
        self.spec = spec
        self.n = spec.n_links
        self.nq = spec.nq
        self.nx = spec.nx

        self._build_symbolic()

    # ------------------------------------------------------------------
    # Symbolic construction (runs once per instance)
    # ------------------------------------------------------------------
    def _build_symbolic(self) -> None:
        """Assemble the Lagrangian EOM and compile CasADi functions."""
        n = self.n
        spec = self.spec

        # Configuration, velocity, control symbols.
        q = ca.SX.sym("q", self.nq)  # [x_cart, theta_1..theta_n]
        qd = ca.SX.sym("qd", self.nq)
        u = ca.SX.sym("u")  # scalar cart force

        x_cart = q[0]
        xd_cart = qd[0]
        theta = [q[1 + i] for i in range(n)]
        thetad = [qd[1 + i] for i in range(n)]

        m_cart = spec.cart_mass_kg
        m = list(spec.link_masses_kg)
        ll = list(spec.link_lengths_m)
        g = spec.gravity_m_s2
        b_cart = spec.damping_cart_n_s_m
        b_link = list(spec.damping_links_n_m_s_rad)

        # --- Kinetic energy ------------------------------------------------
        # Cart (horizontal DOF only).
        T = 0.5 * m_cart * xd_cart**2

        # Cumulative base position of each link's proximal joint.
        # Link 0 hangs/stands off the cart pivot at (x_cart, 0). COM of link i:
        #   p_i.x = x + sum_{j<i} l_j*sin(theta_j) + 0.5*l_i*sin(theta_i)
        #   p_i.y =     sum_{j<i} l_j*cos(theta_j) + 0.5*l_i*cos(theta_i)
        # so theta=0 means COM above the pivot (y>0), matching "0 => up".
        base_x = x_cart
        base_y = ca.SX(0)
        for i in range(n):
            li = ll[i]
            # COM position.
            com_x = base_x + 0.5 * li * ca.sin(theta[i])
            com_y = base_y + 0.5 * li * ca.cos(theta[i])
            # COM velocity = d/dt of the above (chain rule via jacobian on q).
            com_vx = ca.jacobian(com_x, q) @ qd
            com_vy = ca.jacobian(com_y, q) @ qd
            v2 = com_vx**2 + com_vy**2

            I_i = m[i] * li**2 / 12.0  # slender rod about COM
            T = T + 0.5 * m[i] * v2 + 0.5 * I_i * thetad[i] ** 2

            # Advance cumulative base to the distal end of link i.
            base_x = base_x + li * ca.sin(theta[i])
            base_y = base_y + li * ca.cos(theta[i])

        # --- Potential energy ---------------------------------------------
        # V = sum m_i * g * (COM height). theta=0 (up) means higher V.
        V = ca.SX(0)
        base_y = ca.SX(0)
        for i in range(n):
            li = ll[i]
            com_y = base_y + 0.5 * li * ca.cos(theta[i])
            V = V + m[i] * g * com_y
            base_y = base_y + li * ca.cos(theta[i])

        L = T - V

        # --- Manipulator-form EOM via Lagrangian --------------------------
        # p = dL/dqd (generalized momenta), M = dp/dqd (mass matrix),
        # bias = (dp/dq) qd - dL/dq + damping, qdd = M^{-1}(Q - bias).
        p = ca.jacobian(L, qd).T  # (nq, 1)
        M = ca.jacobian(p, qd)  # (nq, nq), symmetric PD
        dL_dq = ca.jacobian(L, q).T  # (nq, 1)
        coriolis_grav = ca.jacobian(p, q) @ qd - dL_dq  # (nq, 1)

        # Damping (Rayleigh): cart linear + per-link rotational.
        damping = ca.vertcat(
            b_cart * xd_cart,
            *[b_link[i] * thetad[i] for i in range(n)],
        )

        # Generalized force: cart force on x only.
        Q = ca.vertcat(u, *[ca.SX(0) for _ in range(n)])

        rhs = Q - coriolis_grav - damping
        qdd = ca.solve(M, rhs)  # (nq, 1), symbolic linear solve

        # Continuous state derivative xdot = [qd, qdd].
        x = ca.vertcat(q, qd)
        xdot = ca.vertcat(qd, qdd)

        # Compiled functions.
        self._x = x
        self._u = u
        self.f = ca.Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])

        A_sym = ca.jacobian(xdot, x)
        B_sym = ca.jacobian(xdot, u)
        self._A_fn = ca.Function("A", [x, u], [A_sym], ["x", "u"], ["A"])
        self._B_fn = ca.Function("B", [x, u], [B_sym], ["x", "u"], ["B"])

        # Energy for sanity checks (total mechanical energy T + V).
        self._energy_fn = ca.Function("E", [x], [T + V], ["x"], ["E"])

    # ------------------------------------------------------------------
    # Equilibria
    # ------------------------------------------------------------------
    def x_equilibrium(self, kind: str = "up") -> np.ndarray:
        """Return the cart-centered equilibrium state.

        ``kind="up"`` means all links upright (theta_i = 0), ``kind="down"``
        means all links hanging (theta_i = pi). All velocities zero, cart at
        origin. Returns a state vector of length ``2 * (n + 1)``.
        """
        x = np.zeros(self.nx)
        if kind == "up":
            angle = 0.0
        elif kind == "down":
            angle = np.pi
        else:
            raise ValueError(f"kind must be 'up' or 'down', got {kind!r}")
        x[1 : 1 + self.n] = angle
        return x

    # ------------------------------------------------------------------
    # Numeric evaluation (all route through compiled functions)
    # ------------------------------------------------------------------
    def f_num(self, x: np.ndarray, u: float) -> np.ndarray:
        """Evaluate the continuous dynamics ``xdot = f(x, u)`` numerically.

        ``u`` is unclipped here. Returns a flat array of length ``nx``.
        """
        out = self.f(x=x, u=u)["xdot"]
        return np.asarray(out).reshape(-1)

    def linearize(self, x: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time Jacobians ``A = df/dx`` and ``B = df/du``.

        Returns ``(A, B)`` with shapes ``(nx, nx)`` and ``(nx, 1)``.
        """
        A = np.asarray(self._A_fn(x=x, u=u)["A"])
        B = np.asarray(self._B_fn(x=x, u=u)["B"]).reshape(self.nx, 1)
        return A, B

    def energy(self, x: np.ndarray) -> float:
        """Total mechanical energy ``T + V`` at state ``x`` (Joules)."""
        return float(self._energy_fn(x=x)["E"])

    def rk4_step(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        """One classical RK4 step of the continuous dynamics.

        Uses ``self.f`` exclusively. Control ``u`` is held constant across the
        step (zero-order hold).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        k1 = self.f_num(x, u)
        k2 = self.f_num(x + 0.5 * dt * k1, u)
        k3 = self.f_num(x + 0.5 * dt * k2, u)
        k4 = self.f_num(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout_zoh(
        self,
        x0: np.ndarray,
        policy: Policy,
        t_final: float,
        control_dt: float,
        rk4_max_step: float,
        seed: int | None = None,
        return_raw: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Deterministic zero-order-hold rollout (the only rollout in repo).

        The control is recomputed once per control tick of length
        ``control_dt`` and held constant. Within each tick the state is
        integrated with RK4 substeps no larger than ``rk4_max_step`` (the
        number of substeps is fixed by ceil so the schedule is deterministic).
        The cart force is clipped to ``+/- force_bound`` at this simulator
        boundary, never inside the symbolic dynamics.

        Args:
            x0: Initial state, length ``nx``.
            policy: Callable ``(state, t) -> force``.
            t_final: Total rollout duration (seconds).
            control_dt: Zero-order-hold control period (seconds).
            rk4_max_step: Maximum RK4 substep (seconds).
            seed: Unused; kept for interface stability (rollout is
                deterministic given its explicit arguments).
            return_raw: When True, also return the raw demanded force from
                the policy before saturation.

        Returns:
            ``(t_log, x_log, u_log)`` where ``t_log`` has shape ``(K+1,)``,
            ``x_log`` has shape ``(K+1, nx)`` (state at each control tick
            boundary, including the initial state), and ``u_log`` has shape
            ``(K,)`` (force applied during each tick). With
            ``return_raw=True`` a fourth array ``u_raw_log`` of shape ``(K,)``
            (raw demanded force before saturation) is appended.
        """
        del seed  # Rollout is deterministic in its explicit arguments.
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        fbound = self.spec.force_bound_n

        n_ticks = int(round(t_final / control_dt))
        # Fixed substep schedule per tick (deterministic, never exceeds max).
        n_sub = max(1, int(np.ceil(control_dt / rk4_max_step)))
        dt_sub = control_dt / n_sub

        x_log = np.empty((n_ticks + 1, self.nx), dtype=float)
        u_log = np.empty(n_ticks, dtype=float)
        u_raw_log = np.empty(n_ticks, dtype=float) if return_raw else None
        t_log = np.empty(n_ticks + 1, dtype=float)
        x_log[0] = x
        t_log[0] = 0.0

        for k in range(n_ticks):
            t = k * control_dt
            u_raw = float(policy(x, t))
            u = float(np.clip(u_raw, -fbound, fbound))
            for _ in range(n_sub):
                x = self.rk4_step(x, u, dt_sub)
            u_log[k] = u
            if u_raw_log is not None:
                u_raw_log[k] = u_raw
            x_log[k + 1] = x
            t_log[k + 1] = (k + 1) * control_dt

        if u_raw_log is None:
            return t_log, x_log, u_log
        return t_log, x_log, u_log, u_raw_log
