"""Adapter contract and shared assessment helpers for the lean generation.

Adapters consume a :class:`RungConfig` loaded from one TOML registry
(``rungs.toml`` beside this package) and return raw demanded forces. The
simulator boundary, never the adapter, clips to ``force_bound_n``: matching
every historical capsule (linearization stays honest, saturation is measured,
not hidden).

Imports come from ``cartpole_capsules.core`` (the future lean spine). Nothing
in this package re-implements dynamics, spec, LQR, TVLQR, or predicates.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from cartpole_capsules.core.rollout import RolloutRecord

if TYPE_CHECKING:  # the lean spine; not importable in this scratch tree yet
    from cartpole_capsules.core.dynamics import NLinkCartPole
    from cartpole_capsules.core.env_spec import CartPoleSpec

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "rungs.toml"

# Optional repository-root override set by the CLI (--root).
ROOT_OVERRIDE: Path | None = None


# --------------------------------------------------------------------------
# Registry data model (one [rungs.nNN] table merged over [defaults])
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NominalSpec:
    """Identity of one frozen nominal artifact."""

    role: str  # "active" | "parent"
    path: str
    sha256: str
    n_nodes: int
    horizon_s: float
    strict_meta: bool = False  # n/force metadata asserted when True
    alias_dialect: bool = False  # n5/n6 npz: assert states==x, forces==u


@dataclass(frozen=True)
class GateSpec:
    """One banked gate record to audit (L1) without rerunning it."""

    file: str
    sha256: str
    seed: int | None
    dialect: str  # "rows" | "n12-evidence" | "todo-aggregate" | "todo-composite"
    controller: str | None = None
    nominal: str | None = None
    nominal_match: str = "name"  # "name" | "exact"
    header_fields: tuple[str, ...] | None = None
    row_fields: tuple[str, ...] | None = None
    handoff_limit_deg: float | None = None
    check_t_pre: bool = False
    check_tracker_ticks: bool = False
    require_saturation: bool = False  # n12: some row must hit the bound
    force_bound_n: float = 150.0
    hold_required_s: float = 5.0
    expected_trials: int | None = None
    expected_successes: int | None = None


@dataclass(frozen=True)
class ExpectedMetric:
    """Release-pinned scalar with an absolute tolerance."""

    value: float
    atol: float


@dataclass(frozen=True)
class RungConfig:
    """Everything a rung needs, from data only."""

    rung: int
    slug: str
    legacy_tag: str
    source_commit: str
    source_tree: str
    adapter: str
    evidence_class: str
    runner: str
    host_sensitive: bool
    n_links: int
    force_bound_n: float
    track_half_length_m: float
    cart_mass_kg: float
    link_mass_kg: float
    link_length_m: float
    gravity_m_s2: float
    control_rate_hz: float
    rk4_max_step_s: float
    theta_tol_deg: float
    theta_rate_tol_rad_s: float
    cart_tol_m: float
    cart_rate_tol_m_s: float
    reject_nonfinite: bool
    force_eps: float
    hold_metric: str  # "suffix" | "longest_run"
    hold_scope: str  # "whole_log" | "from_switch"
    hold_required_s: float
    start_state: str  # "hanging" | "nominal_first"
    post_horizon_s: float
    switch_tick: int | None
    total_ticks: int | None
    nominals: tuple[NominalSpec, ...]
    gates: tuple[GateSpec, ...]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_rate_hz


@dataclass(frozen=True)
class RunResult:
    """Adapter outcome: metrics, verdict, and named failures."""

    rung: int
    record: RolloutRecord
    metrics: dict[str, Any]
    passed: bool
    failures: tuple[str, ...]


class Strategy(Protocol):
    """What every adapter stack exposes to the rollout helper."""

    def control(self, tick: int, state: np.ndarray, time_s: float) -> float: ...

    def phase(self, tick: int) -> str: ...


# --------------------------------------------------------------------------
# Paths and registry loading
# --------------------------------------------------------------------------
def repository_root() -> Path:
    """The monorepo root (two levels above this scratch package)."""
    if ROOT_OVERRIDE is not None:
        return Path(ROOT_OVERRIDE).resolve()
    return Path(__file__).resolve().parents[3]


def rung_root(cfg: RungConfig) -> Path:
    """Return the data directory for one rung."""
    return repository_root() / "rungs" / cfg.slug


def load_registry(path: Path | None = None) -> dict[int, RungConfig]:
    """Parse ``rungs.toml`` into validated per-rung configs."""
    with (path or REGISTRY_PATH).open("rb") as handle:
        raw = tomllib.load(handle)
    defaults = raw.get("defaults", {})
    rungs = {
        int(key.removeprefix("n")): _freeze_rung(int(key.removeprefix("n")), defaults, table)
        for key, table in raw["rungs"].items()
    }
    if sorted(rungs) != list(range(min(rungs), max(rungs) + 1)):
        raise ValueError(f"registry rungs are not contiguous: {sorted(rungs)}")
    return rungs


def _freeze_nominal(raw: dict[str, Any]) -> NominalSpec:
    return NominalSpec(
        role=raw["role"],
        path=raw["path"],
        sha256=raw["sha256"],
        n_nodes=int(raw["n_nodes"]),
        horizon_s=float(raw["horizon_s"]),
        strict_meta=bool(raw.get("strict_meta", False)),
        alias_dialect=bool(raw.get("alias_dialect", False)),
    )


def _freeze_gate(raw: dict[str, Any]) -> GateSpec:
    return GateSpec(
        file=raw["file"],
        sha256=raw["sha256"],
        seed=raw.get("seed"),
        dialect=raw["dialect"],
        controller=raw.get("controller"),
        nominal=raw.get("nominal"),
        nominal_match=raw.get("nominal_match", "name"),
        header_fields=tuple(raw["header_fields"]) if "header_fields" in raw else None,
        row_fields=tuple(raw["row_fields"]) if "row_fields" in raw else None,
        handoff_limit_deg=raw.get("handoff_limit_deg"),
        check_t_pre=bool(raw.get("check_t_pre", False)),
        check_tracker_ticks=bool(raw.get("check_tracker_ticks", False)),
        require_saturation=bool(raw.get("require_saturation", False)),
        force_bound_n=float(raw.get("force_bound_n", 150.0)),
        hold_required_s=float(raw.get("hold_required_s", 5.0)),
        expected_trials=raw.get("expected_trials"),
        expected_successes=raw.get("expected_successes"),
    )


def _freeze_rung(rung: int, defaults: dict[str, Any], raw: dict[str, Any]) -> RungConfig:
    merged: dict[str, Any] = {**defaults, **raw}
    return RungConfig(
        rung=rung,
        slug=merged["slug"],
        legacy_tag=merged["legacy_tag"],
        source_commit=merged["source_commit"],
        source_tree=merged["source_tree"],
        adapter=merged["adapter"],
        evidence_class=merged["evidence_class"],
        runner=merged["runner"],
        host_sensitive=bool(merged.get("host_sensitive", False)),
        n_links=int(merged["n_links"]),
        force_bound_n=float(merged["force_bound_n"]),
        track_half_length_m=float(merged["track_half_length_m"]),
        cart_mass_kg=float(merged.get("cart_mass_kg", 1.0)),
        link_mass_kg=float(merged.get("link_mass_kg", 0.10)),
        link_length_m=float(merged.get("link_length_m", 0.50)),
        gravity_m_s2=float(merged.get("gravity_m_s2", 9.81)),
        control_rate_hz=float(merged.get("control_rate_hz", 1000.0)),
        rk4_max_step_s=float(merged.get("rk4_max_step_s", 0.00025)),
        theta_tol_deg=float(merged["theta_tol_deg"]),
        theta_rate_tol_rad_s=float(merged["theta_rate_tol_rad_s"]),
        cart_tol_m=float(merged["cart_tol_m"]),
        cart_rate_tol_m_s=float(merged["cart_rate_tol_m_s"]),
        reject_nonfinite=bool(merged.get("reject_nonfinite", False)),
        force_eps=float(merged.get("force_eps", 1e-9)),
        hold_metric=merged["hold_metric"],
        hold_scope=merged.get("hold_scope", "whole_log"),
        hold_required_s=float(merged["hold_required_s"]),
        start_state=merged.get("start_state", "hanging"),
        post_horizon_s=float(merged.get("post_horizon_s", 0.0)),
        switch_tick=merged.get("switch_tick"),
        total_ticks=merged.get("total_ticks"),
        nominals=tuple(_freeze_nominal(n) for n in merged.get("nominals", [])),
        gates=tuple(_freeze_gate(g) for g in merged.get("gates", [])),
        extras=merged.get("extras", {}),
    )


# --------------------------------------------------------------------------
# Shared model construction (data only; no YAML loader)
# --------------------------------------------------------------------------
def build_spec(cfg: RungConfig) -> CartPoleSpec:
    """Build the frozen spec from registry data."""
    from cartpole_capsules.core.env_spec import CartPoleSpec

    return CartPoleSpec(
        n_links=cfg.n_links,
        cart_mass_kg=cfg.cart_mass_kg,
        link_masses_kg=[cfg.link_mass_kg] * cfg.n_links,
        link_lengths_m=[cfg.link_length_m] * cfg.n_links,
        gravity_m_s2=cfg.gravity_m_s2,
        damping_cart_n_s_m=0.0,
        damping_links_n_m_s_rad=[0.0] * cfg.n_links,
        force_bound_n=cfg.force_bound_n,
        track_half_length_m=cfg.track_half_length_m,
        control_rate_hz=cfg.control_rate_hz,
        rk4_max_step_s=cfg.rk4_max_step_s,
    )


def build_model(cfg: RungConfig) -> NLinkCartPole:
    from cartpole_capsules.core.dynamics import NLinkCartPole

    return NLinkCartPole(build_spec(cfg))


# --------------------------------------------------------------------------
# Shared assessment helpers (verdict semantics of every historical capsule)
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """SHA-256 of raw file bytes (staged here; moves to core/evidence.py)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def success_mask(model: NLinkCartPole, states: np.ndarray, cfg: RungConfig) -> np.ndarray:
    """Per-sample locked hold-set mask (n14 adds the non-finite reject)."""
    from cartpole_capsules.core.lqr import wrap_state_error

    n = model.n
    upright = model.x_equilibrium("up")
    mask = np.empty(len(states), dtype=bool)
    theta_tol = math.radians(cfg.theta_tol_deg)
    for index, state in enumerate(states):
        if cfg.reject_nonfinite and not np.isfinite(state).all():
            mask[index] = False
            continue
        error = wrap_state_error(state, upright, n)
        mask[index] = bool(
            np.all(np.abs(error[1 : 1 + n]) <= theta_tol)
            and np.all(np.abs(state[n + 2 :]) <= cfg.theta_rate_tol_rad_s)
            and abs(state[0]) <= cfg.cart_tol_m
            and abs(state[n + 1]) <= cfg.cart_rate_tol_m_s
        )
    return mask


def trailing_hold_s(mask: np.ndarray, control_dt_s: float) -> float:
    """Suffix hold: a final run of N in-set samples spans (N-1) intervals."""
    samples = 0
    for value in np.asarray(mask, dtype=bool)[::-1]:
        if not value:
            break
        samples += 1
    return max(0, samples - 1) * control_dt_s


def longest_run_hold_s(mask: np.ndarray, control_dt_s: float) -> float:
    """Longest-run hold: the max consecutive in-set run, same accounting."""
    best = run = 0
    for value in np.asarray(mask, dtype=bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return max(0, best - 1) * control_dt_s


def trailing_run_length(mask: np.ndarray) -> int:
    """Consecutive true samples at the end (n14 required-success-states)."""
    count = 0
    for value in np.asarray(mask, dtype=bool)[::-1]:
        if not value:
            break
        count += 1
    return count


def longest_run_span(mask: np.ndarray) -> tuple[int, int]:
    """First index and length of the longest true run (n14 metrics)."""
    best_first, best_count = -1, 0
    current_first, current_count = 0, 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if current_count == 0:
                current_first = index
            current_count += 1
            if current_count > best_count:
                best_first, best_count = current_first, current_count
        else:
            current_count = 0
    return best_first, best_count


def event(
    tick: int,
    control_dt_s: float,
    value: float | None = None,
    quarter: int | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """One first-occurrence event record."""
    record: dict[str, Any] = {"tick": tick, "time_s": tick * control_dt_s}
    if value is not None:
        record["value"] = float(value)
    if quarter is not None:
        record["quarter"] = quarter
    if kind is not None:
        record["kind"] = kind
    return record


def first_event(
    flags: np.ndarray,
    values: np.ndarray | None = None,
    tick_offset: int = 0,
    control_dt_s: float = 0.001,
) -> dict[str, Any] | None:
    indexes = np.flatnonzero(flags)
    if not len(indexes):
        return None
    return event(
        tick_offset + int(indexes[0]),
        control_dt_s,
        None if values is None else values[indexes[0]],
    )


def force_stats(record: RolloutRecord, cfg: RungConfig) -> dict[str, Any]:
    """Raw-versus-applied force facts: clipping lives at the boundary."""
    delta = record.applied - record.raw
    over = np.abs(record.raw) > cfg.force_bound_n
    return {
        "raw_peak_abs_n": float(np.max(np.abs(record.raw))),
        "applied_peak_abs_n": float(np.max(np.abs(record.applied))),
        "first_raw_over_force_bound": first_event(over, record.raw, control_dt_s=cfg.control_dt_s),
        "first_clipping": first_event(np.abs(delta) > 0.0, delta, control_dt_s=cfg.control_dt_s),
        "clip_ticks": int(np.count_nonzero(np.abs(record.raw) > cfg.force_bound_n + cfg.force_eps)),
    }


def track_stats(record: RolloutRecord, cfg: RungConfig) -> dict[str, Any]:
    peak = float(np.max(np.abs(record.states[:, 0])))
    return {
        "peak_abs_cart_m": peak,
        "bound_abs_m": cfg.track_half_length_m,
        "first_exceedance": first_event(
            np.abs(record.states[:, 0]) > cfg.track_half_length_m,
            np.abs(record.states[:, 0]),
            control_dt_s=cfg.control_dt_s,
        ),
    }


def handoff_angle_error_deg(model: NLinkCartPole, state: np.ndarray) -> float:
    """Max wrapped angle error against upright at the handoff state."""
    from cartpole_capsules.core.lqr import wrap_state_error

    error = wrap_state_error(state, model.x_equilibrium("up"), model.n)
    return float(np.rad2deg(np.max(np.abs(error[1 : 1 + model.n]))))


def phase_schedule(record: RolloutRecord, cfg: RungConfig) -> bool:
    """Exactly one phase before the switch and one after (n12 gate)."""
    if cfg.switch_tick is None:
        return len(set(record.phases)) <= 1
    return record.phases[: cfg.switch_tick] == (
        record.phases[0],
    ) * cfg.switch_tick and record.phases[cfg.switch_tick :] == (
        record.phases[cfg.switch_tick],
    ) * (len(record.phases) - cfg.switch_tick)


def byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return bool(
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    delta = a - b
    return float(np.max(np.abs(delta))) if np.all(np.isfinite(delta)) else float("inf")


def isclose(value: float, target: float, atol: float) -> bool:
    return math.isclose(value, target, rel_tol=0.0, abs_tol=atol)


def to_jsonable(value: Any) -> Any:
    """Lossless JSON conversion for numpy scalars, arrays, dicts, lists."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def expected_metric_failures(
    metrics: dict[str, Any], expected: dict[str, ExpectedMetric]
) -> list[str]:
    """Release-pinned drift checks (n5 baseline, n14 expected-witness)."""
    return [
        key
        for key, spec in expected.items()
        if not isinstance(metrics.get(key), (int, float))
        or abs(float(metrics[key]) - spec.value) > spec.atol
    ]


# --------------------------------------------------------------------------
# Adapter dispatch
# --------------------------------------------------------------------------
def build_adapter(cfg: RungConfig) -> Any:
    """Resolve the registry adapter name to its module (lazy: no cycles)."""
    from . import composite_n12, discrete_stack, legacy_tvlqr, proof_n13, witness_n14

    modules = {
        "legacy_tvlqr": legacy_tvlqr,
        "discrete_stack": discrete_stack,
        "composite_n12": composite_n12,
        "proof_n13": proof_n13,
        "witness_n14": witness_n14,
    }
    if cfg.adapter not in modules:
        raise ValueError(f"unknown adapter {cfg.adapter!r} for rung {cfg.rung}")
    return modules[cfg.adapter]
