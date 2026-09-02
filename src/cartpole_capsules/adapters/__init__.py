"""Scratch package marker for the adapter prototype.

These modules are a prototype for the lean generation's
``src/cartpole_capsules/adapters/`` package. In the real tree this file
re-exports the public API; here it exists so relative imports resolve when
the package directory is executed directly.
"""

from .base import (
    ExpectedMetric,
    GateSpec,
    NominalSpec,
    RolloutRecord,
    RungConfig,
    RunResult,
    Strategy,
    build_adapter,
    build_model,
    build_spec,
    load_registry,
)

__all__ = [
    "GateSpec",
    "NominalSpec",
    "RolloutRecord",
    "RungConfig",
    "RunResult",
    "ExpectedMetric",
    "Strategy",
    "build_adapter",
    "build_model",
    "build_spec",
    "load_registry",
]
