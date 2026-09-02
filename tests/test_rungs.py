"""Registry, legacy-tag, and historical-evidence tests for all rungs."""

from __future__ import annotations

from pathlib import Path

import pytest

from cartpole_capsules.adapters import base
from cartpole_capsules.adapters.verification import audit_gates
from cartpole_capsules.legacy import audit_legacy_source

EXPECTED_COUNTS = {
    5: (112, 112),
    6: (48, 48),
    7: (66, 72),
    8: (72, 96),
    9: (72, 72),
    10: (72, 72),
    11: (72, 72),
    12: (72, 72),
    13: (0, 0),
    14: (0, 0),
}


@pytest.fixture(scope="module")
def registry() -> dict[int, base.RungConfig]:
    return base.load_registry()


def test_registry_is_contiguous_and_data_driven(registry: dict[int, base.RungConfig]) -> None:
    assert sorted(registry) == list(range(5, 15))
    assert registry[5].force_bound_n == registry[6].force_bound_n == 60.0
    assert all(registry[rung].force_bound_n == 150.0 for rung in range(7, 15))
    assert {registry[rung].adapter for rung in registry} == {
        "legacy_tvlqr",
        "discrete_stack",
        "composite_n12",
        "proof_n13",
        "witness_n14",
    }


@pytest.mark.parametrize("rung", range(5, 15))
def test_legacy_tag_preserves_source_release(
    rung: int, registry: dict[int, base.RungConfig]
) -> None:
    result = audit_legacy_source(Path.cwd(), registry[rung])
    assert result["commit"] == registry[rung].source_commit
    assert result["tree"] == registry[rung].source_tree


@pytest.mark.parametrize("rung", range(5, 15))
def test_historical_evidence_audits_without_rerun(
    rung: int, registry: dict[int, base.RungConfig]
) -> None:
    config = registry[rung]
    result = audit_gates(config, base.capsule_root(config))
    assert (result["total_successes"], result["total_trials"]) == EXPECTED_COUNTS[rung]
    assert all(gate.dialect not in {"todo-aggregate", "todo-composite"} for gate in config.gates)
