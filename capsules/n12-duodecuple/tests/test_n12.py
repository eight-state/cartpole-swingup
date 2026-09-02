from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from n12_cartpole import demo, simulator, success, verifier
from n12_cartpole.simulator import N_LINKS

RELEASE_SOURCE_HASHES = {
    "src/cartpole_race/__init__.py": "92f02f32168d383b97f3bc2d853456427b14219a239609de480d5c400cc6b5a3",
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/cartpole_race/lqr.py": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
    "src/n12_cartpole/fast_pieces.py": "e49c94f4d763a89911fa6e55fd9a460f14748246c0096d49694429501e1e20a9",
}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model() -> NLinkCartPole:
    return NLinkCartPole(
        CartPoleSpec(
            n_links=N_LINKS,
            link_masses_kg=[0.1] * N_LINKS,
            link_lengths_m=[0.5] * N_LINKS,
            damping_links_n_m_s_rad=[0.0] * N_LINKS,
        )
    )


def test_frozen_numerical_modules_match_the_release() -> None:
    assert {
        path: _sha256(path) for path in RELEASE_SOURCE_HASHES
    } == RELEASE_SOURCE_HASHES


def test_evidence_audit_rederives_all_banked_gate_totals() -> None:
    audit = verifier.audit_loaded_evidence()

    assert audit["frozen_nominal"]["state_shape"] == [2501, 26]
    assert audit["frozen_nominal"]["control_shape"] == [2500]
    assert audit["banked_gate"] == {
        "seeds": [12345, 777, 2024],
        "successes": 72,
        "trials": 72,
    }
    assert audit["evidence"]["capabilities"]["nominal_synthesis"] is False
    assert audit["evidence"]["capabilities"]["perturbed_gate_rerun"] is False
    assert audit["historical_summary"]["classification"] == (
        "immutable_historical_summary_not_live_evidence"
    )
    assert audit["historical_summary"]["legacy_input_field_mapping"] == {
        "continuous_hold_s": "sampled_hold_s"
    }
    assert audit["historical_summary"]["provenance_limitations"] == {
        "historic_nominal_path": "runs/r2/nom_n12_4ms_fast.npz",
        "historic_nominal_path_present_in_checkout": False,
        "historic_nominal_digest_retained": False,
        "primary_trial_inputs_retained": False,
        "primary_trial_traces_retained": False,
    }


def test_demo_and_verifier_share_the_one_live_stack() -> None:
    assert demo.run_live_rollout is simulator.run_live_rollout
    assert verifier.run_live_rollout is simulator.run_live_rollout


def test_wilson_interval_is_rederived_from_24_of_24() -> None:
    assert verifier.wilson95(24, 24) == (0.862, 1.0)


def test_locked_success_predicate_rejects_each_boundary() -> None:
    model = _model()
    upright = model.x_equilibrium("up")
    assert success.in_success_set(model, upright)

    angle = upright.copy()
    angle[1] = np.deg2rad(5.1)
    assert not success.in_success_set(model, angle)

    link_rate = upright.copy()
    link_rate[N_LINKS + 2] = 0.5001
    assert not success.in_success_set(model, link_rate)

    cart_position = upright.copy()
    cart_position[0] = 2.0001
    assert not success.in_success_set(model, cart_position)

    cart_rate = upright.copy()
    cart_rate[N_LINKS + 1] = 0.5001
    assert not success.in_success_set(model, cart_rate)


def test_live_success_schema_uses_sampled_1khz_fields(monkeypatch) -> None:
    model = _model()
    upright = model.x_equilibrium("up")
    monkeypatch.setattr(success, "SWITCH_TICK", 1)
    monkeypatch.setattr(success, "TOTAL_TICKS", 2)
    rollout = SimpleNamespace(
        model=model,
        states=np.repeat(upright[None, :], 3, axis=0),
        applied_forces=np.zeros(2),
        raw_forces=np.zeros(2),
        phases=("tvlqr", "static_care"),
        times=np.arange(3) * success.CONTROL_DT_S,
    )

    metrics = success.assess_rollout(rollout)["success_set"]

    assert metrics["sampled_hold_s"] == 0.001
    assert metrics["sampled_hold_samples"] == 2
    assert metrics["every_1khz_sample_from_switch_through_final_in_success_set"]
    assert "continuous_hold_s" not in metrics
    assert "continuous_hold_samples" not in metrics


def test_verifier_keeps_six_historical_deltas_diagnostic(monkeypatch) -> None:
    model = _model()
    historic = {
        "duration_s": 0.0,
        "raw_and_applied_force_peak_n": 0.0,
        "peak_abs_cart_m": 0.0,
        "switch_max_wrapped_link_angle_deg": 0.0,
        "switch_max_link_rate_rad_s": 0.0,
        "sampled_hold_s": 0.0,
    }
    loaded = {
        "evidence": {
            "capabilities": {
                "nominal_synthesis": False,
                "perturbed_gate_rerun": False,
            }
        },
        "frozen_nominal": {},
        "banked_gate": {},
        "historical_summary": {"stored_observation": historic},
    }
    rollout = SimpleNamespace(
        dense_states=np.zeros((10_001, 26)),
        dense_controls=np.zeros(10_000),
        nominal=SimpleNamespace(controls=np.zeros(2_500)),
        states=np.array([model.x_equilibrium("down")]),
        model=model,
    )
    live = {
        "execution": {
            "duration_s": 1.0,
            "phase_sequence_valid": True,
            "time_grid_exact_1khz": True,
        },
        "finite": {"all_states": True},
        "forces": {
            "raw_peak_abs_n": 2.0,
            "first_raw_over_force_bound": None,
            "first_clipping": None,
        },
        "track": {"peak_abs_cart_m": 3.0, "first_exceedance": None},
        "success_set": {
            "switch_state": {
                "in_success_set": True,
                "max_wrapped_link_angle_deg": 4.0,
                "max_link_rate_rad_s": 5.0,
            },
            "sampled_hold_s": 6.0,
            "every_1khz_sample_from_switch_through_final_in_success_set": True,
        },
    }
    monkeypatch.setattr(verifier, "audit_loaded_evidence", lambda: loaded)
    monkeypatch.setattr(verifier, "run_live_rollout", lambda: rollout)
    monkeypatch.setattr(verifier, "assess_rollout", lambda _: live)

    payload = verifier.run_verifier()

    assert set(payload["recomputed"]["banked_observation_deltas"]) == set(historic)
    assert "platform_stable_witness_matches" not in payload["checks"]
    assert payload["verdict"] == "PASS"


def test_default_generated_paths_are_ignored_working_artifacts() -> None:
    assert demo.DEFAULT_OUTPUT.parts[-2:] == (".working", "n12-demo.gif")
    assert verifier.DEFAULT_OUTPUT.parts[-2:] == (".working", "n12-verify.json")
