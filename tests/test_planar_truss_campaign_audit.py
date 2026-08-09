"""Tests for frozen planar-truss campaign auditing."""

from __future__ import annotations

import pytest

from studies.planar_truss.audit_campaign import audit_campaign, audit_frozen_protocol


def _campaign() -> dict:
    run = {
        "algorithm_seed": 7,
        "gradient_weight": 0.0,
        "svr_loss": "squared_epsilon",
        "open_sees_limit_state_calls": 95,
        "adaptive_candidate_pool_size": 131072,
        "evidence_history": [
            {"n_samples": n_samples} for n_samples in (15, 35, 55, 75, 95)
        ],
        "validation": {
            "surrogate_pf_mean": 0.0015,
            "reference_evaluations_counted_as_absvr_calls": False,
            "confusion_counts": {
                "true_positive": 90,
                "false_positive": 5,
                "false_negative": 10,
                "true_negative": 895,
            },
        },
    }
    return {
        "schema_version": 2,
        "campaign_stage": "confirmation",
        "configuration_development_used_reference_probability": True,
        "development_algorithm_seeds": [11, 53, 101],
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "training_protocol": {
            "response_preserves_original_failure_event": True,
            "added_points": 80,
            "candidate_pool_size": 131072,
            "hyperparameter_retune_interval": 20,
            "svr_loss": "squared_epsilon",
        },
        "independent_reference": {
            "mean_pf": 0.00152,
            "published_direct_mcs_pf": 0.00152,
        },
        "runs": [run],
    }


def _reference() -> dict:
    return {"mean_pf": 0.00152, "confidence_interval_95": [0.0015, 0.00154]}


def _frozen_protocol() -> dict:
    return {
        "status": "frozen_before_confirmation",
        "development_algorithm_seeds": [11, 53, 101],
        "confirmation_algorithm_seeds": [7],
        "configuration": {
            "svr_loss": "squared_epsilon",
            "gradient_weight": 0.0,
            "initial_design": "normal LHS",
            "response": "log response",
            "response_preserves_original_failure_event": True,
            "total_opensees_calls_per_run": 95,
            "added_points": 80,
            "candidate_pool_size": 131072,
            "hyperparameter_selection": "evidence grid",
            "hyperparameter_retune_interval": 20,
        },
        "validation": {
            "replications": 8,
            "samples_per_replication": 524288,
            "base_seed": 20260808,
        },
        "frozen_algorithm_commit": "abc1234",
    }


def test_campaign_audit_checks_protocol_and_metrics() -> None:
    result = audit_campaign(_campaign(), _reference())
    assert result["audit_passed"] is True
    assert result["fixed_open_sees_calls_per_run"] == 95
    assert result["evidence_schedule"] == [15, 35, 55, 75, 95]
    assert result["pooled_sensitivity"] == 0.9


def test_campaign_audit_rejects_reference_aware_selection() -> None:
    campaign = _campaign()
    campaign["selection_uses_reference_probability"] = True
    with pytest.raises(ValueError, match="blind"):
        audit_campaign(campaign, _reference())


def test_campaign_audit_rejects_development_confirmation_seed_overlap() -> None:
    campaign = _campaign()
    campaign["development_algorithm_seeds"] = [7]
    with pytest.raises(ValueError, match="disjoint"):
        audit_campaign(campaign, _reference())


def test_frozen_protocol_audit_checks_precommitted_configuration() -> None:
    campaign = _campaign()
    campaign["training_protocol"]["initial_design"] = "normal LHS"
    campaign["training_protocol"]["response"] = "log response"
    campaign["training_protocol"]["hyperparameter_selection"] = "evidence grid"
    campaign["validation_protocol"] = {
        "replications": 8,
        "log2_samples": 19,
        "base_seed": 20260808,
    }
    result = audit_frozen_protocol(campaign, _frozen_protocol())
    assert result["frozen_protocol_passed"] is True

    campaign["runs"][0]["open_sees_limit_state_calls"] = 94
    with pytest.raises(ValueError, match="call count"):
        audit_frozen_protocol(campaign, _frozen_protocol())
