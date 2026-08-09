"""Tests for frozen planar-truss campaign auditing."""

from __future__ import annotations

import pytest

from studies.planar_truss.audit_campaign import (
    audit_campaign,
    audit_frozen_protocol,
    audit_uqlab_confirmation,
)


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


def test_uqlab_confirmation_audit_checks_native_method_and_call_ledger() -> None:
    run = {
        "seed": 139,
        "profile": "original_akmcs",
        "uqlab_reliability_method": "AKMCS",
        "convergence": "stopU",
        "convergence_threshold": 2.0,
        "initial_design": {"sampling": "LHS", "size": 30},
        "pf": 0.0015,
        "model_evaluations_uqlab": 380,
        "model_evaluations_opensees_ledger": 380,
    }
    summary = {
        "algorithm_seeds": [139],
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "individual_runs": [run],
        "mean_pf": 0.0015,
        "beta_from_mean_pf": 2.9677,
        "mean_pf_student_t_95_percent_interval_across_seeds": [0.0015, 0.0015],
        "coefficient_of_variation_pf_across_seeds_percent": 0.0,
        "relative_error_of_mean_vs_independent_rqmc_percent": 0.5,
        "relative_error_of_mean_vs_published_mcs_percent": 1.0,
        "mean_absolute_run_error_vs_independent_rqmc_percent": 0.5,
        "median_absolute_run_error_vs_independent_rqmc_percent": 0.5,
        "maximum_absolute_run_error_vs_independent_rqmc_percent": 0.5,
        "mean_open_sees_calls": 380.0,
        "minimum_open_sees_calls": 380,
        "maximum_open_sees_calls": 380,
        "total_open_sees_calls": 380,
    }
    frozen = {
        "status": "frozen_before_confirmation",
        "algorithm_seeds": [139],
        "development_seed_excluded": 127,
        "runs_expected": 1,
        "profile": "original_akmcs",
        "uqlab_reliability_method": "AKMCS",
        "convergence": {"criterion": "stopU", "threshold": 2.0},
        "initial_design": {"sampling": "LHS", "size": 30},
    }

    result = audit_uqlab_confirmation(summary, frozen)
    assert result["audit_passed"] is True
    assert result["mean_open_sees_calls"] == 380.0

    run["model_evaluations_opensees_ledger"] = 379
    with pytest.raises(ValueError, match="ledgers disagree"):
        audit_uqlab_confirmation(summary, frozen)
