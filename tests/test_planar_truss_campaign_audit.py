"""Tests for frozen planar-truss campaign auditing."""

from __future__ import annotations

import pytest

from studies.planar_truss.audit_campaign import audit_campaign


def _campaign() -> dict:
    run = {
        "algorithm_seed": 7,
        "gradient_weight": 0.0,
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
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "training_protocol": {
            "response_preserves_original_failure_event": True,
            "added_points": 80,
            "candidate_pool_size": 131072,
            "hyperparameter_retune_interval": 20,
        },
        "independent_reference": {
            "mean_pf": 0.00152,
            "published_direct_mcs_pf": 0.00152,
        },
        "runs": [run],
    }


def _reference() -> dict:
    return {"mean_pf": 0.00152, "confidence_interval_95": [0.0015, 0.00154]}


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
