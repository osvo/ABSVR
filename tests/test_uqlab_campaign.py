"""Unit tests for the frozen UQLab campaign aggregator."""

from __future__ import annotations

import pytest

from studies.planar_truss.run_uqlab_campaign import summarize_results


def _result(seed: int, pf: float, calls: int = 100) -> dict:
    return {
        "seed": seed,
        "profile": "paper_like",
        "pf": pf,
        "model_evaluations_opensees_ledger": calls,
    }


def test_summary_uses_every_frozen_seed_and_true_call_ledger() -> None:
    results = [_result(11, 0.0014, 91), _result(19, 0.0016, 109)]
    summary = summarize_results(
        results,
        seeds=(11, 19),
        profile="paper_like",
        reference_pf=0.0015,
        published_pf=0.00152,
    )

    assert summary["mean_pf"] == pytest.approx(0.0015)
    assert summary["relative_error_of_mean_vs_independent_rqmc_percent"] == pytest.approx(0.0)
    assert summary["mean_open_sees_calls"] == pytest.approx(100.0)
    assert summary["minimum_open_sees_calls"] == 91
    assert summary["maximum_open_sees_calls"] == 109
    assert summary["total_open_sees_calls"] == 200
    assert summary["selection_uses_reference_probability"] is False


def test_summary_refuses_missing_or_reordered_seed() -> None:
    with pytest.raises(ValueError, match="every frozen seed"):
        summarize_results(
            [_result(19, 0.0015), _result(11, 0.0015)],
            seeds=(11, 19),
            profile="paper_like",
            reference_pf=0.0015,
            published_pf=0.00152,
        )

