"""Unit tests for the frozen UQLab campaign aggregator."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize("pf", [0.0, 1.0])
def test_summary_serializes_endpoint_probability_without_nonstandard_float(
    pf: float,
) -> None:
    summary = summarize_results(
        [_result(11, pf, 31)],
        seeds=(11,),
        profile="paper_like",
        reference_pf=0.0015,
        published_pf=0.00152,
    )

    assert summary["beta_from_mean_pf"] is None
    assert summary["beta_from_mean_pf_is_infinite"] is True


@pytest.mark.parametrize("pf", [-0.1, 1.1, float("nan"), float("inf")])
def test_summary_refuses_invalid_failure_probability(pf: float) -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        summarize_results(
            [_result(11, pf)],
            seeds=(11,),
            profile="paper_like",
            reference_pf=0.0015,
            published_pf=0.00152,
        )


def test_original_profile_uses_uqlab_native_akmcs_stop_u() -> None:
    wrapper = (
        Path(__file__).parents[1]
        / "studies"
        / "planar_truss"
        / "uqlab"
        / "run_uqlab_akmcs.m"
    ).read_text(encoding="utf-8")

    assert "analysisOptions.Method = 'AKMCS';" in wrapper
    assert "analysisOptions.AKMCS.Convergence = 'stopU';" in wrapper
    assert "analysisOptions.AKMCS.ConvThres" not in wrapper
