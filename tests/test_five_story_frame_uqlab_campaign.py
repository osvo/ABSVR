"""Tests for the frame UQLab AK-MCS campaign aggregator."""

from __future__ import annotations

import pytest

from studies.five_story_frame.run_uqlab_akmcs_campaign import summarize_results


def _result(seed: int, pf: float, calls: int, converged: bool = True) -> dict:
    return {
        "seed": seed,
        "pf": pf,
        "converged": converged,
        "model_evaluations_opensees_ledger": calls,
    }


def test_uqlab_frame_summary_uses_all_declared_runs() -> None:
    summary = summarize_results(
        [_result(1, 0.0015, 210), _result(2, 0.0016, 250)],
        seeds=(1, 2),
        reference_pf=0.00151,
    )
    assert summary["runs"] == 2
    assert summary["converged_runs"] == 2
    assert summary["mean_pf"] == pytest.approx(0.00155)
    assert summary["mean_open_sees_calls"] == pytest.approx(230.0)
    assert summary["minimum_open_sees_calls"] == 210
    assert summary["maximum_open_sees_calls"] == 250


def test_uqlab_frame_summary_rejects_reordered_seeds() -> None:
    with pytest.raises(ValueError, match="seed order"):
        summarize_results(
            [_result(2, 0.0015, 200), _result(1, 0.0016, 200)],
            seeds=(1, 2),
            reference_pf=0.00151,
        )
