from __future__ import annotations

import pytest

from studies.five_story_frame.run_uqlab_abpce_campaign import summarize_results


def _result(seed: int, pf: float, calls: int, converged: bool = True) -> dict:
    return {
        "seed": seed,
        "pf": pf,
        "converged": converged,
        "model_evaluations_opensees_ledger": calls,
    }


def test_summarize_abpce_counts_only_true_model_calls() -> None:
    summary = summarize_results(
        [_result(11, 0.0015, 235), _result(53, 0.0016, 241, False)],
        seeds=(11, 53),
        reference_pf=0.00155,
    )

    assert summary["runs"] == 2
    assert summary["converged_runs"] == 1
    assert summary["mean_open_sees_calls"] == pytest.approx(238.0)
    assert summary["total_open_sees_calls"] == 476
    assert summary["relative_error_of_mean_vs_independent_rqmc_percent"] == pytest.approx(0.0)


def test_summarize_abpce_rejects_seed_order_mismatch() -> None:
    with pytest.raises(ValueError, match="seed order"):
        summarize_results(
            [_result(11, 0.0015, 235)], seeds=(53,), reference_pf=0.00155
        )
