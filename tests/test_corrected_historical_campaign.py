"""Tests for the corrected six-benchmark campaign protocol and audit."""

from __future__ import annotations

import pytest

from studies.historical_benchmarks.run_corrected_campaign import (
    PROFILES,
    summarize_benchmark,
)


def test_profiles_match_the_manuscript_launch_table() -> None:
    assert PROFILES["eg1"].c_init == pytest.approx(9.25e4)
    assert PROFILES["eg2"].tail_policy == "down"
    assert PROFILES["eg2"].tail_alpha == pytest.approx(1.95)
    assert PROFILES["eg3"].kernel == "polynomial"
    assert PROFILES["eg4"].kernel == "laplacian"
    assert PROFILES["eg4"].tail_policy == "explore"
    assert PROFILES["eg5"].epsilon_init == pytest.approx(6.04e-3)
    assert PROFILES["eg6"].gamma is None


def test_summary_compares_same_frozen_seeds_without_deletion() -> None:
    current = [
        {"seed": 0, "pf": 0.0046, "evals": 70},
        {"seed": 1, "pf": 0.0048, "evals": 72},
    ]
    previous = [
        {"seed": 0, "pf": 0.0030, "evals": 65},
        {"seed": 1, "pf": 0.0032, "evals": 67},
    ]
    summary = summarize_benchmark(
        current,
        previous,
        profile=PROFILES["eg1"],
        seeds=(0, 1),
    )

    assert summary["runs"] == 2
    assert summary["corrected_mean_pf"] == pytest.approx(0.0047)
    assert summary["corrected_mean_calls"] == pytest.approx(71.0)
    assert summary["previous_mean_calls"] == pytest.approx(66.0)
    assert summary["relative_error_improved"] is True
    assert summary["all_runs_retained"] is True


def test_summary_refuses_reordered_or_missing_seed() -> None:
    with pytest.raises(ValueError, match="every frozen seed"):
        summarize_benchmark(
            [{"seed": 1, "pf": 0.0047, "evals": 70}],
            [{"seed": 0, "pf": 0.0047, "evals": 70}],
            profile=PROFILES["eg1"],
            seeds=(0,),
        )

