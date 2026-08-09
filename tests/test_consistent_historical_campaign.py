"""Tests for the common-profile analytical benchmark campaign."""

from __future__ import annotations

import numpy as np
import pytest

from studies.historical_benchmarks.run_consistent_campaign import (
    _physical_samples,
    _summarize,
)


def test_physical_samples_apply_normal_and_lognormal_transforms() -> None:
    unit = np.array([[0.5, 0.5]])
    mu = np.array([1.0, -1.0])
    sigma = np.array([2.0, 3.0])
    np.testing.assert_allclose(_physical_samples(unit, mu, sigma, "normal"), [[1.0, -1.0]])
    np.testing.assert_allclose(
        _physical_samples(unit, mu, sigma, "lognormal"),
        np.exp([[1.0, -1.0]]),
    )


def test_consistent_campaign_summary_retains_all_runs() -> None:
    runs = [
        {
            "true_limit_state_calls": 95,
            "validation": {
                "surrogate_pf_mean": 0.009,
                "relative_error_vs_reference_percent": 10.0,
            },
        },
        {
            "true_limit_state_calls": 95,
            "validation": {
                "surrogate_pf_mean": 0.011,
                "relative_error_vs_reference_percent": 10.0,
            },
        },
    ]
    summary = _summarize(runs, 0.01)
    assert summary["runs"] == 2
    assert summary["mean_surrogate_pf"] == pytest.approx(0.01)
    assert summary["relative_error_of_mean_pf_percent"] == pytest.approx(0.0, abs=1.0e-12)
    assert summary["maximum_absolute_run_error_percent"] == 10.0
