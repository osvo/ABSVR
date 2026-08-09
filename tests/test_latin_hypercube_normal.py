"""Tests for isoprobabilistic normal Latin hypercube sampling."""

from __future__ import annotations

import numpy as np
from scipy import stats

from utilities import lhs_normal, reset_global_seed


def test_normal_lhs_stratifies_each_probability_axis() -> None:
    n_samples = 32
    reset_global_seed(20260808)
    samples, density = lhs_normal(
        np.array([2.0, -1.0]),
        np.array([0.5, 3.0]),
        n_samples,
    )

    probabilities = stats.norm.cdf((samples - [2.0, -1.0]) / [0.5, 3.0])
    for column in probabilities.T:
        strata = np.floor(column * n_samples).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(n_samples))

    expected_density = np.prod(
        stats.norm.pdf((samples - [2.0, -1.0]) / [0.5, 3.0]) / [0.5, 3.0],
        axis=1,
    )
    np.testing.assert_allclose(density, expected_density)


def test_normal_lhs_is_reproducible_with_shared_seed() -> None:
    reset_global_seed(61)
    first, _ = lhs_normal(np.zeros(3), np.ones(3), 15)
    reset_global_seed(61)
    second, _ = lhs_normal(np.zeros(3), np.ones(3), 15)
    np.testing.assert_array_equal(first, second)
