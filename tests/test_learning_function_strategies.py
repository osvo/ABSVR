import numpy as np
import pytest

from absvr_core.learning_function import compute_learning_function


def _inputs():
    pool = np.array([[0.0], [1.0], [2.0]])
    doe = np.array([[-1.0]])
    prediction = np.array([0.2, 0.1, 0.4])
    variance = np.array([0.01, 0.04, 0.01])
    density = np.array([0.1, 0.5, 1.0])
    return pool, doe, prediction, variance, density


def test_u_strategy_is_absolute_mean_over_standard_deviation():
    pool, doe, prediction, variance, density = _inputs()

    score, region, indices = compute_learning_function(
        pool,
        doe,
        prediction,
        variance,
        density,
        current_pf=0.5,
        w_grad=0.0,
        strategy="u",
    )

    np.testing.assert_array_equal(indices, np.array([1, 2]))
    np.testing.assert_array_equal(region, pool[1:])
    np.testing.assert_allclose(score, np.array([0.5, 4.0]))


def test_unknown_learning_strategy_is_rejected():
    pool, doe, prediction, variance, density = _inputs()

    with pytest.raises(ValueError, match="strategy"):
        compute_learning_function(
            pool,
            doe,
            prediction,
            variance,
            density,
            current_pf=0.5,
            strategy="unknown",
        )
