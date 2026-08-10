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


def test_u_distance_prefers_more_novel_point_at_equal_u_score():
    pool = np.array([[0.0], [1.0], [3.0]])
    doe = np.array([[-1.0]])
    prediction = np.array([0.1, 0.1, 0.1])
    variance = np.array([0.01, 0.01, 0.01])
    density = np.array([0.1, 0.5, 1.0])

    score, region, indices = compute_learning_function(
        pool,
        doe,
        prediction,
        variance,
        density,
        current_pf=0.5,
        w_grad=0.0,
        strategy="u_distance",
    )

    np.testing.assert_array_equal(indices, np.array([1, 2]))
    np.testing.assert_array_equal(region, pool[1:])
    assert score[1] < score[0]


def test_u_distance_rejects_duplicate_even_when_its_u_score_is_smaller():
    pool = np.array([[0.0], [1.0], [2.0]])
    doe = np.array([[1.0]])
    prediction = np.array([0.2, 0.0, 0.1])
    variance = np.array([0.01, 0.01, 0.01])
    density = np.array([0.1, 0.5, 1.0])

    score, _, indices = compute_learning_function(
        pool,
        doe,
        prediction,
        variance,
        density,
        current_pf=0.5,
        w_grad=0.0,
        strategy="u_distance",
    )

    np.testing.assert_array_equal(indices, np.array([1, 2]))
    assert score[1] < score[0]
