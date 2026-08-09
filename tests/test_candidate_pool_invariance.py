"""Regression tests for the fixed Monte Carlo candidate population."""

from __future__ import annotations

import numpy as np

from absvr_core.candidate_selection import select_best_candidate
from absvr_core.learning_function import compute_learning_function


def test_selected_candidate_does_not_shrink_monte_carlo_population() -> None:
    pool = np.array([[0.0], [1.0], [2.0]])
    pdf = np.array([0.5, 0.3, 0.2])
    region_indices = np.array([0, 1, 2])
    doe = np.array([[-1.0]])
    g = np.array([1.0])

    result = select_best_candidate(
        np.array([3.0, 1.0, 2.0]),
        pool,
        region_indices,
        doe,
        g,
        pool,
        pdf,
        pool.shape[0],
        0,
        lambda points, _params: points[:, 0] - 1.5,
        {},
    )
    updated_doe, updated_g, updated_pool, updated_pdf, n_mc, n_added = result

    np.testing.assert_array_equal(updated_pool, pool)
    np.testing.assert_array_equal(updated_pdf, pdf)
    np.testing.assert_array_equal(updated_doe[-1], pool[1])
    assert updated_g[-1] == -0.5
    assert n_mc == pool.shape[0]
    assert n_added == 1


def test_existing_doe_point_is_ineligible_for_reselection() -> None:
    pool = np.array([[0.0], [1.0], [2.0]])
    doe = np.array([[1.0]])
    learning, region, indices = compute_learning_function(
        pool,
        doe,
        g_predict=np.array([0.2, 0.0, -0.2]),
        g_mse=np.ones(3),
        v_pdf_pool=np.array([1.0, 2.0, 3.0]),
        current_pf=0.5,
        model=None,
        w_grad=0.0,
    )

    duplicate_position = int(np.flatnonzero(indices == 1)[0])
    assert np.array_equal(region[duplicate_position], doe[0])
    assert np.isinf(learning[duplicate_position])
