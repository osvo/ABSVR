"""Regression tests for the fixed Monte Carlo candidate population."""

from __future__ import annotations

import importlib

import numpy as np

from absvr_core.candidate_selection import select_best_candidate
from absvr_core.checkpointing import load_checkpoint
from absvr_core.learning_function import compute_learning_function


def test_selected_candidate_does_not_shrink_monte_carlo_population() -> None:
    pool = np.array([[0.0], [1.0], [2.0]])
    pdf = np.array([0.5, 0.3, 0.2])
    region_indices = np.array([0, 1, 2])
    doe = np.array([[-1.0]])
    g = np.array([1.0])
    eligible = np.ones(pool.shape[0], dtype=bool)

    result = select_best_candidate(
        np.array([3.0, 1.0, 2.0]),
        pool,
        region_indices,
        doe,
        g,
        pool,
        pdf,
        eligible,
        pool.shape[0],
        0,
        lambda points, _params: points[:, 0] - 1.5,
        {},
    )
    (
        updated_doe,
        updated_g,
        updated_pool,
        updated_pdf,
        updated_eligible,
        n_mc,
        n_added,
    ) = result

    np.testing.assert_array_equal(updated_pool, pool)
    np.testing.assert_array_equal(updated_pdf, pdf)
    np.testing.assert_array_equal(updated_doe[-1], pool[1])
    assert updated_g[-1] == -0.5
    assert n_mc == pool.shape[0]
    assert n_added == 1
    np.testing.assert_array_equal(updated_eligible, [True, False, True])


def test_candidate_eligibility_excludes_selected_point_without_reindexing() -> None:
    pool = np.array([[0.0], [1.0], [2.0]])
    doe = np.array([[1.0]])
    eligible = np.array([True, False, True])
    learning, region, indices = compute_learning_function(
        pool,
        doe,
        g_predict=np.array([0.2, 0.0, -0.2]),
        g_mse=np.ones(3),
        v_pdf_pool=np.array([1.0, 2.0, 3.0]),
        current_pf=0.5,
        model=None,
        w_grad=0.0,
        candidate_eligible=eligible,
    )

    assert 1 not in indices
    assert not np.any(np.all(region == doe[0], axis=1))
    assert learning.shape[0] == region.shape[0] == indices.shape[0]


def test_logical_candidate_removal_matches_physical_removal_scores() -> None:
    pool = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    pdf = np.array([0.2, 0.5, 1.0, 0.5, 0.2])
    predicted = np.array([0.4, 0.2, 0.05, -0.2, -0.4])
    mse = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    eligible = np.array([True, True, False, True, True])
    doe = np.array([[3.0]])

    logical_lf, logical_region, logical_indices = compute_learning_function(
        pool,
        doe,
        predicted,
        mse,
        pdf,
        current_pf=0.4,
        model=None,
        w_grad=0.0,
        candidate_eligible=eligible,
    )
    physical_lf, physical_region, physical_indices = compute_learning_function(
        pool[eligible],
        doe,
        predicted[eligible],
        mse[eligible],
        pdf[eligible],
        current_pf=0.4,
        model=None,
        w_grad=0.0,
    )

    np.testing.assert_allclose(logical_lf, physical_lf, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(logical_region, physical_region)
    np.testing.assert_array_equal(
        logical_indices,
        np.flatnonzero(eligible)[physical_indices],
    )


def test_adaptive_checkpoint_separates_integration_and_candidate_sets(
    monkeypatch, tmp_path
) -> None:
    adaptive_module = importlib.import_module("absvr_core.adaptive_loop")
    monkeypatch.setattr(adaptive_module, "N_MCS", 32)
    monkeypatch.setattr(adaptive_module, "MAX_MCS_POOL_SIZE", 32)
    checkpoint = tmp_path / "separated-pools.npz"
    problem = (
        np.zeros(1),
        np.ones(1),
        1,
        {},
        lambda x, _params: np.asarray(x)[:, 0],
        0.5,
        "normal",
    )

    result = adaptive_module.run_adaptive_svr(
        "separated-pools",
        1,
        problem_definition=problem,
        n0=5,
        random_seed=20260808,
        learning_w_grad=0.0,
        checkpoint_path=str(checkpoint),
        verbose=False,
    )
    state = load_checkpoint(checkpoint)

    assert result.total_evaluations == 6
    assert state["mc_pool"].shape[0] == 32
    assert state["v_pdf_pool"].shape[0] == 32
    assert state["candidate_eligible"].shape == (32,)
    assert np.count_nonzero(state["candidate_eligible"]) == 31
