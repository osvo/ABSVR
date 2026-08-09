"""Regression tests for candidate-population cap handling."""

from __future__ import annotations

import importlib

import numpy as np


def _linear_limit_state(x: np.ndarray, _params: dict) -> np.ndarray:
    return np.asarray(x, dtype=float)[:, 0]


def test_pool_is_evaluated_when_initial_size_equals_cap(monkeypatch) -> None:
    adaptive_module = importlib.import_module("absvr_core.adaptive_loop")
    monkeypatch.setattr(adaptive_module, "N_MCS", 8)
    monkeypatch.setattr(adaptive_module, "MAX_MCS_POOL_SIZE", 8)
    problem = (
        np.zeros(1),
        np.ones(1),
        1,
        {},
        _linear_limit_state,
        0.5,
        "normal",
    )

    result = adaptive_module.run_adaptive_svr(
        "cap-regression",
        0,
        problem_definition=problem,
        n0=5,
        initial_design="normal_lhs",
        random_seed=123,
        learning_w_grad=0.0,
        verbose=False,
    )

    assert result.phase_counts["model_training"] == 1
    assert result.phase_counts["surrogate_prediction"] == 1
    assert 0.0 <= result.probability_of_failure <= 1.0
    assert result.total_evaluations == 5
