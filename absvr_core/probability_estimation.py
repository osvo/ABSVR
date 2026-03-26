"""Failure probability estimation."""

import numpy as np


def estimate_probability_of_failure(
    g_predict: np.ndarray, g_doe: np.ndarray, n_mc: int
) -> float:
    """Estimate probability of failure from the MC pool.

    Following the paper's definition, only counts failures from the Monte Carlo
    pool predictions, not the DOE evaluations.

    Parameters
    ----------
    g_predict : np.ndarray
        Surrogate predictions on the MC pool (g <= 0 indicates failure).
    g_doe : np.ndarray
        Actual limit-state evaluations on DOE points (unused in paper formula).
    n_mc : int
        Size of the Monte Carlo pool.

    Returns
    -------
    float
        Estimated probability of failure: P(g <= 0) ≈ N_failures / N_mc
    """
    failures_predict = np.count_nonzero(g_predict < 0.0)
    return failures_predict / n_mc
