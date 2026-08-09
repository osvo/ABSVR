"""Utilities to enrich the design of experiments."""

import numpy as np


def select_best_candidate(
    lf: np.ndarray,
    mc_pool_region: np.ndarray,
    region_indices: np.ndarray,
    doe: np.ndarray,
    g: np.ndarray,
    mc_pool: np.ndarray,
    v_pdf_pool: np.ndarray,
    n_mc: int,
    n_samples_added: int,
    fun,
    fun_par,
):
    """Select and evaluate the best candidate sample."""

    if lf.size == 0:
        return doe, g, mc_pool, v_pdf_pool, n_mc, n_samples_added

    best_idx = int(np.argmin(lf))
    best_point = mc_pool_region[best_idx]
    g_value = np.asarray(fun(best_point[None, :], fun_par), dtype=float).reshape(-1)
    doe = np.vstack([doe, best_point])
    g = np.concatenate([g, g_value])

    # Keep the Monte Carlo population fixed.  Removing adaptively selected
    # points would preferentially deplete samples near the limit-state surface
    # and bias the empirical failure probability.  The learning function masks
    # points already present in the DoE, so they cannot be selected twice.
    n_samples_added += 1
    return doe, g, mc_pool, v_pdf_pool, n_mc, n_samples_added
