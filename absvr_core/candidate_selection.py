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
    candidate_eligible: np.ndarray,
    n_mc: int,
    n_samples_added: int,
    fun,
    fun_par,
):
    """Select and evaluate the best candidate sample."""

    if lf.size == 0:
        return doe, g, mc_pool, v_pdf_pool, candidate_eligible, n_mc, n_samples_added

    best_idx = int(np.argmin(lf))
    best_point = mc_pool_region[best_idx]
    g_value = np.asarray(fun(best_point[None, :], fun_par), dtype=float).reshape(-1)
    doe = np.vstack([doe, best_point])
    g = np.concatenate([g, g_value])

    original_idx = int(region_indices[best_idx])
    candidate_eligible = np.asarray(candidate_eligible, dtype=bool).copy()
    if candidate_eligible.shape != (n_mc,):
        raise ValueError("candidate_eligible must match the Monte Carlo population.")
    candidate_eligible[original_idx] = False

    # Keep the integration population fixed while removing the evaluated point
    # only from the logical candidate set. This preserves an unbiased empirical
    # Pf and the intended no-reselection behavior of adaptive enrichment.
    n_samples_added += 1
    return doe, g, mc_pool, v_pdf_pool, candidate_eligible, n_mc, n_samples_added
