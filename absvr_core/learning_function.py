"""Computation of the adaptive learning function used for enrichment."""

import numpy as np

try:
    from surrogate.svr.grad import svr_mean_grad
except ImportError:  # pragma: no cover
    from ..surrogate.svr.grad import svr_mean_grad  # type: ignore


SAMPLING_REGION_FACTOR = 0.1
SLF_PENALTY_FACTOR = 100.0
NUMERICAL_STABILITY_TERM = 1e-8


def compute_learning_function(
    mc_pool: np.ndarray,
    doe: np.ndarray,
    g_predict: np.ndarray,
    g_mse: np.ndarray,
    v_pdf_pool: np.ndarray,
    current_pf: float,
    model=None,
    w_grad: float = 1.0,
):
    """Return the learning function and associated pool subset."""

    grad_weight = float(w_grad)
    if grad_weight < 0.0:
        grad_weight = 0.0

    v_pdf = v_pdf_pool
    v_joint_sorted = np.sort(v_pdf)
    threshold_position = int(np.ceil(SAMPLING_REGION_FACTOR * current_pf * mc_pool.shape[0]))
    if threshold_position <= 0:
        threshold_position = 1
    if threshold_position > v_joint_sorted.size:
        threshold_position = v_joint_sorted.size
    pdf_threshold = v_joint_sorted[threshold_position - 1]
    region_indices = np.where(v_pdf > pdf_threshold)[0]
    mc_pool_region = mc_pool[region_indices]
    g_predict_region = g_predict[region_indices]
    g_mse_region = g_mse[region_indices]
    v_joint_region = v_pdf[region_indices]

    if not region_indices.size:
        return np.array([]), mc_pool_region, region_indices

    v_joint_region_norm = v_joint_region / np.max(v_joint_region)
    sqrt_g_mse = np.sqrt(np.maximum(g_mse_region, 0.0))
    max_std = np.max(sqrt_g_mse)
    if max_std <= 0.0 or not np.isfinite(max_std):
        g_mse_region_norm = np.ones_like(sqrt_g_mse, dtype=float)
    else:
        g_mse_region_norm = sqrt_g_mse / max_std

    # Compute \|x - y\|^2 using the identity \|x\|^2 + \|y\|^2 - 2 x·y
    doe_sq = np.sum(doe**2, axis=1)
    region_sq = np.sum(mc_pool_region**2, axis=1, keepdims=True)
    pairwise_sq = region_sq + doe_sq[None, :] - 2.0 * mc_pool_region @ doe.T
    pairwise_sq = np.maximum(pairwise_sq, 0.0)
    min_distance = np.sqrt(np.min(pairwise_sq, axis=1))

    grad_factor = np.ones_like(min_distance, dtype=float)
    if model is not None and grad_weight > 0.0 and mc_pool_region.size:
        try:
            gradients = svr_mean_grad(mc_pool_region, model)
        except Exception:
            gradients = None
        if gradients is not None and gradients.size:
            gradients = np.asarray(gradients, dtype=float)
            if gradients.ndim == 1:
                gradients = gradients.reshape(-1, 1)
            grad_norm = np.linalg.norm(gradients, axis=1)
            grad_norm = np.where(np.isfinite(grad_norm), grad_norm, 0.0)
            if grad_norm.size:
                max_grad = np.max(grad_norm)
                if max_grad > 0.0 and np.isfinite(max_grad):
                    grad_norm = grad_norm / max_grad
                    grad_norm = np.clip(grad_norm, 0.0, 1.0)
                    grad_factor = 1.0 + grad_weight * grad_norm

    max_abs_g = np.max(np.abs(g_predict_region))
    if max_abs_g == 0:
        scaled = np.full_like(g_predict_region, np.inf)
    else:
        scaled = np.abs(g_predict_region) / max_abs_g
    numerator = 1.0 + np.exp(SLF_PENALTY_FACTOR * scaled)
    denominator = NUMERICAL_STABILITY_TERM + (g_mse_region_norm * v_joint_region_norm) * min_distance * grad_factor
    lf = numerator / denominator

    return lf, mc_pool_region, region_indices
