"""Convenience wrapper to train the SVR surrogate."""

import os
import math

import numpy as np
from scipy.spatial.distance import pdist

from .kernel_utils import normalize_kernel_name
from .loss_utils import LEGACY_LOSS, SQUARE_LOSS, normalize_loss
from .model import svr_model


def _data_driven_defaults(
    X: np.ndarray, Y: np.ndarray, n_dim: int
) -> tuple[float, float, np.ndarray]:
    """Compute data-driven initial hyperparameters.

    These heuristics give the box-min optimizer an informed starting point
    instead of arbitrary fixed values.

    Returns ``(c_init, epsilon_init, theta_init_vector)``.

    References
    ----------
    Cherkassky & Ma (2004) "Practical selection of SVM parameters and noise
        estimation for SVM regression", *Neural Networks*, 17(1), 113-126.
    Garreau, Jitkrittum & Kanagawa (2017) "Large sample analysis of the
        median heuristic", arXiv:1707.07269.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    n = Y.size

    # ── Normalize (mirrors svr_model internalization) ────────────────
    mu_x = X.mean(axis=0)
    s_x = X.std(axis=0, ddof=0)
    s_x[s_x == 0.0] = 1.0
    Xn = (X - mu_x) / s_x

    mu_y = Y.mean()
    s_y = float(Y.std(ddof=0)) or 1.0
    Yn = (Y - mu_y) / s_y

    # ── C: range-based (Cherkassky & Ma 2004) ───────────────────────
    # C = max(|ȳ + 3σ_y|, |ȳ - 3σ_y|) on normalized data (ȳ≈0, σ≈1)
    c_init = float(max(abs(Yn.mean() + 3.0 * Yn.std(ddof=0)),
                       abs(Yn.mean() - 3.0 * Yn.std(ddof=0))))
    c_init = max(c_init, 1.0)  # ensure positive lower bound

    # ── ε: noise-adaptive (Cherkassky & Ma 2004) ────────────────────
    # σ_noise ≈ MAD / 0.6745, then ε = σ_noise * √(ln(n)/n)
    residuals = Yn - np.median(Yn)
    mad = float(np.median(np.abs(residuals)))
    sigma_noise = mad / 0.6745 if mad > 0 else 0.01
    if n > 1:
        epsilon_init = sigma_noise * math.sqrt(math.log(n) / n)
    else:
        epsilon_init = 0.01
    epsilon_init = max(epsilon_init, 1e-7)  # floor

    # ── θ (gamma): median heuristic (Garreau et al. 2017) ───────────
    # θ = 1 / (2 · median(‖x_i - x_j‖²)) per dimension
    if n > 1:
        sq_dists = pdist(Xn, "sqeuclidean")
        med_sq = float(np.median(sq_dists))
        if med_sq > 0:
            theta_scalar = 1.0 / (2.0 * med_sq / n_dim)
        else:
            theta_scalar = 1.0
    else:
        theta_scalar = 1.0
    theta_init = np.full(n_dim, theta_scalar)

    return c_init, epsilon_init, theta_init


def train_svr(
    X: np.ndarray,
    Y: np.ndarray,
    n_dim: int,
    *,
    kernel: str = "gaussian",
    theta_init: float | None = None,
    c_init: float | None = None,
    epsilon_init: float | None = None,
    bounds_mode: str | None = None,
    loss: str = LEGACY_LOSS,
) -> dict:
    """Train the SVR surrogate with reliability-aligned hyperparameter bounds.

    Parameters
    ----------
    X : np.ndarray
        Training inputs with shape ``(n_samples, n_dim)``.
    Y : np.ndarray
        Training targets with shape ``(n_samples,)``.
    n_dim : int
        Number of input dimensions.
    kernel : str, optional
        Kernel type: ``"gaussian"``, ``"laplacian"``, or ``"polynomial"``.
    theta_init : float, optional
        Initial kernel length scale per dimension.  If *None*, computed
        from the median heuristic (Garreau et al. 2017).
    c_init : float, optional
        Initial regularization parameter C.  If *None*, computed from
        the data range (Cherkassky & Ma 2004).
    epsilon_init : float, optional
        Initial ε-insensitive tube width.  If *None*, computed from
        the noise-adaptive rule (Cherkassky & Ma 2004).
    bounds_mode : str, optional
        Bound profile to use (``"python"`` or ``"baseline"``). Defaults to
        env ``ABSVR_SVR_BOUNDS_MODE`` then falls back to ``"python"``.

    Returns
    -------
    dict
        Trained SVR model dictionary.
    """
    loss = normalize_loss(loss)
    mode = (bounds_mode or os.getenv("ABSVR_SVR_BOUNDS_MODE", "python")).strip().lower()
    if mode not in {"python", "baseline"}:
        raise ValueError("bounds_mode must be 'python' or 'baseline'.")

    if mode == "baseline":
        svr_c_lb = 10.0
        svr_c_ub = 1e10
        svr_epsilon_lb = 1e-7
        svr_epsilon_ub = 0.01
        svr_theta_lb = 1e-5
        svr_theta_ub = 1e5
    else:
        svr_c_lb = 1.0
        svr_c_ub = 50.0
        svr_epsilon_lb = 0.05
        svr_epsilon_ub = 0.2
        svr_theta_lb = 1e-5
        svr_theta_ub = 1e5

    # Compute data-driven defaults
    dd_c, dd_eps, dd_theta = _data_driven_defaults(X, Y, n_dim)

    # User-provided values override data-driven defaults
    svr_c_init = float(c_init) if c_init is not None else dd_c
    if loss == SQUARE_LOSS:
        svr_epsilon_init = 0.0
        svr_epsilon_lb = 0.0
        svr_epsilon_ub = 0.0
    else:
        svr_epsilon_init = float(epsilon_init) if epsilon_init is not None else dd_eps

    if theta_init is not None:
        svr_theta_init = np.full(n_dim, float(theta_init))
    else:
        svr_theta_init = dd_theta

    # Clamp initial values to the bounds
    svr_c_init = float(np.clip(svr_c_init, svr_c_lb, svr_c_ub))
    svr_epsilon_init = float(np.clip(svr_epsilon_init, svr_epsilon_lb, svr_epsilon_ub))
    svr_theta_init = np.clip(svr_theta_init, svr_theta_lb, svr_theta_ub)

    # Bounds for hyperparameter optimization (internal boxmin search)
    svr_theta_lb_vec = np.full(n_dim, float(svr_theta_lb))
    svr_theta_ub_vec = np.full(n_dim, float(svr_theta_ub))

    hyperparameters = np.concatenate([[svr_c_init, svr_epsilon_init], svr_theta_init])
    lb = np.concatenate([[svr_c_lb, svr_epsilon_lb], svr_theta_lb_vec])
    ub = np.concatenate([[svr_c_ub, svr_epsilon_ub], svr_theta_ub_vec])

    _, covariance = normalize_kernel_name(kernel)

    return svr_model(
        X,
        Y,
        hyperparameters,
        lb,
        ub,
        covariance,
        loss=loss,
    )

