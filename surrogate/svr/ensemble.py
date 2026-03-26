"""Lightweight SVR ensembling helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from utilities.random_stream import global_random_state
except ImportError:  # pragma: no cover
    from ..utilities.random_stream import global_random_state  # type: ignore

from .kernel_utils import normalize_kernel_name
from .predict import svr_predict
from .train import train_svr


def _prepare_rng(seed: int | None):
    if seed is None:
        return global_random_state()
    return np.random.RandomState(seed)


def train_svr_ensemble(
    X: np.ndarray,
    Y: np.ndarray,
    n_dim: int,
    n_models: int = 8,
    bootstrap: bool = True,
    seed: int | None = None,
    kernel: str = "gaussian",
    theta_init: float | None = None,
    c_init: float | None = None,
    epsilon_init: float | None = None,
    bounds_mode: str | None = None,
) -> dict:
    """Train a bagged SVR ensemble.

    Parameters
    ----------
    X, Y:
        Training data in the original design space.
    n_dim:
        Feature dimension, mirrored from :func:`train_svr`.
    n_models:
        Number of base learners to train.
    bootstrap:
        Whether to resample with replacement for each member.
    seed:
        Optional seed to reinitialize the shared random stream before training.
    c_init:
        Initial SVR regularization parameter C.
    epsilon_init:
        Initial ε-insensitive tube width.
    bounds_mode:
        Bound profile passed to :func:`train_svr`.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    if n_models <= 0:
        raise ValueError("n_models must be positive.")

    rng = _prepare_rng(seed)
    canonical_kernel, covariance = normalize_kernel_name(kernel)
    members: list[dict] = []
    n_samples = X.shape[0]

    for _ in range(int(n_models)):
        if bootstrap and n_samples:
            indices = rng.randint(0, n_samples, size=n_samples)
        else:
            indices = np.arange(n_samples, dtype=int)
        member = train_svr(
            X[indices],
            Y[indices],
            n_dim,
            kernel=canonical_kernel,
            theta_init=theta_init,
            c_init=c_init,
            epsilon_init=epsilon_init,
            bounds_mode=bounds_mode,
        )
        members.append(member)

    ensemble = {
        "type": "svr_ensemble",
        "models": members,
        "bootstrap": bool(bootstrap),
        "n_models": len(members),
        "kernel": canonical_kernel,
        "covariance": covariance,
    }
    if members:
        ensemble["primary_model"] = members[0]
    return ensemble


def svr_predict_ensemble(X: np.ndarray, ensemble: dict) -> tuple[np.ndarray, np.ndarray]:
    """Predict the mean and variance aggregated across an SVR ensemble."""

    models: Iterable[dict] = ensemble.get("models", [])
    models = list(models)
    if not models:
        return np.array([]), np.array([])

    X = np.asarray(X, dtype=float)
    predictions = []
    variances = []
    for model in models:
        mean, variance = svr_predict(X, model)
        predictions.append(mean)
        variances.append(variance)

    pred_stack = np.stack(predictions, axis=0)
    var_stack = np.stack(variances, axis=0)
    mean_pred = pred_stack.mean(axis=0)
    epistemic = pred_stack.var(axis=0, ddof=0)
    aleatoric = var_stack.mean(axis=0)
    combined_var = np.maximum(0.0, aleatoric + epistemic)
    return mean_pred, combined_var
