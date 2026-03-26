"""Gradient utilities for SVR mean predictions."""

from __future__ import annotations

import numpy as np

from .kernel_utils import _determine_tile_rows, expand_theta


def svr_mean_grad(X: np.ndarray, model: dict) -> np.ndarray:
    """Return the gradient of the SVR predictive mean at the query points.

    Parameters
    ----------
    X:
        Query locations with shape ``(n_samples, n_dim)`` or ``(n_dim,)``.
    model:
        Trained SVR model dictionary produced by :func:`surrogate.svr.train_svr`.

    Returns
    -------
    np.ndarray
        Array with the same leading dimension as ``X`` containing the gradient
        of the predictive mean with respect to the original (unnormalized)
        inputs.
    """

    X = np.asarray(X, dtype=float)
    was_1d = X.ndim == 1
    if was_1d:
        X = X[None, :]

    if X.size == 0:
        zero = np.zeros_like(X, dtype=float)
        return zero[0] if was_1d else zero

    def _format(arr: np.ndarray) -> np.ndarray:
        return arr[0] if was_1d else arr

    sv = model.get("SV")
    if sv is None or sv.size == 0:
        return _format(np.zeros_like(X, dtype=float))

    covariance = model.get("Covariance", "Gaussian")
    if covariance != "Gaussian":
        return _format(np.zeros_like(X, dtype=float))

    X_train = np.asarray(model["Input"], dtype=float)
    beta = np.asarray(model["parameter"], dtype=float)
    X_sv = X_train[sv]
    beta_sv = beta[sv]
    if X_sv.size == 0:
        return _format(np.zeros_like(X, dtype=float))

    input_moment = np.asarray(model["Inputmoment"], dtype=float)
    output_moment = np.asarray(model["Outputmoment"], dtype=float)
    M_input = input_moment[0, :]
    S_input = input_moment[1, :]
    # Guard against degenerate standard deviations produced by constant columns.
    S_input_safe = np.where(S_input == 0.0, 1.0, S_input)
    X_normalized = (X - M_input) / S_input_safe

    theta = np.asarray(model["theta"], dtype=float)
    n_dim = X.shape[1]
    theta_vec = expand_theta(theta, n_dim)

    tile_rows = _determine_tile_rows(X_normalized.shape[0], X_sv.shape[0], n_dim)
    if tile_rows <= 0:
        tile_rows = X_normalized.shape[0]

    grad_normalized = np.zeros_like(X_normalized, dtype=float)
    theta_row = theta_vec.reshape(1, 1, -1)
    beta_row = beta_sv.reshape(1, -1)

    for start in range(0, X_normalized.shape[0], tile_rows):
        end = min(start + tile_rows, X_normalized.shape[0])
        x_tile = X_normalized[start:end]
        differences = x_tile[:, None, :] - X_sv[None, :, :]
        squared = differences**2
        kernel_values = np.exp(-np.sum(theta_vec * squared, axis=2))
        weights = kernel_values * beta_row
        grad_tile = np.sum(weights[..., None] * (-2.0 * theta_row) * differences, axis=1)
        grad_normalized[start:end] = grad_tile

    S_output = output_moment[1, :]
    # Map gradients back to the original input space.
    output_scale = float(S_output[0]) if S_output.ndim else float(S_output)
    input_scale = S_input_safe.reshape(1, -1)
    grad = grad_normalized * (output_scale / input_scale)
    return _format(grad)
