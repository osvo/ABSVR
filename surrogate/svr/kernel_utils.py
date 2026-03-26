"""Kernel computation helpers targeting NumPy."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


_EXPLICIT_TILE_ROWS = os.getenv("ABSVR_KERNEL_TILE_ROWS")
# Limit the working set the kernel builder uses; defaults to 128 MiB worth of data.
_MAX_TILE_BYTES = int(os.getenv("ABSVR_KERNEL_TILE_BYTES", str(128 * 1024 * 1024)))

# Global polynomial hyperparameters; degree >= 1 and coef0 >= 0.
_POLY_DEGREE = int(os.getenv("ABSVR_POLY_DEGREE", "3"))
_POLY_COEF0 = float(os.getenv("ABSVR_POLY_COEF0", "1.0"))
_POLY_DOT_CLIP = float(os.getenv("ABSVR_POLY_DOT_CLIP", "50.0"))

_KERNEL_ALIAS_TO_COVARIANCE: dict[str, str] = {
    "gaussian": "Gaussian",
    "rbf": "Gaussian",
    "laplacian": "Laplacian",
    "exponential": "Laplacian",
    "exp": "Laplacian",
    "polynomial": "Polynomial",
    "poly": "Polynomial",
}

_COVARIANCE_TO_CANONICAL: dict[str, str] = {
    "Gaussian": "gaussian",
    "Laplacian": "laplacian",
    "Polynomial": "polynomial",
}


def normalize_kernel_name(kernel: str) -> Tuple[str, str]:
    """Return the canonical kernel identifier and covariance name."""

    key = (kernel or "").strip().lower()
    covariance = _KERNEL_ALIAS_TO_COVARIANCE.get(key)
    if covariance is None:
        valid = ", ".join(sorted({v for v in _COVARIANCE_TO_CANONICAL.values()}))
        raise ValueError(f"Unsupported kernel '{kernel}'. Valid options: {valid}")
    canonical = _COVARIANCE_TO_CANONICAL[covariance]
    return canonical, covariance


def supported_kernel_names() -> tuple[str, ...]:
    """Names the CLI and workflow expose for SVR kernels."""

    return tuple(sorted({name for name in _COVARIANCE_TO_CANONICAL.values()}))


def expand_theta(theta, dimension: int):
    """Ensure ``theta`` is a length-``dimension`` vector."""

    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    if theta_arr.size == 1:
        return np.full(dimension, float(theta_arr[0]), dtype=float)
    if theta_arr.size != dimension:
        raise ValueError(f"theta must have length 1 or {dimension}")
    return theta_arr


def _determine_tile_rows(n_rows: int, n_cols: int, n_dim: int) -> int:
    if n_rows <= 0:
        return 0

    if _EXPLICIT_TILE_ROWS:
        try:
            value = int(_EXPLICIT_TILE_ROWS)
        except ValueError:
            value = n_rows
        return max(1, min(n_rows, value))

    bytes_per_row = n_cols * n_dim * 8
    if bytes_per_row <= 0:
        return n_rows

    if _MAX_TILE_BYTES <= 0:
        return n_rows

    max_rows = max(1, _MAX_TILE_BYTES // bytes_per_row)
    return max(1, min(n_rows, max_rows))


def _compute_gaussian_block(differences: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
    expo = -np.sum(theta_vec * (differences**2), axis=2)
    return np.exp(expo)


def _compute_laplacian_block(differences: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
    gamma = np.sqrt(np.maximum(theta_vec, 0.0))
    expo = -np.sum(gamma * np.abs(differences), axis=2)
    return np.exp(expo)


def _compute_polynomial_block(x1_block: np.ndarray, x2_arr: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(theta_vec, 1e-12))
    x1_scaled = x1_block * scale
    x2_scaled = x2_arr * scale
    dim = max(x1_block.shape[1], 1)
    dot = (x1_scaled @ x2_scaled.T) / float(dim)
    if np.isfinite(_POLY_DOT_CLIP) and _POLY_DOT_CLIP > 0.0:
        dot = np.clip(dot, -_POLY_DOT_CLIP, _POLY_DOT_CLIP)
    return (dot + _POLY_COEF0) ** _POLY_DEGREE


def compute_kernel(x1, x2, theta, covariance: str):
    """Dense kernel matrix between ``x1`` and ``x2`` (NumPy only).

    The computation is tiled along rows of ``x1`` to keep intermediate buffers
    manageable for large candidate pools.
    """

    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)
    if x1_arr.shape[1] != x2_arr.shape[1]:
        raise ValueError("Dimension mismatch between x1 and x2")

    theta_vec = expand_theta(theta, x1_arr.shape[1])
    n_rows = x1_arr.shape[0]
    n_cols = x2_arr.shape[0]
    result = np.empty((n_rows, n_cols), dtype=float)

    tile_rows = _determine_tile_rows(n_rows, n_cols, x1_arr.shape[1])
    if tile_rows <= 0:
        tile_rows = n_rows

    if covariance == "Polynomial":
        for start in range(0, n_rows, tile_rows):
            end = min(start + tile_rows, n_rows)
            block = _compute_polynomial_block(x1_arr[start:end], x2_arr, theta_vec)
            result[start:end] = block
        return result

    for start in range(0, n_rows, tile_rows):
        end = min(start + tile_rows, n_rows)
        differences = x1_arr[start:end, None, :] - x2_arr[None, :, :]
        if covariance == "Gaussian":
            block = _compute_gaussian_block(differences, theta_vec)
        elif covariance == "Laplacian":
            block = _compute_laplacian_block(differences, theta_vec)
        else:
            raise ValueError(f"Unknown covariance family: {covariance}")
        result[start:end] = block

    return result
