"""Out-of-distribution scoring helpers for candidate selection."""

from __future__ import annotations

import numpy as np

NORMALIZATION_EPS = 1e-9
SCORE_CLIP = 10.0


def score_standard_distance(points: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Return the L2 distance in the standardized input space."""

    if reference is None:
        return np.zeros(points.shape[0], dtype=float)

    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(points, dtype=float)
    if ref.ndim != 2 or ref.shape[0] == 0:
        return np.zeros(pts.shape[0], dtype=float)
    if pts.ndim != 2:
        pts = np.atleast_2d(pts)

    mean = ref.mean(axis=0)
    std = ref.std(axis=0, ddof=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    z_points = (pts - mean) / std_safe
    return np.linalg.norm(z_points, axis=1)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Return robustly normalised scores centred at zero."""

    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) + NORMALIZATION_EPS
    normalized = (arr - median) / mad
    return np.clip(normalized, -SCORE_CLIP, SCORE_CLIP)


def apply_tail_policy(
    acquisition: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float,
    policy: str,
) -> np.ndarray:
    """Adjust acquisition scores according to the requested tail policy."""

    policy = (policy or "none").lower()
    if policy not in {"down", "explore"}:
        return acquisition

    normalized = normalize_scores(scores)
    alpha = float(max(alpha, 0.0))

    if policy == "down":
        weights = np.exp(-alpha * normalized)
    else:
        weights = np.exp(alpha * normalized)

    return np.asarray(acquisition, dtype=float) * weights
