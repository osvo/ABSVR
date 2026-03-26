"""Random sampling utilities for Latin hypercube sampling."""

import numpy as np

from .random_stream import global_random_state


def _randperm(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Generate a random permutation using the shared RNG."""

    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(np.floor(rng.random_sample() * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def lhs_uniform(mean: np.ndarray, sigma: np.ndarray, n_samples: int) -> tuple:
    """Generate Latin Hypercube samples mapped to a uniform box using the shared RNG."""

    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mean.shape != sigma.shape:
        raise ValueError("mean and sigma must have matching shapes")

    dimension = mean.size
    rng = global_random_state()

    uniform = rng.random_sample(n_samples * dimension).reshape((n_samples, dimension), order='F')
    strata = (np.arange(n_samples)[:, None] + uniform) / n_samples

    samples_unit = np.empty_like(strata)
    for j in range(dimension):
        perm = _randperm(rng, n_samples)
        samples_unit[:, j] = strata[perm, j]

    lower = mean - np.sqrt(3.0) * sigma
    upper = mean + np.sqrt(3.0) * sigma
    width = upper - lower
    samples = lower + samples_unit * width
    pdf_value = float(1.0 / np.prod(width))
    pdf = np.full(n_samples, pdf_value)

    return samples, pdf
