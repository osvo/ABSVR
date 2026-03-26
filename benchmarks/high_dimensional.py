"""High-dimensional benchmark limit-state function.

This benchmark tests the algorithm's scalability to problems with many
random variables (default n=40).
"""

import numpy as np


def high_dimensional(x: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate the high-dimensional limit-state function.

    The limit-state function is defined as:
        g(x₁, ..., xₙ) = n + 3σ√n - 1.05·Σᵢ₌₁ⁿ √xᵢ

    where the inputs xᵢ are typically lognormal random variables
    (ensuring positivity for the square root).

    Failure occurs when g(x) ≤ 0.

    Parameters
    ----------
    x : np.ndarray
        Candidate points with shape ``(n_samples, n_dim)``. Each row contains
        the n-dimensional input vector. Values must be positive for the
        square root operation.
    params : dict
        Configuration dictionary with keys:
        - ``n_dim`` : int, number of dimensions (default 40)
        - ``sigma`` : float, standard deviation parameter (default 0.22)

    Returns
    -------
    np.ndarray
        Limit-state evaluations ``g(x)`` with shape ``(n_samples,)``.
        Negative values indicate failure.

    Raises
    ------
    ValueError
        If input dimensions don't match ``n_dim`` or if negative values
        are encountered.

    Notes
    -----
    This problem is designed to test high-dimensional reliability analysis.
    The inputs should be sampled from lognormal distributions to ensure
    positivity. With n=40 and σ=1.0, this gives Pf ≈ 5.16e-3 (MCS-verified).

    For standard usage with lognormal inputs:
    - Mean of underlying normal: μ_ln = 0
    - Std of underlying normal: σ_ln = 1
    - Resulting lognormal mean: exp(0.5) ≈ 1.649
    - Resulting lognormal std: sqrt((exp(1)-1)*exp(1)) ≈ 2.161

    The sigma parameter in the limit-state function controls the safety margin:
    - sigma=0.5: Pf ≈ 30%
    - sigma=1.0: Pf ≈ 5e-3 (recommended)
    - sigma=1.1: Pf ≈ 1.7e-3

    Reference: Results.tex Section 3.4
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_dim = params.get("n_dim", 40)
    sigma = params.get("sigma", 0.22)

    if x.shape[1] != n_dim:
        raise ValueError(
            f"Expected {n_dim} columns in x, received {x.shape[1]}"
        )

    # Handle potential numerical issues with small/negative values
    # Clip to small positive value to avoid sqrt of negative numbers
    x_safe = np.maximum(x, 1e-10)

    # g(x) = n + 3σ√n - 1.05·Σ√xᵢ
    sqrt_sum = np.sum(np.sqrt(x_safe), axis=1)
    g = n_dim + 3.0 * sigma * np.sqrt(n_dim) - 1.05 * sqrt_sum

    return g
