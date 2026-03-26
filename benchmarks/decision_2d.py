"""2D Decision function benchmark limit-state function.

This benchmark combines exponential decay and polynomial terms, creating
a smooth but non-trivial failure boundary.
"""

import numpy as np


def decision_2d(x: np.ndarray, parameter: float = 0.0) -> np.ndarray:
    """Evaluate the 2D decision limit-state function.

    The limit-state function is defined as:
        g(x₁, x₂) = 2 - x₂ + exp(-x₁²/10) + (x₁/5)⁴

    Failure occurs when g(x) ≤ 0.

    Parameters
    ----------
    x : np.ndarray
        Candidate points with shape ``(n_samples, 2)``. Each row contains
        ``[x₁, x₂]`` coordinates in standard normal space.
    parameter : float, optional
        Offset parameter added to the limit-state value. Default is 0.0.

    Returns
    -------
    np.ndarray
        Limit-state evaluations ``g(x)`` with shape ``(n_samples,)``.
        Negative values indicate failure.

    Notes
    -----
    This function creates a curved failure boundary where:
    - The exponential term exp(-x₁²/10) provides a Gaussian-like bump
    - The quartic term (x₁/5)⁴ dominates at large |x₁|
    - The linear -x₂ term shifts the boundary based on the second variable

    Reference: Results.tex Section 3.3
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in x, received {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    # g(x) = 2 - x₂ + exp(-x₁²/10) + (x₁/5)⁴
    g = 2.0 - x2 + np.exp(-x1**2 / 10.0) + (x1 / 5.0) ** 4

    return g + parameter
