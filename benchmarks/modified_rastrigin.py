"""Modified Rastrigin function benchmark limit-state function.

This benchmark features multiple failure regions due to the oscillatory
nature of the Rastrigin function, making it challenging for surrogate-based
reliability methods.
"""

import numpy as np


def modified_rastrigin(x: np.ndarray, parameter: float = 0.0) -> np.ndarray:
    """Evaluate the modified Rastrigin limit-state function.

    The limit-state function is defined as:
        g(x₁, x₂) = 10 - Σᵢ₌₁² (xᵢ² - 5·cos(2πxᵢ))

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
    The standard Rastrigin function is a common optimization benchmark with
    many local minima. This modified version creates multiple disconnected
    failure regions, testing the algorithm's ability to identify complex
    failure boundaries.

    Reference: Results.tex Section 3.2
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in x, received {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    # g(x) = 10 - Σ(xᵢ² - 5·cos(2πxᵢ))
    term1 = x1**2 - 5.0 * np.cos(2.0 * np.pi * x1)
    term2 = x2**2 - 5.0 * np.cos(2.0 * np.pi * x2)

    return 10.0 - term1 - term2 + parameter
