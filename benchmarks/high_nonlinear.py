"""High nonlinearity benchmark limit-state function."""

import numpy as np


def high_nonlinear(x: np.ndarray, parameter: float) -> np.ndarray:
    """Evaluate the high nonlinearity limit-state.

    Parameters
    ----------
    x : numpy.ndarray
        Candidate points with shape (n_samples, 2).
    parameter : float
        Threshold parameter ``d`` for this benchmark definition.

    Returns
    -------
    numpy.ndarray
        Limit-state evaluations ``g(x)`` with shape (n_samples,).
    """

    u1 = x[:, 0]
    u2 = x[:, 1]
    return parameter - ((u1**2 + 4.0) * (u2 - 1.0) / 20.0) + np.sin(2.5 * u1)
