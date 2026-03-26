"""Dynamic response of a nonlinear oscillator benchmark."""

import numpy as np


def nonlinear_oscillator(x: np.ndarray, parameter: float) -> np.ndarray:
    """Evaluate the nonlinear oscillator limit-state.

    Parameters
    ----------
    x : numpy.ndarray
        Candidate points with shape (n_samples, 6) ordered as
        ``[m, c1, c2, r, F1, t1]``.
    parameter : float
        Threshold parameter controlling the limit-state level.

    Returns
    -------
    numpy.ndarray
        Limit-state evaluations ``g(x)`` with shape (n_samples,).
    """

    if x.shape[1] != 6:
        raise ValueError(f"Expected 6 columns in x, received {x.shape[1]}")

    m = x[:, 0]
    c1 = x[:, 1]
    c2 = x[:, 2]
    r = x[:, 3]
    f1 = x[:, 4]
    t1 = x[:, 5]

    w0 = np.sqrt((c1 + c2) / m)
    return parameter * r - np.abs(2.0 * f1 / m / (w0**2) * np.sin(w0 * t1 / 2.0))
