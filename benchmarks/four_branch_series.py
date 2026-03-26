"""Four-branch series system benchmark."""

import numpy as np


def four_branch_series(x: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate the four-branch series limit-state.

    Parameters
    ----------
    x : numpy.ndarray
        Candidate points with shape (n_samples, 2).
    params : dict
        Dictionary with keys ``type``, ``a``, ``b`` and ``f0`` controlling the
        configuration used in the benchmark definition.

    Returns
    -------
    numpy.ndarray
        Limit-state evaluations ``g(x)`` with shape (n_samples,).
    """

    fbss_type = params.get("type", 1)
    a = params["a"]
    b = params["b"]
    f0 = params.get("f0", 0.0)

    x1 = x[:, 0]
    x2 = x[:, 1]

    f1 = a + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2.0)
    f2 = a + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2.0)
    f3 = (x1 - x2) + b / np.sqrt(2.0)
    f4 = (x2 - x1) + b / np.sqrt(2.0)

    if fbss_type == 1:
        g = np.minimum.reduce([f1, f2, f3, f4])
    elif fbss_type == 2:
        g = f1
    elif fbss_type == 3:
        g = f2
    elif fbss_type == 4:
        g = f3
    elif fbss_type == 5:
        g = f4
    elif fbss_type == 6:
        y1 = np.minimum.reduce([f1, f2, f3, f4])
        y2 = np.maximum.reduce([f1, f2, f3, f4])
        g = np.where(y1 > 0.0, y2, y1)
    else:
        raise ValueError(f"Unknown FBSS type: {fbss_type}")

    return g - f0
