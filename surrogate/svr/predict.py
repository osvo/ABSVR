"""SVR prediction helper (NumPy only)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .kernel_utils import compute_kernel


def svr_predict(X1: np.ndarray, model: dict) -> tuple:
    """Predict outputs and variance using the trained SVR model."""

    X1 = np.asarray(X1, dtype=float)
    beta = model["parameter"]

    input_moment = model["Inputmoment"]
    output_moment = model["Outputmoment"]
    MInput = input_moment[0, :]
    SInput = input_moment[1, :]
    MOutput = output_moment[0, :]
    SOutput = output_moment[1, :]

    Xn = (X1 - MInput) / SInput
    X_train = model["Input"]
    theta = model["theta"]
    covariance = model["Covariance"]

    sv = model["SV"]

    # Compute predictions using only support vectors for efficiency.
    X_train_sv = X_train[sv] if sv.size else X_train[:0]
    beta_sv = beta[sv] if sv.size else beta[:0]

    Hsv = compute_kernel(Xn, X_train_sv, theta, covariance)
    y = Hsv @ beta_sv + model["bias"]

    if sv.size:
        kernel1 = model["Kernelmatrix1"]
        chol = model.get("Kernelmatrix1_cholesky")
        if chol is None:
            K = kernel1[np.ix_(sv, sv)]
            try:
                chol = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                chol = None
            else:
                model["Kernelmatrix1_cholesky"] = chol
        Hnum = Hsv
        if chol is not None:
            # Triangular solves are significantly cheaper than a full dense solve.
            temp = solve_triangular(chol, Hnum.T, lower=True, check_finite=False)
            temp = solve_triangular(chol.T, temp, lower=False, check_finite=False)
        else:
            temp = np.linalg.solve(kernel1[np.ix_(sv, sv)], Hnum.T)
        variance = 1.0 - np.sum(Hnum.T * temp, axis=0)
        variance = np.maximum(variance, 0.0)
    else:
        variance = np.ones(X1.shape[0])

    SOutput_scalar = SOutput[0]
    MOutput_scalar = MOutput[0]
    y = y * SOutput_scalar + MOutput_scalar
    variance = variance * (SOutput_scalar**2)

    return y, variance
