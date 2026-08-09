"""Internal SVR training routine for epsilon-SVR."""

from typing import Optional

import numpy as np
from scipy import optimize

try:  # Optional CVXOPT acceleration
    from cvxopt import matrix as cvx_matrix, solvers as cvx_solvers

    cvx_solvers.options["show_progress"] = False
    cvx_solvers.options["abstol"] = 1e-10
    cvx_solvers.options["reltol"] = 1e-9
    cvx_solvers.options["feastol"] = 1e-9
    cvx_solvers.options["refinement"] = 2
    cvx_solvers.options["maxiters"] = 200
    _HAVE_CVXOPT = True
except ImportError:  # pragma: no cover - optional
    _HAVE_CVXOPT = False

from .kernel_utils import compute_kernel


def _solve_qp(
    H: np.ndarray, f: np.ndarray, Aeq: np.ndarray, beq: float, upper_bound: Optional[float]
) -> np.ndarray:
    """Solve the quadratic program using CVXOPT when available."""

    H = 0.5 * (H + H.T)
    n = H.shape[0]

    if upper_bound is not None:
        if not np.isfinite(upper_bound):
            upper_bound = None
        elif upper_bound <= 0.0:
            raise ValueError("upper_bound must be positive when specified.")

    if _HAVE_CVXOPT:
        P = cvx_matrix(H)
        q = cvx_matrix(f.astype(float))
        A = cvx_matrix(Aeq.reshape(1, -1))
        b = cvx_matrix([float(beq)])
        if upper_bound is None:
            G = cvx_matrix(-np.eye(n))
            h = cvx_matrix(np.zeros(n))
        else:
            G = cvx_matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h = cvx_matrix(
                np.concatenate([np.zeros(n, dtype=float), np.full(n, upper_bound, dtype=float)])
            )
        try:
            solution = cvx_solvers.qp(P, q, G, h, A, b)
        except Exception:
            solution = None
        else:
            if solution.get("status") == "optimal":  # type: ignore[arg-type]
                return np.array(solution["x"], dtype=float).ravel()

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * alpha @ (H @ alpha) + f @ alpha

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return H @ alpha + f

    def hessian(alpha: np.ndarray) -> np.ndarray:
        return H

    linear_constraint = optimize.LinearConstraint(Aeq[np.newaxis, :], beq, beq)
    ub = np.inf if upper_bound is None else float(upper_bound)
    lb = np.zeros(n)
    bounds = optimize.Bounds(lb, np.full(n, ub))

    result = optimize.minimize(
        objective,
        np.zeros(n),
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        constraints=[linear_constraint],
        bounds=bounds,
        options={"gtol": 1e-7, "xtol": 1e-7, "maxiter": 500, "verbose": 0},
    )

    if not result.success:
        slsqp_bounds = [(0.0, None if np.isinf(ub) else ub)] * n
        result = optimize.minimize(
            objective,
            np.zeros(n),
            jac=gradient,
            bounds=slsqp_bounds,
            constraints=[
                {"type": "eq", "fun": lambda alpha: Aeq @ alpha - beq, "jac": lambda alpha: Aeq}
            ],
            method="SLSQP",
            options={"ftol": 1e-7, "maxiter": 2000, "disp": False},
        )
    if not result.success:
        raise RuntimeError(f"SVR QP solver failed: {result.message}")
    return result.x


def svr_train_internal(par: dict, hyperparameters: np.ndarray, covariance: str) -> dict:
    """Train an epsilon-SVR model for fixed hyperparameters."""

    C = float(hyperparameters[0])
    epsilon = float(hyperparameters[1])
    theta = hyperparameters[2:]

    X = par["X"]
    Y = par["Y"]
    m = X.shape[0]

    kernel = compute_kernel(X, X, theta, covariance)
    mu = (10.0 + m) * np.finfo(float).eps
    kernel[np.diag_indices_from(kernel)] += 2.0 * mu
    kernel1 = kernel + np.diag(np.full(m, 1.0 / C))

    # Dual Hessian for epsilon-insensitive squared loss.  The 1/C term
    # penalizes alpha and alpha-star separately, so it belongs only on the
    # diagonal blocks.  Unlike L1 epsilon-SVR, the multipliers have no C box
    # constraint in this L2 formulation.
    hb = np.block([[kernel1, -kernel], [-kernel, kernel1]])
    f = np.concatenate([(epsilon - Y), (epsilon + Y)])

    aeq = np.concatenate([np.ones(m), -np.ones(m)])
    beq = 0.0

    alpha = _solve_qp(hb, f, aeq, beq, None)
    alpha_pos = alpha[:m]
    alpha_neg = alpha[m:]

    beta = alpha_pos - alpha_neg

    prediction = kernel @ beta

    # KKT conditions for epsilon-insensitive squared loss:
    # y_i - f_i - b = epsilon + alpha_i/C for positive multipliers, and
    # y_i - f_i - b = -epsilon - alpha_i_star/C for negative multipliers.
    sv_tol = 1e-10
    active_pos = alpha_pos > sv_tol
    active_neg = alpha_neg > sv_tol
    b_candidates = []
    if np.any(active_pos):
        b_candidates.append(
            Y[active_pos] - prediction[active_pos] - epsilon - alpha_pos[active_pos] / C
        )
    if np.any(active_neg):
        b_candidates.append(
            Y[active_neg] - prediction[active_neg] + epsilon + alpha_neg[active_neg] / C
        )
    if b_candidates:
        bias = float(np.median(np.concatenate(b_candidates)))
    else:
        bias = float(np.mean(Y - prediction))

    # Interior-point QP solvers can leave tiny positive multipliers on samples
    # that are still strictly inside the epsilon tube.  The EISLF curvature
    # matrix uses only loss-active samples, so identify support vectors from
    # the KKT residual after estimating the bias instead of from alpha alone.
    residual = Y - (prediction + bias)
    support_vector_mask = ((beta > 0.0) & (residual > epsilon)) | (
        (beta < 0.0) & (residual < -epsilon)
    )
    sv_indices = np.nonzero(support_vector_mask)[0]

    if sv_indices.size:
        # Cache the Cholesky factor for the active support vectors to reuse during prediction.
        try:
            kernel1_cholesky = np.linalg.cholesky(kernel1[np.ix_(sv_indices, sv_indices)])
        except np.linalg.LinAlgError:
            kernel1_cholesky = None
    else:
        kernel1_cholesky = None

    return {
        "Input": X,
        "Output": Y,
        "parameter": beta,
        "Covariance": covariance,
        "C": C,
        "epsilon": epsilon,
        "SV": sv_indices,
        "bias": bias,
        "Kernelmatrix": kernel,
        "Kernelmatrix1": kernel1,
        "Kernelmatrix1_cholesky": kernel1_cholesky,
        "Inputmoment": par["Inputmoment"],
        "Outputmoment": par["Outputmoment"],
        "theta": theta,
    }
