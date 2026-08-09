"""Deterministic evidence-based hyperparameter selection for Bayesian SVR."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Sequence

import numpy as np

from .kernel_utils import normalize_kernel_name
from .model import _svr_likelihood
from .train_internal import svr_train_internal


def _normalized_training_parameters(X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Build the normalized parameter dictionary used by the SVR internals."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != Y.size:
        raise ValueError("X and Y must contain the same number of training samples.")

    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0, ddof=0)
    input_std[input_std == 0.0] = 1.0
    X_normalized = (X - input_mean) / input_std

    output_mean = float(Y.mean())
    output_std = float(Y.std(ddof=0))
    if output_std == 0.0:
        output_std = 1.0
    Y_normalized = (Y - output_mean) / output_std

    row_idx, col_idx = np.triu_indices(X.shape[0], k=1)
    return {
        "D": X_normalized[row_idx] - X_normalized[col_idx],
        "ij": np.column_stack([row_idx, col_idx]),
        "Y": Y_normalized,
        "X": X_normalized,
        "Inputmoment": np.vstack([input_mean, input_std]),
        "Outputmoment": np.array([[output_mean], [output_std]]),
    }


def _unique_clipped(values: Sequence[float], lower: float, upper: float) -> tuple[float, ...]:
    clipped = np.clip(np.asarray(values, dtype=float), lower, upper)
    return tuple(float(value) for value in np.unique(clipped))


@dataclass
class PeriodicEvidenceGridTrainer:
    """Select isotropic Gaussian-SVR hyperparameters using model evidence.

    A broad deterministic grid is used on the initial design. Later searches
    are local multiplicative refinements around the previous evidence optimum.
    Only the current design and its limit-state responses enter the objective.
    """

    retune_interval: int = 20
    initial_c_grid: tuple[float, ...] = (1.0e2, 1.0e3, 1.0e4)
    epsilon_grid: tuple[float, ...] = (1.0e-5, 1.0e-3, 1.0e-2)
    initial_theta_grid: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    c_factors: tuple[float, ...] = (0.1, 1.0, 10.0)
    theta_factors: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    c_bounds: tuple[float, float] = (10.0, 1.0e5)
    theta_bounds: tuple[float, float] = (1.0e-4, 1.0)
    history: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _best: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_tuned_samples: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.retune_interval <= 0:
            raise ValueError("retune_interval must be positive.")
        if not self.initial_c_grid or not self.epsilon_grid or not self.initial_theta_grid:
            raise ValueError("Evidence grids cannot be empty.")
        if self.c_bounds[0] <= 0.0 or self.c_bounds[0] > self.c_bounds[1]:
            raise ValueError("c_bounds must be positive and ordered.")
        if self.theta_bounds[0] <= 0.0 or self.theta_bounds[0] > self.theta_bounds[1]:
            raise ValueError("theta_bounds must be positive and ordered.")

    @property
    def best_hyperparameters(self) -> np.ndarray | None:
        """Return a defensive copy of ``[C, epsilon, isotropic theta]``."""

        return None if self._best is None else self._best.copy()

    def _candidate_grid(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        if self._best is None:
            c_values = _unique_clipped(self.initial_c_grid, *self.c_bounds)
            theta_values = _unique_clipped(self.initial_theta_grid, *self.theta_bounds)
        else:
            c_values = _unique_clipped(
                [self._best[0] * factor for factor in self.c_factors],
                *self.c_bounds,
            )
            theta_values = _unique_clipped(
                [self._best[2] * factor for factor in self.theta_factors],
                *self.theta_bounds,
            )
        epsilon_values = tuple(float(value) for value in self.epsilon_grid)
        return c_values, epsilon_values, theta_values

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_dim: int,
        *,
        kernel: str = "gaussian",
        theta_init: float | None = None,
        c_init: float | None = None,
        epsilon_init: float | None = None,
        bounds_mode: str | None = None,
    ) -> dict[str, Any]:
        """Fit a model, retuning on schedule with the corrected evidence."""

        del theta_init, c_init, epsilon_init, bounds_mode
        canonical, covariance = normalize_kernel_name(kernel)
        if canonical != "gaussian":
            raise ValueError("PeriodicEvidenceGridTrainer supports only the Gaussian kernel.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != int(n_dim):
            raise ValueError("n_dim must match the number of columns in X.")
        par = _normalized_training_parameters(X, Y)
        sample_count = X.shape[0]
        should_retune = (
            self._best is None
            or sample_count - self._last_tuned_samples >= self.retune_interval
        )

        if should_retune:
            c_values, epsilon_values, theta_values = self._candidate_grid()
            best_score = float("inf")
            best = None
            evaluated = 0
            for c_value, epsilon_value, theta_value in product(
                c_values,
                epsilon_values,
                theta_values,
            ):
                hyperparameters = np.concatenate(
                    [[c_value, epsilon_value], np.full(n_dim, theta_value)]
                )
                try:
                    score = float(_svr_likelihood(hyperparameters, par, covariance))
                except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                    score = float("inf")
                evaluated += 1
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best = np.array([c_value, epsilon_value, theta_value], dtype=float)

            if best is None:
                raise RuntimeError("No finite SVR evidence value was found on the grid.")
            self._best = best
            self._last_tuned_samples = sample_count
            self.history.append(
                {
                    "n_samples": int(sample_count),
                    "C": float(best[0]),
                    "epsilon": float(best[1]),
                    "theta": float(best[2]),
                    "negative_log_evidence": float(best_score),
                    "candidates_evaluated": int(evaluated),
                }
            )

        assert self._best is not None
        fixed_hyperparameters = np.concatenate(
            [[self._best[0], self._best[1]], np.full(n_dim, self._best[2])]
        )
        return svr_train_internal(par, fixed_hyperparameters, covariance)


__all__ = ["PeriodicEvidenceGridTrainer"]
