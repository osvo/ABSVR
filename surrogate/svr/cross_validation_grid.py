"""Deterministic cross-validated hyperparameter selection for SVR."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from .evidence_grid import _normalized_training_parameters
from .kernel_utils import normalize_kernel_name
from .loss_utils import LEGACY_LOSS, normalize_loss
from .predict import svr_predict
from .train_internal import svr_train_internal


def _stratified_folds(y: np.ndarray, n_folds: int) -> np.ndarray:
    """Assign deterministic round-robin folds within each response-sign stratum."""

    response = np.asarray(y, dtype=float).reshape(-1)
    folds = np.empty(response.size, dtype=int)
    for mask in (response <= 0.0, response > 0.0):
        indices = np.flatnonzero(mask)
        if indices.size:
            ordered = indices[np.argsort(response[indices], kind="stable")]
            folds[ordered] = np.arange(ordered.size) % n_folds
    return folds


@dataclass
class PeriodicCrossValidationGridTrainer:
    """Select hyperparameters by out-of-fold failure-classification error.

    Balanced sign error is the primary criterion because structural
    reliability depends on the sign of the limit-state response. Normalized
    response RMSE breaks ties without using a reference failure probability.
    """

    retune_interval: int = 20
    n_folds: int = 5
    c_grid: tuple[float, ...] = (10.0, 100.0, 1.0e3, 1.0e4, 1.0e5)
    epsilon_grid: tuple[float, ...] = (1.0e-5, 1.0e-3)
    theta_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
    loss: str = LEGACY_LOSS
    history: list[dict[str, float | int | str]] = field(default_factory=list, init=False)
    _best: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_tuned_samples: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.loss = normalize_loss(self.loss)
        if self.retune_interval <= 0:
            raise ValueError("retune_interval must be positive.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two.")
        grids = (self.c_grid, self.epsilon_grid, self.theta_grid)
        if any(not grid or any(value <= 0.0 for value in grid) for grid in grids):
            raise ValueError("Cross-validation grids must contain positive values.")

    @property
    def best_hyperparameters(self) -> np.ndarray | None:
        return None if self._best is None else self._best.copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_hyperparameters": None if self._best is None else self._best.tolist(),
            "last_tuned_samples": int(self._last_tuned_samples),
            "history": [dict(item) for item in self.history],
            "loss": self.loss,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_loss = normalize_loss(state.get("loss", self.loss))
        if stored_loss != self.loss:
            raise ValueError(
                f"Checkpoint cross-validation loss {stored_loss!r} does not match {self.loss!r}."
            )
        raw_best = state.get("best_hyperparameters")
        best = None if raw_best is None else np.asarray(raw_best, dtype=float).reshape(-1)
        if best is not None and (best.size != 3 or np.any(best <= 0.0)):
            raise ValueError("Invalid cross-validation hyperparameters in checkpoint.")
        history = state.get("history", [])
        if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
            raise ValueError("Invalid cross-validation history in checkpoint.")
        self._best = best
        self._last_tuned_samples = int(state.get("last_tuned_samples", -1))
        self.history = [dict(item) for item in history]

    def _score_candidate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        covariance: str,
        hyperparameters: np.ndarray,
    ) -> tuple[float, float]:
        folds = _stratified_folds(Y, min(self.n_folds, Y.size))
        prediction = np.empty(Y.size, dtype=float)
        for fold in np.unique(folds):
            test = folds == fold
            train = ~test
            if np.count_nonzero(train) < 2:
                return float("inf"), float("inf")
            par = _normalized_training_parameters(X[train], Y[train])
            model = svr_train_internal(
                par,
                hyperparameters,
                covariance,
                loss=self.loss,
            )
            prediction[test] = svr_predict(X[test], model)[0]

        scale = float(np.std(Y, ddof=0)) or 1.0
        nrmse = float(np.sqrt(np.mean((prediction - Y) ** 2)) / scale)
        failure = Y <= 0.0
        safe = ~failure
        false_negative = float(np.mean(prediction[failure] > 0.0)) if np.any(failure) else 0.0
        false_positive = float(np.mean(prediction[safe] <= 0.0)) if np.any(safe) else 0.0
        balanced_sign_error = 0.5 * (false_negative + false_positive)
        return nrmse, balanced_sign_error

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
        del theta_init, c_init, epsilon_init, bounds_mode
        _, covariance = normalize_kernel_name(kernel)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape != (Y.size, int(n_dim)):
            raise ValueError("X, Y, and n_dim are inconsistent.")
        sample_count = X.shape[0]
        should_retune = (
            self._best is None
            or sample_count - self._last_tuned_samples >= self.retune_interval
        )
        if should_retune:
            best_key = (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
            best = None
            evaluated = 0
            for c_value, epsilon_value, theta_value in product(
                self.c_grid,
                self.epsilon_grid,
                self.theta_grid,
            ):
                hyperparameters = np.concatenate(
                    [[c_value, epsilon_value], np.full(n_dim, theta_value)]
                )
                try:
                    nrmse, sign_error = self._score_candidate(
                        X,
                        Y,
                        covariance,
                        hyperparameters,
                    )
                except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                    nrmse, sign_error = float("inf"), float("inf")
                evaluated += 1
                key = (sign_error, nrmse, c_value, epsilon_value, theta_value)
                if key < best_key:
                    best_key = key
                    best = np.array([c_value, epsilon_value, theta_value], dtype=float)
            if best is None or not np.isfinite(best_key[0]):
                raise RuntimeError("No finite cross-validation score was found on the grid.")
            self._best = best
            self._last_tuned_samples = sample_count
            self.history.append(
                {
                    "n_samples": int(sample_count),
                    "C": float(best[0]),
                    "epsilon": float(best[1]),
                    "theta": float(best[2]),
                    "cross_validated_nrmse": float(best_key[1]),
                    "cross_validated_balanced_sign_error": float(best_key[0]),
                    "folds": int(min(self.n_folds, sample_count)),
                    "candidates_evaluated": int(evaluated),
                    "loss": self.loss,
                }
            )

        assert self._best is not None
        par = _normalized_training_parameters(X, Y)
        hyperparameters = np.concatenate(
            [[self._best[0], self._best[1]], np.full(n_dim, self._best[2])]
        )
        return svr_train_internal(
            par,
            hyperparameters,
            covariance,
            loss=self.loss,
        )


__all__ = ["PeriodicCrossValidationGridTrainer"]
