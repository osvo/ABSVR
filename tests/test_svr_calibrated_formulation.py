"""Regression tests for the calibrated ABSVR epsilon-SVR formulation."""

from __future__ import annotations

import numpy as np

from surrogate.svr.predict import svr_predict
from surrogate.svr.train_internal import svr_train_internal


def _training_parameters(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "X": x,
        "Y": y,
        "Inputmoment": np.vstack([np.zeros(x.shape[1]), np.ones(x.shape[1])]),
        "Outputmoment": np.array([[0.0], [1.0]]),
    }


def test_calibrated_dual_preserves_equality_and_c_box() -> None:
    x = np.linspace(-2.0, 2.0, 17).reshape(-1, 1)
    y = np.sin(x[:, 0])
    c_value = 50.0
    model = svr_train_internal(
        _training_parameters(x, y),
        np.array([c_value, 0.02, 0.8]),
        "Gaussian",
    )

    beta = np.asarray(model["parameter"])
    assert abs(float(np.sum(beta))) < 2.0e-7
    assert np.max(np.abs(beta)) <= c_value + 2.0e-7
    assert np.asarray(model["SV"]).size > 0


def test_calibrated_svr_fits_smooth_training_data() -> None:
    x = np.linspace(-2.0, 2.0, 21).reshape(-1, 1)
    y = np.sin(x[:, 0]) + 0.1 * x[:, 0]
    model = svr_train_internal(
        _training_parameters(x, y),
        np.array([100.0, 1.0e-3, 1.0]),
        "Gaussian",
    )
    prediction, variance = svr_predict(x, model)
    assert float(np.sqrt(np.mean((prediction - y) ** 2))) < 5.0e-3
    assert np.all(np.isfinite(variance))
    assert np.all(variance >= 0.0)

