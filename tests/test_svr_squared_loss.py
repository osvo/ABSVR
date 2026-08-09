"""Tests for the epsilon-insensitive squared-loss SVR dual."""

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


def test_squared_loss_dual_satisfies_equality_and_kkt_residuals() -> None:
    x = np.linspace(-2.0, 2.0, 17).reshape(-1, 1)
    y = np.sin(x[:, 0])
    c_value = 50.0
    epsilon = 0.02
    model = svr_train_internal(
        _training_parameters(x, y),
        np.array([c_value, epsilon, 0.8]),
        "Gaussian",
    )

    beta = np.asarray(model["parameter"])
    assert abs(float(np.sum(beta))) < 2.0e-7

    prediction, _ = svr_predict(x, model)
    residual = y - prediction
    active = np.zeros(beta.size, dtype=bool)
    active[np.asarray(model["SV"], dtype=int)] = True
    expected_magnitude = epsilon + np.abs(beta[active]) / c_value
    np.testing.assert_allclose(
        np.abs(residual[active]), expected_magnitude, rtol=2.0e-4, atol=2.0e-6
    )


def test_squared_loss_svr_fits_smooth_training_data() -> None:
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
