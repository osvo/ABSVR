"""Tests for deterministic cross-validated SVR hyperparameter selection."""

from __future__ import annotations

import numpy as np

from surrogate.svr import PeriodicCrossValidationGridTrainer, svr_predict
from surrogate.svr.cross_validation_grid import _stratified_folds


def _settings() -> dict:
    return {
        "retune_interval": 2,
        "n_folds": 3,
        "c_grid": (100.0,),
        "epsilon_grid": (1.0e-3,),
        "theta_grid": (0.1,),
        "loss": "legacy",
    }


def test_stratified_folds_distribute_each_response_sign() -> None:
    y = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    folds = _stratified_folds(y, 3)
    assert set(folds[y <= 0.0]) == {0, 1, 2}
    assert set(folds[y > 0.0]) == {0, 1, 2}


def test_cross_validation_trainer_retunes_and_resumes() -> None:
    x = np.linspace(-2.0, 2.0, 9)[:, None]
    y = x[:, 0] ** 2 - 1.0
    trainer = PeriodicCrossValidationGridTrainer(**_settings())
    model_7 = trainer(x[:7], y[:7], 1)
    trainer(x[:8], y[:8], 1)
    assert len(trainer.history) == 1
    model_9 = trainer(x, y, 1)
    assert len(trainer.history) == 2
    assert trainer.history[-1]["n_samples"] == 9
    assert trainer.history[-1]["candidates_evaluated"] == 1
    np.testing.assert_array_equal(trainer.best_hyperparameters, [100.0, 1.0e-3, 0.1])

    for model in (model_7, model_9):
        prediction, variance = svr_predict(x, model)
        assert np.all(np.isfinite(prediction))
        assert np.all(np.isfinite(variance))

    restored = PeriodicCrossValidationGridTrainer(**_settings())
    restored.load_state_dict(trainer.state_dict())
    assert restored.state_dict() == trainer.state_dict()


def test_balanced_sign_error_takes_priority_over_response_rmse(monkeypatch) -> None:
    trainer = PeriodicCrossValidationGridTrainer(
        retune_interval=2,
        n_folds=3,
        c_grid=(10.0, 100.0),
        epsilon_grid=(1.0e-3,),
        theta_grid=(0.1,),
        loss="legacy",
    )

    def synthetic_scores(_x, _y, _covariance, hyperparameters):
        if hyperparameters[0] == 10.0:
            return 0.01, 0.40
        return 0.20, 0.10

    monkeypatch.setattr(trainer, "_score_candidate", synthetic_scores)
    x = np.linspace(-1.0, 1.0, 6)[:, None]
    y = x[:, 0]
    trainer(x, y, 1)

    np.testing.assert_array_equal(trainer.best_hyperparameters, [100.0, 1.0e-3, 0.1])
    assert trainer.history[0]["cross_validated_balanced_sign_error"] == 0.10
    assert trainer.history[0]["cross_validated_nrmse"] == 0.20
