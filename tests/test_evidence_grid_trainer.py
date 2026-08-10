"""Tests for deterministic periodic evidence tuning."""

from __future__ import annotations

import numpy as np

from surrogate.svr import PeriodicEvidenceGridTrainer, svr_predict


def test_evidence_grid_retunes_only_on_its_schedule() -> None:
    trainer = PeriodicEvidenceGridTrainer(
        retune_interval=2,
        initial_c_grid=(100.0,),
        epsilon_grid=(1.0e-3,),
        initial_theta_grid=(0.1,),
        c_factors=(1.0,),
        theta_factors=(1.0,),
    )
    X = np.linspace(-1.0, 1.0, 8)[:, None]
    Y = X[:, 0] ** 2 - 0.25

    model_6 = trainer(X[:6], Y[:6], 1)
    assert len(trainer.history) == 1
    assert trainer.history[0]["candidates_evaluated"] == 1

    trainer(X[:7], Y[:7], 1)
    assert len(trainer.history) == 1

    model_8 = trainer(X, Y, 1)
    assert len(trainer.history) == 2
    assert trainer.history[-1]["n_samples"] == 8
    np.testing.assert_array_equal(trainer.best_hyperparameters, [100.0, 1.0e-3, 0.1])

    for model in (model_6, model_8):
        prediction, variance = svr_predict(X, model)
        assert np.all(np.isfinite(prediction))
        assert np.all(np.isfinite(variance))


def test_evidence_grid_state_round_trip_preserves_schedule() -> None:
    settings = {
        "retune_interval": 2,
        "initial_c_grid": (100.0,),
        "epsilon_grid": (1.0e-3,),
        "initial_theta_grid": (0.1,),
        "c_factors": (1.0,),
        "theta_factors": (1.0,),
    }
    X = np.linspace(-1.0, 1.0, 8)[:, None]
    Y = X[:, 0] ** 2 - 0.25
    original = PeriodicEvidenceGridTrainer(**settings)
    original(X[:6], Y[:6], 1)

    restored = PeriodicEvidenceGridTrainer(**settings)
    restored.load_state_dict(original.state_dict())
    restored(X[:7], Y[:7], 1)
    assert len(restored.history) == 1
    restored(X, Y, 1)
    assert len(restored.history) == 2
    np.testing.assert_array_equal(
        restored.best_hyperparameters,
        original.best_hyperparameters,
    )


def test_squared_epsilon_evidence_choice_is_explicit_and_resumable() -> None:
    settings = {
        "retune_interval": 2,
        "initial_c_grid": (100.0,),
        "epsilon_grid": (1.0e-3,),
        "initial_theta_grid": (0.1,),
        "c_factors": (1.0,),
        "theta_factors": (1.0,),
        "loss": "squared_epsilon",
    }
    x = np.linspace(-1.0, 1.0, 6)[:, None]
    y = x[:, 0] ** 2 - 0.25
    trainer = PeriodicEvidenceGridTrainer(**settings)
    model = trainer(x, y, 1)

    assert model["Loss"] == "squared_epsilon"
    assert trainer.history[0]["loss"] == "squared_epsilon"
    assert trainer.state_dict()["loss"] == "squared_epsilon"

    restored = PeriodicEvidenceGridTrainer(**settings)
    restored.load_state_dict(trainer.state_dict())
    assert restored.state_dict() == trainer.state_dict()


def test_square_loss_evidence_has_no_epsilon_search_dimension() -> None:
    trainer = PeriodicEvidenceGridTrainer(
        retune_interval=2,
        initial_c_grid=(10.0, 100.0),
        epsilon_grid=(1.0e-5, 1.0e-3, 1.0e-2),
        initial_theta_grid=(0.1, 0.2),
        c_factors=(1.0,),
        theta_factors=(1.0,),
        loss="square",
    )
    x = np.linspace(-1.0, 1.0, 8)[:, None]
    y = np.cos(x[:, 0]) - 0.7
    model = trainer(x, y, 1)

    assert model["Loss"] == "square"
    assert model["epsilon"] == 0.0
    assert trainer.history[0]["epsilon"] == 0.0
    assert trainer.history[0]["candidates_evaluated"] == 4
