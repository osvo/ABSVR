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
