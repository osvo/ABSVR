"""Tests for posthoc planar-truss call-budget reconstruction."""

from __future__ import annotations

import numpy as np

import studies.planar_truss.evaluate_budget_prefixes as budget_module


def test_fit_prefix_replays_retune_schedule_without_future_samples(monkeypatch) -> None:
    instances = []

    class FakeTrainer:
        def __init__(self, *, retune_interval, loss):
            self.retune_interval = retune_interval
            self.loss = loss
            self.history = []
            self.calls = []
            instances.append(self)

        def __call__(self, x, y, n_dim, *, kernel):
            self.calls.append(x.shape[0])
            self.history.append({"n_samples": x.shape[0], "loss": self.loss})
            return {"samples": x.shape[0], "kernel": kernel, "n_dim": n_dim}

    monkeypatch.setattr(budget_module, "PeriodicEvidenceGridTrainer", FakeTrainer)
    doe = np.arange(100, dtype=float).reshape(50, 2)
    response = np.linspace(-1.0, 1.0, 50)
    model, history = budget_module._fit_prefix(
        doe,
        response,
        calls=48,
        initial_samples=15,
        retune_interval=20,
        loss="squared_epsilon",
    )

    assert instances[0].calls == [15, 35, 48]
    assert model == {"samples": 48, "kernel": "gaussian", "n_dim": 2}
    assert [item["n_samples"] for item in history] == [15, 35, 48]
