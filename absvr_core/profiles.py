"""Convenience profiles for baseline settings."""

from __future__ import annotations


def baseline_profile_kwargs() -> dict[str, object]:
    """Return kwargs that mirror the baseline settings."""

    return {
        "tail_policy": "none",
        "tail_alpha": 1.0,
        "learning_w_grad": 0.0,
        "svr_bounds_mode": "baseline",
        "random_seed": 60,
    }
