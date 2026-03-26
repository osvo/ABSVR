"""Shared random number stream with a fixed baseline seed (60)."""

import numpy as np

_DEFAULT_SEED = 60
_global_rng: np.random.RandomState | None = None


def reset_global_seed(seed: int = _DEFAULT_SEED) -> None:
    """Reset the shared RNG to the provided seed."""

    global _global_rng
    _global_rng = np.random.RandomState(seed)


def global_random_state() -> np.random.RandomState:
    """Return the shared RandomState, creating it if necessary."""

    global _global_rng
    if _global_rng is None:
        reset_global_seed()
    return _global_rng  # type: ignore[return-value]


def get_default_seed() -> int:
    """Expose the default seed used for the shared RNG."""

    return _DEFAULT_SEED
