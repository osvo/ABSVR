"""Loss-formulation identifiers used by the Bayesian SVR implementation."""

from __future__ import annotations


LEGACY_LOSS = "legacy"
SQUARE_LOSS = "square"
SQUARED_EPSILON_LOSS = "squared_epsilon"

_ALIASES = {
    "legacy": LEGACY_LOSS,
    "calibrated": LEGACY_LOSS,
    "square": SQUARE_LOSS,
    "squared": SQUARE_LOSS,
    "l2": SQUARE_LOSS,
    "squared_epsilon": SQUARED_EPSILON_LOSS,
    "squared-epsilon": SQUARED_EPSILON_LOSS,
    "l2_epsilon": SQUARED_EPSILON_LOSS,
}


def normalize_loss(loss: str) -> str:
    """Return the canonical loss identifier or raise for an unknown value."""

    token = str(loss).strip().lower()
    try:
        return _ALIASES[token]
    except KeyError as exc:
        choices = ", ".join(sorted({LEGACY_LOSS, SQUARE_LOSS, SQUARED_EPSILON_LOSS}))
        raise ValueError(f"Unknown SVR loss {loss!r}; expected one of: {choices}.") from exc


__all__ = ["LEGACY_LOSS", "SQUARE_LOSS", "SQUARED_EPSILON_LOSS", "normalize_loss"]
