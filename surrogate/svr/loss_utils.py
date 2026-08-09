"""Loss-formulation identifiers used by the Bayesian SVR implementation."""

from __future__ import annotations


LEGACY_LOSS = "legacy"
SQUARED_EPSILON_LOSS = "squared_epsilon"

_ALIASES = {
    "legacy": LEGACY_LOSS,
    "calibrated": LEGACY_LOSS,
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
        choices = ", ".join(sorted({LEGACY_LOSS, SQUARED_EPSILON_LOSS}))
        raise ValueError(f"Unknown SVR loss {loss!r}; expected one of: {choices}.") from exc


__all__ = ["LEGACY_LOSS", "SQUARED_EPSILON_LOSS", "normalize_loss"]
