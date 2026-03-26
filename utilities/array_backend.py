"""CPU-only array utilities.

The project formerly attempted to switch between NumPy and GPU-oriented
backends (CuPy, cuNumeric). That path proved fragile and provided limited
payoff, so the helpers now deliberately expose NumPy-only behaviour while
preserving the public API expected by the rest of the codebase.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_default_array_module():
    """Return the sole supported array module, NumPy."""

    return np


def get_backend_name() -> str:
    """Human-readable backend name for status banners."""

    return "numpy"


def gpu_available() -> bool:
    """GPU acceleration is no longer supported."""

    return False


def get_array_module(*_arrays: Any):
    """Compatibility shim that always yields NumPy."""

    return np


def as_xp(array: Any, xp=None, dtype=None):
    """Convert *array* to a NumPy ndarray."""

    return np.asarray(array, dtype=dtype)


def to_cpu(array: Any):
    """Identity helper retained for backwards compatibility."""

    return np.asarray(array)

