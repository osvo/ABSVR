"""Utility routines for the ABSVR Python implementation."""

from importlib import import_module
import os
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIRNAME = Path(".cache") / "absvr"
DEFAULT_RESULTS_DIRNAME = Path("results") / "surrogate"
ABSVR_CACHE_DIR = Path(os.getenv("ABSVR_CACHE_DIR", str(DEFAULT_CACHE_DIRNAME))).expanduser()
ABSVR_RESULTS_DIR = Path(os.getenv("ABSVR_RESULTS_DIR", str(DEFAULT_RESULTS_DIRNAME))).expanduser()


def ensure_cache_dir() -> Path:
    """Create the shared cache directory if it does not already exist."""

    ABSVR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ABSVR_CACHE_DIR


def ensure_results_dir() -> Path:
    """Create the default directory used to store run artifacts."""

    ABSVR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return ABSVR_RESULTS_DIR


__all__ = [
    "maybe_print_memory_usage",
    "lhs_uniform",
    "global_random_state",
    "reset_global_seed",
    "get_default_seed",
    "ABSVR_CACHE_DIR",
    "ensure_cache_dir",
    "ABSVR_RESULTS_DIR",
    "ensure_results_dir",
]


_MODULE_MAP = {
    "maybe_print_memory_usage": "utilities.memory",
    "lhs_uniform": "utilities.latin_hypercube",
    "global_random_state": "utilities.random_stream",
    "reset_global_seed": "utilities.random_stream",
    "get_default_seed": "utilities.random_stream",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module 'utilities' has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
