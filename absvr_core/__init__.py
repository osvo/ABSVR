"""Adaptive SVR workflow building blocks."""

import os


def _supports_shared_memory() -> bool:
    """Return ``True`` when the platform exposes POSIX shared memory support."""

    try:
        from multiprocessing import shared_memory
    except ImportError:
        return False

    try:
        shm = shared_memory.SharedMemory(create=True, size=1)
    except Exception:
        return False
    else:
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        return True


# Only force the sequential layer when we detect that POSIX shared memory
# segments are unavailable. Otherwise let MKL default to its multithreaded
# configuration so the workflow benefits from multi-core BLAS out of the box.
if os.getenv("MKL_THREADING_LAYER") is None and not _supports_shared_memory():
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

from .adaptive_loop import AdaptiveSVRResult, adaptive_svr, run_adaptive_svr
from .candidate_selection import select_best_candidate
from .learning_function import compute_learning_function
from .profiles import baseline_profile_kwargs
from .probability_estimation import estimate_probability_of_failure
from .problem_setup import load_custom_benchmark, prepare_problem_definition

__all__ = [
    "AdaptiveSVRResult",
    "adaptive_svr",
    "run_adaptive_svr",
    "select_best_candidate",
    "compute_learning_function",
    "baseline_profile_kwargs",
    "estimate_probability_of_failure",
    "prepare_problem_definition",
    "load_custom_benchmark",
]
