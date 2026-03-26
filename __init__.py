"""Python implementation of the ABSVR surrogate modeling core."""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("ABSVR requires Python 3.10 or later.")

from .absvr_cli import run_cli
from .absvr_core import AdaptiveSVRResult, adaptive_svr, run_adaptive_svr

__all__ = ["AdaptiveSVRResult", "adaptive_svr", "run_adaptive_svr", "run_cli"]
