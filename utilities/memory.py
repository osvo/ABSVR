"""Memory profiling helper."""

import os

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def maybe_print_memory_usage(enabled: bool) -> bool:
    """Print current process memory usage when enabled."""

    if not enabled:
        return False

    if psutil is None:
        print("Memory profiling requires the 'psutil' package; disabling.")
        return False

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"Memory used by Python: {mem_mb:.2f} MB")
    return True
