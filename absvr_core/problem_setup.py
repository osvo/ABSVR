"""Benchmark initialization helpers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import numpy as np

try:
    from benchmarks import (
        four_branch_series,
        high_nonlinear,
        nonlinear_oscillator,
        modified_rastrigin,
        decision_2d,
        high_dimensional,
    )
except ImportError:
    from ..benchmarks import (
        four_branch_series,
        high_nonlinear,
        nonlinear_oscillator,
        modified_rastrigin,
        decision_2d,
        high_dimensional,
    )


def prepare_problem_definition(test_example: str, *args: int) -> tuple:
    """Return benchmark configuration for the built-in examples.

    Parameters
    ----------
    test_example : str
        Benchmark identifier (``'eg1'`` through ``'eg6'``).
    *args : int
        Optional case identifier used by ``eg3``.

    Returns
    -------
    tuple
        ``(mu, sigma, n_dim, fun_parameter, fun_handle, pf_reference, dist_type)``.

        dist_type is ``'normal'`` for Gaussian inputs, or ``'lognormal'``
        when the limit-state function expects lognormal random variables.
    """
    dist_type = "normal"  # Default to normal distribution

    if test_example == "eg1":
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        n_dim = 2
        fun_par = 1.2
        fun = high_nonlinear
        pf_ref = 4.71e-3
    elif test_example == "eg2":
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        n_dim = 2
        fun_par = {"type": 1, "a": 3.0, "b": 7.0, "f0": 0.0}
        fun = four_branch_series
        pf_ref = 2.221e-3
    elif test_example == "eg3":
        n_dim = 6
        mu = np.array([1.0, 1.0, 0.1, 0.5, 1.0, 1.0])
        sigma = np.array([0.05, 0.1, 0.01, 0.05, 0.2, 0.2])
        pf_ref = 2.859e-2
        fun_par = 3.0
        fun = nonlinear_oscillator
    elif test_example == "eg4":
        # Modified Rastrigin function (2D with multiple failure regions)
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        n_dim = 2
        fun_par = 0.0
        fun = modified_rastrigin
        pf_ref = 7.28e-2  # MCS-verified reference Pf
    elif test_example == "eg5":
        # 2D Decision function
        mu = np.array([0.0, 0.0])
        sigma = np.array([1.0, 1.0])
        n_dim = 2
        fun_par = 0.0
        fun = decision_2d
        pf_ref = 1.88e-3  # MCS-verified reference Pf
    elif test_example == "eg6":
        # High-dimensional problem (n=40, lognormal inputs)
        n_dim = 40
        # Lognormal inputs: sample from N(0,1), then transform via exp()
        # For lognormal(0, 1): mean = exp(0.5) ≈ 1.649, std ≈ 2.161
        mu = np.zeros(n_dim)  # Mean of underlying normal
        sigma = np.ones(n_dim)  # Std of underlying normal
        # sigma parameter in LSF: controls failure probability
        # sigma=1.0 gives Pf ≈ 5e-3 (MCS-verified)
        fun_par = {"n_dim": n_dim, "sigma": 1.0}
        fun = high_dimensional
        pf_ref = 5.16e-3  # MCS-verified reference Pf
        dist_type = "lognormal"
    else:
        raise ValueError(
            "Unknown test_example. Use 'eg1', 'eg2', 'eg3', 'eg4', 'eg5', or 'eg6'."
        )

    return mu, sigma, n_dim, fun_par, fun, pf_ref, dist_type


def load_custom_benchmark(spec: str, *, factory: str | None = None) -> tuple:
    """Load a custom benchmark definition from a module or file path.

    The factory function must return:
        (mu, sigma, n_dim, fun_par, fun_handle, pf_reference, dist_type)

    ``dist_type`` defaults to "normal" when omitted.
    """

    if not spec or not isinstance(spec, str):
        raise ValueError("Custom benchmark spec must be a non-empty string.")

    resolved_spec = spec.strip()
    factory_name = factory
    if "::" in resolved_spec and factory_name is None:
        resolved_spec, factory_name = resolved_spec.split("::", 1)
        resolved_spec = resolved_spec.strip()
        factory_name = factory_name.strip() or None

    factory_name = factory_name or "get_problem_definition"

    module = None
    spec_path = Path(resolved_spec).expanduser()
    if spec_path.suffix == ".py" or spec_path.exists():
        module_name = f"custom_benchmark_{spec_path.stem}"
        resolved_path = spec_path.resolve()
        module_spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load module from {resolved_path}.")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(resolved_spec)

    factory_fn = getattr(module, factory_name, None)
    if factory_fn is None or not callable(factory_fn):
        raise AttributeError(f"Custom benchmark factory '{factory_name}' was not found.")

    result = factory_fn()
    if isinstance(result, tuple):
        values = list(result)
    else:
        raise ValueError("Custom benchmark factory must return a tuple.")

    if len(values) == 6:
        values.append("normal")
    if len(values) != 7:
        raise ValueError(
            "Custom benchmark factory must return 6 or 7 values: "
            "(mu, sigma, n_dim, fun_par, fun_handle, pf_reference[, dist_type])."
        )

    mu, sigma, n_dim, fun_par, fun, pf_ref, dist_type = values

    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    sigma_arr = np.asarray(sigma, dtype=float).reshape(-1)
    if mu_arr.shape != sigma_arr.shape:
        raise ValueError("Custom benchmark mu and sigma must share the same shape.")
    if n_dim is None:
        n_dim_val = int(mu_arr.size)
    else:
        n_dim_val = int(n_dim)
    if mu_arr.size != n_dim_val:
        raise ValueError("Custom benchmark n_dim does not match mu/sigma length.")
    if not callable(fun):
        raise ValueError("Custom benchmark fun_handle must be callable.")

    dist_type_val = str(dist_type or "normal").lower()
    if dist_type_val not in {"normal", "lognormal"}:
        raise ValueError("dist_type must be 'normal' or 'lognormal'.")

    return mu_arr, sigma_arr, n_dim_val, fun_par, fun, pf_ref, dist_type_val
