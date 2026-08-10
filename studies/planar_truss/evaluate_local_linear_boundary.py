"""Evaluate a frozen local-linear boundary correction for an ABSVR campaign.

The correction fits a weighted affine limit-state approximation in independent
standard-normal coordinates.  Because the fitted boundary is affine and the
inputs are standard normal, its failure probability is available analytically.
The independent reference is used only after fitting to audit the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

from absvr_core.checkpointing import load_checkpoint
from studies.planar_truss.run_absvr_campaign import _load_reference


def fit_local_linear_boundary(
    inputs: np.ndarray,
    response: np.ndarray,
    *,
    weight_power: float = 1.0,
    ridge_alpha: float = 0.1,
) -> dict[str, Any]:
    """Fit the scale-invariant weighted affine response model."""

    x = np.asarray(inputs, dtype=float)
    y = np.asarray(response, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("inputs and response must contain the same samples.")
    if x.shape[0] <= x.shape[1]:
        raise ValueError("the affine fit requires more samples than dimensions.")
    if weight_power < 0.0 or ridge_alpha <= 0.0:
        raise ValueError("weight_power must be non-negative and ridge_alpha positive.")

    response_scale = float(np.median(np.abs(y)))
    if not np.isfinite(response_scale) or response_scale <= 0.0:
        response_scale = float(np.std(y, ddof=0))
    if not np.isfinite(response_scale) or response_scale <= 0.0:
        raise ValueError("response must contain non-constant finite values.")

    weights = np.exp(-float(weight_power) * np.abs(y) / response_scale)
    design = np.column_stack([np.ones(x.shape[0]), x])
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_response = y * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_response,
    )

    intercept = float(coefficients[0])
    gradient = np.asarray(coefficients[1:], dtype=float)
    gradient_norm = float(np.linalg.norm(gradient))
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("the fitted boundary has a zero or non-finite gradient.")
    reliability_index = intercept / gradient_norm
    failure_probability = float(norm.cdf(-reliability_index))
    effective_sample_size = float(weights.sum() ** 2 / np.sum(weights**2))
    return {
        "failure_probability": failure_probability,
        "reliability_index": float(reliability_index),
        "intercept": intercept,
        "gradient": gradient.tolist(),
        "response_scale_median_abs": response_scale,
        "effective_weighted_sample_size": effective_sample_size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=55)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/planar_truss/reference_qmc.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    reference_pf, reference_report = _load_reference(args.reference)
    evaluated_runs: list[dict[str, Any]] = []
    for run in campaign["runs"]:
        checkpoint = Path(run["checkpoint"])
        state = load_checkpoint(checkpoint)
        inputs = np.asarray(state["doe"], dtype=float)
        response = np.asarray(state["g"], dtype=float).reshape(-1)
        if inputs.shape[0] < args.calls:
            raise ValueError(f"{checkpoint} contains fewer than {args.calls} calls.")
        fit = fit_local_linear_boundary(
            inputs[: args.calls],
            response[: args.calls],
            weight_power=args.weight_power,
            ridge_alpha=args.ridge_alpha,
        )
        estimate = float(fit["failure_probability"])
        evaluated_runs.append(
            {
                "algorithm_seed": int(run["algorithm_seed"]),
                "open_sees_limit_state_calls": int(args.calls),
                "source_checkpoint": checkpoint.as_posix(),
                "fit": fit,
                "relative_error_vs_independent_reference_percent": float(
                    100.0 * abs(estimate - reference_pf) / reference_pf
                ),
            }
        )

    estimates = np.array(
        [item["fit"]["failure_probability"] for item in evaluated_runs], dtype=float
    )
    errors = np.array(
        [item["relative_error_vs_independent_reference_percent"] for item in evaluated_runs],
        dtype=float,
    )
    report = {
        "schema_version": 1,
        "purpose": "development evaluation of a frozen local-linear boundary correction",
        "configuration_uses_reference_probability": False,
        "reference_used_only_for_postfit_audit": True,
        "source_campaign": args.campaign.as_posix(),
        "configuration": {
            "calls": int(args.calls),
            "weight": "exp(-weight_power * abs(g) / median(abs(g)))",
            "weight_power": float(args.weight_power),
            "ridge_alpha": float(args.ridge_alpha),
            "probability_formula": "Phi(-intercept / l2_norm(gradient))",
        },
        "independent_reference": {
            "path": args.reference.as_posix(),
            "mean_pf": reference_pf,
            "confidence_interval_95": reference_report.get("confidence_interval_95"),
        },
        "summary": {
            "runs": int(estimates.size),
            "mean_pf": float(np.mean(estimates)),
            "standard_deviation_pf": (
                float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0
            ),
            "relative_error_of_mean_pf_percent": float(
                100.0 * abs(np.mean(estimates) - reference_pf) / reference_pf
            ),
            "mean_absolute_run_error_percent": float(np.mean(errors)),
            "median_absolute_run_error_percent": float(np.median(errors)),
            "minimum_absolute_run_error_percent": float(np.min(errors)),
            "maximum_absolute_run_error_percent": float(np.max(errors)),
            "mean_open_sees_limit_state_calls": float(args.calls),
        },
        "runs": evaluated_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
