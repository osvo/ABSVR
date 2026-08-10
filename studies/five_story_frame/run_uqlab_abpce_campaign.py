"""Run and aggregate direct UQLab A-bPCE analyses on the frame."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
from scipy import stats

from studies.five_story_frame.run_uqlab_akmcs_campaign import (
    DEFAULT_SEEDS,
    _atomic_write_json,
    _find_matlab,
    _matlab_quote,
    _parse_seeds,
    _validate_uqlab_core,
)


def _load_result(path: Path, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("profile") != "published_abpce" or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Result metadata mismatch: {path}")
    if int(result["model_evaluations_uqlab_design"]) != int(
        result["model_evaluations_opensees_ledger"]
    ):
        raise ValueError(f"UQLab/OpenSees call-count mismatch: {path}")
    expected = {
        "bootstrap_replications": 100,
        "initial_design": {"sampling": "LHS", "size": 40},
        "maximum_total_evaluations": 300,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Metadata mismatch for {key!r}: {path}")
    if result.get("internal_mcs", {}).get("size") != 1_000_000:
        raise ValueError(f"Internal MCS size mismatch: {path}")
    return result


def summarize_results(
    results: Sequence[dict[str, Any]], *, seeds: Sequence[int], reference_pf: float
) -> dict[str, Any]:
    if tuple(int(result["seed"]) for result in results) != tuple(seeds):
        raise ValueError("Results do not match the declared seed order.")
    estimates = np.array([result["pf"] for result in results], dtype=float)
    calls = np.array(
        [result["model_evaluations_opensees_ledger"] for result in results], dtype=int
    )
    if np.any(~np.isfinite(estimates)) or np.any((estimates <= 0.0) | (estimates >= 1.0)):
        raise ValueError("Every A-bPCE probability must be finite and non-degenerate.")
    mean_pf = float(np.mean(estimates))
    errors = 100.0 * np.abs(estimates - reference_pf) / reference_pf
    if len(results) > 1:
        interval = stats.t.interval(
            0.95, df=len(results) - 1, loc=mean_pf, scale=stats.sem(estimates)
        )
        standard_deviation = float(np.std(estimates, ddof=1))
    else:
        interval = (mean_pf, mean_pf)
        standard_deviation = 0.0
    return {
        "schema_version": 1,
        "method": "direct UQLab A-bPCE with OpenSeesPy",
        "profile": "published_abpce",
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "algorithm_seeds": list(seeds),
        "runs": len(results),
        "converged_runs": int(sum(bool(result["converged"]) for result in results)),
        "mean_pf": mean_pf,
        "mean_pf_student_t_95_percent_interval_across_seeds": [
            float(interval[0]),
            float(interval[1]),
        ],
        "beta_from_mean_pf": -NormalDist().inv_cdf(mean_pf),
        "standard_deviation_pf_across_seeds": standard_deviation,
        "relative_error_of_mean_vs_independent_rqmc_percent": float(
            100.0 * abs(mean_pf - reference_pf) / reference_pf
        ),
        "mean_absolute_run_error_vs_independent_rqmc_percent": float(np.mean(errors)),
        "median_absolute_run_error_vs_independent_rqmc_percent": float(np.median(errors)),
        "maximum_absolute_run_error_vs_independent_rqmc_percent": float(np.max(errors)),
        "mean_open_sees_calls": float(np.mean(calls)),
        "minimum_open_sees_calls": int(np.min(calls)),
        "maximum_open_sees_calls": int(np.max(calls)),
        "total_open_sees_calls": int(np.sum(calls)),
        "independent_rqmc_reference_pf": reference_pf,
        "individual_runs": list(results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uqlab-core", required=True, type=Path)
    parser.add_argument("--matlab", type=Path, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/five_story_frame/reference_rqmc.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/five_story_frame/uqlab_abpce"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    wrapper_directory = Path(__file__).resolve().parent / "uqlab"
    uqlab_core = _validate_uqlab_core(args.uqlab_core)
    matlab = _find_matlab(args.matlab)
    output_directory = args.output_directory.resolve()
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_campaign",
        "source_method": "Marelli and Sudret (2018), DOI 10.1016/j.strusafe.2018.06.003",
        "algorithm_seeds": list(args.seeds),
        "initial_design": {"sampling": "LHS", "size": 40},
        "method": "A-bPCE controller with direct UQLab sparse PCE and fast bootstrap",
        "pce": {
            "method": "LARS",
            "degree": list(range(1, 11)),
            "q_norm": 0.75,
            "maximum_interaction": 2,
        },
        "bootstrap_replications": 100,
        "internal_mcs": {"sampling": "MC", "size": 1_000_000, "common_random_numbers": True},
        "enrichment_points_per_iteration": 1,
        "convergence": {"relative_pf_range_tolerance": 0.15, "consecutive_iterations": 2},
        "maximum_total_evaluations": 300,
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "matlab_executable": str(matlab),
        "uqlab_core": str(uqlab_core),
    }
    _atomic_write_json(output_directory / "campaign_protocol.json", protocol)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        result_path = output_directory / f"uqlab_abpce_seed_{seed}.json"
        if not (args.resume and result_path.is_file()):
            expression = (
                f"addpath('{_matlab_quote(uqlab_core)}');"
                f"addpath('{_matlab_quote(wrapper_directory)}');"
                f"run_uqlab_abpce({seed},'{_matlab_quote(result_path)}');"
            )
            log_path = output_directory / f"uqlab_abpce_seed_{seed}_matlab.log"
            output_directory.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    [str(matlab), "-batch", expression],
                    cwd=repo_root,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        results.append(_load_result(result_path, seed))

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_pf = float(reference["summary"]["failure_probability"])
    summary = summarize_results(results, seeds=args.seeds, reference_pf=reference_pf)
    summary["protocol"] = protocol
    _atomic_write_json(output_directory / "campaign_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
