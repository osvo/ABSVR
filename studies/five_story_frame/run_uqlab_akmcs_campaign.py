"""Run and aggregate independently seeded native UQLab AK-MCS analyses."""

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


DEFAULT_SEEDS = (269, 281, 293, 307, 313, 331, 347, 359, 373, 389)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("Provide unique non-negative seeds.")
    return seeds


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _find_matlab(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit
    elif os.environ.get("MATLAB_EXE"):
        candidate = Path(os.environ["MATLAB_EXE"])
    elif shutil.which("matlab"):
        candidate = Path(shutil.which("matlab") or "")
    else:
        installations = sorted(
            Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"), reverse=True
        )
        if not installations:
            raise FileNotFoundError("MATLAB was not found.")
        candidate = installations[0]
    if not candidate.is_file():
        raise FileNotFoundError(f"MATLAB executable does not exist: {candidate}")
    return candidate.resolve()


def _validate_uqlab_core(path: Path) -> Path:
    resolved = path.resolve()
    if not (resolved / "uqlab.m").is_file():
        raise FileNotFoundError(f"Expected uqlab.m in {resolved}")
    return resolved


def _load_result(path: Path, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("profile") != "native_akmcs" or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Result metadata mismatch: {path}")
    if int(result["model_evaluations_uqlab"]) != int(
        result["model_evaluations_opensees_ledger"]
    ):
        raise ValueError(f"UQLab/OpenSees call-count mismatch: {path}")
    expected = {
        "learning_function": "U",
        "kriging_covariance": "Gaussian",
        "convergence": "stopU",
        "convergence_threshold": 2.0,
        "internal_mcs_size": 1_000_000,
        "initial_design": {"sampling": "LHS", "size": 40},
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Metadata mismatch for {key!r}: {path}")
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
        raise ValueError("Every AK-MCS probability must be finite and non-degenerate.")
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
        "method": "direct UQLab native AK-MCS with OpenSeesPy",
        "profile": "native_akmcs",
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "algorithm_seeds": list(seeds),
        "runs": len(results),
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
        default=Path("results/five_story_frame/uqlab_akmcs"),
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
        "algorithm_seeds": list(args.seeds),
        "initial_design": {"sampling": "LHS", "size": 40},
        "method": "native UQLab AKMCS",
        "kriging_covariance": "Gaussian",
        "learning_function": "U",
        "convergence": {"criterion": "stopU", "threshold": 2.0},
        "internal_mcs_size": 1_000_000,
        "max_added_design": 960,
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "matlab_executable": str(matlab),
        "uqlab_core": str(uqlab_core),
    }
    _atomic_write_json(output_directory / "campaign_protocol.json", protocol)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        result_path = output_directory / f"uqlab_native_akmcs_seed_{seed}.json"
        if not (args.resume and result_path.is_file()):
            expression = (
                f"addpath('{_matlab_quote(uqlab_core)}');"
                f"addpath('{_matlab_quote(wrapper_directory)}');"
                f"run_uqlab_akmcs({seed},'{_matlab_quote(result_path)}');"
            )
            log_path = output_directory / f"uqlab_native_akmcs_seed_{seed}_matlab.log"
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
