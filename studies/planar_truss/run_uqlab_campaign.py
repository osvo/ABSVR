"""Run and aggregate a frozen, independently seeded UQLab AK-MCS campaign."""

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


DEFAULT_SEEDS = (11, 19, 31, 43, 53, 67, 79, 89, 101, 113)
DEFAULT_REFERENCE = Path("results/planar_truss/reference_qmc.json")
DEFAULT_OUTPUT_DIRECTORY = Path("results/planar_truss/uqlab")


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("Provide at least one non-negative seed.")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Campaign seeds must be unique.")
    return seeds


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"),
            reverse=True,
        )
        if not installations:
            raise FileNotFoundError(
                "MATLAB was not found. Pass --matlab or set MATLAB_EXE."
            )
        candidate = installations[0]
    if not candidate.is_file():
        raise FileNotFoundError(f"MATLAB executable does not exist: {candidate}")
    return candidate.resolve()


def _validate_uqlab_core(path: Path) -> Path:
    resolved = path.resolve()
    if not (resolved / "uqlab.m").is_file():
        raise FileNotFoundError(
            f"Expected uqlab.m in the supplied UQLab core directory: {resolved}"
        )
    return resolved


def _result_path(output_directory: Path, profile: str, seed: int) -> Path:
    return output_directory / f"uqlab_{profile}_seed_{seed}.json"


def _load_completed_result(path: Path, *, profile: str, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("profile") != profile or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Result metadata does not match its campaign slot: {path}")
    if int(result["model_evaluations_uqlab"]) != int(
        result["model_evaluations_opensees_ledger"]
    ):
        raise ValueError(f"UQLab/OpenSees call-count mismatch in {path}")
    return result


def summarize_results(
    results: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
    profile: str,
    reference_pf: float,
    published_pf: float,
) -> dict[str, Any]:
    if not results:
        raise ValueError("A campaign must contain at least one result.")
    actual_seeds = tuple(int(result["seed"]) for result in results)
    if actual_seeds != tuple(seeds):
        raise ValueError("Results must contain every frozen seed in its declared order.")
    if any(result["profile"] != profile for result in results):
        raise ValueError("All results must use the frozen campaign profile.")

    estimates = np.array([result["pf"] for result in results], dtype=float)
    if not np.all(np.isfinite(estimates)) or np.any((estimates < 0.0) | (estimates > 1.0)):
        raise ValueError("Every failure-probability estimate must be finite and in [0, 1].")
    calls = np.array(
        [result["model_evaluations_opensees_ledger"] for result in results],
        dtype=int,
    )
    individual_errors = 100.0 * np.abs(estimates - reference_pf) / reference_pf
    mean_pf = float(np.mean(estimates))
    beta_from_mean_pf = (
        -NormalDist().inv_cdf(mean_pf) if 0.0 < mean_pf < 1.0 else None
    )
    return {
        "schema_version": 1,
        "method": "direct UQLab ALR / AK-MCS with OpenSeesPy",
        "profile": profile,
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "algorithm_seeds": list(seeds),
        "runs": len(results),
        "mean_pf": mean_pf,
        "beta_from_mean_pf": beta_from_mean_pf,
        "beta_from_mean_pf_is_infinite": mean_pf in (0.0, 1.0),
        "standard_deviation_pf_across_seeds": (
            float(np.std(estimates, ddof=1)) if len(results) > 1 else 0.0
        ),
        "relative_error_of_mean_vs_independent_rqmc_percent": float(
            100.0 * abs(mean_pf - reference_pf) / reference_pf
        ),
        "relative_error_of_mean_vs_published_mcs_percent": float(
            100.0 * abs(mean_pf - published_pf) / published_pf
        ),
        "mean_absolute_run_error_vs_independent_rqmc_percent": float(
            np.mean(individual_errors)
        ),
        "median_absolute_run_error_vs_independent_rqmc_percent": float(
            np.median(individual_errors)
        ),
        "maximum_absolute_run_error_vs_independent_rqmc_percent": float(
            np.max(individual_errors)
        ),
        "mean_open_sees_calls": float(np.mean(calls)),
        "minimum_open_sees_calls": int(np.min(calls)),
        "maximum_open_sees_calls": int(np.max(calls)),
        "total_open_sees_calls": int(np.sum(calls)),
        "independent_rqmc_reference_pf": reference_pf,
        "published_direct_mcs_reference_pf": published_pf,
        "individual_runs": list(results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uqlab-core", required=True, type=Path)
    parser.add_argument("--matlab", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=("paper_like", "original_akmcs"),
        default="paper_like",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated frozen algorithm seeds.",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    uqlab_core = _validate_uqlab_core(args.uqlab_core)
    matlab = _find_matlab(args.matlab)
    wrapper_directory = Path(__file__).resolve().parent / "uqlab"
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    protocol = {
        "schema_version": 1,
        "profile": args.profile,
        "algorithm_seeds": list(args.seeds),
        "runs_expected": len(args.seeds),
        "initial_design": {"sampling": "LHS", "size": 30},
        "kriging_covariance": "Gaussian",
        "learning_function": "U",
        "convergence": (
            {"criterion": "StopPfBound", "threshold": 0.10, "iterations": 2}
            if args.profile == "paper_like"
            else {"criterion": "StopLF", "threshold": 2.0, "iterations": 1}
        ),
        "internal_mcs_size": 1_000_000,
        "max_candidate_size": 1_000_000,
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "matlab_executable": str(matlab),
        "uqlab_core": str(uqlab_core),
    }
    _atomic_write_json(output_directory / "campaign_protocol.json", protocol)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        result_path = _result_path(output_directory, args.profile, seed)
        if not (args.resume and result_path.is_file()):
            matlab_expression = (
                f"addpath('{_matlab_quote(uqlab_core)}');"
                f"addpath('{_matlab_quote(wrapper_directory)}');"
                f"run_uqlab_akmcs({seed},'{_matlab_quote(result_path)}',"
                f"'{_matlab_quote(args.profile)}');"
            )
            subprocess.run(
                [str(matlab), "-batch", matlab_expression],
                cwd=repo_root,
                check=True,
            )
        results.append(
            _load_completed_result(result_path, profile=args.profile, seed=seed)
        )

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_pf = float(reference["mean_pf"])
    published_pf = float(
        reference.get("published_pf", reference.get("published_reference_pf", 1.52e-3))
    )
    summary = summarize_results(
        results,
        seeds=args.seeds,
        profile=args.profile,
        reference_pf=reference_pf,
        published_pf=published_pf,
    )
    summary["protocol"] = protocol
    _atomic_write_json(output_directory / "campaign_summary.json", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
