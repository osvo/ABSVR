"""Rerun the six manuscript benchmarks after the corrected ABSVR formulation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from absvr_core import run_adaptive_svr


DEFAULT_BENCHMARKS = ("eg1", "eg2", "eg3", "eg4", "eg5", "eg6")
DEFAULT_SEEDS = tuple(range(100))
DEFAULT_OUTPUT_DIRECTORY = Path("results/corrected_historical")


@dataclass(frozen=True)
class BenchmarkProfile:
    benchmark: str
    kernel: str
    tail_policy: str
    tail_alpha: float
    c_init: float
    epsilon_init: float
    cov_pf_tol: float
    gamma: float | None
    reference_pf: float
    previous_results: str


PROFILES = {
    "eg1": BenchmarkProfile(
        "eg1", "gaussian", "none", 1.0, 9.25e4, 2.19e-4, 0.020, 1.0,
        4.7094e-3, "results/results_100seeds_eg1_v3.csv",
    ),
    "eg2": BenchmarkProfile(
        "eg2", "gaussian", "down", 1.95, 8.63e1, 2.72e-4, 0.049, 1.0,
        2.2199e-3, "results/results_100seeds_eg2.csv",
    ),
    "eg3": BenchmarkProfile(
        "eg3", "polynomial", "none", 1.0, 2.91e2, 2.21e-4, 0.046, 1.0,
        2.8614e-2, "results/results_100seeds_eg3.csv",
    ),
    "eg4": BenchmarkProfile(
        "eg4", "laplacian", "explore", 4.91, 1.93e2, 8.90e-4, 0.019, 1.0,
        7.2958e-2, "results/results_100seeds_eg4.csv",
    ),
    "eg5": BenchmarkProfile(
        "eg5", "laplacian", "down", 2.45, 1.97e2, 6.04e-3, 0.017, 1.0,
        1.8508e-3, "results/results_100seeds_eg5.csv",
    ),
    "eg6": BenchmarkProfile(
        "eg6", "gaussian", "none", 1.0, 1.00e2, 1.00e-2, 0.050, None,
        5.0800e-3, "results/results_100seeds_eg6.csv",
    ),
}


def _parse_benchmarks(value: str) -> tuple[str, ...]:
    benchmarks = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(benchmarks) - set(PROFILES))
    if not benchmarks or unknown:
        raise argparse.ArgumentTypeError(
            "Benchmarks must be a non-empty comma-separated subset of eg1,...,eg6."
        )
    if len(set(benchmarks)) != len(benchmarks):
        raise argparse.ArgumentTypeError("Benchmark identifiers must be unique.")
    return benchmarks


def _parse_seeds(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError("Seed ranges must be increasing.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    seeds = tuple(values)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers.")
    return seeds


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_previous_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "seed": int(row["seed"]),
            "pf": float(row["pf"]),
            "beta": float(row["beta"]),
            "evals": int(row["evals"]),
        }
        for row in rows
    ]


def _load_run(path: Path, *, benchmark: str, seed: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("benchmark") != benchmark or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Completed run metadata does not match its slot: {path}")
    return result


def summarize_benchmark(
    current_rows: Sequence[dict[str, Any]],
    previous_rows: Sequence[dict[str, Any]],
    *,
    profile: BenchmarkProfile,
    seeds: Sequence[int],
) -> dict[str, Any]:
    expected = tuple(seeds)
    current_seeds = tuple(int(row["seed"]) for row in current_rows)
    if current_seeds != expected:
        raise ValueError("Current results must contain every frozen seed in order.")
    previous_by_seed = {int(row["seed"]): row for row in previous_rows}
    if any(seed not in previous_by_seed for seed in expected):
        raise ValueError("Previous results do not contain every frozen seed.")
    comparable_previous = [previous_by_seed[seed] for seed in expected]

    current_pf = np.array([row["pf"] for row in current_rows], dtype=float)
    previous_pf = np.array([row["pf"] for row in comparable_previous], dtype=float)
    current_calls = np.array([row["evals"] for row in current_rows], dtype=float)
    previous_calls = np.array([row["evals"] for row in comparable_previous], dtype=float)
    current_mean = float(np.mean(current_pf))
    previous_mean = float(np.mean(previous_pf))
    current_error = float(abs(current_mean - profile.reference_pf) / profile.reference_pf)
    previous_error = float(abs(previous_mean - profile.reference_pf) / profile.reference_pf)
    return {
        "benchmark": profile.benchmark,
        "runs": len(current_rows),
        "algorithm_seeds": list(expected),
        "reference_pf": profile.reference_pf,
        "corrected_mean_pf": current_mean,
        "corrected_relative_error_percent": 100.0 * current_error,
        "corrected_mean_calls": float(np.mean(current_calls)),
        "corrected_standard_deviation_pf": (
            float(np.std(current_pf, ddof=1)) if len(current_pf) > 1 else 0.0
        ),
        "corrected_standard_deviation_calls": (
            float(np.std(current_calls, ddof=1)) if len(current_calls) > 1 else 0.0
        ),
        "previous_mean_pf": previous_mean,
        "previous_relative_error_percent": 100.0 * previous_error,
        "previous_mean_calls": float(np.mean(previous_calls)),
        "relative_error_improved": current_error < previous_error,
        "relative_error_ratio_corrected_over_previous": (
            current_error / previous_error if previous_error > 0.0 else float("inf")
        ),
        "all_runs_retained": True,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "pf", "beta", "evals"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", type=_parse_benchmarks, default=DEFAULT_BENCHMARKS
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_iter < 50:
        raise ValueError("max-iter must be at least the historical 50-point burn-in.")
    repo_root = Path(__file__).resolve().parents[2]
    output_directory = args.output_directory.resolve()
    runs_directory = output_directory / "runs"
    checkpoints_directory = output_directory / "checkpoints"

    protocol = {
        "schema_version": 1,
        "purpose": (
            "rerun manuscript examples after unbiased candidate-pool and "
            "polynomial-kernel corrections"
        ),
        "benchmarks": list(args.benchmarks),
        "algorithm_seeds": list(args.seeds),
        "runs_expected_per_benchmark": len(args.seeds),
        "max_added_samples": args.max_iter,
        "initial_sample_rule": "max(15, 2*n_dim + 1)",
        "initial_design": "historical bounded uniform-box LHS",
        "candidate_pool": "fixed unscrambled Sobol; 5e5 for dimensions <=6, 1e5 for 40D",
        "gradient_coefficient": 1.0,
        "svr_bounds_mode": "baseline",
        "profiles": {name: asdict(PROFILES[name]) for name in args.benchmarks},
        "selection_uses_corrected_campaign_results": False,
        "all_runs_retained": True,
    }
    _atomic_write_json(output_directory / "campaign_protocol.json", protocol)

    benchmark_summaries: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        profile = PROFILES[benchmark]
        current_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            run_path = runs_directory / benchmark / f"seed_{seed}.json"
            checkpoint_path = checkpoints_directory / benchmark / f"seed_{seed}.npz"
            if args.resume and run_path.is_file():
                row = _load_run(run_path, benchmark=benchmark, seed=seed)
            else:
                resume_path = str(checkpoint_path) if args.resume and checkpoint_path.is_file() else None
                result = run_adaptive_svr(
                    benchmark,
                    args.max_iter,
                    random_seed=seed,
                    svr_kernel=profile.kernel,
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_frequency=5,
                    resume_from=resume_path,
                    learning_w_grad=1.0,
                    svr_bounds_mode="baseline",
                    tail_policy=profile.tail_policy,
                    tail_alpha=profile.tail_alpha,
                    n0=0,
                    vm=3.0,
                    cov_pf_tol=profile.cov_pf_tol,
                    gamma=profile.gamma,
                    c_init=profile.c_init,
                    epsilon_init=profile.epsilon_init,
                    min_samples=0,
                    initial_design="uniform_box",
                    verbose=False,
                )
                row = {
                    "schema_version": 1,
                    "benchmark": benchmark,
                    "seed": seed,
                    "pf": result.probability_of_failure,
                    "beta": result.reliability_index,
                    "evals": result.total_evaluations,
                    "kernel": profile.kernel,
                    "all_runs_retained": True,
                }
                _atomic_write_json(run_path, row)
            current_rows.append(row)
            _write_csv(output_directory / f"{benchmark}_corrected.csv", current_rows)

        previous_path = repo_root / profile.previous_results
        previous_rows = _read_previous_rows(previous_path)
        benchmark_summaries.append(
            summarize_benchmark(
                current_rows,
                previous_rows,
                profile=profile,
                seeds=args.seeds,
            )
        )
        _atomic_write_json(
            output_directory / "campaign_summary.json",
            {
                "schema_version": 1,
                "protocol": protocol,
                "completed_benchmarks": [item["benchmark"] for item in benchmark_summaries],
                "benchmark_summaries": benchmark_summaries,
                "campaign_complete": len(benchmark_summaries) == len(args.benchmarks),
            },
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
