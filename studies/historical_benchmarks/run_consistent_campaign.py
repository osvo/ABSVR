"""Run all analytical benchmarks with one reference-blind ABSVR configuration."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from absvr_core import run_adaptive_svr
from absvr_core.checkpointing import load_checkpoint
from absvr_core.problem_setup import prepare_problem_definition
from surrogate.svr import PeriodicEvidenceGridTrainer, svr_predict

from .run_corrected_campaign import DEFAULT_BENCHMARKS, PROFILES, _parse_benchmarks, _parse_seeds


def _physical_samples(
    unit: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    distribution: str,
) -> np.ndarray:
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    normal = mu + sigma * z
    return np.exp(normal) if distribution == "lognormal" else normal


def _validate_model(
    model: dict[str, Any],
    problem: tuple,
    *,
    reference_pf: float,
    log2_samples: int,
    replications: int,
    base_seed: int,
) -> dict[str, Any]:
    mu, sigma, n_dim, fun_par, fun, _, distribution = problem
    items = []
    for index in range(replications):
        unit = qmc.Sobol(d=n_dim, scramble=True, seed=base_seed + index).random_base2(
            log2_samples
        )
        samples = _physical_samples(unit, mu, sigma, distribution)
        true_failure = np.asarray(fun(samples, fun_par), dtype=float) <= 0.0
        prediction, _ = svr_predict(samples, model)
        predicted_failure = np.asarray(prediction, dtype=float) <= 0.0
        items.append(
            {
                "seed": int(base_seed + index),
                "samples": int(1 << log2_samples),
                "true_pf": float(np.mean(true_failure)),
                "surrogate_pf": float(np.mean(predicted_failure)),
                "true_positive": int(np.count_nonzero(true_failure & predicted_failure)),
                "false_positive": int(np.count_nonzero(~true_failure & predicted_failure)),
                "false_negative": int(np.count_nonzero(true_failure & ~predicted_failure)),
                "true_negative": int(np.count_nonzero(~true_failure & ~predicted_failure)),
            }
        )
    estimates = np.array([item["surrogate_pf"] for item in items], dtype=float)
    true_estimates = np.array([item["true_pf"] for item in items], dtype=float)
    surrogate_pf = float(np.mean(estimates))
    return {
        "method": "independently scrambled Sobol RQMC after model freezing",
        "replications": int(replications),
        "samples_per_replication": int(1 << log2_samples),
        "posthoc_true_limit_state_evaluations": int(replications * (1 << log2_samples)),
        "posthoc_evaluations_counted_as_absvr_calls": False,
        "surrogate_pf_mean": surrogate_pf,
        "same_points_true_pf_mean": float(np.mean(true_estimates)),
        "relative_error_vs_reference_percent": float(
            100.0 * abs(surrogate_pf - reference_pf) / reference_pf
        ),
        "individual_replications": items,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summarize(runs: Sequence[dict[str, Any]], reference_pf: float) -> dict[str, Any]:
    estimates = np.array(
        [run["validation"]["surrogate_pf_mean"] for run in runs], dtype=float
    )
    errors = np.array(
        [run["validation"]["relative_error_vs_reference_percent"] for run in runs],
        dtype=float,
    )
    calls = np.array([run["true_limit_state_calls"] for run in runs], dtype=float)
    return {
        "runs": len(runs),
        "mean_surrogate_pf": float(np.mean(estimates)),
        "relative_error_of_mean_pf_percent": float(
            100.0 * abs(float(np.mean(estimates)) - reference_pf) / reference_pf
        ),
        "mean_absolute_run_error_percent": float(np.mean(errors)),
        "median_absolute_run_error_percent": float(np.median(errors)),
        "maximum_absolute_run_error_percent": float(np.max(errors)),
        "mean_true_limit_state_calls": float(np.mean(calls)),
        "minimum_true_limit_state_calls": int(np.min(calls)),
        "maximum_true_limit_state_calls": int(np.max(calls)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=_parse_benchmarks, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--seeds", type=_parse_seeds, default=(127,))
    parser.add_argument("--max-added", type=int, default=80)
    parser.add_argument("--pool-log2", type=int, default=17)
    parser.add_argument("--validation-log2", type=int, default=18)
    parser.add_argument("--validation-replications", type=int, default=4)
    parser.add_argument("--validation-base-seed", type=int, default=20260809)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/consistent_historical/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/consistent_historical"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    adaptive_module = importlib.import_module("absvr_core.adaptive_loop")
    pool_size = 1 << args.pool_log2
    adaptive_module.N_MCS = pool_size
    adaptive_module.MCS_ENRICH_SIZE = pool_size
    adaptive_module.MAX_MCS_POOL_SIZE = pool_size
    runs_by_benchmark: dict[str, list[dict[str, Any]]] = {
        benchmark: [] for benchmark in args.benchmarks
    }

    for benchmark in args.benchmarks:
        problem = prepare_problem_definition(benchmark)
        reference_pf = float(PROFILES[benchmark].reference_pf)
        for seed in args.seeds:
            run_path = args.output_dir / "runs" / benchmark / f"seed_{seed}.json"
            checkpoint = args.checkpoint_dir / benchmark / f"seed_{seed}.npz"
            if args.resume and run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
            else:
                trainer = PeriodicEvidenceGridTrainer(
                    retune_interval=20,
                    loss="squared_epsilon",
                )
                result = run_adaptive_svr(
                    benchmark,
                    args.max_added,
                    problem_definition=problem,
                    random_seed=seed,
                    n0=0,
                    min_samples=args.max_added,
                    initial_design="normal_lhs",
                    svr_trainer=trainer,
                    learning_w_grad=0.0,
                    checkpoint_path=str(checkpoint),
                    resume_from=(str(checkpoint) if args.resume and checkpoint.is_file() else None),
                    checkpoint_frequency=1,
                    verbose=False,
                )
                state = load_checkpoint(checkpoint)
                doe = np.asarray(state["doe"], dtype=float)
                response = np.asarray(state["g"], dtype=float)
                model = trainer(doe, response, doe.shape[1], kernel="gaussian")
                validation = _validate_model(
                    model,
                    problem,
                    reference_pf=reference_pf,
                    log2_samples=args.validation_log2,
                    replications=args.validation_replications,
                    base_seed=args.validation_base_seed,
                )
                run = {
                    "benchmark": benchmark,
                    "algorithm_seed": int(seed),
                    "svr_loss": "squared_epsilon",
                    "kernel": "gaussian",
                    "gradient_weight": 0.0,
                    "true_limit_state_calls": int(result.total_evaluations),
                    "candidate_pool_size": int(pool_size),
                    "evidence_history": trainer.history,
                    "validation": validation,
                    "checkpoint": checkpoint.as_posix(),
                }
                _atomic_write_json(run_path, run)
            runs_by_benchmark[benchmark].append(run)
            print(
                f"{benchmark} seed={seed}: Pf={run['validation']['surrogate_pf_mean']:.8g}, "
                f"error={run['validation']['relative_error_vs_reference_percent']:.3f}%, "
                f"calls={run['true_limit_state_calls']}"
            )

    summaries = {
        benchmark: _summarize(runs, float(PROFILES[benchmark].reference_pf))
        for benchmark, runs in runs_by_benchmark.items()
    }
    report = {
        "schema_version": 1,
        "purpose": "benchmark-independent squared-loss ABSVR pilot",
        "current_benchmark_reference_used_for_training_or_model_selection": False,
        "configuration_origin": "theory correction and planar-truss development",
        "training_protocol": {
            "svr_loss": "squared_epsilon",
            "kernel": "gaussian",
            "gradient_weight": 0.0,
            "initial_design": "isoprobabilistic normal LHS",
            "added_points": int(args.max_added),
            "candidate_pool_size": int(pool_size),
            "hyperparameter_selection": "periodic deterministic Bayesian-evidence grid",
            "hyperparameter_retune_interval": 20,
        },
        "algorithm_seeds": list(args.seeds),
        "reference_probabilities": {
            benchmark: float(PROFILES[benchmark].reference_pf)
            for benchmark in args.benchmarks
        },
        "validation_protocol": {
            "log2_samples": int(args.validation_log2),
            "replications": int(args.validation_replications),
            "base_seed": int(args.validation_base_seed),
            "posthoc_only": True,
        },
        "summaries": summaries,
        "runs": [run for benchmark in args.benchmarks for run in runs_by_benchmark[benchmark]],
    }
    _atomic_write_json(args.output_dir / "campaign_summary.json", report)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
