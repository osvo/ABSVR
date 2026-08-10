"""Repeated, auditable ABSVR campaign for the five-storey frame."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from absvr_core import run_adaptive_svr
from absvr_core.checkpointing import load_checkpoint
from benchmarks.five_story_frame_opensees import (
    PUBLISHED_REFERENCE_FAILURE_PROBABILITY,
    five_story_frame_limit_state,
    get_problem_definition,
)
from surrogate.svr import PeriodicEvidenceGridTrainer, svr_predict
from surrogate.svr.loss_utils import normalize_loss


DEFAULT_DEVELOPMENT_SEEDS = (11, 53, 101)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_reference(path: Path) -> tuple[float, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("The independent reference is not complete.")
    probability = float(report["summary"]["failure_probability"])
    if not 0.0 < probability < 1.0:
        raise ValueError("Reference probability must lie between zero and one.")
    return probability, report


def _problem_with_log_response() -> tuple:
    problem = list(get_problem_definition())
    settings = dict(problem[3])
    settings["solver"] = "opensees"
    settings["response"] = "log_ratio"
    problem[3] = settings
    return tuple(problem)


def _validation_replication(
    model: dict[str, Any], *, log2_samples: int, seed: int
) -> dict[str, int | float]:
    sample_count = 1 << log2_samples
    unit = qmc.Sobol(d=21, scramble=True, seed=seed).random_base2(log2_samples)
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    true_response = five_story_frame_limit_state(
        z, {"solver": "banded", "response": "log_ratio"}
    )
    predicted_response, _ = svr_predict(z, model)
    true_failure = np.asarray(true_response) <= 0.0
    predicted_failure = np.asarray(predicted_response) <= 0.0
    true_positive = int(np.count_nonzero(true_failure & predicted_failure))
    false_positive = int(np.count_nonzero(~true_failure & predicted_failure))
    false_negative = int(np.count_nonzero(true_failure & ~predicted_failure))
    return {
        "seed": int(seed),
        "samples": int(sample_count),
        "true_failures": int(np.count_nonzero(true_failure)),
        "predicted_failures": int(np.count_nonzero(predicted_failure)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": int(
            sample_count - true_positive - false_positive - false_negative
        ),
        "true_pf": float(np.mean(true_failure)),
        "surrogate_pf": float(np.mean(predicted_failure)),
    }


def _validate_model(
    model: dict[str, Any],
    *,
    reference_pf: float,
    log2_samples: int,
    replications: int,
    base_seed: int,
) -> dict[str, Any]:
    items = [
        _validation_replication(
            model, log2_samples=log2_samples, seed=base_seed + index
        )
        for index in range(replications)
    ]
    estimates = np.array([item["surrogate_pf"] for item in items], dtype=float)
    same_point_truth = np.array([item["true_pf"] for item in items], dtype=float)
    totals = {
        key: int(sum(int(item[key]) for item in items))
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    surrogate_pf = float(np.mean(estimates))
    sensitivity_denominator = totals["true_positive"] + totals["false_negative"]
    precision_denominator = totals["true_positive"] + totals["false_positive"]
    return {
        "method": "independently scrambled Sobol RQMC after model freezing",
        "replications": int(replications),
        "samples_per_replication": int(1 << log2_samples),
        "posthoc_true_limit_state_evaluations": int(replications * (1 << log2_samples)),
        "posthoc_evaluations_counted_as_absvr_calls": False,
        "surrogate_pf_mean": surrogate_pf,
        "surrogate_pf_standard_deviation": (
            float(np.std(estimates, ddof=1)) if replications > 1 else 0.0
        ),
        "same_points_true_pf_mean": float(np.mean(same_point_truth)),
        "relative_error_vs_independent_reference_percent": float(
            100.0 * abs(surrogate_pf - reference_pf) / reference_pf
        ),
        "confusion_counts": totals,
        "sensitivity": (
            totals["true_positive"] / sensitivity_denominator
            if sensitivity_denominator
            else float("nan")
        ),
        "precision": (
            totals["true_positive"] / precision_denominator
            if precision_denominator
            else float("nan")
        ),
        "individual_replications": items,
    }


def _run_one(
    *,
    seed: int,
    initial_samples: int,
    added_points: int,
    pool_log2: int,
    checkpoint_dir: Path,
    resume: bool,
    reference_pf: float,
    validation_log2: int,
    validation_replications: int,
    validation_base_seed: int,
    verbose: bool,
    learning_strategy: str,
    loss: str,
) -> dict[str, Any]:
    adaptive_module = importlib.import_module("absvr_core.adaptive_loop")
    pool_size = 1 << pool_log2
    adaptive_module.N_MCS = pool_size
    adaptive_module.MCS_ENRICH_SIZE = pool_size
    adaptive_module.MAX_MCS_POOL_SIZE = pool_size

    trainer = PeriodicEvidenceGridTrainer(
        retune_interval=20, loss=loss
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"seed_{seed}_{learning_strategy}_{loss}.npz"
    result = run_adaptive_svr(
        "five_story_frame",
        added_points,
        problem_definition=_problem_with_log_response(),
        random_seed=seed,
        n0=initial_samples,
        min_samples=added_points,
        initial_design="normal_lhs",
        svr_trainer=trainer,
        learning_w_grad=0.0,
        learning_strategy=learning_strategy,
        checkpoint_path=str(checkpoint),
        resume_from=(str(checkpoint) if resume and checkpoint.is_file() else None),
        checkpoint_frequency=1,
        verbose=verbose,
    )
    state = load_checkpoint(checkpoint)
    design = np.asarray(state["doe"], dtype=float)
    response = np.asarray(state["g"], dtype=float)
    model = trainer(design, response, design.shape[1], kernel="gaussian")
    validation = _validate_model(
        model,
        reference_pf=reference_pf,
        log2_samples=validation_log2,
        replications=validation_replications,
        base_seed=validation_base_seed,
    )
    return {
        "algorithm_seed": int(seed),
        "true_limit_state_calls": int(result.total_evaluations),
        "initial_samples": int(initial_samples),
        "added_points": int(added_points),
        "candidate_pool_size": int(pool_size),
        "learning_strategy": learning_strategy,
        "svr_loss": loss,
        "adaptive_pool_pf_diagnostic": float(result.probability_of_failure),
        "evidence_history": trainer.history,
        "validation": validation,
        "checkpoint": checkpoint.as_posix(),
    }


def _summarize(runs: Sequence[dict[str, Any]], reference_pf: float) -> dict[str, Any]:
    estimates = np.array(
        [run["validation"]["surrogate_pf_mean"] for run in runs], dtype=float
    )
    errors = np.array(
        [
            run["validation"]["relative_error_vs_independent_reference_percent"]
            for run in runs
        ],
        dtype=float,
    )
    calls = np.array([run["true_limit_state_calls"] for run in runs], dtype=float)
    return {
        "runs": len(runs),
        "mean_surrogate_pf": float(np.mean(estimates)),
        "standard_deviation_surrogate_pf_across_algorithm_seeds": (
            float(np.std(estimates, ddof=1)) if len(runs) > 1 else 0.0
        ),
        "relative_error_of_mean_pf_percent": float(
            100.0 * abs(float(np.mean(estimates)) - reference_pf) / reference_pf
        ),
        "mean_absolute_run_error_percent": float(np.mean(errors)),
        "median_absolute_run_error_percent": float(np.median(errors)),
        "minimum_absolute_run_error_percent": float(np.min(errors)),
        "maximum_absolute_run_error_percent": float(np.max(errors)),
        "mean_true_limit_state_calls": float(np.mean(calls)),
        "minimum_true_limit_state_calls": int(np.min(calls)),
        "maximum_true_limit_state_calls": int(np.max(calls)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_DEVELOPMENT_SEEDS)
    parser.add_argument("--campaign-stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--development-seeds", type=_parse_int_list, default=())
    parser.add_argument("--configuration-development-used-reference-probability", action="store_true")
    parser.add_argument("--initial-samples", type=int, default=43)
    parser.add_argument("--added-points", type=int, default=80)
    parser.add_argument("--loss", default="squared_epsilon")
    parser.add_argument("--pool-log2", type=int, default=17)
    parser.add_argument(
        "--learning-strategy",
        choices=("slf", "u", "u_distance"),
        default="slf",
    )
    parser.add_argument("--validation-log2", type=int, default=18)
    parser.add_argument("--validation-replications", type=int, default=4)
    parser.add_argument("--validation-base-seed", type=int, default=20260810)
    parser.add_argument("--reference", type=Path, default=Path("results/five_story_frame/reference_rqmc.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/five_story_frame/absvr_development_checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("results/five_story_frame/absvr_development.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.loss = normalize_loss(args.loss)

    if min(args.initial_samples, args.added_points, args.pool_log2) < 1:
        raise SystemExit("Initial size, added points, and pool exponent must be positive.")
    if args.validation_log2 < 1 or args.validation_replications < 1:
        raise SystemExit("Validation size and replication count must be positive.")
    if args.campaign_stage == "confirmation":
        if not args.configuration_development_used_reference_probability:
            raise SystemExit("Confirmation must disclose reference-aware development.")
        overlap = sorted(set(args.seeds) & set(args.development_seeds))
        if overlap:
            raise SystemExit(f"Development and confirmation seeds overlap: {overlap}")

    reference_pf, reference_report = _load_reference(args.reference)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"Running frame ABSVR seed={seed}")
        run = _run_one(
            seed=seed,
            initial_samples=args.initial_samples,
            added_points=args.added_points,
            pool_log2=args.pool_log2,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            reference_pf=reference_pf,
            validation_log2=args.validation_log2,
            validation_replications=args.validation_replications,
            validation_base_seed=args.validation_base_seed,
            verbose=args.verbose,
            learning_strategy=args.learning_strategy,
            loss=args.loss,
        )
        runs.append(run)
        print(
            "  Pf={pf:.8g}, error={error:.3f}%, calls={calls}".format(
                pf=run["validation"]["surrogate_pf_mean"],
                error=run["validation"]["relative_error_vs_independent_reference_percent"],
                calls=run["true_limit_state_calls"],
            )
        )

    report = {
        "schema_version": 1,
        "benchmark": "published five-storey three-bay frame evaluated with OpenSeesPy",
        "campaign_stage": args.campaign_stage,
        "configuration_development_used_reference_probability": bool(
            args.configuration_development_used_reference_probability
        ),
        "development_algorithm_seeds": list(args.development_seeds),
        "algorithm_seeds": list(args.seeds),
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "primary_metrics": [
            "relative_error_vs_independent_reference_percent",
            "true_limit_state_calls",
        ],
        "training_protocol": {
            "initial_design": f"{args.initial_samples}-point isoprobabilistic normal LHS",
            "response": "log(displacement_limit / abs(top_displacement))",
            "response_preserves_original_failure_event": True,
            "added_points": int(args.added_points),
            "candidate_pool_size": int(1 << args.pool_log2),
            "svr_loss": args.loss,
            "kernel": "gaussian",
            "gradient_weight": 0.0,
            "learning_strategy": args.learning_strategy,
            "hyperparameter_selection": "periodic deterministic Bayesian-evidence grid",
            "hyperparameter_retune_interval": 20,
        },
        "independent_reference": {
            "path": args.reference.as_posix(),
            "mean_pf": reference_pf,
            "confidence_interval_95": reference_report["summary"]["confidence_interval_95"],
            "published_importance_sampling_pf": PUBLISHED_REFERENCE_FAILURE_PROBABILITY,
        },
        "validation_protocol": {
            "log2_samples": int(args.validation_log2),
            "replications": int(args.validation_replications),
            "base_seed": int(args.validation_base_seed),
            "posthoc_only": True,
        },
        "software": {
            "numpy": np.__version__,
            "git_revision": _git_revision(),
        },
        "summary": _summarize(runs, reference_pf),
        "runs": runs,
    }
    _atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
