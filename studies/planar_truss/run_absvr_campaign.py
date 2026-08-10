"""Repeated, reference-blind ABSVR campaign for the planar truss benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from absvr_core import run_adaptive_svr
from absvr_core.checkpointing import load_checkpoint
from benchmarks.planar_truss_opensees import (
    REFERENCE_FAILURE_PROBABILITY,
    get_problem_definition,
    planar_truss_limit_state,
)
from surrogate.svr import (
    PeriodicCrossValidationGridTrainer,
    PeriodicEvidenceGridTrainer,
    svr_predict,
)
from surrogate.svr.loss_utils import normalize_loss


DEFAULT_SEEDS = (11, 23, 37, 41, 53, 61, 73, 89, 97, 101)


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return parsed


def _parse_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("At least one non-negative weight is required.")
    return parsed


def _gradient_label(weight: float) -> str:
    return f"{weight:g}".replace("-", "m").replace(".", "p")


def _load_reference(path: Path) -> tuple[float, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    reference_pf = float(report["mean_pf"])
    if not 0.0 < reference_pf < 1.0:
        raise ValueError("Reference probability must lie strictly between zero and one.")
    return reference_pf, report


def _validation_replication(
    model: dict[str, Any],
    *,
    log2_samples: int,
    seed: int,
) -> dict[str, int | float]:
    n_samples = 1 << log2_samples
    unit = qmc.Sobol(d=10, scramble=True, seed=seed).random_base2(log2_samples)
    z = ndtri(
        np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    )
    true_response = planar_truss_limit_state(
        z,
        {"solver": "vectorized", "response": "log_ratio"},
    )
    predicted_response, _ = svr_predict(z, model)
    true_failure = np.asarray(true_response) <= 0.0
    predicted_failure = np.asarray(predicted_response) <= 0.0

    true_positive = int(np.count_nonzero(true_failure & predicted_failure))
    false_positive = int(np.count_nonzero(~true_failure & predicted_failure))
    false_negative = int(np.count_nonzero(true_failure & ~predicted_failure))
    true_negative = n_samples - true_positive - false_positive - false_negative
    return {
        "seed": int(seed),
        "samples": int(n_samples),
        "true_failures": int(np.count_nonzero(true_failure)),
        "predicted_failures": int(np.count_nonzero(predicted_failure)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": int(true_negative),
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
            model,
            log2_samples=log2_samples,
            seed=base_seed + index,
        )
        for index in range(replications)
    ]
    surrogate_estimates = np.array([item["surrogate_pf"] for item in items], dtype=float)
    true_estimates = np.array([item["true_pf"] for item in items], dtype=float)
    totals = {
        key: int(sum(int(item[key]) for item in items))
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    sensitivity_denominator = totals["true_positive"] + totals["false_negative"]
    precision_denominator = totals["true_positive"] + totals["false_positive"]
    surrogate_pf = float(np.mean(surrogate_estimates))
    return {
        "method": "independently scrambled Sobol RQMC, post-training",
        "replications": int(replications),
        "samples_per_replication": int(1 << log2_samples),
        "posthoc_reference_evaluations": int(replications * (1 << log2_samples)),
        "reference_evaluations_counted_as_absvr_calls": False,
        "surrogate_pf_mean": surrogate_pf,
        "surrogate_pf_standard_deviation": (
            float(np.std(surrogate_estimates, ddof=1)) if replications > 1 else 0.0
        ),
        "same_points_true_pf_mean": float(np.mean(true_estimates)),
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


def _summarize_runs(
    runs: Iterable[dict[str, Any]],
    reference_pf: float,
) -> dict[str, Any]:
    run_list = list(runs)
    errors = np.array(
        [run["validation"]["relative_error_vs_independent_reference_percent"] for run in run_list],
        dtype=float,
    )
    calls = np.array([run["open_sees_limit_state_calls"] for run in run_list], dtype=float)
    estimates = np.array(
        [run["validation"]["surrogate_pf_mean"] for run in run_list],
        dtype=float,
    )
    return {
        "runs": len(run_list),
        "mean_surrogate_pf": float(np.mean(estimates)),
        "standard_deviation_surrogate_pf_across_algorithm_seeds": (
            float(np.std(estimates, ddof=1)) if len(run_list) > 1 else 0.0
        ),
        "mean_relative_error_percent": float(np.mean(errors)),
        "relative_error_of_mean_pf_percent": float(
            100.0 * abs(np.mean(estimates) - reference_pf) / reference_pf
        ),
        "median_relative_error_percent": float(np.median(errors)),
        "minimum_relative_error_percent": float(np.min(errors)),
        "maximum_relative_error_percent": float(np.max(errors)),
        "mean_open_sees_limit_state_calls": float(np.mean(calls)),
        "minimum_open_sees_limit_state_calls": int(np.min(calls)),
        "maximum_open_sees_limit_state_calls": int(np.max(calls)),
    }


def _problem_with_response(response: str) -> tuple:
    problem = list(get_problem_definition())
    settings = dict(problem[3])
    settings["response"] = response
    problem[3] = settings
    return tuple(problem)


def run_one(
    *,
    seed: int,
    gradient_weight: float,
    max_added: int,
    pool_log2: int,
    checkpoint_dir: Path,
    resume: bool,
    reference_pf: float,
    validation_log2: int,
    validation_replications: int,
    validation_base_seed: int,
    verbose: bool,
    loss: str,
    hyperparameter_selection: str,
    learning_strategy: str,
    fixed_c: float,
    fixed_epsilon: float,
    fixed_theta: float,
    response: str,
) -> dict[str, Any]:
    adaptive_module = importlib.import_module("absvr_core.adaptive_loop")
    pool_size = 1 << pool_log2
    adaptive_module.N_MCS = pool_size
    adaptive_module.MCS_ENRICH_SIZE = pool_size
    # The campaign uses a fixed candidate population for every algorithm seed.
    adaptive_module.MAX_MCS_POOL_SIZE = pool_size

    if hyperparameter_selection == "cross_validation":
        trainer = PeriodicCrossValidationGridTrainer(retune_interval=20, loss=loss)
    elif hyperparameter_selection == "fixed":
        trainer = PeriodicEvidenceGridTrainer(
            retune_interval=20,
            initial_c_grid=(fixed_c,),
            epsilon_grid=(fixed_epsilon,),
            initial_theta_grid=(fixed_theta,),
            c_factors=(1.0,),
            theta_factors=(1.0,),
            c_bounds=(fixed_c, fixed_c),
            theta_bounds=(fixed_theta, fixed_theta),
            loss=loss,
        )
    else:
        trainer = PeriodicEvidenceGridTrainer(retune_interval=20, loss=loss)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    strategy_suffix = "" if learning_strategy == "slf" else f"_learning_{learning_strategy}"
    fixed_suffix = (
        "_c_{c}_epsilon_{epsilon}_theta_{theta}".format(
            c=_gradient_label(fixed_c),
            epsilon=_gradient_label(fixed_epsilon),
            theta=_gradient_label(fixed_theta),
        )
        if hyperparameter_selection == "fixed"
        else ""
    )
    response_suffix = "" if response == "log_ratio" else f"_response_{response}"
    checkpoint = checkpoint_dir / (
        f"seed_{seed}_gradient_{_gradient_label(gradient_weight)}_loss_{loss}_"
        f"tuning_{hyperparameter_selection}{fixed_suffix}{strategy_suffix}"
        f"{response_suffix}.npz"
    )
    resume_source = str(checkpoint) if resume and checkpoint.exists() else None
    result = run_adaptive_svr(
        "eg7",
        max_added,
        problem_definition=_problem_with_response(response),
        random_seed=seed,
        n0=15,
        min_samples=max_added,
        initial_design="normal_lhs",
        svr_trainer=trainer,
        learning_w_grad=gradient_weight,
        learning_strategy=learning_strategy,
        checkpoint_path=str(checkpoint),
        resume_from=resume_source,
        checkpoint_frequency=1,
        verbose=verbose,
    )

    state = load_checkpoint(checkpoint)
    doe = np.asarray(state["doe"], dtype=float)
    training_response = np.asarray(state["g"], dtype=float)
    model = trainer(doe, training_response, doe.shape[1], kernel="gaussian")
    validation = _validate_model(
        model,
        reference_pf=reference_pf,
        log2_samples=validation_log2,
        replications=validation_replications,
        base_seed=validation_base_seed,
    )
    return {
        "algorithm_seed": int(seed),
        "gradient_weight": float(gradient_weight),
        "svr_loss": loss,
        "hyperparameter_selection": hyperparameter_selection,
        "fixed_hyperparameters": (
            {
                "C": float(fixed_c),
                "epsilon": float(fixed_epsilon),
                "theta": float(fixed_theta),
            }
            if hyperparameter_selection == "fixed"
            else None
        ),
        "learning_strategy": learning_strategy,
        "response": response,
        "open_sees_limit_state_calls": int(result.total_evaluations),
        "adaptive_candidate_pool_size": int(pool_size),
        "adaptive_pool_pf_diagnostic": float(result.probability_of_failure),
        "evidence_history": trainer.history,
        "validation": validation,
        "checkpoint": checkpoint.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated algorithm seeds.",
    )
    parser.add_argument(
        "--gradient-weights",
        type=_parse_float_list,
        default=(0.0, 1.0),
        help="Comma-separated gradient weights for the ablation.",
    )
    parser.add_argument("--max-added", type=int, default=80)
    parser.add_argument("--loss", default="squared_epsilon")
    parser.add_argument(
        "--hyperparameter-selection",
        choices=("evidence", "cross_validation", "fixed"),
        default="evidence",
    )
    parser.add_argument("--fixed-c", type=float, default=1.0e3)
    parser.add_argument("--fixed-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--fixed-theta", type=float, default=6.25e-3)
    parser.add_argument(
        "--response",
        choices=("difference", "log_ratio"),
        default="log_ratio",
    )
    parser.add_argument(
        "--learning-strategy",
        choices=("slf", "u", "u_distance"),
        default="slf",
    )
    parser.add_argument(
        "--campaign-stage",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument(
        "--configuration-development-used-reference-probability",
        action="store_true",
    )
    parser.add_argument(
        "--development-seeds",
        type=_parse_int_list,
        default=(),
    )
    parser.add_argument("--pool-log2", type=int, default=17)
    parser.add_argument("--validation-log2", type=int, default=20)
    parser.add_argument("--validation-replications", type=int, default=16)
    parser.add_argument("--validation-base-seed", type=int, default=20260808)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/planar_truss/reference_qmc.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/planar_truss/campaign_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/planar_truss/absvr_campaign.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.loss = normalize_loss(args.loss)
    if args.campaign_stage == "confirmation":
        if not args.configuration_development_used_reference_probability:
            raise SystemExit(
                "Confirmation must disclose whether development used the reference probability."
            )
        overlap = sorted(set(args.seeds) & set(args.development_seeds))
        if overlap:
            raise SystemExit(f"Development and confirmation seeds overlap: {overlap}")
    if args.max_added < 1 or args.pool_log2 < 1:
        raise SystemExit("Training budget and pool exponent must be positive.")
    if min(args.fixed_c, args.fixed_epsilon, args.fixed_theta) <= 0.0:
        raise SystemExit("Fixed SVR hyperparameters must be positive.")
    if args.validation_log2 < 1 or args.validation_replications < 1:
        raise SystemExit("Validation size and replication count must be positive.")

    reference_pf, reference_report = _load_reference(args.reference)
    runs: list[dict[str, Any]] = []
    for gradient_weight in args.gradient_weights:
        for seed in args.seeds:
            print(f"Running seed={seed}, gradient_weight={gradient_weight:g}")
            run = run_one(
                seed=seed,
                gradient_weight=gradient_weight,
                max_added=args.max_added,
                pool_log2=args.pool_log2,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                reference_pf=reference_pf,
                validation_log2=args.validation_log2,
                validation_replications=args.validation_replications,
                validation_base_seed=args.validation_base_seed,
                verbose=args.verbose,
                loss=args.loss,
                hyperparameter_selection=args.hyperparameter_selection,
                learning_strategy=args.learning_strategy,
                fixed_c=args.fixed_c,
                fixed_epsilon=args.fixed_epsilon,
                fixed_theta=args.fixed_theta,
                response=args.response,
            )
            runs.append(run)
            print(
                "  Pf={pf:.8g}, relative error={error:.3f}%, calls={calls}".format(
                    pf=run["validation"]["surrogate_pf_mean"],
                    error=run["validation"][
                        "relative_error_vs_independent_reference_percent"
                    ],
                    calls=run["open_sees_limit_state_calls"],
                )
            )

    summaries = {
        _gradient_label(weight): _summarize_runs(
            (run for run in runs if np.isclose(run["gradient_weight"], weight)),
            reference_pf,
        )
        for weight in args.gradient_weights
    }
    report = {
        "schema_version": 2,
        "benchmark": "published 23-bar planar truss evaluated with OpenSeesPy",
        "campaign_stage": args.campaign_stage,
        "configuration_development_used_reference_probability": bool(
            args.configuration_development_used_reference_probability
        ),
        "development_algorithm_seeds": list(args.development_seeds),
        "selection_uses_reference_probability": False,
        "selection_uses_validation_set": False,
        "primary_metrics": [
            "relative_error_vs_independent_reference_percent",
            "open_sees_limit_state_calls",
        ],
        "training_protocol": {
            "initial_design": "15-point isoprobabilistic normal LHS",
            "response": (
                "displacement_limit - abs(midspan_displacement)"
                if args.response == "difference"
                else "log(displacement_limit / abs(midspan_displacement))"
            ),
            "response_identifier": args.response,
            "response_preserves_original_failure_event": True,
            "added_points": int(args.max_added),
            "candidate_pool_size": int(1 << args.pool_log2),
            "hyperparameter_selection": (
                "periodic deterministic cross-validated grid"
                if args.hyperparameter_selection == "cross_validation"
                else (
                    "single fixed development profile"
                    if args.hyperparameter_selection == "fixed"
                    else "periodic deterministic Bayesian-evidence grid"
                )
            ),
            "fixed_hyperparameters": (
                {
                    "C": float(args.fixed_c),
                    "epsilon": float(args.fixed_epsilon),
                    "theta": float(args.fixed_theta),
                }
                if args.hyperparameter_selection == "fixed"
                else None
            ),
            "hyperparameter_retune_interval": 20,
            "svr_loss": args.loss,
            "learning_strategy": args.learning_strategy,
        },
        "independent_reference": {
            "path": args.reference.as_posix(),
            "mean_pf": reference_pf,
            "confidence_interval_95": reference_report.get("confidence_interval_95"),
            "published_direct_mcs_pf": REFERENCE_FAILURE_PROBABILITY,
        },
        "validation_protocol": {
            "log2_samples": int(args.validation_log2),
            "replications": int(args.validation_replications),
            "base_seed": int(args.validation_base_seed),
            "posthoc_only": True,
        },
        "summaries_by_gradient_weight": summaries,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
