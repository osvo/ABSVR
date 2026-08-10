"""Evaluate stored frame-design prefixes without new structural calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from absvr_core.checkpointing import load_checkpoint
from surrogate.svr import PeriodicEvidenceGridTrainer

from .run_absvr_campaign import (
    _load_reference,
    _parse_int_list,
    _summarize,
    _validate_model,
)


def _fit_prefixes(
    design: np.ndarray,
    response: np.ndarray,
    *,
    call_budgets: tuple[int, ...],
    initial_samples: int,
    retune_interval: int,
) -> list[tuple[int, dict[str, Any], list[dict[str, Any]]]]:
    """Replay the original evidence schedule at requested design prefixes."""

    if call_budgets[0] < initial_samples or call_budgets[-1] > design.shape[0]:
        raise ValueError("A requested budget lies outside the stored design.")
    trainer = PeriodicEvidenceGridTrainer(
        retune_interval=retune_interval, loss="squared_epsilon"
    )
    retune_counts = set(
        range(initial_samples, call_budgets[-1] + 1, retune_interval)
    )
    schedule = sorted(retune_counts | set(call_budgets))
    results: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for sample_count in schedule:
        model = trainer(
            design[:sample_count],
            response[:sample_count],
            design.shape[1],
            kernel="gaussian",
        )
        if sample_count in call_budgets:
            results.append(
                (sample_count, model, [dict(item) for item in trainer.history])
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_int_list, default=(11, 53, 101))
    parser.add_argument(
        "--calls", type=_parse_int_list, default=(123, 143, 163, 183, 203, 223, 235)
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/five_story_frame/u_distance_development_checkpoints"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/five_story_frame/reference_rqmc.json"),
    )
    parser.add_argument("--validation-log2", type=int, default=18)
    parser.add_argument("--validation-replications", type=int, default=4)
    parser.add_argument("--validation-base-seed", type=int, default=20260810)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/five_story_frame/u_distance_budget_prefixes.json"),
    )
    args = parser.parse_args(argv)

    call_budgets = tuple(sorted(set(args.calls)))
    reference_pf, reference_report = _load_reference(args.reference)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        checkpoint = args.checkpoint_dir / f"seed_{seed}_u_distance.npz"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing completed checkpoint: {checkpoint}")
        state = load_checkpoint(checkpoint)
        design = np.asarray(state["doe"], dtype=float)
        response = np.asarray(state["g"], dtype=float).reshape(-1)
        if design.shape[0] != response.size:
            raise ValueError(f"Mismatched design and response in {checkpoint}")

        for calls, model, evidence_history in _fit_prefixes(
            design,
            response,
            call_budgets=call_budgets,
            initial_samples=43,
            retune_interval=20,
        ):
            validation = _validate_model(
                model,
                reference_pf=reference_pf,
                log2_samples=args.validation_log2,
                replications=args.validation_replications,
                base_seed=args.validation_base_seed,
            )
            runs.append(
                {
                    "algorithm_seed": int(seed),
                    "true_limit_state_calls": int(calls),
                    "additional_true_limit_state_calls_for_prefix_analysis": 0,
                    "learning_strategy": "u_distance",
                    "evidence_history": evidence_history,
                    "validation": validation,
                    "source_checkpoint": checkpoint.as_posix(),
                }
            )
            print(
                f"seed={seed}, calls={calls}: "
                f"Pf={validation['surrogate_pf_mean']:.8g}, "
                f"error={validation['relative_error_vs_independent_reference_percent']:.3f}%"
            )

    summaries = {
        str(calls): _summarize(
            [run for run in runs if run["true_limit_state_calls"] == calls],
            reference_pf,
        )
        for calls in call_budgets
    }
    eligible = [
        calls
        for calls in call_budgets
        if summaries[str(calls)]["relative_error_of_mean_pf_percent"] <= 3.0
        and summaries[str(calls)]["maximum_absolute_run_error_percent"] <= 5.0
    ]
    selected = min(eligible) if eligible else None
    report = {
        "schema_version": 1,
        "purpose": "development-only posthoc call-budget selection",
        "configuration_development_uses_reference_probability": True,
        "confirmation_seeds_must_be_disjoint": True,
        "algorithm_seeds": list(args.seeds),
        "call_budgets": list(call_budgets),
        "additional_true_limit_state_calls": 0,
        "selection_rule": {
            "rule": "minimum budget meeting both development accuracy thresholds",
            "maximum_relative_error_of_mean_pf_percent": 3.0,
            "maximum_individual_run_error_percent": 5.0,
            "selected_calls": selected,
        },
        "independent_reference": {
            "path": args.reference.as_posix(),
            "mean_pf": reference_pf,
            "confidence_interval_95": reference_report["summary"]["confidence_interval_95"],
        },
        "validation_protocol": {
            "log2_samples": int(args.validation_log2),
            "replications": int(args.validation_replications),
            "base_seed": int(args.validation_base_seed),
        },
        "summaries_by_call_budget": summaries,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_rule": report["selection_rule"], "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
