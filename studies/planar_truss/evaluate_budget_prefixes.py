"""Evaluate frozen call-budget prefixes from completed planar-truss runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from absvr_core.checkpointing import load_checkpoint
from surrogate.svr import PeriodicEvidenceGridTrainer
from surrogate.svr.loss_utils import normalize_loss

from .run_absvr_campaign import (
    _gradient_label,
    _load_reference,
    _parse_int_list,
    _summarize_runs,
    _validate_model,
)


def _fit_prefix(
    doe: np.ndarray,
    response: np.ndarray,
    *,
    calls: int,
    initial_samples: int,
    retune_interval: int,
    loss: str,
) -> tuple[dict[str, Any], list[dict[str, float | int | str]]]:
    """Replay the evidence schedule and return the model at ``calls``."""

    if not initial_samples <= calls <= doe.shape[0]:
        raise ValueError("Requested call budget is outside the stored DOE prefix.")
    trainer = PeriodicEvidenceGridTrainer(
        retune_interval=retune_interval,
        loss=loss,
    )
    schedule = list(range(initial_samples, calls + 1, retune_interval))
    if schedule[-1] != calls:
        schedule.append(calls)
    model = None
    for sample_count in schedule:
        model = trainer(
            doe[:sample_count],
            response[:sample_count],
            doe.shape[1],
            kernel="gaussian",
        )
    assert model is not None
    return model, [dict(item) for item in trainer.history]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_int_list, required=True)
    parser.add_argument("--calls", type=_parse_int_list, default=(35, 55, 75, 95))
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--loss", default="squared_epsilon")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/planar_truss/reference_qmc.json"),
    )
    parser.add_argument("--validation-log2", type=int, default=19)
    parser.add_argument("--validation-replications", type=int, default=8)
    parser.add_argument("--validation-base-seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    loss = normalize_loss(args.loss)
    call_budgets = tuple(sorted(set(args.calls)))
    if call_budgets[0] < 15:
        raise ValueError("Every call budget must include the 15-point initial design.")
    reference_pf, reference_report = _load_reference(args.reference)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        checkpoint = args.checkpoint_dir / (
            f"seed_{seed}_gradient_{_gradient_label(args.gradient_weight)}_loss_{loss}_"
            "tuning_evidence.npz"
        )
        if not checkpoint.is_file():
            checkpoint = args.checkpoint_dir / (
                f"seed_{seed}_gradient_{_gradient_label(args.gradient_weight)}_loss_{loss}.npz"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing completed checkpoint: {checkpoint}")
        state = load_checkpoint(checkpoint)
        doe = np.asarray(state["doe"], dtype=float)
        response = np.asarray(state["g"], dtype=float).reshape(-1)
        if doe.shape[0] != response.size or doe.shape[0] < call_budgets[-1]:
            raise ValueError(f"Checkpoint {checkpoint} does not contain every requested prefix.")

        for calls in call_budgets:
            model, evidence_history = _fit_prefix(
                doe,
                response,
                calls=calls,
                initial_samples=15,
                retune_interval=20,
                loss=loss,
            )
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
                    "gradient_weight": float(args.gradient_weight),
                    "svr_loss": loss,
                    "open_sees_limit_state_calls": int(calls),
                    "additional_open_sees_calls_for_budget_analysis": 0,
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
        str(calls): _summarize_runs(
            (run for run in runs if run["open_sees_limit_state_calls"] == calls),
            reference_pf,
        )
        for calls in call_budgets
    }
    report = {
        "schema_version": 1,
        "purpose": "development-only posthoc call-budget selection",
        "configuration_development_uses_reference_probability": True,
        "confirmation_seeds_must_be_disjoint": True,
        "algorithm_seeds": list(args.seeds),
        "call_budgets": list(call_budgets),
        "svr_loss": loss,
        "additional_open_sees_calls": 0,
        "independent_reference": {
            "path": args.reference.as_posix(),
            "mean_pf": reference_pf,
            "confidence_interval_95": reference_report.get("confidence_interval_95"),
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
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
