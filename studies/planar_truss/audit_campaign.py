"""Audit a frozen planar-truss campaign and build its literature comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy import stats


def audit_campaign(campaign: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Validate protocol invariants and return aggregate reliability metrics."""

    if campaign.get("selection_uses_reference_probability") is not False:
        raise ValueError("Campaign must explicitly be blind to the reference probability.")
    if campaign.get("selection_uses_validation_set") is not False:
        raise ValueError("Campaign must explicitly be blind to the validation set.")
    protocol = campaign["training_protocol"]
    if protocol.get("response_preserves_original_failure_event") is not True:
        raise ValueError("Campaign response must preserve the original failure event.")
    schema_version = int(campaign.get("schema_version", 1))
    raw_loss = protocol.get("svr_loss")
    expected_loss = None if raw_loss is None else str(raw_loss)
    if schema_version >= 2 and expected_loss not in {"legacy", "squared_epsilon"}:
        raise ValueError("Campaign must declare a recognized SVR loss formulation.")

    initial_samples = 15
    added_points = int(protocol["added_points"])
    expected_calls = initial_samples + added_points
    interval = int(protocol["hyperparameter_retune_interval"])
    expected_schedule = list(range(initial_samples, expected_calls + 1, interval))
    runs = list(campaign["runs"])
    if not runs:
        raise ValueError("Campaign contains no runs.")
    seeds = [int(run["algorithm_seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Algorithm seeds must be unique within the frozen campaign.")

    for run in runs:
        if expected_loss is not None and run.get("svr_loss") != expected_loss:
            raise ValueError("A run does not match the frozen SVR loss formulation.")
        if int(run["open_sees_limit_state_calls"]) != expected_calls:
            raise ValueError("A run does not match the fixed OpenSees call budget.")
        if int(run["adaptive_candidate_pool_size"]) != int(protocol["candidate_pool_size"]):
            raise ValueError("A run does not match the frozen candidate-pool size.")
        schedule = [int(item["n_samples"]) for item in run["evidence_history"]]
        if schedule != expected_schedule:
            raise ValueError(f"Unexpected evidence schedule for seed {run['algorithm_seed']}.")
        validation = run["validation"]
        if validation.get("reference_evaluations_counted_as_absvr_calls") is not False:
            raise ValueError("Post-training reference evaluations cannot count as ABSVR calls.")

    reference_pf = float(reference["mean_pf"])
    campaign_reference_pf = float(campaign["independent_reference"]["mean_pf"])
    if not np.isclose(reference_pf, campaign_reference_pf, rtol=0.0, atol=1.0e-15):
        raise ValueError("Campaign and independent reference probabilities differ.")
    published_pf = float(campaign["independent_reference"]["published_direct_mcs_pf"])
    estimates = np.array(
        [run["validation"]["surrogate_pf_mean"] for run in runs],
        dtype=float,
    )
    absolute_errors = 100.0 * np.abs(estimates - reference_pf) / reference_pf
    mean_pf = float(np.mean(estimates))
    sample_std = float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0
    if estimates.size > 1:
        half_width = float(
            stats.t.ppf(0.975, estimates.size - 1)
            * sample_std
            / np.sqrt(estimates.size)
        )
    else:
        half_width = float("nan")

    confusion = {
        key: int(
            sum(run["validation"]["confusion_counts"][key] for run in runs)
        )
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    sensitivity_denominator = confusion["true_positive"] + confusion["false_negative"]
    precision_denominator = confusion["true_positive"] + confusion["false_positive"]
    return {
        "audit_passed": True,
        "algorithm_seeds": seeds,
        "runs": len(runs),
        "fixed_open_sees_calls_per_run": expected_calls,
        "total_open_sees_calls": int(expected_calls * len(runs)),
        "candidate_pool_size": int(protocol["candidate_pool_size"]),
        "evidence_schedule": expected_schedule,
        "mean_pf": mean_pf,
        "beta_from_mean_pf": -NormalDist().inv_cdf(mean_pf),
        "standard_deviation_pf_across_algorithm_seeds": sample_std,
        "coefficient_of_variation_across_algorithm_seeds_percent": float(
            100.0 * sample_std / mean_pf
        ),
        "t_interval_95_for_mean_across_algorithm_seeds": [
            mean_pf - half_width,
            mean_pf + half_width,
        ],
        "relative_error_of_mean_vs_independent_rqmc_percent": float(
            100.0 * abs(mean_pf - reference_pf) / reference_pf
        ),
        "relative_error_of_mean_vs_published_mcs_percent": float(
            100.0 * abs(mean_pf - published_pf) / published_pf
        ),
        "mean_absolute_run_error_vs_independent_rqmc_percent": float(
            np.mean(absolute_errors)
        ),
        "median_absolute_run_error_vs_independent_rqmc_percent": float(
            np.median(absolute_errors)
        ),
        "maximum_absolute_run_error_vs_independent_rqmc_percent": float(
            np.max(absolute_errors)
        ),
        "pooled_confusion_counts": confusion,
        "pooled_sensitivity": confusion["true_positive"] / sensitivity_denominator,
        "pooled_precision": confusion["true_positive"] / precision_denominator,
        "independent_rqmc_reference_pf": reference_pf,
        "independent_rqmc_confidence_interval_95": reference["confidence_interval_95"],
        "published_direct_mcs_reference_pf": published_pf,
    }


def build_comparison_rows(
    literature_path: Path,
    audit: dict[str, Any],
) -> list[dict[str, str | float]]:
    """Append this study to the immutable literature comparison data."""

    with literature_path.open(newline="", encoding="utf-8") as stream:
        literature = list(csv.DictReader(stream))
    rows: list[dict[str, str | float]] = []
    reference_pf = float(audit["independent_rqmc_reference_pf"])
    for item in literature:
        literature_pf = float(item["pf"])
        rows.append(
            {
                "method": item["method"],
                "pf": literature_pf,
                "beta": float(item["beta"]),
                "mean_model_evaluations": float(item["mean_model_evaluations"]),
                "relative_error_vs_published_mcs_percent": float(
                    item["relative_error_percent"]
                ),
                "relative_error_vs_independent_rqmc_percent": float(
                    100.0 * abs(literature_pf - reference_pf) / reference_pf
                ),
                "runs": "",
                "source": item["source"],
            }
        )
    rows.append(
        {
            "method": "ABSVR (this study, mean)",
            "pf": audit["mean_pf"],
            "beta": audit["beta_from_mean_pf"],
            "mean_model_evaluations": audit["fixed_open_sees_calls_per_run"],
            "relative_error_vs_published_mcs_percent": audit[
                "relative_error_of_mean_vs_published_mcs_percent"
            ],
            "relative_error_vs_independent_rqmc_percent": audit[
                "relative_error_of_mean_vs_independent_rqmc_percent"
            ],
            "runs": audit["runs"],
            "source": "This study",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/planar_truss/confirmation_campaign.json"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/planar_truss/reference_qmc.json"),
    )
    parser.add_argument(
        "--literature",
        type=Path,
        default=Path("studies/planar_truss/literature_results.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("results/planar_truss/campaign_audit.json"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("results/planar_truss/comparison_table.csv"),
    )
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    audit = audit_campaign(campaign, reference)
    rows = build_comparison_rows(args.literature, audit)

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    with args.comparison_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
