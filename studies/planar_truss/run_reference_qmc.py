"""Independent randomized-QMC reference for the planar-truss probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy import stats
from scipy.special import ndtri
from scipy.stats import qmc

from benchmarks.planar_truss_opensees import (
    DEFAULT_DISPLACEMENT_LIMIT_M,
    REFERENCE_FAILURE_PROBABILITY,
    midspan_displacement_vectorized,
    standard_normal_to_physical,
)


def one_replication(log2_samples: int, seed: int) -> dict[str, float | int]:
    n_samples = 1 << log2_samples
    unit = qmc.Sobol(d=10, scramble=True, seed=seed).random_base2(log2_samples)
    tiny = np.finfo(float).tiny
    z = ndtri(np.clip(unit, tiny, 1.0 - np.finfo(float).eps))
    physical = standard_normal_to_physical(z)
    displacement = np.asarray(midspan_displacement_vectorized(physical))
    failure_count = int(np.count_nonzero(np.abs(displacement) >= DEFAULT_DISPLACEMENT_LIMIT_M))
    pf = failure_count / n_samples
    return {"seed": seed, "samples": n_samples, "failures": failure_count, "pf": pf}


def summarize(replications: list[dict[str, float | int]]) -> dict[str, object]:
    estimates = np.array([float(item["pf"]) for item in replications])
    n_replications = estimates.size
    mean_pf = float(np.mean(estimates))
    standard_deviation = float(np.std(estimates, ddof=1)) if n_replications > 1 else 0.0
    standard_error = standard_deviation / np.sqrt(n_replications) if n_replications > 1 else 0.0
    if n_replications > 1:
        t_critical = float(stats.t.ppf(0.975, df=n_replications - 1))
    else:
        t_critical = float("nan")
    half_width = t_critical * standard_error if n_replications > 1 else float("nan")
    beta = -NormalDist().inv_cdf(mean_pf)

    return {
        "method": "randomized Sobol QMC with independent scrambles",
        "replications": n_replications,
        "samples_per_replication": int(replications[0]["samples"]),
        "total_limit_state_evaluations": int(
            n_replications * int(replications[0]["samples"])
        ),
        "mean_pf": mean_pf,
        "standard_deviation_across_scrambles": standard_deviation,
        "standard_error_of_mean": standard_error,
        "confidence_interval_95": [mean_pf - half_width, mean_pf + half_width],
        "beta_from_mean_pf": beta,
        "published_pf": REFERENCE_FAILURE_PROBABILITY,
        "relative_difference_from_published_percent": 100.0
        * abs(mean_pf - REFERENCE_FAILURE_PROBABILITY)
        / REFERENCE_FAILURE_PROBABILITY,
        "individual_replications": replications,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log2-samples", type=int, default=20)
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/planar_truss/reference_qmc.json"),
    )
    args = parser.parse_args()
    if args.log2_samples < 1 or args.replications < 1:
        raise SystemExit("Sample exponent and replication count must be positive.")

    replications = [
        one_replication(args.log2_samples, args.base_seed + index)
        for index in range(args.replications)
    ]
    report = summarize(replications)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
