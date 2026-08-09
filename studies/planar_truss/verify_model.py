"""Write a deterministic cross-solver verification report for the truss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.planar_truss_opensees import (
    ALL_MEMBERS,
    INPUT_MEANS,
    NODE_COORDINATES,
    midspan_displacement_numpy,
    midspan_displacement_opensees,
    midspan_displacement_vectorized,
    standard_normal_to_physical,
)


def build_report(seed: int, random_cases: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = np.vstack([np.zeros(10), rng.normal(size=(random_cases, 10))])
    physical = standard_normal_to_physical(z)
    numpy_response = np.asarray(midspan_displacement_numpy(physical))
    vectorized_response = np.asarray(midspan_displacement_vectorized(physical))
    opensees_response = np.asarray(midspan_displacement_opensees(physical))

    scale = np.maximum(np.abs(numpy_response), np.finfo(float).tiny)
    opensees_relative = np.abs(opensees_response - numpy_response) / scale
    vectorized_relative = np.abs(vectorized_response - numpy_response) / scale

    return {
        "seed": seed,
        "random_cases": random_cases,
        "node_count": len(NODE_COORDINATES),
        "member_count": len(ALL_MEMBERS),
        "mean_input_displacement_m": float(
            midspan_displacement_opensees(INPUT_MEANS)
        ),
        "max_relative_error_opensees_vs_numpy": float(np.max(opensees_relative)),
        "max_relative_error_vectorized_vs_numpy": float(np.max(vectorized_relative)),
        "passed": bool(
            np.max(opensees_relative) < 2.0e-11
            and np.max(vectorized_relative) < 2.0e-11
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/planar_truss/model_verification.json"),
    )
    args = parser.parse_args()

    report = build_report(args.seed, args.random_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Cross-solver verification failed.")


if __name__ == "__main__":
    main()
