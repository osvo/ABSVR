"""Independent replicated-RQMC reference for the five-storey frame."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtri
from scipy.stats import qmc, t

from benchmarks.five_story_frame_opensees import (
    DEFAULT_DISPLACEMENT_LIMIT_M,
    PUBLISHED_REFERENCE_FAILURE_PROBABILITY,
    standard_normal_to_physical,
    top_displacement_banded,
)


DEFAULT_OUTPUT = Path("results/five_story_frame/reference_rqmc.json")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_reference(
    *, power: int, replicates: int, seed: int, chunk_size: int, output: Path
) -> dict[str, object]:
    if power < 1:
        raise ValueError("power must be positive.")
    if replicates < 2:
        raise ValueError("At least two independent scrambles are required.")

    n_per_replicate = 2**power
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(replicates)
    replicate_results: list[dict[str, object]] = []

    for replicate_index, child_sequence in enumerate(child_sequences):
        scramble_seed = int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        unit = qmc.Sobol(d=21, scramble=True, seed=scramble_seed).random_base2(power)
        independent_normal = ndtri(
            np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        )
        physical = standard_normal_to_physical(independent_normal)
        displacement = np.asarray(
            top_displacement_banded(physical, chunk_size=chunk_size), dtype=float
        )
        failed = displacement >= DEFAULT_DISPLACEMENT_LIMIT_M
        replicate_results.append(
            {
                "replicate": replicate_index,
                "scramble_seed": scramble_seed,
                "n_calls": n_per_replicate,
                "failure_count": int(np.count_nonzero(failed)),
                "failure_probability": float(np.mean(failed)),
                "response_mean_m": float(np.mean(displacement)),
                "response_standard_deviation_m": float(np.std(displacement, ddof=1)),
            }
        )

        checkpoint = {
            "status": "running",
            "protocol": {
                "method": "independently scrambled Sobol RQMC",
                "power": power,
                "replicates": replicates,
                "master_seed": seed,
                "n_per_replicate": n_per_replicate,
                "displacement_limit_m": DEFAULT_DISPLACEMENT_LIMIT_M,
                "solver": "exact symmetric-banded finite element",
                "chunk_size": chunk_size,
            },
            "completed_replicates": replicate_results,
        }
        _atomic_write_json(output, checkpoint)

    estimates = np.array(
        [result["failure_probability"] for result in replicate_results], dtype=float
    )
    mean_estimate = float(np.mean(estimates))
    replicate_standard_deviation = float(np.std(estimates, ddof=1))
    standard_error = replicate_standard_deviation / np.sqrt(replicates)
    critical_value = float(t.ppf(0.975, df=replicates - 1))
    confidence_interval = [
        float(mean_estimate - critical_value * standard_error),
        float(mean_estimate + critical_value * standard_error),
    ]

    payload: dict[str, object] = {
        "status": "complete",
        "protocol": {
            "method": "independently scrambled Sobol RQMC",
            "power": power,
            "replicates": replicates,
            "master_seed": seed,
            "n_per_replicate": n_per_replicate,
            "total_true_model_calls": replicates * n_per_replicate,
            "displacement_limit_m": DEFAULT_DISPLACEMENT_LIMIT_M,
            "solver": "exact symmetric-banded finite element",
            "chunk_size": chunk_size,
            "confidence_interval": "Student-t interval across independent scrambles",
        },
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git_revision": _git_revision(),
        },
        "replicates": replicate_results,
        "summary": {
            "failure_probability": mean_estimate,
            "replicate_standard_deviation": replicate_standard_deviation,
            "standard_error": standard_error,
            "confidence_interval_95": confidence_interval,
            "published_is_probability": PUBLISHED_REFERENCE_FAILURE_PROBABILITY,
            "relative_difference_from_published_is": float(
                abs(mean_estimate - PUBLISHED_REFERENCE_FAILURE_PROBABILITY)
                / PUBLISHED_REFERENCE_FAILURE_PROBABILITY
            ),
            "response_mean_m": float(
                np.mean([result["response_mean_m"] for result in replicate_results])
            ),
            "response_standard_deviation_m": float(
                np.mean(
                    [
                        result["response_standard_deviation_m"]
                        for result in replicate_results
                    ]
                )
            ),
        },
    }
    _atomic_write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=int, default=17)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--chunk-size", type=int, default=16_384)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run_reference(
        power=arguments.power,
        replicates=arguments.replicates,
        seed=arguments.seed,
        chunk_size=arguments.chunk_size,
        output=arguments.output,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
