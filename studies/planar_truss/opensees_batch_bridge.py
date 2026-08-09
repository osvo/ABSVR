"""CSV bridge from MATLAB/UQLab to the planar-truss OpenSeesPy model.

UQLab supplies samples in the physical variable order
``A1, A2, E1, E2, P1, ..., P6``. This module evaluates the conventional
limit-state function ``g = 0.12 - abs(displacement)`` and writes one response
per row. An optional append-only ledger makes the number and content of true
model evaluations independently auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmarks.planar_truss_opensees import (
    DEFAULT_DISPLACEMENT_LIMIT_M,
    INPUT_NAMES,
    midspan_displacement_opensees,
)


def _as_physical_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(INPUT_NAMES):
        raise ValueError(
            "Expected physical inputs with shape (n_samples, 10) in the order "
            + ", ".join(INPUT_NAMES)
            + "."
        )
    if matrix.shape[0] == 0:
        raise ValueError("At least one physical input row is required.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Physical inputs must all be finite.")
    if np.any(matrix <= 0.0):
        raise ValueError("Areas, elastic moduli, and loads must be positive.")
    return matrix


def evaluate_physical_inputs(
    physical_inputs: np.ndarray,
    *,
    displacement_limit_m: float = DEFAULT_DISPLACEMENT_LIMIT_M,
) -> np.ndarray:
    """Evaluate conventional limit-state values using OpenSeesPy."""

    matrix = _as_physical_matrix(physical_inputs)
    if not np.isfinite(displacement_limit_m) or displacement_limit_m <= 0.0:
        raise ValueError("displacement_limit_m must be finite and positive.")
    displacement = np.asarray(midspan_displacement_opensees(matrix), dtype=float)
    return displacement_limit_m - np.abs(displacement.reshape(-1))


def _array_digest(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _append_ledger(
    ledger_path: Path,
    *,
    physical_inputs: np.ndarray,
    responses: np.ndarray,
    displacement_limit_m: float,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": 1,
        "model": "23-bar planar truss / OpenSeesPy",
        "input_order": list(INPUT_NAMES),
        "evaluations": int(physical_inputs.shape[0]),
        "displacement_limit_m": float(displacement_limit_m),
        "input_sha256_float64_le": _array_digest(physical_inputs),
        "response_sha256_float64_le": _array_digest(responses),
    }
    with ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Headerless 10-column CSV.")
    parser.add_argument("--output", required=True, type=Path, help="One-column response CSV.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help=(
            "Optional JSONL evaluation ledger. If omitted, the "
            "ABSVR_UQLAB_CALL_LEDGER environment variable is used when set."
        ),
    )
    parser.add_argument(
        "--displacement-limit-m",
        type=float,
        default=DEFAULT_DISPLACEMENT_LIMIT_M,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    physical = _as_physical_matrix(np.loadtxt(args.input, delimiter=","))
    responses = evaluate_physical_inputs(
        physical,
        displacement_limit_m=args.displacement_limit_m,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, responses.reshape(-1, 1), delimiter=",", fmt="%.17g")

    ledger_value = args.ledger
    if ledger_value is None:
        environment_ledger = os.environ.get("ABSVR_UQLAB_CALL_LEDGER", "").strip()
        ledger_value = Path(environment_ledger) if environment_ledger else None
    if ledger_value is not None:
        _append_ledger(
            ledger_value,
            physical_inputs=physical,
            responses=responses,
            displacement_limit_m=args.displacement_limit_m,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    raise SystemExit(main())

