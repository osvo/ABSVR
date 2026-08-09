"""Tests for the UQLab-to-OpenSeesPy CSV bridge."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.planar_truss_opensees import INPUT_MEANS, midspan_displacement_opensees
from studies.planar_truss.opensees_batch_bridge import evaluate_physical_inputs, main


def test_bridge_matches_direct_opensees_limit_state() -> None:
    pytest.importorskip("openseespy.opensees")
    physical = np.vstack([INPUT_MEANS, INPUT_MEANS * 1.01])
    expected = 0.12 - np.abs(midspan_displacement_opensees(physical))
    actual = evaluate_physical_inputs(physical)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)


def test_bridge_cli_writes_responses_and_auditable_ledger(tmp_path) -> None:
    pytest.importorskip("openseespy.opensees")
    physical = np.vstack([INPUT_MEANS, INPUT_MEANS * 0.99])
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    np.savetxt(input_path, physical, delimiter=",", fmt="%.17g")

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ledger",
            str(ledger_path),
        ]
    )

    assert exit_code == 0
    actual = np.atleast_1d(np.loadtxt(output_path, delimiter=","))
    expected = evaluate_physical_inputs(physical)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)

    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["evaluations"] == 2
    assert entries[0]["input_order"] == [
        "A1", "A2", "E1", "E2", "P1", "P2", "P3", "P4", "P5", "P6"
    ]
    assert len(entries[0]["input_sha256_float64_le"]) == 64
    assert len(entries[0]["response_sha256_float64_le"]) == 64


def test_bridge_rejects_wrong_physical_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate_physical_inputs(np.ones((3, 9)))

