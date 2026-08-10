"""Tests for the UQLab-to-frame OpenSeesPy CSV bridge."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.five_story_frame_opensees import (
    INPUT_MEANS,
    top_displacement_opensees,
)
from studies.five_story_frame.opensees_batch_bridge import (
    evaluate_physical_inputs,
    main,
)


def test_frame_bridge_matches_direct_opensees_limit_state() -> None:
    pytest.importorskip("openseespy.opensees")
    physical = np.vstack([INPUT_MEANS, INPUT_MEANS * 1.01])
    expected = 0.05 - np.abs(top_displacement_opensees(physical))
    actual = evaluate_physical_inputs(physical)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)


def test_frame_bridge_cli_writes_responses_and_ledger(tmp_path) -> None:
    pytest.importorskip("openseespy.opensees")
    physical = np.vstack([INPUT_MEANS, INPUT_MEANS * 0.99])
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    np.savetxt(input_path, physical, delimiter=",", fmt="%.17g")

    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ledger",
            str(ledger_path),
        ]
    ) == 0
    actual = np.atleast_1d(np.loadtxt(output_path, delimiter=","))
    np.testing.assert_allclose(
        actual, evaluate_physical_inputs(physical), rtol=0.0, atol=1.0e-15
    )
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["evaluations"] == 2
    assert entries[0]["input_order"] == list(
        (
            "P1", "P2", "P3", "E4", "E5", "I6", "I7", "I8", "I9",
            "I10", "I11", "I12", "I13", "A14", "A15", "A16", "A17",
            "A18", "A19", "A20", "A21",
        )
    )


def test_frame_bridge_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate_physical_inputs(np.ones((2, 20)))
