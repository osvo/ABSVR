"""Tests for crash-safe checkpoint replacement."""

from __future__ import annotations

import numpy as np
import pytest

from absvr_core.checkpointing import load_checkpoint, save_checkpoint


def _state(value: float) -> dict:
    return {
        "test_example": "atomic-test",
        "doe": np.array([[value]]),
        "g": np.array([value]),
    }


def test_checkpoint_round_trip(tmp_path) -> None:
    target = tmp_path / "checkpoint.npz"
    save_checkpoint(target, _state(1.25))
    restored = load_checkpoint(target)
    np.testing.assert_array_equal(restored["doe"], [[1.25]])
    np.testing.assert_array_equal(restored["g"], [1.25])


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch) -> None:
    target = tmp_path / "checkpoint.npz"
    save_checkpoint(target, _state(1.0))

    def fail_after_partial_write(stream, **_payload) -> None:
        stream.write(b"incomplete")
        raise OSError("simulated interruption")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated interruption"):
        save_checkpoint(target, _state(2.0))

    restored = load_checkpoint(target)
    np.testing.assert_array_equal(restored["g"], [1.0])
    assert list(tmp_path.glob(".checkpoint.npz.*.tmp")) == []
