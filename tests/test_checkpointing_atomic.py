"""Tests for crash-safe checkpoint replacement."""

from __future__ import annotations

import numpy as np
import pytest

import absvr_core.checkpointing as checkpointing
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


def test_checkpoint_retries_transient_windows_replace_lock(tmp_path, monkeypatch) -> None:
    target = tmp_path / "checkpoint.npz"
    real_replace = checkpointing.os.replace
    attempts = 0

    def transient_lock(source, destination) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated transient lock")
        real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", transient_lock)
    monkeypatch.setattr(checkpointing.time, "sleep", lambda _delay: None)
    save_checkpoint(target, _state(3.0))

    assert attempts == 3
    np.testing.assert_array_equal(load_checkpoint(target)["g"], [3.0])
