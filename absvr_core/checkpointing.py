"""Checkpoint persistence helpers for the adaptive SVR absvr_core."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

import numpy as np

CHECKPOINT_VERSION = 1

_ARRAY_FIELDS = (
    "doe",
    "g",
    "mc_pool",
    "v_pdf_pool",
    "pf_history",
    "pf_sequence",
)


def _ensure_parent_directory(path: Path) -> None:
    """Ensure the checkpoint directory exists."""

    if path.parent.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy scalars/containers to built-in Python types."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    """Persist the adaptive SVR state to ``path``."""

    target = Path(path)
    _ensure_parent_directory(target)

    arrays: Dict[str, np.ndarray] = {}
    for field in _ARRAY_FIELDS:
        if field not in state:
            continue
        arrays[field] = np.asarray(state[field])

    meta_keys = set(state.keys()) - set(_ARRAY_FIELDS)
    meta: Dict[str, Any] = {key: _to_builtin(state[key]) for key in meta_keys}
    meta["version"] = CHECKPOINT_VERSION

    payload = {name: array for name, array in arrays.items()}
    payload["meta"] = np.array(json.dumps(meta))

    # Write beside the destination and atomically replace it only after the
    # compressed archive has been flushed. An interruption therefore leaves
    # either the preceding valid checkpoint or the complete new checkpoint,
    # never a partially written NPZ at the public path.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load the adaptive SVR state stored at ``path``."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        raw_meta = data["meta"]
        if isinstance(raw_meta, np.ndarray):
            meta_json = raw_meta.item() if raw_meta.shape == () else str(raw_meta)
        else:
            meta_json = str(raw_meta)
        meta: Dict[str, Any] = json.loads(meta_json)
        version = int(meta.get("version", -1))
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Incompatible checkpoint version {version}; expected {CHECKPOINT_VERSION}. "
                "Please regenerate the checkpoint."
            )

        state: Dict[str, Any] = {key: meta[key] for key in meta if key != "version"}

        for field in _ARRAY_FIELDS:
            if field not in data:
                continue
            state[field] = np.asarray(data[field])

    return state

