"""Small immutable artifact writers shared by workflows and reports."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes


def write_immutable_bytes(path: Path, payload: bytes) -> str:
    """Create an artifact once, or reuse it when the bytes are identical."""

    path = Path(path)
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError as exc:
        if path.read_bytes() == payload:
            return "reused_identical"
        raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}") from exc
    return "created"


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a table without binding callers to a temporary filename."""

    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    """Create a Parquet artifact, comparing existing files by table content."""

    path = Path(path)
    if path.exists():
        if pd.read_parquet(path).equals(frame):
            return "reused_identical"
        raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
    return write_immutable_bytes(path, parquet_bytes(frame))


def json_bytes(value: Any) -> bytes:
    """Render deterministic UTF-8 JSON with one trailing LF."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    """Convert nested NumPy scalars and non-finite values for report JSON."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_immutable_json(path: Path, value: Any) -> str:
    """Create a deterministic JSON artifact without overwriting it."""

    return write_immutable_bytes(path, json_bytes(value))
