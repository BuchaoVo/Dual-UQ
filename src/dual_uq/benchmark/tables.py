"""Canonical table ordering, validation, and transactional Parquet writes."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from dual_uq.benchmark.schema_registry import SchemaRegistry
from dual_uq.benchmark.validation import validate_frame_contract
from dual_uq.core.atomic_io import atomic_write_new_bytes


def canonical_columns(table: str, registry: SchemaRegistry) -> tuple[str, ...]:
    return registry.get(table).properties


def validate_frame(frame: pd.DataFrame, table: str, registry: SchemaRegistry) -> pd.DataFrame:
    validate_frame_contract(frame, table, registry)
    ordered = frame.copy()
    schema = registry.get(table)
    for column in schema.properties:
        if column not in ordered:
            ordered[column] = None
    return ordered.loc[:, list(canonical_columns(table, registry))].copy()


def _parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove transient DataFrame metadata before Parquet serialization."""

    prepared = frame.copy(deep=False)
    prepared.attrs = {}
    return prepared


def write_parquet_transactional(frame: pd.DataFrame, table: str, destination: str | Path, registry: SchemaRegistry) -> None:
    ordered = _parquet_frame(validate_frame(frame, table, registry))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        ordered.to_parquet(temp, index=False)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def write_parquet_bundle_transactional(
    frames: dict[str, pd.DataFrame],
    destinations: dict[str, str | Path],
    registry: SchemaRegistry,
    *,
    replace_existing: bool = False,
) -> dict[str, str]:
    """Validate and stage a related table bundle before publishing any file."""
    if set(frames) != set(destinations):
        raise ValueError("frames and destinations must have identical table names")
    stage = Path(tempfile.mkdtemp(prefix=".canonical-materialization."))
    payloads: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    try:
        for name, frame in frames.items():
            ordered = _parquet_frame(validate_frame(frame, name, registry))
            buffer = io.BytesIO()
            ordered.to_parquet(buffer, index=False)
            payload = buffer.getvalue()
            staged = stage / f"{name}.parquet"
            staged.write_bytes(payload)
            validate_frame(pd.read_parquet(staged), name, registry)
            payloads[name] = payload
            digests[name] = hashlib.sha256(payload).hexdigest()
        resolved = {name: Path(path) for name, path in destinations.items()}
        for name, path in resolved.items():
            if (
                path.is_file()
                and not replace_existing
                and hashlib.sha256(path.read_bytes()).hexdigest() != digests[name]
            ):
                raise ValueError(f"immutable conflict: {path}")
        for name, path in resolved.items():
            if not path.exists() or replace_existing:
                path.parent.mkdir(parents=True, exist_ok=True)
                if replace_existing and path.exists():
                    os.replace(stage / f"{name}.parquet", path)
                else:
                    atomic_write_new_bytes(path, payloads[name])
        return digests
    finally:
        shutil.rmtree(stage, ignore_errors=True)
