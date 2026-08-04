"""Streaming manifest input and deterministic record-level task planning."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from dual_uq.core.hashing import derive_seed, sha256_canonical

from .task import DatasetTask


@dataclass(frozen=True)
class PlanningOptions:
    stage: str
    stage_version: str
    config: Mapping[str, Any] = field(default_factory=dict)
    base_seed: int = 0
    shard_index: int = 0
    num_shards: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if type(self.num_shards) is not int or self.num_shards <= 0:
            raise ValueError("num_shards must be a positive integer")
        if (
            type(self.shard_index) is not int
            or self.shard_index < 0
            or self.shard_index >= self.num_shards
        ):
            raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    @property
    def config_digest(self) -> str:
        return sha256_canonical(dict(self.config))


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if value is None:
        value = record.get("protein_id")
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("record_id is required (legacy protein_id is accepted explicitly)")
    return value


def _normalized(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["record_id"] = _record_id(record)
    return normalized


def _task(record: Mapping[str, Any], options: PlanningOptions) -> DatasetTask:
    normalized = _normalized(record)
    record_id = str(normalized["record_id"])
    dependencies = normalized.get("dependencies", {})
    seed = derive_seed(
        base_seed=options.base_seed,
        pipeline_version=options.stage_version,
        protein_id=record_id,
        stage=options.stage,
        candidate_id=None,
        replicate=0,
    )
    return DatasetTask(
        record_id=record_id,
        stage=options.stage,
        stage_version=options.stage_version,
        record=normalized,
        input_digest=sha256_canonical(normalized),
        config_digest=options.config_digest,
        dependency_digest=sha256_canonical(dependencies),
        seed=seed,
    )


def iter_planned_tasks(
    records: Iterable[Mapping[str, Any]], options: PlanningOptions
) -> Iterator[DatasetTask]:
    seen: set[str] = set()
    for record in records:
        task = _task(record, options)
        if task.record_id in seen:
            raise ValueError(f"duplicate record_id: {task.record_id}")
        seen.add(task.record_id)
        if task.shard_index(options.num_shards) == options.shard_index:
            yield task


def plan_tasks(
    records: Iterable[Mapping[str, Any]], options: PlanningOptions
) -> tuple[DatasetTask, ...]:
    """Materialize a deterministic plan for inspection and small/medium panels."""
    return tuple(
        sorted(
            iter_planned_tasks(records, options),
            key=lambda task: (task.record_id, task.stage),
        )
    )


def _records_from_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    return tuple(frame.where(pd.notna(frame), None).to_dict(orient="records"))


def iter_manifest_records(
    path: Path, *, chunk_size: int = 1000
) -> Iterator[tuple[dict[str, Any], ...]]:
    """Yield bounded manifest chunks without rescanning the manifest per record."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunk_size):
            yield _records_from_frame(batch.to_pandas())
        return
    if suffix in {".tsv", ".csv"}:
        separator = "\t" if suffix == ".tsv" else ","
        for frame in pd.read_csv(path, sep=separator, chunksize=chunk_size):
            yield _records_from_frame(frame)
        return
    if suffix in {".jsonl", ".ndjson"}:
        chunk: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"manifest line {line_number} must be a JSON object")
                chunk.append(value)
                if len(chunk) == chunk_size:
                    yield tuple(chunk)
                    chunk = []
        if chunk:
            yield tuple(chunk)
        return
    raise ValueError(f"unsupported manifest format: {suffix or '<none>'}")


def flatten_manifest_chunks(
    chunks: Iterable[Sequence[Mapping[str, Any]]],
) -> Iterator[Mapping[str, Any]]:
    for chunk in chunks:
        yield from chunk
