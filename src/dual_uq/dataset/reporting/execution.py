"""Deterministic reporting from structured task execution results only."""

from __future__ import annotations

import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_bytes, atomic_write_json
from dual_uq.dataset.pipeline.task import TaskExecutionResult


def execution_summary(results: tuple[TaskExecutionResult, ...]) -> dict[str, object]:
    statuses = Counter(result.execution_status.value for result in results)
    failures = Counter(
        result.failure.code for result in results if result.failure is not None
    )
    return {
        "record_count": len(results),
        "execution_status_counts": dict(sorted(statuses.items())),
        "failure_code_counts": dict(sorted(failures.items())),
    }


def execution_frame(results: tuple[TaskExecutionResult, ...]) -> pd.DataFrame:
    rows = []
    for result in sorted(results, key=lambda item: (item.record_id, item.stage)):
        payload = result.as_dict()
        payload["outputs"] = json.dumps(
            payload["outputs"], sort_keys=True, separators=(",", ":")
        )
        payload["metrics"] = json.dumps(
            payload["metrics"], sort_keys=True, separators=(",", ":")
        )
        payload["failure"] = (
            None
            if payload["failure"] is None
            else json.dumps(payload["failure"], sort_keys=True, separators=(",", ":"))
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def write_execution_report(
    results: tuple[TaskExecutionResult, ...],
    *,
    table_path: Path,
    summary_path: Path,
) -> None:
    buffer = BytesIO()
    execution_frame(results).to_parquet(buffer, index=False)
    atomic_write_bytes(table_path, buffer.getvalue())
    atomic_write_json(summary_path, execution_summary(results))
