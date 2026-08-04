"""Canonical manifest-driven Dataset command implementation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths

from .paths import DatasetPaths
from .pipeline import (
    DatasetTask,
    PlanningOptions,
    RetryPolicy,
    TaskExecutionOutput,
    TaskStatusStore,
    execute_tasks,
    flatten_manifest_chunks,
    iter_manifest_records,
    iter_planned_tasks,
)
from .reporting import write_execution_report
from .stages.manifest import build_manifest_stage_handler

COMMAND_STAGES = {
    "census": "census",
    "resolve": "resolution",
    "acquire": "acquisition",
    "derive": "derivation",
    "validate": "validation",
    "release": "release",
}

StageHandler = Callable[[DatasetTask], TaskExecutionOutput]
_STAGE_HANDLERS: dict[str, StageHandler] = {}


def register_stage_handler(stage: str, handler: StageHandler) -> None:
    """Register a reusable record primitive without coupling it to the executor."""
    if stage not in set(COMMAND_STAGES.values()):
        raise ValueError(f"unsupported Dataset stage: {stage}")
    _STAGE_HANDLERS[stage] = handler


def _handler(
    stage: str, *, project: ProjectPaths, dataset_paths: DatasetPaths
) -> StageHandler:
    if stage in _STAGE_HANDLERS:
        return _STAGE_HANDLERS[stage]
    return build_manifest_stage_handler(
        stage, project=project, dataset_paths=dataset_paths
    )


def run_manifest_stage(
    *,
    manifest_path: Path,
    stage: str,
    batch_size: int,
    chunk_size: int,
    shard_index: int,
    num_shards: int,
    retry_failed: bool,
    handler: StageHandler | None = None,
) -> tuple:
    """Plan and run one manifest stage through the common serial executor."""
    project = ProjectPaths.discover(anchor=Path(__file__))
    dataset_paths = DatasetPaths.from_project(project)
    manifest_digest = sha256_file(manifest_path)
    options = PlanningOptions(
        stage=stage,
        stage_version="dataset-manifest-stage.v1",
        config={"stage": stage},
        base_seed=0,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    chunks = iter_manifest_records(manifest_path, chunk_size=chunk_size)
    tasks = iter_planned_tasks(flatten_manifest_chunks(chunks), options)
    layout = f"shards-{num_shards:05d}"
    status_path = (
        dataset_paths.runs
        / stage
        / manifest_digest
        / layout
        / f"shard-{shard_index:05d}.status.json"
    )
    results = execute_tasks(
        tasks,
        (
            _handler(stage, project=project, dataset_paths=dataset_paths)
            if handler is None
            else handler
        ),
        paths=project,
        status_store=TaskStatusStore(status_path),
        retry_policy=RetryPolicy(retry_failed=retry_failed),
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    report_dir = (
        dataset_paths.reports / "execution" / stage / manifest_digest / layout
    )
    prefix = f"shard-{shard_index:05d}"
    write_execution_report(
        results,
        table_path=report_dir / f"{prefix}.parquet",
        summary_path=report_dir / f"{prefix}.summary.json",
    )
    return results


def add_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    dataset = subparsers.add_parser("dataset", help="manifest-driven Dataset pipeline")
    commands = dataset.add_subparsers(dest="dataset_command", required=True)
    for command in COMMAND_STAGES:
        parser = commands.add_parser(command)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--chunk-size", type=int, default=1000)
        parser.add_argument("--shard-index", type=int, default=0)
        parser.add_argument("--num-shards", type=int, default=1)
        parser.add_argument("--retry-failed", action="store_true")


def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command != "dataset":
        parser.error("a command is required")
    if args.shard_index < 0 or args.num_shards <= 0 or args.shard_index >= args.num_shards:
        parser.error("shard bounds require 0 <= shard-index < num-shards")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("batch-size and chunk-size must be positive")
    results = run_manifest_stage(
        manifest_path=args.manifest,
        stage=COMMAND_STAGES[args.dataset_command],
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        retry_failed=args.retry_failed,
    )
    return 1 if any(result.execution_status.value.startswith("failed") for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dual-uq")
    commands = parser.add_subparsers(dest="command", required=True)
    add_dataset_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return dispatch(args, parser)
