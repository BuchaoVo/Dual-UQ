from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.models import LogicalAssetRef
from dual_uq.dataset.pipeline import (
    DatasetTaskError,
    PlanningOptions,
    RetryPolicy,
    TaskExecutionOutput,
    TaskStatusStore,
    execute_tasks,
    iter_manifest_records,
    plan_tasks,
)


def _paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "src/dual_uq").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return ProjectPaths.discover(project_root=tmp_path)


def _records(count: int) -> list[dict[str, object]]:
    return [
        {
            "record_id": f"protein-{index:03d}",
            "value": index,
            "eligibility": "eligible",
            "disposition": "candidate",
        }
        for index in range(1, count + 1)
    ]


def _options(
    *,
    stage: str = "derive",
    config: dict[str, object] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> PlanningOptions:
    return PlanningOptions(
        stage=stage,
        stage_version="dataset-stage.v1",
        config={} if config is None else config,
        base_seed=20260730,
        shard_index=shard_index,
        num_shards=num_shards,
    )


def _scientific_view(results):
    return [
        (
            result.record_id,
            result.execution_status.value,
            dict(result.metrics),
            result.scientific_disposition,
            result.failure.code if result.failure else None,
        )
        for result in results
    ]


def test_empty_manifest_plans_and_executes_as_empty(tmp_path: Path) -> None:
    tasks = plan_tasks([], _options())

    results = execute_tasks(
        tasks,
        lambda task: TaskExecutionOutput(metrics={"value": task.record["value"]}),
        paths=_paths(tmp_path),
        status_store=TaskStatusStore(tmp_path / "status.json"),
    )

    assert tasks == ()
    assert results == ()


def test_batch_size_one_multi_record_and_reordered_inputs_are_equivalent(
    tmp_path: Path,
) -> None:
    records = _records(7)

    def handler(task):
        return TaskExecutionOutput(
            metrics={"scientific_value": int(task.record["value"]) ** 2},
            scientific_disposition=str(task.record["disposition"]),
        )

    baseline = execute_tasks(
        plan_tasks(records, _options()),
        handler,
        paths=_paths(tmp_path / "one"),
        status_store=TaskStatusStore(tmp_path / "one-status.json"),
        batch_size=1,
        chunk_size=2,
    )
    reordered = execute_tasks(
        plan_tasks(list(reversed(records)), _options()),
        handler,
        paths=_paths(tmp_path / "many"),
        status_store=TaskStatusStore(tmp_path / "many-status.json"),
        batch_size=4,
        chunk_size=7,
    )

    assert _scientific_view(baseline) == _scientific_view(reordered)


def test_deterministic_shards_recombine_to_same_task_set() -> None:
    records = _records(41)
    unsharded = plan_tasks(records, _options())
    shards = [
        plan_tasks(
            list(reversed(records)),
            _options(shard_index=index, num_shards=5),
        )
        for index in range(5)
    ]

    combined = [task for shard in shards for task in shard]
    assert sorted(task.task_key for task in combined) == sorted(
        task.task_key for task in unsharded
    )
    assert len({task.task_key for task in combined}) == len(unsharded)
    assert all(task.shard_index(5) == index for index, shard in enumerate(shards) for task in shard)


def test_recombined_shard_results_equal_unsharded_results(tmp_path: Path) -> None:
    records = list(reversed(_records(19)))

    def handler(task):
        return TaskExecutionOutput(metrics={"value": int(task.record["value"]) * 3})

    whole = execute_tasks(
        plan_tasks(records, _options()),
        handler,
        paths=_paths(tmp_path / "whole"),
        status_store=TaskStatusStore(tmp_path / "whole-status.json"),
        batch_size=5,
    )
    sharded = []
    for shard_index in range(4):
        sharded.extend(
            execute_tasks(
                plan_tasks(
                    records,
                    _options(shard_index=shard_index, num_shards=4),
                ),
                handler,
                paths=_paths(tmp_path / f"shard-{shard_index}"),
                status_store=TaskStatusStore(
                    tmp_path / f"shard-{shard_index}.status.json"
                ),
                batch_size=2,
            )
        )

    assert _scientific_view(whole) == _scientific_view(
        sorted(sharded, key=lambda result: result.record_id)
    )


def test_duplicate_record_id_is_rejected_before_execution() -> None:
    records = _records(2)
    records.append(dict(records[0]))

    with pytest.raises(ValueError, match="duplicate record_id"):
        plan_tasks(records, _options())


def test_partial_failure_is_isolated_and_structured(tmp_path: Path) -> None:
    def handler(task):
        if task.record_id == "protein-002":
            raise DatasetTaskError("fixture_failure", "intentional fixture failure")
        return TaskExecutionOutput(metrics={"value": task.record["value"]})

    results = execute_tasks(
        plan_tasks(_records(3), _options()),
        handler,
        paths=_paths(tmp_path),
        status_store=TaskStatusStore(tmp_path / "status.json"),
    )

    assert [result.record_id for result in results] == [
        "protein-001",
        "protein-002",
        "protein-003",
    ]
    assert [result.execution_status.value for result in results] == [
        "complete",
        "failed_runtime",
        "complete",
    ]
    assert results[1].failure is not None
    assert results[1].failure.code == "fixture_failure"


def test_partial_resume_and_failed_retry_policy_are_record_level(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    fail_once = {"protein-002"}

    def handler(task):
        calls[task.record_id] += 1
        if task.record_id in fail_once:
            fail_once.remove(task.record_id)
            raise DatasetTaskError("transient_fixture", "retry me")
        return TaskExecutionOutput(metrics={"value": task.record["value"]})

    tasks = plan_tasks(_records(3), _options())
    paths = _paths(tmp_path)
    store = TaskStatusStore(tmp_path / "status.json")
    first = execute_tasks(tasks, handler, paths=paths, status_store=store)
    second = execute_tasks(
        tasks,
        handler,
        paths=paths,
        status_store=store,
        retry_policy=RetryPolicy(retry_failed=False),
    )
    third = execute_tasks(
        tasks,
        handler,
        paths=paths,
        status_store=store,
        retry_policy=RetryPolicy(retry_failed=True),
    )

    assert [result.execution_status.value for result in first] == [
        "complete",
        "failed_runtime",
        "complete",
    ]
    assert [result.execution_status.value for result in second] == [
        "skipped_validated",
        "failed_runtime",
        "skipped_validated",
    ]
    assert [result.execution_status.value for result in third] == [
        "skipped_validated",
        "complete",
        "skipped_validated",
    ]
    assert calls == Counter({"protein-002": 2, "protein-001": 1, "protein-003": 1})


@pytest.mark.parametrize(
    ("changed_records", "changed_options"),
    [
        ([{"record_id": "protein-001", "value": 99}], _options()),
        (_records(1), _options(config={"threshold": 2})),
    ],
)
def test_input_and_config_drift_rerun_only_changed_record(
    tmp_path: Path,
    changed_records: list[dict[str, object]],
    changed_options: PlanningOptions,
) -> None:
    calls: Counter[str] = Counter()

    def handler(task):
        calls[task.record_id] += 1
        return TaskExecutionOutput(metrics={"value": task.record["value"]})

    paths = _paths(tmp_path)
    store = TaskStatusStore(tmp_path / "status.json")
    execute_tasks(
        plan_tasks(_records(2), _options()), handler, paths=paths, status_store=store
    )
    changed = changed_records + [_records(2)[1]]
    results = execute_tasks(
        plan_tasks(changed, changed_options), handler, paths=paths, status_store=store
    )

    assert results[0].execution_status.value == "complete"
    assert results[0].resume_reason in {"input_drift", "config_drift"}
    assert calls["protein-001"] == 2
    if changed_options.config_digest == _options().config_digest:
        assert results[1].execution_status.value == "skipped_validated"
        assert calls["protein-002"] == 1
    else:
        assert results[1].execution_status.value == "complete"
        assert calls["protein-002"] == 2


def test_dependency_drift_reruns_changed_record(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()

    def handler(task):
        calls[task.record_id] += 1
        return TaskExecutionOutput(metrics={"value": task.record["value"]})

    paths = _paths(tmp_path)
    store = TaskStatusStore(tmp_path / "status.json")
    original = [{"record_id": "protein-001", "value": 1, "dependencies": {"p0": "a"}}]
    changed = [{"record_id": "protein-001", "value": 1, "dependencies": {"p0": "b"}}]
    execute_tasks(
        plan_tasks(original, _options()), handler, paths=paths, status_store=store
    )
    result = execute_tasks(
        plan_tasks(changed, _options()), handler, paths=paths, status_store=store
    )[0]

    assert result.execution_status.value == "complete"
    assert result.resume_reason == "input_and_dependency_drift"
    assert calls == Counter({"protein-001": 2})


def test_corrupt_declared_output_forces_only_that_record_to_rerun(
    tmp_path: Path,
) -> None:
    calls: Counter[str] = Counter()
    paths = _paths(tmp_path)

    def handler(task):
        calls[task.record_id] += 1
        logical_path = f"artifacts/dataset/results/{task.record_id}.txt"
        output = paths.resolve_logical(logical_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{task.record_id}:{calls[task.record_id]}\n")
        return TaskExecutionOutput(
            outputs=(
                LogicalAssetRef(
                    asset_type="result",
                    logical_path=logical_path,
                    sha256=sha256_file(output),
                    provenance="fixture",
                ),
            )
        )

    tasks = plan_tasks(_records(2), _options())
    store = TaskStatusStore(tmp_path / "status.json")
    execute_tasks(tasks, handler, paths=paths, status_store=store)
    paths.resolve_logical("artifacts/dataset/results/protein-002.txt").write_text(
        "corrupt\n"
    )

    results = execute_tasks(tasks, handler, paths=paths, status_store=store)

    assert [result.execution_status.value for result in results] == [
        "skipped_validated",
        "complete",
    ]
    assert results[1].resume_reason == "output_validation_failed"
    assert calls == Counter({"protein-002": 2, "protein-001": 1})


def test_manifest_reader_streams_tsv_and_parquet_in_chunks(tmp_path: Path) -> None:
    frame = pd.DataFrame(_records(5))
    tsv = tmp_path / "records.tsv"
    parquet = tmp_path / "records.parquet"
    frame.to_csv(tsv, sep="\t", index=False)
    frame.to_parquet(parquet, index=False)

    for path in (tsv, parquet):
        chunks = list(iter_manifest_records(path, chunk_size=2))
        assert [len(chunk) for chunk in chunks] == [2, 2, 1]
        assert [row["record_id"] for chunk in chunks for row in chunk] == list(
            frame["record_id"]
        )


def test_legacy_protein_id_is_explicitly_normalized_to_record_id() -> None:
    tasks = plan_tasks(
        [{"protein_id": "index103", "value": 1}],
        _options(),
    )

    assert tasks[0].record_id == "index103"
    assert tasks[0].record["record_id"] == "index103"


def test_task_record_is_deeply_immutable() -> None:
    source = {
        "record_id": "protein-001",
        "nested": {"values": [1, 2]},
    }
    task = plan_tasks([source], _options())[0]
    source["nested"]["values"][0] = 99

    assert task.record["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        task.record["nested"]["values"] = (3,)  # type: ignore[index]


@pytest.mark.parametrize("field", ["batch_size", "chunk_size"])
def test_execution_sizes_must_be_positive(tmp_path: Path, field: str) -> None:
    arguments = {field: 0}
    with pytest.raises(ValueError, match=field):
        execute_tasks(
            plan_tasks(_records(1), _options()),
            lambda task: TaskExecutionOutput(),
            paths=_paths(tmp_path),
            status_store=TaskStatusStore(tmp_path / "status.json"),
            **arguments,
        )
