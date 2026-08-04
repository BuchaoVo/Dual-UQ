"""Generic deterministic batch/subset orchestration for dataset derivation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths

from ..models import (
    CandidateContext,
    CandidateDerivationResult,
    DerivationConfig,
    DerivationRunResult,
    FailureRecord,
)
from .status import LifecycleStatus, TaskStatusStore
from .task import (
    DatasetTask,
    DatasetTaskError,
    TaskExecutionOutput,
    TaskExecutionResult,
)


@dataclass(frozen=True)
class RetryPolicy:
    retry_failed: bool = False


def _positive(value: int, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _batched(values: Iterable[DatasetTask], size: int) -> Iterator[tuple[DatasetTask, ...]]:
    batch: list[DatasetTask] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _outputs_valid(result: TaskExecutionResult, paths: ProjectPaths) -> bool:
    for output in result.outputs:
        if output.sha256 is None:
            return False
        try:
            path = paths.resolve_logical(output.logical_path)
        except ProjectPathError:
            return False
        if not path.is_file() or path.is_symlink() or sha256_file(path) != output.sha256:
            return False
    return True


def _historical_result(
    task: DatasetTask,
    raw: dict[str, Any] | None,
) -> TaskExecutionResult | None:
    if raw is None:
        return None
    try:
        result = TaskExecutionResult.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None
    if result.task_key != task.task_key:
        return None
    return result


def _result(
    task: DatasetTask,
    *,
    status: LifecycleStatus,
    output: TaskExecutionOutput | None = None,
    failure: FailureRecord | None = None,
    resume_reason: str | None = None,
) -> TaskExecutionResult:
    value = TaskExecutionOutput() if output is None else output
    return TaskExecutionResult(
        record_id=task.record_id,
        task_key=task.task_key,
        task_digest=task.task_digest,
        stage=task.stage,
        stage_version=task.stage_version,
        input_digest=task.input_digest,
        config_digest=task.config_digest,
        dependency_digest=task.dependency_digest,
        execution_status=status,
        outputs=value.outputs,
        metrics=value.metrics,
        scientific_disposition=value.scientific_disposition,
        failure=failure,
        resume_reason=resume_reason,
    )


def _execute_task(
    task: DatasetTask,
    handler: Callable[[DatasetTask], TaskExecutionOutput],
    *,
    resume_reason: str,
) -> TaskExecutionResult:
    try:
        output = handler(task)
        if not isinstance(output, TaskExecutionOutput):
            raise TypeError("stage handler must return TaskExecutionOutput")
        return _result(
            task,
            status=LifecycleStatus.COMPLETE,
            output=output,
            resume_reason=resume_reason,
        )
    except DatasetTaskError as exc:
        return _result(
            task,
            status=LifecycleStatus.FAILED_RUNTIME,
            failure=exc.failure,
            resume_reason=resume_reason,
        )
    # Stage adapters may raise third-party parser/runtime exceptions. The record
    # boundary deliberately converts every such exception into failure evidence.
    except Exception as exc:  # noqa: BLE001
        return _result(
            task,
            status=LifecycleStatus.FAILED_RUNTIME,
            failure=FailureRecord(
                code=f"unexpected_{type(exc).__name__}",
                message=str(exc) or type(exc).__name__,
            ),
            resume_reason=resume_reason,
        )


def execute_tasks(
    tasks: Iterable[DatasetTask],
    handler: Callable[[DatasetTask], TaskExecutionOutput],
    *,
    paths: ProjectPaths,
    status_store: TaskStatusStore,
    retry_policy: RetryPolicy | None = None,
    batch_size: int = 1,
    chunk_size: int = 1000,
) -> tuple[TaskExecutionResult, ...]:
    """Execute bounded serial batches with record-level validated reuse."""
    checked_batch_size = _positive(batch_size, "batch_size")
    checked_chunk_size = _positive(chunk_size, "chunk_size")
    policy = RetryPolicy() if retry_policy is None else retry_policy
    raw_status = status_store.read()
    results: list[TaskExecutionResult] = []

    for chunk in _batched(tasks, checked_chunk_size):
        for batch in _batched(chunk, checked_batch_size):
            for task in batch:
                historical = _historical_result(task, raw_status.get(task.task_key))
                if historical is not None:
                    successful = historical.execution_status in {
                        LifecycleStatus.COMPLETE,
                        LifecycleStatus.SKIPPED_VALIDATED,
                    }
                    if (
                        successful
                        and historical.task_digest == task.task_digest
                        and _outputs_valid(historical, paths)
                    ):
                        current = replace(
                            historical,
                            execution_status=LifecycleStatus.SKIPPED_VALIDATED,
                            resume_reason="validated_existing_task",
                        )
                    elif (
                        historical.execution_status
                        in {LifecycleStatus.FAILED_RUNTIME, LifecycleStatus.FAILED_VALIDATION}
                        and historical.task_digest == task.task_digest
                        and not policy.retry_failed
                    ):
                        current = replace(
                            historical,
                            resume_reason="previous_failure_not_retried",
                        )
                    else:
                        if (
                            historical.execution_status
                            in {
                                LifecycleStatus.FAILED_RUNTIME,
                                LifecycleStatus.FAILED_VALIDATION,
                            }
                            and historical.task_digest == task.task_digest
                        ):
                            reason = "retry_previous_failure"
                        elif historical.task_digest != task.task_digest:
                            changed = []
                            for name in (
                                "stage_version",
                                "input_digest",
                                "config_digest",
                                "dependency_digest",
                            ):
                                if getattr(historical, name) != getattr(task, name):
                                    changed.append(name.removesuffix("_digest"))
                            reason = "_and_".join(changed) + "_drift"
                        else:
                            reason = "output_validation_failed"
                        current = _execute_task(task, handler, resume_reason=reason)
                else:
                    current = _execute_task(task, handler, resume_reason="new_task")
                raw_status[task.task_key] = current.as_dict()
                status_store.write(raw_status)
                results.append(current)

    return tuple(sorted(results, key=lambda result: (result.record_id, result.stage)))


def derive_candidate(
    context: CandidateContext,
    config: DerivationConfig,
    paths: ProjectPaths,
) -> CandidateDerivationResult:
    """Late-bound candidate executor keeps pipeline orchestration capability-only."""
    from ..stages.candidate_derivation import derive_candidate as execute_candidate

    return execute_candidate(context, config, paths)


def _known_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _descriptive_attrition(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        "canonical_sequence_length",
        "mapped_residue_count",
        "mapping_coverage",
        "gap_count",
        "segment_count",
        "global_pLDDT_proxy",
        "metadata_record_count",
        "fragment_count",
    ]
    groups: dict[str, Any] = {}
    for name, rows in (
        (
            "mechanism_observable",
            [row for row in candidates if row["mechanism_observable"]],
        ),
        (
            "not_mechanism_observable",
            [row for row in candidates if not row["mechanism_observable"]],
        ),
    ):
        medians: dict[str, float | None] = {}
        for field in fields:
            values = [
                float(row[field])
                for row in rows
                if _known_number(row.get(field)) is not None
            ]
            medians[field] = float(np.median(values)) if values else None
        groups[name] = {
            "count": len(rows),
            "medians": medians,
            "pae_available_count": sum(bool(row.get("pae_bound")) for row in rows),
        }
    return {
        "scope": (
            f"descriptive stress-test only; N={len(candidates)}; "
            "no significance or prevalence inference"
        ),
        "groups": groups,
        "requested_proxies_not_available": [],
    }


def _summary(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attrition_classes = {
        str(row["attrition_class"])
        for row in candidates
        if row.get("attrition_class")
    }
    if "implementation_or_schema_issue" in attrition_classes:
        verdict = "IMPLEMENTATION_BUG"
    elif "protocol_eligibility_issue" in attrition_classes:
        verdict = "PROTOCOL_CONFLICT"
    elif all(row["mechanism_observable"] for row in candidates) and not any(
        row.get("scientific_attrition_flags") for row in candidates
    ):
        verdict = "DERIVATION_PIPELINE_PASS"
    else:
        verdict = "DERIVATION_PIPELINE_PASS_WITH_DATA_FAILURES"
    return {
        "candidate_count": len(candidates),
        "canonical_sequence_complete_count": sum(
            bool(row["canonical_sequence_complete"]) for row in candidates
        ),
        "pair_qc_complete_count": sum(
            bool(row["pair_qc_complete"]) for row in candidates
        ),
        "pair_qc_pass_count": sum(
            row.get("pair_qc_status") == "pair_qc_pass" for row in candidates
        ),
        "mapping_complete_count": sum(
            bool(row["mapping_complete"]) for row in candidates
        ),
        "fragment_resolved_count": sum(
            bool(row["fragment_resolved"]) for row in candidates
        ),
        "pae_bound_count": sum(bool(row["pae_bound"]) for row in candidates),
        "confidence_bound_count": sum(
            bool(row["confidence_bound"]) for row in candidates
        ),
        "mechanism_observable_count": sum(
            bool(row["mechanism_observable"]) for row in candidates
        ),
        "primary_failure_count": sum(
            row.get("primary_failure_stage") is not None for row in candidates
        ),
        "pair_qc_fail_count": sum(
            row.get("pair_qc_status") == "pair_qc_fail" for row in candidates
        ),
        "protocol_eligibility_issue_count": sum(
            "protocol_eligibility_issue"
            in row.get("scientific_attrition_classes", [])
            for row in candidates
        ),
        "pipeline_verdict": verdict,
    }


def run_derivation(
    panel: Sequence[CandidateContext],
    config: DerivationConfig,
    paths: ProjectPaths,
) -> DerivationRunResult:
    """Run one generic scientific implementation over any registered subset."""
    if not panel:
        raise ValueError("derivation panel must not be empty")
    indices = [context.candidate_index for context in panel]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate_index in derivation panel")
    identities = [context.identity.pair_id for context in panel]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate biological identity in derivation panel")
    biological_keys = [
        (
            context.identity.pdb_id,
            context.identity.chain_id,
            context.identity.uniprot_accession,
            context.identity.polymer_entity_id,
        )
        for context in panel
    ]
    if len(biological_keys) != len(set(biological_keys)):
        raise ValueError("duplicate biological identity in derivation panel")
    if any(context.protocol_binding != config.protocol_binding for context in panel):
        raise ValueError("candidate protocol binding does not match derivation config")

    candidates = tuple(
        derive_candidate(context, config, paths)
        for context in sorted(panel, key=lambda item: item.candidate_index)
    )
    records = [dict(candidate.report_record) for candidate in candidates]
    return DerivationRunResult(
        schema_version=config.schema_version,
        preflight=config.preflight,
        candidates=candidates,
        summary=_summary(records),
        attrition_bias_probe=_descriptive_attrition(records),
        scope=config.scope,
        selection_policy=config.selection_policy,
        report_metadata=config.report_metadata,
    )
