"""Thin manifest-record adapters for the shared Dataset task executor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.models import LogicalAssetRef
from dual_uq.dataset.paths import DatasetPaths
from dual_uq.dataset.pipeline import (
    DatasetTask,
    DatasetTaskError,
    TaskExecutionOutput,
    read_stage_manifest,
)

from .acquisition import run_plan
from .census import CENSUS_CONFIG, evaluate_protein
from .derivation import P1_STAGE_NAME, run_p1
from .resolution import P0_STAGE_NAME, run_p0

ManifestStageHandler = Callable[[DatasetTask], TaskExecutionOutput]
_STAGES = {
    "census",
    "resolution",
    "acquisition",
    "derivation",
    "validation",
    "release",
}


def _mapping(value: object, *, field: str, default: Mapping[str, Any]) -> dict[str, Any]:
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise DatasetTaskError("invalid_manifest_field", f"{field} must be a mapping")
    return dict(value)


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value or value != value.strip():
        raise DatasetTaskError("missing_manifest_field", f"{field} is required")
    return value


def _stage_outputs(
    project: ProjectPaths, stage_dir: Path
) -> tuple[LogicalAssetRef, ...]:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        return ()
    manifest = read_stage_manifest(manifest_path)
    outputs = []
    for declaration in manifest.declared_outputs:
        path = stage_dir / declaration.relative_path
        outputs.append(
            LogicalAssetRef(
                asset_type=declaration.logical_name,
                logical_path=project.logical_ref(path),
                sha256=declaration.sha256,
                provenance=f"{manifest.stage}:{manifest.run_id}",
            )
        )
    return tuple(outputs)


def _census_handler(
    project: ProjectPaths, dataset_paths: DatasetPaths
) -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        config = _mapping(
            task.record.get("stage_config"), field="stage_config", default=CENSUS_CONFIG
        )
        result = evaluate_protein(
            dict(task.record),
            project_root=project.repository_root,
            census_stage_root=dataset_paths.runs / "census",
            config=config,
            pipeline_version=task.stage_version,
            run_id=str(task.record.get("run_id") or "manifest-census"),
        )
        return TaskExecutionOutput(
            metrics=asdict(result), scientific_disposition=result.outcome
        )

    return handler


def _resolution_handler(
    project: ProjectPaths, dataset_paths: DatasetPaths
) -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        stage_dir = dataset_paths.runs / "records" / task.record_id / P0_STAGE_NAME
        result = run_p0(
            manifest_row=task.record,
            project_root=project.repository_root,
            stage_dir=stage_dir,
            config=_mapping(
                task.record.get("stage_config"),
                field="stage_config",
                default={
                    "mapping_policy": "explicit_auth_label",
                    "fragment_policy": "full_coverage",
                },
            ),
            pipeline_version=task.stage_version,
            run_id=str(task.record.get("run_id") or "manifest-resolution"),
        )
        return TaskExecutionOutput(
            outputs=_stage_outputs(project, stage_dir),
            metrics={
                "validation": result.validation.as_dict(),
                "failure_code": result.failure_code,
                "selected_model_entity_id": result.selected_model_entity_id,
            },
            scientific_disposition=result.status.value,
        )

    return handler


def _acquisition_handler(project: ProjectPaths) -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        authorization = _text(task.record, "acquisition_authorization_digest")
        if len(authorization) != 64:
            raise DatasetTaskError(
                "invalid_acquisition_authorization",
                "acquisition_authorization_digest must be SHA-256",
            )
        result = run_plan(
            {"records": [dict(task.record)], "input_bindings": {"authorization": authorization}},
            project.repository_root,
        )
        candidate = result["candidate_completeness"][0]
        outputs = []
        for record in result["records"]:
            if record.get("status") not in {"reused_valid", "downloaded_new"}:
                continue
            logical_path = str(record["local_path"])
            path = project.resolve_logical(logical_path)
            outputs.append(
                LogicalAssetRef(
                    asset_type=str(record["asset_type"]),
                    logical_path=logical_path,
                    sha256=str(record.get("SHA256") or sha256_file(path)),
                    provenance=str(record["source_url"]),
                )
            )
        return TaskExecutionOutput(
            outputs=tuple(outputs),
            metrics={"summary": result["summary"]},
            scientific_disposition=str(candidate["completeness"]),
        )

    return handler


def _derivation_handler(
    project: ProjectPaths, dataset_paths: DatasetPaths
) -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        record_root = dataset_paths.runs / "records" / task.record_id
        p0_stage_dir = record_root / P0_STAGE_NAME
        stage_dir = record_root / P1_STAGE_NAME
        result = run_p1(
            project_root=project.repository_root,
            p0_stage_dir=p0_stage_dir,
            stage_dir=stage_dir,
            config=_mapping(
                task.record.get("stage_config"),
                field="stage_config",
                default={
                    "pairing_policy": "frozen_mapping_auth_only",
                    "output_chain": "A",
                },
            ),
            pipeline_version=task.stage_version,
            run_id=str(task.record.get("run_id") or "manifest-derivation"),
        )
        return TaskExecutionOutput(
            outputs=_stage_outputs(project, stage_dir),
            metrics={
                "validation": result.validation.as_dict(),
                "failure_code": result.failure_code,
                "residue_count": result.residue_count,
            },
            scientific_disposition=result.status.value,
        )

    return handler


def _validation_handler(project: ProjectPaths) -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        resources = task.record.get("resources")
        if not isinstance(resources, (list, tuple)):
            raise DatasetTaskError(
                "missing_manifest_field", "resources must be a list"
            )
        for position, resource in enumerate(resources):
            if not isinstance(resource, Mapping):
                raise DatasetTaskError(
                    "invalid_manifest_field", "resource entries must be mappings"
                )
            logical_path = _text(resource, "logical_path")
            expected = _text(resource, "sha256")
            path = project.resolve_logical(logical_path)
            if not path.is_file():
                raise DatasetTaskError(
                    "resource_missing", f"resource {position} is missing"
                )
            if sha256_file(path) != expected:
                raise DatasetTaskError(
                    "resource_hash_mismatch", f"resource {position} hash mismatch"
                )
        disposition = task.record.get("disposition")
        return TaskExecutionOutput(
            metrics={"checked_resource_count": len(resources)},
            scientific_disposition=(
                str(disposition) if isinstance(disposition, str) else None
            ),
        )

    return handler


def _release_handler() -> ManifestStageHandler:
    def handler(task: DatasetTask) -> TaskExecutionOutput:
        release_id = _text(task.record, "release_id")
        disposition = task.record.get("disposition")
        return TaskExecutionOutput(
            metrics={"release_id": release_id},
            scientific_disposition=(
                str(disposition) if isinstance(disposition, str) else None
            ),
        )

    return handler


def build_manifest_stage_handler(
    stage: str,
    *,
    project: ProjectPaths,
    dataset_paths: DatasetPaths,
) -> ManifestStageHandler:
    """Bind one canonical stage to its existing reusable record primitive."""
    if stage not in _STAGES:
        raise ValueError(f"unsupported Dataset stage: {stage}")
    if stage == "census":
        return _census_handler(project, dataset_paths)
    if stage == "resolution":
        return _resolution_handler(project, dataset_paths)
    if stage == "acquisition":
        return _acquisition_handler(project)
    if stage == "derivation":
        return _derivation_handler(project, dataset_paths)
    if stage == "validation":
        return _validation_handler(project)
    return _release_handler()
