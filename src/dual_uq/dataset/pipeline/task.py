"""Immutable record-level task and execution-result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dual_uq.core.hashing import sha256_canonical
from dual_uq.dataset.models import FailureRecord, LogicalAssetRef

from .status import LifecycleStatus


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _digest(value: object, field_name: str) -> str:
    checked = _text(value, field_name)
    if len(checked) != 64 or any(character not in "0123456789abcdef" for character in checked):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return checked


def _immutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_immutable_value(item) for item in value)
    return value


def _immutable(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("task record and metrics must be mappings")
    return _immutable_value(value)


@dataclass(frozen=True)
class DatasetTask:
    record_id: str
    stage: str
    stage_version: str
    record: Mapping[str, Any]
    input_digest: str
    config_digest: str
    dependency_digest: str
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("record_id", "stage", "stage_version"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        for field_name in ("input_digest", "config_digest", "dependency_digest"):
            object.__setattr__(self, field_name, _digest(getattr(self, field_name), field_name))
        if type(self.seed) is not int or self.seed <= 0:
            raise ValueError("seed must be a positive integer")
        object.__setattr__(self, "record", _immutable(self.record))

    @property
    def task_key(self) -> str:
        return sha256_canonical({"record_id": self.record_id, "stage": self.stage})

    @property
    def task_digest(self) -> str:
        return sha256_canonical(
            {
                "record_id": self.record_id,
                "stage": self.stage,
                "stage_version": self.stage_version,
                "input_digest": self.input_digest,
                "config_digest": self.config_digest,
                "dependency_digest": self.dependency_digest,
                "seed": self.seed,
            }
        )

    def shard_index(self, num_shards: int) -> int:
        if type(num_shards) is not int or num_shards <= 0:
            raise ValueError("num_shards must be a positive integer")
        return int(self.task_key[:16], 16) % num_shards


@dataclass(frozen=True)
class TaskExecutionOutput:
    outputs: tuple[LogicalAssetRef, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    scientific_disposition: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(output, LogicalAssetRef) for output in self.outputs):
            raise TypeError("outputs must contain LogicalAssetRef values")
        object.__setattr__(self, "metrics", _immutable(self.metrics))
        if self.scientific_disposition is not None:
            object.__setattr__(
                self,
                "scientific_disposition",
                _text(self.scientific_disposition, "scientific_disposition"),
            )


@dataclass(frozen=True)
class TaskExecutionResult:
    record_id: str
    task_key: str
    task_digest: str
    stage: str
    stage_version: str
    input_digest: str
    config_digest: str
    dependency_digest: str
    execution_status: LifecycleStatus
    outputs: tuple[LogicalAssetRef, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    scientific_disposition: str | None = None
    failure: FailureRecord | None = None
    resume_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_status", LifecycleStatus(self.execution_status))
        object.__setattr__(self, "metrics", _immutable(self.metrics))

    @property
    def output_digest(self) -> str:
        return sha256_canonical(
            [
                {
                    "asset_type": output.asset_type,
                    "logical_path": output.logical_path,
                    "sha256": output.sha256,
                    "expected_model_identity": output.expected_model_identity,
                }
                for output in sorted(
                    self.outputs,
                    key=lambda item: (item.asset_type, item.logical_path),
                )
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_key": self.task_key,
            "task_digest": self.task_digest,
            "stage": self.stage,
            "stage_version": self.stage_version,
            "input_digest": self.input_digest,
            "config_digest": self.config_digest,
            "dependency_digest": self.dependency_digest,
            "execution_status": self.execution_status.value,
            "output_digest": self.output_digest,
            "outputs": [
                {
                    "asset_type": output.asset_type,
                    "logical_path": output.logical_path,
                    "sha256": output.sha256,
                    "provenance": output.provenance,
                    "expected_model_identity": output.expected_model_identity,
                }
                for output in self.outputs
            ],
            "metrics": dict(self.metrics),
            "scientific_disposition": self.scientific_disposition,
            "failure": (
                None
                if self.failure is None
                else {
                    "code": self.failure.code,
                    "message": self.failure.message,
                    "details": dict(self.failure.details),
                }
            ),
            "resume_reason": self.resume_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskExecutionResult:
        failure_value = value.get("failure")
        failure = (
            None
            if failure_value is None
            else FailureRecord(
                code=str(failure_value["code"]),
                message=str(failure_value["message"]),
                details=dict(failure_value.get("details", {})),
            )
        )
        return cls(
            record_id=str(value["record_id"]),
            task_key=str(value["task_key"]),
            task_digest=str(value["task_digest"]),
            stage=str(value["stage"]),
            stage_version=str(value["stage_version"]),
            input_digest=str(value["input_digest"]),
            config_digest=str(value["config_digest"]),
            dependency_digest=str(value["dependency_digest"]),
            execution_status=LifecycleStatus(str(value["execution_status"])),
            outputs=tuple(
                LogicalAssetRef(
                    asset_type=str(output["asset_type"]),
                    logical_path=str(output["logical_path"]),
                    sha256=output.get("sha256"),
                    provenance=str(output["provenance"]),
                    expected_model_identity=output.get("expected_model_identity"),
                )
                for output in value.get("outputs", [])
            ),
            metrics=dict(value.get("metrics", {})),
            scientific_disposition=value.get("scientific_disposition"),
            failure=failure,
            resume_reason=value.get("resume_reason"),
        )


class DatasetTaskError(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.failure = FailureRecord(code=code, message=message, details=details)
        super().__init__(message)
