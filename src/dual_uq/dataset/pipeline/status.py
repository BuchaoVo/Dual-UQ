from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from dual_uq.core.atomic_io import atomic_write_json
from dual_uq.core.hashing import MAX_TOOL_SEED, MIN_TOOL_SEED

STAGE_MANIFEST_SCHEMA_VERSION = "dataset_a_stage_manifest.v1"
VALIDATION_SCHEMA_VERSION = "dataset_a_validation.v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class LifecycleStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED_VALIDATED = "skipped_validated"
    NOT_RUN_BY_TIER = "not_run_by_tier"
    FAILED_VALIDATION = "failed_validation"
    FAILED_RUNTIME = "failed_runtime"
    BLOCKED_UPSTREAM = "blocked_upstream"
    BLOCKED_INPUT_DRIFT = "blocked_input_drift"
    BLOCKED_INSUFFICIENT_SITES = "blocked_insufficient_sites"


_FAILURE_STATUSES = frozenset(
    {
        LifecycleStatus.FAILED_VALIDATION,
        LifecycleStatus.FAILED_RUNTIME,
        LifecycleStatus.BLOCKED_UPSTREAM,
        LifecycleStatus.BLOCKED_INPUT_DRIFT,
        LifecycleStatus.BLOCKED_INSUFFICIENT_SITES,
    }
)


def _canonical_identity(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    if not value or value != value.strip() or "\0" in value:
        raise ValueError(
            f"{field_name} must be non-empty, canonical, and contain no NUL."
        )
    return value


def validate_sha256_digest(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a SHA-256 string.")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters.")
    return value


def _validate_relative_output_path(value: object) -> str:
    relative_path = _canonical_identity(value, "relative_path")
    if "\\" in relative_path:
        raise ValueError("relative_path must use portable POSIX separators.")
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    raw_parts = relative_path.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError("relative_path must be relative without path traversal.")
    return posix.as_posix()


def _validate_timestamp(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    checked = _canonical_identity(value, field_name)
    try:
        parsed = datetime.fromisoformat(checked.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return checked


def _codes(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple of strings.")
    checked = tuple(_canonical_identity(value, field_name) for value in values)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return checked


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], field_name: str
) -> None:
    missing = expected.difference(payload)
    extra = set(payload).difference(expected)
    if missing or extra:
        raise ValueError(
            f"{field_name} fields invalid; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


@dataclass(frozen=True)
class ValidationRecord:
    validation_pass: bool
    error_codes: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.validation_pass) is not bool:
            raise TypeError("validation_pass must be a boolean.")
        schema_version = _canonical_identity(self.schema_version, "schema_version")
        if schema_version != VALIDATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {schema_version}")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "error_codes", _codes(self.error_codes, "error_codes"))
        object.__setattr__(self, "warning_codes", _codes(self.warning_codes, "warning_codes"))
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dictionary.")
        object.__setattr__(self, "details", dict(self.details))
        if self.validation_pass and self.error_codes:
            raise ValueError("validation_pass=true requires empty error_codes.")
        if not self.validation_pass and not self.error_codes:
            raise ValueError("validation_pass=false requires at least one error_codes entry.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "validation_pass": self.validation_pass,
            "error_codes": list(self.error_codes),
            "warning_codes": list(self.warning_codes),
            "details": self.details,
        }


@dataclass(frozen=True)
class SeedIdentity:
    seed_digest: str
    seed_int: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed_digest",
            validate_sha256_digest(self.seed_digest, "seed_digest"),
        )
        if type(self.seed_int) is not int:
            raise TypeError("seed_int must be an integer.")
        if not MIN_TOOL_SEED <= self.seed_int <= MAX_TOOL_SEED:
            raise ValueError("seed_int is outside the shared tool-compatible range.")

    def as_dict(self) -> dict[str, Any]:
        return {"seed_digest": self.seed_digest, "seed_int": self.seed_int}


@dataclass(frozen=True)
class OutputDeclaration:
    logical_name: str
    relative_path: str
    sha256: str | None
    required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_name", _canonical_identity(self.logical_name, "logical_name")
        )
        object.__setattr__(
            self, "relative_path", _validate_relative_output_path(self.relative_path)
        )
        if type(self.required) is not bool:
            raise TypeError("required must be a boolean.")
        if self.sha256 is not None:
            object.__setattr__(
                self, "sha256", validate_sha256_digest(self.sha256, "sha256")
            )
        if self.required and self.sha256 is None:
            raise ValueError("A required output must declare sha256.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "required": self.required,
        }


@dataclass(frozen=True)
class StageManifest:
    pipeline_name: str
    pipeline_version: str
    schema_version: str
    run_id: str
    protein_id: str
    tier: int
    stage: str
    status: LifecycleStatus
    input_digest: str
    config_digest: str
    seed_identity: SeedIdentity | None
    declared_outputs: tuple[OutputDeclaration, ...]
    validation_pass: bool
    failure_code: str | None = None
    failure_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "pipeline_name",
            "pipeline_version",
            "schema_version",
            "run_id",
            "protein_id",
            "stage",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_identity(getattr(self, field_name), field_name),
            )
        if self.schema_version != STAGE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if type(self.tier) is not int or self.tier not in {1, 2}:
            raise ValueError("tier must be integer 1 or 2.")
        try:
            status = LifecycleStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"Invalid lifecycle status: {self.status}") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "input_digest",
            validate_sha256_digest(self.input_digest, "input_digest"),
        )
        object.__setattr__(
            self,
            "config_digest",
            validate_sha256_digest(self.config_digest, "config_digest"),
        )
        if self.seed_identity is not None and not isinstance(
            self.seed_identity, SeedIdentity
        ):
            raise TypeError("seed_identity must be SeedIdentity or None.")
        if not isinstance(self.declared_outputs, (list, tuple)):
            raise TypeError("declared_outputs must be a list or tuple.")
        outputs = tuple(self.declared_outputs)
        if any(not isinstance(output, OutputDeclaration) for output in outputs):
            raise TypeError("declared_outputs must contain OutputDeclaration values.")
        if len({output.logical_name for output in outputs}) != len(outputs):
            raise ValueError("declared_outputs contains duplicate logical_name values.")
        if len({output.relative_path for output in outputs}) != len(outputs):
            raise ValueError("declared_outputs contains duplicate relative_path values.")
        object.__setattr__(
            self,
            "declared_outputs",
            tuple(sorted(outputs, key=lambda output: (output.logical_name, output.relative_path))),
        )
        if type(self.validation_pass) is not bool:
            raise TypeError("validation_pass must be a boolean.")
        if status in {LifecycleStatus.COMPLETE, LifecycleStatus.SKIPPED_VALIDATED}:
            if not self.validation_pass:
                raise ValueError(f"status={status.value} requires validation_pass=true.")
        elif self.validation_pass:
            raise ValueError(f"status={status.value} cannot have validation_pass=true.")
        if self.failure_code is not None:
            object.__setattr__(
                self,
                "failure_code",
                _canonical_identity(self.failure_code, "failure_code"),
            )
        if self.failure_message is not None and type(self.failure_message) is not str:
            raise TypeError("failure_message must be a string or None.")
        if status in _FAILURE_STATUSES:
            if self.failure_code is None:
                raise ValueError(f"status={status.value} requires failure_code.")
        elif self.failure_code is not None or self.failure_message is not None:
            raise ValueError(f"status={status.value} cannot have failure metadata.")
        object.__setattr__(
            self, "started_at", _validate_timestamp(self.started_at, "started_at")
        )
        object.__setattr__(
            self, "finished_at", _validate_timestamp(self.finished_at, "finished_at")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "protein_id": self.protein_id,
            "tier": self.tier,
            "stage": self.stage,
            "status": self.status.value,
            "input_digest": self.input_digest,
            "config_digest": self.config_digest,
            "seed_identity": (
                None if self.seed_identity is None else self.seed_identity.as_dict()
            ),
            "declared_outputs": [output.as_dict() for output in self.declared_outputs],
            "validation_pass": self.validation_pass,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageManifest:
        required_fields = {
            "pipeline_name",
            "pipeline_version",
            "schema_version",
            "run_id",
            "protein_id",
            "tier",
            "stage",
            "status",
            "input_digest",
            "config_digest",
            "seed_identity",
            "declared_outputs",
            "validation_pass",
            "failure_code",
            "failure_message",
            "started_at",
            "finished_at",
        }
        if not isinstance(payload, Mapping):
            raise TypeError("Stage manifest must be a mapping.")
        missing = required_fields.difference(payload)
        extra = set(payload).difference(required_fields)
        if missing or extra:
            raise ValueError(
                f"Stage manifest fields invalid; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        seed_payload = payload["seed_identity"]
        seed = None
        if seed_payload is not None:
            if not isinstance(seed_payload, Mapping):
                raise TypeError("seed_identity must be a mapping or null.")
            _require_exact_fields(
                seed_payload, {"seed_digest", "seed_int"}, "seed_identity"
            )
            seed = SeedIdentity(
                seed_digest=seed_payload["seed_digest"],
                seed_int=seed_payload["seed_int"],
            )
        output_payloads = payload["declared_outputs"]
        if not isinstance(output_payloads, list):
            raise TypeError("declared_outputs must be a list.")
        outputs_list: list[OutputDeclaration] = []
        output_fields = {"logical_name", "relative_path", "sha256", "required"}
        for output in output_payloads:
            if not isinstance(output, Mapping):
                raise TypeError("declared_outputs entries must be mappings.")
            _require_exact_fields(output, output_fields, "declared_outputs entry")
            outputs_list.append(
                OutputDeclaration(
                    logical_name=output["logical_name"],
                    relative_path=output["relative_path"],
                    sha256=output["sha256"],
                    required=output["required"],
                )
            )
        outputs = tuple(outputs_list)
        return cls(
            pipeline_name=payload["pipeline_name"],
            pipeline_version=payload["pipeline_version"],
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            protein_id=payload["protein_id"],
            tier=payload["tier"],
            stage=payload["stage"],
            status=payload["status"],
            input_digest=payload["input_digest"],
            config_digest=payload["config_digest"],
            seed_identity=seed,
            declared_outputs=outputs,
            validation_pass=payload["validation_pass"],
            failure_code=payload["failure_code"],
            failure_message=payload["failure_message"],
            started_at=payload["started_at"],
            finished_at=payload["finished_at"],
        )


def validate_stage_manifest(payload: Mapping[str, Any]) -> ValidationRecord:
    try:
        StageManifest.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return ValidationRecord(
            validation_pass=False,
            error_codes=("stage_manifest_invalid",),
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
    return ValidationRecord(validation_pass=True)


def write_stage_manifest(path: Path, manifest: StageManifest) -> None:
    atomic_write_json(path, manifest.as_dict())


def read_stage_manifest(path: Path) -> StageManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid stage manifest file {path}: {exc}") from exc
    try:
        return StageManifest.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid stage manifest schema in {path}: {exc}") from exc
