from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dual_uq.core.hashing import sha256_file

from .status import (
    LifecycleStatus,
    OutputDeclaration,
    ValidationRecord,
    read_stage_manifest,
    validate_sha256_digest,
)

TERMINAL_STATUSES = frozenset(
    {
        LifecycleStatus.COMPLETE,
        LifecycleStatus.SKIPPED_VALIDATED,
        LifecycleStatus.NOT_RUN_BY_TIER,
        LifecycleStatus.FAILED_VALIDATION,
        LifecycleStatus.FAILED_RUNTIME,
        LifecycleStatus.BLOCKED_UPSTREAM,
        LifecycleStatus.BLOCKED_INPUT_DRIFT,
        LifecycleStatus.BLOCKED_INSUFFICIENT_SITES,
    }
)
SUCCESSFUL_UPSTREAM_STATUSES = frozenset(
    {LifecycleStatus.COMPLETE, LifecycleStatus.SKIPPED_VALIDATED}
)

_ALLOWED_TRANSITIONS = {
    LifecycleStatus.PLANNED: frozenset(
        {
            LifecycleStatus.RUNNING,
            LifecycleStatus.NOT_RUN_BY_TIER,
            LifecycleStatus.BLOCKED_UPSTREAM,
            LifecycleStatus.BLOCKED_INPUT_DRIFT,
            LifecycleStatus.BLOCKED_INSUFFICIENT_SITES,
        }
    ),
    LifecycleStatus.RUNNING: frozenset(
        {
            LifecycleStatus.COMPLETE,
            LifecycleStatus.FAILED_VALIDATION,
            LifecycleStatus.FAILED_RUNTIME,
        }
    ),
    LifecycleStatus.COMPLETE: frozenset({LifecycleStatus.SKIPPED_VALIDATED}),
    LifecycleStatus.SKIPPED_VALIDATED: frozenset(
        {LifecycleStatus.SKIPPED_VALIDATED}
    ),
}


@dataclass(frozen=True)
class ResumeDecision:
    can_skip: bool
    target_status: LifecycleStatus
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpstreamBlocker:
    stage: str
    status: LifecycleStatus


@dataclass(frozen=True)
class UpstreamDecision:
    ready: bool
    target_status: LifecycleStatus | None
    reason_code: str
    blockers: tuple[UpstreamBlocker, ...]


def _status(value: LifecycleStatus | str) -> LifecycleStatus:
    try:
        return LifecycleStatus(value)
    except ValueError as exc:
        raise ValueError(f"Invalid lifecycle status: {value}") from exc


def is_terminal_status(status: LifecycleStatus | str) -> bool:
    return _status(status) in TERMINAL_STATUSES


def validate_status_transition(
    from_status: LifecycleStatus | str,
    to_status: LifecycleStatus | str,
) -> None:
    source = _status(from_status)
    target = _status(to_status)
    if target not in _ALLOWED_TRANSITIONS.get(source, frozenset()):
        raise ValueError(
            f"Invalid lifecycle transition: {source.value} -> {target.value}"
        )


def _record_integrity_issue(
    issues: list[dict[str, Any]],
    error_codes: list[str],
    *,
    code: str,
    output: OutputDeclaration,
    path: Path,
) -> None:
    issues.append(
        {
            "code": code,
            "logical_name": output.logical_name,
            "relative_path": output.relative_path,
            "resolved_path": str(path),
        }
    )
    if code not in error_codes:
        error_codes.append(code)


def verify_declared_outputs(
    stage_dir: Path,
    declared_outputs: Sequence[OutputDeclaration],
) -> ValidationRecord:
    """Verify declared files and raw SHA values without modifying history."""
    stage_root = stage_dir.resolve()
    issues: list[dict[str, Any]] = []
    error_codes: list[str] = []
    for output in sorted(
        declared_outputs, key=lambda item: (item.logical_name, item.relative_path)
    ):
        path = stage_dir / output.relative_path
        cursor = stage_dir
        has_symlink_component = False
        for component in Path(output.relative_path).parts:
            cursor /= component
            if cursor.is_symlink():
                has_symlink_component = True
                break
        if has_symlink_component:
            _record_integrity_issue(
                issues,
                error_codes,
                code="output_symlink_rejected",
                output=output,
                path=path,
            )
            continue
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(stage_root)
        except ValueError:
            _record_integrity_issue(
                issues,
                error_codes,
                code="output_path_escape",
                output=output,
                path=path,
            )
            continue
        if not path.exists():
            if output.required:
                code = "required_output_missing"
            elif output.sha256 is not None:
                code = "optional_output_missing"
            else:
                continue
            _record_integrity_issue(
                issues, error_codes, code=code, output=output, path=path
            )
            continue
        if not path.is_file():
            _record_integrity_issue(
                issues,
                error_codes,
                code="output_not_regular_file",
                output=output,
                path=path,
            )
            continue
        if output.sha256 is None:
            _record_integrity_issue(
                issues,
                error_codes,
                code="output_hash_missing",
                output=output,
                path=path,
            )
            continue
        observed = sha256_file(path)
        if observed != output.sha256:
            _record_integrity_issue(
                issues,
                error_codes,
                code="output_hash_mismatch",
                output=output,
                path=path,
            )
            issues[-1]["expected_sha256"] = output.sha256
            issues[-1]["observed_sha256"] = observed

    if error_codes:
        return ValidationRecord(
            validation_pass=False,
            error_codes=tuple(error_codes),
            details={"issues": issues},
        )
    return ValidationRecord(
        validation_pass=True,
        details={"checked_output_count": len(declared_outputs)},
    )


def _cannot_skip(
    target_status: LifecycleStatus,
    reason_code: str,
    **details: Any,
) -> ResumeDecision:
    return ResumeDecision(False, target_status, reason_code, details)


def evaluate_resume(
    *,
    stage_dir: Path,
    current_input_digest: str,
    current_config_digest: str,
    current_validation: ValidationRecord,
) -> ResumeDecision:
    """Read-only decision for validated reuse of one historical stage."""
    validate_sha256_digest(current_input_digest, "current_input_digest")
    validate_sha256_digest(current_config_digest, "current_config_digest")
    if not isinstance(current_validation, ValidationRecord):
        raise TypeError("current_validation must be a ValidationRecord.")

    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        return _cannot_skip(LifecycleStatus.PLANNED, "manifest_missing")
    try:
        manifest = read_stage_manifest(manifest_path)
    except (TypeError, ValueError):
        return _cannot_skip(LifecycleStatus.FAILED_VALIDATION, "manifest_invalid")

    if manifest.status not in SUCCESSFUL_UPSTREAM_STATUSES:
        if manifest.status in {LifecycleStatus.PLANNED, LifecycleStatus.RUNNING}:
            return _cannot_skip(
                LifecycleStatus.PLANNED,
                "incomplete_previous_execution",
                previous_status=manifest.status.value,
            )
        return _cannot_skip(
            manifest.status,
            "previous_status_not_reusable",
            previous_status=manifest.status.value,
        )

    input_drift = manifest.input_digest != current_input_digest
    config_drift = manifest.config_digest != current_config_digest
    if input_drift or config_drift:
        reason_code = {
            (True, False): "input_digest_mismatch",
            (False, True): "config_digest_mismatch",
            (True, True): "input_and_config_digest_mismatch",
        }[(input_drift, config_drift)]
        return _cannot_skip(
            LifecycleStatus.BLOCKED_INPUT_DRIFT,
            reason_code,
            input_drift=input_drift,
            config_drift=config_drift,
            historical_input_digest=manifest.input_digest,
            current_input_digest=current_input_digest,
            historical_config_digest=manifest.config_digest,
            current_config_digest=current_config_digest,
        )

    integrity = verify_declared_outputs(stage_dir, manifest.declared_outputs)
    if not integrity.validation_pass:
        return _cannot_skip(
            LifecycleStatus.FAILED_VALIDATION,
            integrity.error_codes[0],
            output_integrity=integrity.as_dict(),
        )
    if not current_validation.validation_pass:
        return _cannot_skip(
            LifecycleStatus.FAILED_VALIDATION,
            "current_validation_failed",
            current_validation=current_validation.as_dict(),
        )
    return ResumeDecision(
        can_skip=True,
        target_status=LifecycleStatus.SKIPPED_VALIDATED,
        reason_code="validated_existing_stage",
        details={
            "warning_codes": list(current_validation.warning_codes),
            "historical_status": manifest.status.value,
        },
    )


def evaluate_upstream_dependencies(
    required_upstreams: Mapping[str, LifecycleStatus | str],
) -> UpstreamDecision:
    blockers: list[UpstreamBlocker] = []
    for stage, value in required_upstreams.items():
        if type(stage) is not str or not stage or stage != stage.strip():
            raise ValueError("Upstream stage must be a non-empty canonical string.")
        status = _status(value)
        if status not in SUCCESSFUL_UPSTREAM_STATUSES:
            blockers.append(UpstreamBlocker(stage=stage, status=status))
    ordered = tuple(sorted(blockers, key=lambda blocker: blocker.stage))
    if ordered:
        return UpstreamDecision(
            ready=False,
            target_status=LifecycleStatus.BLOCKED_UPSTREAM,
            reason_code="required_upstream_not_ready",
            blockers=ordered,
        )
    return UpstreamDecision(
        ready=True,
        target_status=None,
        reason_code="upstream_ready",
        blockers=(),
    )
