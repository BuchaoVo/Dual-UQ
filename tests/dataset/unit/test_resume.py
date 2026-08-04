from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dual_uq.core.atomic_io import atomic_write_json
from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.pipeline.resume import (
    evaluate_resume,
    is_terminal_status,
    validate_status_transition,
    verify_declared_outputs,
)
from dual_uq.dataset.pipeline.status import (
    STAGE_MANIFEST_SCHEMA_VERSION,
    LifecycleStatus,
    OutputDeclaration,
    SeedIdentity,
    StageManifest,
    ValidationRecord,
    read_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)

INPUT_DIGEST = "a" * 64
CONFIG_DIGEST = "b" * 64
SEED_DIGEST = "c" * 64


def _current_validation(*, passed: bool = True) -> ValidationRecord:
    return ValidationRecord(
        validation_pass=passed,
        error_codes=() if passed else ("scientific_validator_failed",),
        warning_codes=("review_warning",) if passed else (),
        details={"validator": "fixture"},
    )


def _declaration(
    stage_dir: Path,
    *,
    logical_name: str = "result",
    relative_path: str = "outputs/result.txt",
    content: bytes | None = b"valid output",
    required: bool = True,
    recorded_hash: str | None = None,
) -> OutputDeclaration:
    path = stage_dir / relative_path
    if content is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = recorded_hash
    if digest is None and content is not None:
        digest = sha256_file(path)
    return OutputDeclaration(
        logical_name=logical_name,
        relative_path=relative_path,
        sha256=digest,
        required=required,
    )


def _manifest(
    outputs: tuple[OutputDeclaration, ...],
    *,
    status: LifecycleStatus = LifecycleStatus.COMPLETE,
    input_digest: str = INPUT_DIGEST,
    config_digest: str = CONFIG_DIGEST,
) -> StageManifest:
    return StageManifest(
        pipeline_name="dataset_a" + "_scale",
        pipeline_version="protocol_v1",
        schema_version=STAGE_MANIFEST_SCHEMA_VERSION,
        run_id="run-001",
        protein_id="index103",
        tier=2,
        stage="P3_cross_score_generated_candidates",
        status=status,
        input_digest=input_digest,
        config_digest=config_digest,
        seed_identity=SeedIdentity(seed_digest=SEED_DIGEST, seed_int=12345),
        declared_outputs=outputs,
        validation_pass=status in {
            LifecycleStatus.COMPLETE,
            LifecycleStatus.SKIPPED_VALIDATED,
        },
        failure_code=(
            "fixture_failure"
            if status
            in {LifecycleStatus.FAILED_RUNTIME, LifecycleStatus.FAILED_VALIDATION}
            else None
        ),
        failure_message=None,
        started_at="2026-08-01T00:00:00+00:00",
        finished_at="2026-08-01T00:01:00+00:00",
    )


def _write_history(stage_dir: Path, manifest: StageManifest) -> Path:
    path = stage_dir / "stage_manifest.json"
    write_stage_manifest(path, manifest)
    return path


def _resume(stage_dir: Path, **overrides: object):
    arguments: dict[str, object] = {
        "stage_dir": stage_dir,
        "current_input_digest": INPUT_DIGEST,
        "current_config_digest": CONFIG_DIGEST,
        "current_validation": _current_validation(),
    }
    arguments.update(overrides)
    return evaluate_resume(**arguments)  # type: ignore[arg-type]


def test_complete_stage_with_matching_identity_outputs_and_validation_can_skip(
    tmp_path: Path,
) -> None:
    output = _declaration(tmp_path)
    manifest_path = _write_history(tmp_path, _manifest((output,)))
    historical_bytes = manifest_path.read_bytes()

    decision = _resume(tmp_path)

    assert decision.can_skip is True
    assert decision.target_status is LifecycleStatus.SKIPPED_VALIDATED
    assert decision.reason_code == "validated_existing_stage"
    assert decision.details["warning_codes"] == ["review_warning"]
    assert manifest_path.read_bytes() == historical_bytes


def test_existing_output_alone_cannot_trigger_skip(tmp_path: Path) -> None:
    _declaration(tmp_path)
    decision = _resume(tmp_path)
    assert decision.can_skip is False
    assert decision.reason_code == "manifest_missing"


def test_invalid_historical_manifest_is_structured_resume_failure(
    tmp_path: Path,
) -> None:
    output = _declaration(tmp_path)
    payload = _manifest((output,)).as_dict()
    del payload["declared_outputs"][0]["sha256"]
    atomic_write_json(tmp_path / "stage_manifest.json", payload)

    decision = _resume(tmp_path)

    assert decision.can_skip is False
    assert decision.target_status is LifecycleStatus.FAILED_VALIDATION
    assert decision.reason_code == "manifest_invalid"


def test_historical_validation_true_is_not_enough_when_current_validation_fails(
    tmp_path: Path,
) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))

    decision = _resume(tmp_path, current_validation=_current_validation(passed=False))

    assert decision.can_skip is False
    assert decision.target_status is LifecycleStatus.FAILED_VALIDATION
    assert decision.reason_code == "current_validation_failed"


def test_missing_required_output_prevents_skip(tmp_path: Path) -> None:
    output = _declaration(tmp_path, content=None, recorded_hash="d" * 64)
    _write_history(tmp_path, _manifest((output,)))
    decision = _resume(tmp_path)
    assert decision.can_skip is False
    assert decision.reason_code == "required_output_missing"


def test_output_hash_mismatch_prevents_skip(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))
    (tmp_path / output.relative_path).write_bytes(b"corrupted")
    decision = _resume(tmp_path)
    assert decision.can_skip is False
    assert decision.reason_code == "output_hash_mismatch"


def test_input_digest_drift_blocks_current_run(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))
    decision = _resume(tmp_path, current_input_digest="d" * 64)
    assert decision.can_skip is False
    assert decision.target_status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert decision.reason_code == "input_digest_mismatch"


def test_config_digest_drift_blocks_current_run(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))
    decision = _resume(tmp_path, current_config_digest="e" * 64)
    assert decision.target_status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert decision.reason_code == "config_digest_mismatch"


def test_input_and_config_drift_are_structured(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))
    decision = _resume(
        tmp_path,
        current_input_digest="d" * 64,
        current_config_digest="e" * 64,
    )
    assert decision.target_status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert decision.reason_code == "input_and_config_digest_mismatch"
    assert decision.details["input_drift"] is True
    assert decision.details["config_drift"] is True


def test_drift_takes_precedence_over_output_reuse(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,)))
    (tmp_path / output.relative_path).write_bytes(b"also corrupted")
    decision = _resume(tmp_path, current_input_digest="d" * 64)
    assert decision.reason_code == "input_digest_mismatch"


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (LifecycleStatus.PLANNED, "incomplete_previous_execution"),
        (LifecycleStatus.RUNNING, "incomplete_previous_execution"),
        (LifecycleStatus.FAILED_RUNTIME, "previous_status_not_reusable"),
        (LifecycleStatus.FAILED_VALIDATION, "previous_status_not_reusable"),
    ],
)
def test_previous_nonreusable_stage_cannot_skip(
    tmp_path: Path, status: LifecycleStatus, reason: str
) -> None:
    output = _declaration(tmp_path)
    _write_history(tmp_path, _manifest((output,), status=status))
    decision = _resume(tmp_path)
    assert decision.can_skip is False
    assert decision.reason_code == reason


def test_previous_skipped_validated_stage_is_revalidated(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    _write_history(
        tmp_path,
        _manifest((output,), status=LifecycleStatus.SKIPPED_VALIDATED),
    )
    assert _resume(tmp_path).can_skip is True


def test_optional_missing_output_without_hash_is_allowed(tmp_path: Path) -> None:
    optional = _declaration(tmp_path, content=None, required=False)
    _write_history(tmp_path, _manifest((optional,)))
    assert _resume(tmp_path).can_skip is True


def test_declared_missing_optional_output_with_hash_is_not_allowed(
    tmp_path: Path,
) -> None:
    optional = _declaration(
        tmp_path,
        content=None,
        required=False,
        recorded_hash="d" * 64,
    )
    _write_history(tmp_path, _manifest((optional,)))
    assert _resume(tmp_path).reason_code == "optional_output_missing"


def test_declared_existing_optional_output_must_match_hash(tmp_path: Path) -> None:
    optional = _declaration(tmp_path, required=False, recorded_hash="d" * 64)
    _write_history(tmp_path, _manifest((optional,)))
    assert _resume(tmp_path).reason_code == "output_hash_mismatch"


def test_existing_optional_output_without_hash_is_not_reusable(tmp_path: Path) -> None:
    optional = _declaration(tmp_path, required=False)
    optional = OutputDeclaration(
        optional.logical_name,
        optional.relative_path,
        None,
        False,
    )
    _write_history(tmp_path, _manifest((optional,)))
    assert _resume(tmp_path).reason_code == "output_hash_missing"


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute.txt", "../outside.txt", "outputs/../../outside.txt", "C:\\outside.txt"],
)
def test_unsafe_output_path_is_rejected(relative_path: str) -> None:
    with pytest.raises(ValueError, match="relative_path"):
        OutputDeclaration(
            logical_name="unsafe",
            relative_path=relative_path,
            sha256="d" * 64,
            required=True,
        )


def test_symlink_output_is_not_validated_for_resume(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"valid output")
    link = tmp_path / "outputs/result.txt"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    output = OutputDeclaration("result", "outputs/result.txt", sha256_file(target), True)
    _write_history(tmp_path, _manifest((output,)))
    assert _resume(tmp_path).reason_code == "output_symlink_rejected"


def test_symlink_parent_directory_is_not_validated_for_resume(tmp_path: Path) -> None:
    real_outputs = tmp_path / "real_outputs"
    real_outputs.mkdir()
    target = real_outputs / "result.txt"
    target.write_bytes(b"valid output")
    try:
        (tmp_path / "outputs").symlink_to(real_outputs, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    output = OutputDeclaration("result", "outputs/result.txt", sha256_file(target), True)
    _write_history(tmp_path, _manifest((output,)))
    assert _resume(tmp_path).reason_code == "output_symlink_rejected"


def test_directory_output_is_rejected_as_non_regular_file(tmp_path: Path) -> None:
    directory = tmp_path / "outputs/result"
    directory.mkdir(parents=True)
    output = OutputDeclaration("result", "outputs/result", "d" * 64, True)
    _write_history(tmp_path, _manifest((output,)))
    assert _resume(tmp_path).reason_code == "output_not_regular_file"


def test_lifecycle_status_vocabulary_is_fixed() -> None:
    assert {status.value for status in LifecycleStatus} == {
        "planned",
        "running",
        "complete",
        "skipped_validated",
        "not_run_by_tier",
        "failed_validation",
        "failed_runtime",
        "blocked_upstream",
        "blocked_input_drift",
        "blocked_insufficient_sites",
    }


def test_terminal_status_classification() -> None:
    assert is_terminal_status(LifecycleStatus.PLANNED) is False
    assert is_terminal_status(LifecycleStatus.RUNNING) is False
    for status in set(LifecycleStatus) - {
        LifecycleStatus.PLANNED,
        LifecycleStatus.RUNNING,
    }:
        assert is_terminal_status(status) is True


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (LifecycleStatus.PLANNED, LifecycleStatus.RUNNING),
        (LifecycleStatus.RUNNING, LifecycleStatus.COMPLETE),
        (LifecycleStatus.RUNNING, LifecycleStatus.FAILED_RUNTIME),
        (LifecycleStatus.RUNNING, LifecycleStatus.FAILED_VALIDATION),
        (LifecycleStatus.PLANNED, LifecycleStatus.BLOCKED_UPSTREAM),
        (LifecycleStatus.PLANNED, LifecycleStatus.BLOCKED_INPUT_DRIFT),
        (LifecycleStatus.PLANNED, LifecycleStatus.NOT_RUN_BY_TIER),
        (LifecycleStatus.PLANNED, LifecycleStatus.BLOCKED_INSUFFICIENT_SITES),
        (LifecycleStatus.COMPLETE, LifecycleStatus.SKIPPED_VALIDATED),
    ],
)
def test_valid_status_transitions(
    source: LifecycleStatus, target: LifecycleStatus
) -> None:
    validate_status_transition(source, target)


def test_invalid_status_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        validate_status_transition(LifecycleStatus.PLANNED, LifecycleStatus.COMPLETE)


def test_stage_manifest_round_trip_and_atomic_json_contract(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    manifest = _manifest((output,))
    path = _write_history(tmp_path, manifest)

    assert read_stage_manifest(path) == manifest
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r" not in payload
    assert json.loads(payload.decode("utf-8"))["protein_id"] == "index103"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_stage_manifest_output_order_is_deterministic(tmp_path: Path) -> None:
    first = _declaration(
        tmp_path, logical_name="zeta", relative_path="outputs/z.txt", content=b"z"
    )
    second = _declaration(
        tmp_path, logical_name="alpha", relative_path="outputs/a.txt", content=b"a"
    )
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    write_stage_manifest(left, _manifest((first, second)))
    write_stage_manifest(right, _manifest((second, first)))
    assert left.read_bytes() == right.read_bytes()


def test_duplicate_output_logical_names_are_rejected(tmp_path: Path) -> None:
    first = _declaration(tmp_path, logical_name="same", relative_path="outputs/a")
    second = _declaration(tmp_path, logical_name="same", relative_path="outputs/b")
    with pytest.raises(ValueError, match="logical_name"):
        _manifest((first, second))


def test_duplicate_output_paths_are_rejected(tmp_path: Path) -> None:
    first = _declaration(tmp_path, logical_name="a", relative_path="outputs/same")
    second = _declaration(tmp_path, logical_name="b", relative_path="outputs/same")
    with pytest.raises(ValueError, match="relative_path"):
        _manifest((first, second))


def test_invalid_sha_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        OutputDeclaration("result", "outputs/result", "not-a-digest", True)


def test_required_flag_must_be_an_actual_boolean() -> None:
    with pytest.raises(TypeError, match="required"):
        OutputDeclaration("result", "outputs/result", "d" * 64, 1)  # type: ignore[arg-type]


def test_stage_identity_digest_must_be_sha256() -> None:
    with pytest.raises(ValueError, match="input_digest"):
        _manifest((), input_digest="not-a-digest")


def test_stage_manifest_validator_returns_structured_failure() -> None:
    result = validate_stage_manifest({"pipeline_name": "dataset_a" + "_scale"})
    assert result.validation_pass is False
    assert result.error_codes == ("stage_manifest_invalid",)


@pytest.mark.parametrize("nested_field", ["seed_identity", "declared_outputs"])
def test_stage_manifest_rejects_unknown_nested_fields(
    tmp_path: Path, nested_field: str
) -> None:
    output = _declaration(tmp_path)
    payload = _manifest((output,)).as_dict()
    if nested_field == "seed_identity":
        payload[nested_field]["unexpected"] = "silently-dropped"  # type: ignore[index]
    else:
        payload[nested_field][0]["unexpected"] = "silently-dropped"  # type: ignore[index]

    result = validate_stage_manifest(payload)

    assert result.validation_pass is False
    assert result.error_codes == ("stage_manifest_invalid",)


def test_validation_record_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ValidationRecord(validation_pass=True, schema_version="future-version")


@pytest.mark.parametrize(
    "status",
    [LifecycleStatus.COMPLETE, LifecycleStatus.SKIPPED_VALIDATED],
)
def test_successful_manifest_rejects_failure_metadata(status: LifecycleStatus) -> None:
    manifest = _manifest((), status=status)
    with pytest.raises(ValueError, match="failure"):
        replace(manifest, failure_code="contradicts_success")
    with pytest.raises(ValueError, match="failure"):
        replace(manifest, failure_message="contradicts success")


@pytest.mark.parametrize(
    "status",
    [
        LifecycleStatus.FAILED_VALIDATION,
        LifecycleStatus.FAILED_RUNTIME,
        LifecycleStatus.BLOCKED_UPSTREAM,
        LifecycleStatus.BLOCKED_INPUT_DRIFT,
        LifecycleStatus.BLOCKED_INSUFFICIENT_SITES,
    ],
)
def test_failure_and_blocked_manifests_require_failure_code(
    status: LifecycleStatus,
) -> None:
    if status in {LifecycleStatus.FAILED_VALIDATION, LifecycleStatus.FAILED_RUNTIME}:
        manifest = _manifest((), status=status)
        with pytest.raises(ValueError, match="failure_code"):
            replace(manifest, failure_code=None)
    else:
        manifest = _manifest((), status=LifecycleStatus.PLANNED)
        with pytest.raises(ValueError, match="failure_code"):
            replace(manifest, status=status)


def test_validation_false_requires_error_code() -> None:
    with pytest.raises(ValueError, match="error_codes"):
        ValidationRecord(validation_pass=False)


def test_atomic_write_json_creates_parseable_lf_file(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    atomic_write_json(path, {"name": "蛋白质", "validation_pass": True})
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "蛋白质"
    assert path.read_bytes().endswith(b"\n")
    assert b"\r" not in path.read_bytes()
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_verify_declared_outputs_ignores_extra_files(tmp_path: Path) -> None:
    output = _declaration(tmp_path)
    (tmp_path / "outputs/debug.tmp").write_text("extra", encoding="utf-8")
    result = verify_declared_outputs(tmp_path, (output,))
    assert result.validation_pass is True
