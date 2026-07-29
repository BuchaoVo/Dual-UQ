from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAIR = "pair"
GEOMETRY = "geometry"
ROBUST = "robust"
SEGMENT_CONTEXT = "segment_context"
STAGES = (PAIR, GEOMETRY, ROBUST, SEGMENT_CONTEXT)

TERMINAL_STATUSES = {
    "complete",
    "skipped_preflight",
    "skipped_complete",
    "successful_no_segments",
    "unsupported_afdb_fragment",
    "failed_pair",
    "failed_geometry",
    "failed_robust",
    "failed_segment_context",
    "blocked_upstream",
}


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    status: str
    reason: str


@dataclass(frozen=True)
class OutputValidation:
    valid: bool
    status: str = "complete"
    checked_paths: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()
    invalid_paths: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StagePlan:
    screening_index: int
    pdb_id: str
    chain_id: str
    uniprot_id: str
    pair_name: str
    provisional_stratum: str | None
    preflight_status: str
    stage: str
    action: str
    reason: str
    seed: int
    command: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurrentStageState:
    screening_index: int
    stage: str
    attempt: int
    status: str
    interrupted: bool
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StatusRecord:
    run_id: str
    screening_index: int
    pdb_id: str
    chain_id: str
    uniprot_id: str
    pair_name: str
    provisional_stratum: str | None
    preflight_status: str
    stage: str
    attempt: int
    status: str
    reason: str
    started_at: str | None
    finished_at: str | None
    runtime_seconds: float | None
    seed: int
    command: tuple[str, ...]
    return_code: int | None
    validated_outputs: tuple[str, ...]
    error_type: str | None
    error_message: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value(candidate: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def classify_candidate_eligibility(
    preflight_status: str,
    *,
    include_warnings: bool = False,
) -> EligibilityDecision:
    status = str(preflight_status or "not_run")
    if status == "pass_full_length":
        return EligibilityDecision(True, "eligible", "pass_full_length")
    if status == "warn_construct_difference" and include_warnings:
        return EligibilityDecision(True, "eligible", "warning_explicitly_included")
    if status == "unsupported_afdb_fragment":
        return EligibilityDecision(
            False,
            "unsupported_afdb_fragment",
            "unsupported_afdb_fragment",
        )
    return EligibilityDecision(False, "skipped_preflight", "skipped_preflight")


def derive_stage_seed(base_seed: int, screening_index: int, stage: str) -> int:
    payload = f"{int(base_seed)}:{int(screening_index)}:{stage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def reduce_status_history(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], CurrentStageState]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for ordinal, source in enumerate(records):
        try:
            index = int(source["screening_index"])
            stage = str(source["stage"])
            attempt = int(source["attempt"])
            status = str(source["status"])
        except (KeyError, TypeError, ValueError):
            continue
        if stage not in STAGES or status not in TERMINAL_STATUSES | {
            "running",
            "pending",
        }:
            continue
        record = dict(source)
        record["_ordinal"] = ordinal
        record["attempt"] = attempt
        grouped.setdefault((index, stage), []).append(record)

    reduced: dict[tuple[int, str], CurrentStageState] = {}
    for key, history in grouped.items():
        ordered = sorted(history, key=lambda item: (item["attempt"], item["_ordinal"]))
        by_attempt: dict[int, list[dict[str, Any]]] = {}
        for record in ordered:
            by_attempt.setdefault(int(record["attempt"]), []).append(record)
        active_attempts = [
            attempt
            for attempt, attempt_records in by_attempt.items()
            if any(
                record["status"] in TERMINAL_STATUSES | {"running"}
                for record in attempt_records
            )
        ]
        latest_attempt = max(active_attempts) if active_attempts else max(by_attempt)
        attempt_records = by_attempt[latest_attempt]
        terminals = [
            record
            for record in attempt_records
            if record["status"] in TERMINAL_STATUSES
        ]
        latest = terminals[-1] if terminals else attempt_records[-1]
        status = (
            "interrupted"
            if not terminals and latest["status"] == "running"
            else latest["status"]
        )
        cleaned = tuple(
            {name: value for name, value in item.items() if name != "_ordinal"}
            for item in ordered
        )
        reduced[key] = CurrentStageState(
            screening_index=key[0],
            stage=key[1],
            attempt=int(latest["attempt"]),
            status=status,
            interrupted=status == "interrupted",
            history=cleaned,
        )
    return reduced


def _expected_outputs(pair_name: str, stage: str) -> tuple[str, ...]:
    base = f"data/processed/pairs/{pair_name}"
    return {
        PAIR: (f"{base}/pair_qc.json", f"{base}/residue_mapping.parquet"),
        GEOMETRY: (
            f"{base}/pair_geometry_qc.json",
            f"{base}/residue_geometry.parquet",
            f"{base}/pairwise_geometry.npz",
        ),
        ROBUST: (
            f"{base}/robust_pair_diagnostics.json",
            f"{base}/disagreement_segments.csv",
            f"{base}/pairwise_strata.csv",
            f"{base}/high_confidence_disagreement.csv",
        ),
        SEGMENT_CONTEXT: (
            f"{base}/segment_context.json",
            f"{base}/segment_context.csv",
        ),
    }[stage]


def build_stage_plan(
    candidate: Mapping[str, Any] | Any,
    status_history: Sequence[Mapping[str, Any]],
    output_state: Mapping[str, OutputValidation],
    *,
    resume: bool,
    force_stages: set[str],
    base_seed: int,
    include_warnings: bool = False,
) -> list[StagePlan]:
    index = int(_value(candidate, "screening_index"))
    pdb_id = str(_value(candidate, "pdb_id")).lower()
    chain_id = str(_value(candidate, "chain_id"))
    uniprot_id = str(_value(candidate, "uniprot_id")).upper()
    pair_name = str(
        _value(candidate, "pair_name", f"{pdb_id}_{chain_id}__{uniprot_id}")
    )
    preflight_status = str(_value(candidate, "preflight_status", "not_run"))
    provisional_stratum = _value(candidate, "provisional_stratum")
    commands = _value(candidate, "commands", {}) or {}
    eligibility = classify_candidate_eligibility(
        preflight_status,
        include_warnings=include_warnings,
    )

    def item(stage: str, action: str, reason: str) -> StagePlan:
        return StagePlan(
            screening_index=index,
            pdb_id=pdb_id,
            chain_id=chain_id,
            uniprot_id=uniprot_id,
            pair_name=pair_name,
            provisional_stratum=(
                None if provisional_stratum is None else str(provisional_stratum)
            ),
            preflight_status=preflight_status,
            stage=stage,
            action=action,
            reason=reason,
            seed=derive_stage_seed(base_seed, index, stage),
            command=tuple(commands.get(stage, ())),
            expected_outputs=_expected_outputs(pair_name, stage),
        )

    if not eligibility.eligible:
        action = (
            "skip_unsupported"
            if eligibility.status == "unsupported_afdb_fragment"
            else "skip_preflight"
        )
        return [item(stage, action, eligibility.reason) for stage in STAGES]

    invalid_force_stages = set(force_stages).difference(STAGES)
    if invalid_force_stages:
        raise ValueError(f"Unknown force stages: {sorted(invalid_force_stages)}")
    force_index = (
        0
        if not resume
        else min((STAGES.index(stage) for stage in force_stages), default=len(STAGES))
    )
    reduced = reduce_status_history(status_history)
    blocked = False
    plan: list[StagePlan] = []
    for stage_index, stage in enumerate(STAGES):
        state = reduced.get((index, stage))
        validation = output_state.get(stage, OutputValidation(False))
        forced = stage_index >= force_index
        if blocked:
            plan.append(item(stage, "blocked_upstream", "awaiting_upstream"))
            continue
        if validation.valid and validation.status == "successful_no_segments":
            plan.append(item(stage, "skip_complete", "successful_no_segments"))
            continue
        if forced:
            plan.append(item(stage, "run", "forced"))
            blocked = True
            continue
        if state is not None and (
            state.interrupted
            or state.status.startswith("failed_")
            or state.status in {"blocked_upstream", "pending"}
        ):
            reason = "interrupted_attempt" if state.interrupted else state.status
            plan.append(item(stage, "run", reason))
            blocked = True
            continue
        if resume and validation.valid:
            plan.append(item(stage, "skip_complete", validation.status))
            continue
        plan.append(item(stage, "run", "missing_or_invalid_outputs"))
        blocked = True
    return plan


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (value, None) if isinstance(value, dict) else (None, "expected JSON object")


def _identity_matches(payload: Mapping[str, Any], candidate: Mapping[str, Any] | Any) -> bool:
    return (
        str(payload.get("pdb_id", "")).lower()
        == str(_value(candidate, "pdb_id")).lower()
        and str(payload.get("chain_id", "")) == str(_value(candidate, "chain_id"))
        and str(payload.get("uniprot_id", "")).upper()
        == str(_value(candidate, "uniprot_id")).upper()
    )


def _is_positive_integer(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _validation(
    *,
    checked: list[Path],
    missing: list[Path],
    invalid: list[Path],
    details: dict[str, Any],
    status: str = "complete",
) -> OutputValidation:
    return OutputValidation(
        valid=not missing and not invalid,
        status=status,
        checked_paths=tuple(str(path) for path in checked),
        missing_paths=tuple(str(path) for path in missing),
        invalid_paths=tuple(str(path) for path in invalid),
        details=details,
    )


def _require_paths(paths: Sequence[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    checked = list(paths)
    missing = [path for path in paths if not path.exists()]
    invalid = [
        path
        for path in paths
        if path.exists() and (not path.is_file() or path.stat().st_size == 0)
    ]
    return checked, missing, invalid


def _read_nonempty_parquet(path: Path) -> str | None:
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if not table.empty else "parquet table is empty"


def _read_segments_csv(
    path: Path,
) -> tuple[pd.DataFrame, str | None, bool]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}", False
    if not content.strip():
        return pd.DataFrame(), None, True
    try:
        return pd.read_csv(path), None, False
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}", False


def validate_stage_outputs(
    stage: str,
    candidate: Mapping[str, Any] | Any,
    project_root: str | Path,
) -> OutputValidation:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    root = Path(project_root)
    pair_name = str(
        _value(
            candidate,
            "pair_name",
            (
                f"{str(_value(candidate, 'pdb_id')).lower()}_"
                f"{_value(candidate, 'chain_id')}__"
                f"{str(_value(candidate, 'uniprot_id')).upper()}"
            ),
        )
    )
    pair_dir = root / "data/processed/pairs" / pair_name
    details: dict[str, Any] = {}

    if stage == PAIR:
        report_path = pair_dir / "pair_qc.json"
        mapping_path = pair_dir / "residue_mapping.parquet"
        checked, missing, invalid = _require_paths([report_path, mapping_path])
        report = None
        if report_path not in missing and report_path not in invalid:
            report, error = _read_json_object(report_path)
            if error:
                invalid.append(report_path)
                details[str(report_path)] = error
        if report is not None:
            unsupported = any(
                report.get(field) == "unsupported_afdb_fragment"
                for field in ("afdb_fragment_status", "afdb_coverage_status", "status")
            )
            required = (
                _identity_matches(report, candidate)
                and _is_positive_integer(report.get("mapped_residue_count"))
                and bool(report.get("afdb_model_entity_id"))
                and report.get("afdb_version") is not None
                and not unsupported
            )
            if not required:
                invalid.append(report_path)
                details[str(report_path)] = "pair report fields are inconsistent"
        if mapping_path not in missing and mapping_path not in invalid:
            error = _read_nonempty_parquet(mapping_path)
            if error:
                invalid.append(mapping_path)
                details[str(mapping_path)] = error
        return _validation(
            checked=checked,
            missing=missing,
            invalid=list(dict.fromkeys(invalid)),
            details=details,
        )

    if stage == GEOMETRY:
        report_path = pair_dir / "pair_geometry_qc.json"
        residue_path = pair_dir / "residue_geometry.parquet"
        pairwise_path = pair_dir / "pairwise_geometry.npz"
        checked, missing, invalid = _require_paths(
            [report_path, residue_path, pairwise_path]
        )
        report = None
        if report_path not in missing and report_path not in invalid:
            report, error = _read_json_object(report_path)
            if error:
                invalid.append(report_path)
                details[str(report_path)] = error
        if report is not None and (
            not _identity_matches(report, candidate)
            or not _is_positive_integer(report.get("mapped_ca_count"))
        ):
            invalid.append(report_path)
            details[str(report_path)] = "geometry report fields are inconsistent"
        if residue_path not in missing and residue_path not in invalid:
            error = _read_nonempty_parquet(residue_path)
            if error:
                invalid.append(residue_path)
                details[str(residue_path)] = error
        if pairwise_path not in missing and pairwise_path not in invalid:
            try:
                with np.load(pairwise_path, allow_pickle=False) as pairwise:
                    required_arrays = {
                        "symmetric_pae",
                        "absolute_pairwise_error",
                        "uniprot_positions",
                    }
                    if not required_arrays.issubset(pairwise.files) or any(
                        pairwise[name].size == 0 for name in required_arrays
                    ):
                        raise ValueError("required pairwise arrays are missing or empty")
            except (OSError, ValueError) as exc:
                invalid.append(pairwise_path)
                details[str(pairwise_path)] = f"{type(exc).__name__}: {exc}"
        return _validation(
            checked=checked,
            missing=missing,
            invalid=list(dict.fromkeys(invalid)),
            details=details,
        )

    if stage == ROBUST:
        report_path = pair_dir / "robust_pair_diagnostics.json"
        segments_path = pair_dir / "disagreement_segments.csv"
        strata_path = pair_dir / "pairwise_strata.csv"
        high_conf_path = pair_dir / "high_confidence_disagreement.csv"
        checked, missing, invalid = _require_paths(
            [report_path, segments_path, strata_path, high_conf_path]
        )
        # Header-only/empty segment outputs are valid and explicitly represent no hits.
        invalid = [path for path in invalid if path != segments_path]
        report = None
        if report_path not in missing and report_path not in invalid:
            report, error = _read_json_object(report_path)
            if error:
                invalid.append(report_path)
                details[str(report_path)] = error
        if report is not None:
            local = report.get("local_plddt_disagreement_test")
            pairwise = report.get("pae_pairwise_error_test")
            report_pair_name = Path(str(report.get("pair_dir", ""))).name
            traceable = all(
                isinstance(test, Mapping)
                and (
                    test.get("n_permutations") is not None
                    or test.get("seed") is not None
                )
                for test in (local, pairwise)
            )
            if report_pair_name != pair_name or not traceable:
                invalid.append(report_path)
                details[str(report_path)] = "robust report fields are inconsistent"
        if segments_path not in missing:
            segment_table, segment_error, _ = _read_segments_csv(segments_path)
            if segment_error is not None:
                invalid.append(segments_path)
                details[str(segments_path)] = segment_error
            elif (
                not segment_table.empty
                and not {"threshold", "residue_count"}.issubset(segment_table.columns)
            ):
                invalid.append(segments_path)
                details[str(segments_path)] = "missing segment selection columns"
        for path in (strata_path, high_conf_path):
            if path in missing or path in invalid:
                continue
            try:
                pd.read_csv(path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                invalid.append(path)
                details[str(path)] = f"{type(exc).__name__}: {exc}"
        return _validation(
            checked=checked,
            missing=missing,
            invalid=list(dict.fromkeys(invalid)),
            details=details,
        )

    json_path = pair_dir / "segment_context.json"
    csv_path = pair_dir / "segment_context.csv"
    segments_path = pair_dir / "disagreement_segments.csv"
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _validation(
                checked=[json_path],
                missing=[],
                invalid=[json_path],
                details={str(json_path): f"{type(exc).__name__}: {exc}"},
            )
        if isinstance(payload, list) and payload:
            return _validation(
                checked=[json_path], missing=[], invalid=[], details={}
            )
        if isinstance(payload, list) and not payload:
            return _validation(
                checked=[json_path],
                missing=[],
                invalid=[],
                details={"reason": "empty_segment_context"},
                status="successful_no_segments",
            )
        return _validation(
            checked=[json_path],
            missing=[],
            invalid=[json_path],
            details={str(json_path): "expected JSON list"},
        )
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            context = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            return _validation(
                checked=[csv_path],
                missing=[],
                invalid=[csv_path],
                details={str(csv_path): f"{type(exc).__name__}: {exc}"},
            )
        return _validation(
            checked=[csv_path],
            missing=[],
            invalid=[],
            details={},
            status="complete" if not context.empty else "successful_no_segments",
        )
    if not segments_path.exists():
        return _validation(
            checked=[],
            missing=[segments_path],
            invalid=[],
            details={"reason": "robust_segments_missing"},
        )
    segments, segment_error, _ = _read_segments_csv(segments_path)
    if segment_error is not None:
        return _validation(
            checked=[segments_path],
            missing=[],
            invalid=[segments_path],
            details={str(segments_path): segment_error},
        )
    threshold = float(_value(candidate, "segment_threshold", 1.0))
    min_length = int(_value(candidate, "segment_min_length", 3))
    required_columns = {"threshold", "residue_count"}
    if segments.empty:
        selected = segments
    elif not required_columns.issubset(segments.columns):
        return _validation(
            checked=[segments_path],
            missing=[],
            invalid=[segments_path],
            details={str(segments_path): "missing segment selection columns"},
        )
    else:
        selected = segments.loc[
            (segments["threshold"] == threshold)
            & (segments["residue_count"] >= min_length)
        ]
    if selected.empty:
        return _validation(
            checked=[segments_path],
            missing=[],
            invalid=[],
            details={"reason": "no_qualifying_segments"},
            status="successful_no_segments",
        )
    return _validation(
        checked=[segments_path],
        missing=[json_path, csv_path],
        invalid=[],
        details={"reason": "segment_context_outputs_missing"},
    )


def make_terminal_status_record(
    plan: StagePlan,
    *,
    run_id: str,
    attempt: int,
    return_code: int,
    runtime_seconds: float,
    validation: OutputValidation,
    error_type: str | None = None,
    error_message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> StatusRecord:
    if return_code == 0 and validation.valid:
        status = validation.status
        reason = validation.status
    else:
        status = f"failed_{plan.stage}"
        reason = "subprocess_failed" if return_code != 0 else "invalid_outputs"
        if error_type is None:
            error_type = (
                "CalledProcessError" if return_code != 0 else "OutputValidationError"
            )
        if error_message is None:
            error_message = json.dumps(
                {
                    "missing_paths": validation.missing_paths,
                    "invalid_paths": validation.invalid_paths,
                    "details": validation.details,
                },
                sort_keys=True,
            )
    return StatusRecord(
        run_id=run_id,
        screening_index=plan.screening_index,
        pdb_id=plan.pdb_id,
        chain_id=plan.chain_id,
        uniprot_id=plan.uniprot_id,
        pair_name=plan.pair_name,
        provisional_stratum=plan.provisional_stratum,
        preflight_status=plan.preflight_status,
        stage=plan.stage,
        attempt=attempt,
        status=status,
        reason=reason,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=float(runtime_seconds),
        seed=plan.seed,
        command=plan.command,
        return_code=int(return_code),
        validated_outputs=validation.checked_paths if validation.valid else (),
        error_type=error_type,
        error_message=error_message,
    )


def append_status_record(path: str | Path, record: StatusRecord) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")


def make_planned_status_record(
    plan: StagePlan,
    *,
    run_id: str,
    attempt: int,
    status: str,
    validated_outputs: Sequence[str] = (),
) -> StatusRecord:
    now = datetime.now(timezone.utc).isoformat()
    return StatusRecord(
        run_id=run_id,
        screening_index=plan.screening_index,
        pdb_id=plan.pdb_id,
        chain_id=plan.chain_id,
        uniprot_id=plan.uniprot_id,
        pair_name=plan.pair_name,
        provisional_stratum=plan.provisional_stratum,
        preflight_status=plan.preflight_status,
        stage=plan.stage,
        attempt=attempt,
        status=status,
        reason=plan.reason,
        started_at=now,
        finished_at=now,
        runtime_seconds=0.0,
        seed=plan.seed,
        command=plan.command,
        return_code=None,
        validated_outputs=tuple(validated_outputs),
        error_type=None,
        error_message=None,
    )


def summarize_status_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    total_runtime: dict[int, float] = {}
    last_errors: dict[int, dict[str, Any]] = {}
    attempt_events: dict[tuple[int, str, int], set[str]] = {}
    for record in records:
        try:
            index = int(record["screening_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if "stage" not in record:
            continue
        try:
            attempt_key = (
                index,
                str(record["stage"]),
                int(record["attempt"]),
            )
            attempt_events.setdefault(attempt_key, set()).add(str(record["status"]))
        except (KeyError, TypeError, ValueError):
            pass
        if record.get("error_type") or record.get("error_message"):
            last_errors[index] = dict(record)
        metadata[index] = {
            "screening_index": index,
            "pair_name": record.get("pair_name"),
            "preflight_status": record.get("preflight_status", "not_run"),
        }
        runtime = record.get("runtime_seconds")
        if runtime is not None and record.get("status") != "running":
            try:
                total_runtime[index] = total_runtime.get(index, 0.0) + float(runtime)
            except (TypeError, ValueError):
                pass

    reduced = reduce_status_history(records)
    rows: list[dict[str, Any]] = []
    interrupted_count = sum(
        "running" in statuses
        and not any(status in TERMINAL_STATUSES for status in statuses)
        for statuses in attempt_events.values()
    )
    for index in sorted(metadata):
        stage_statuses: dict[str, str] = {}
        for stage in STAGES:
            state = reduced.get((index, stage))
            if state is None:
                stage_statuses[stage] = "not_started"
                continue
            if state.interrupted:
                stage_statuses[stage] = "in_progress"
            else:
                stage_statuses[stage] = state.status

        values = set(stage_statuses.values())
        failed = next(
            (
                stage_statuses[stage]
                for stage in STAGES
                if stage_statuses[stage].startswith("failed_")
            ),
            None,
        )
        if "unsupported_afdb_fragment" in values:
            terminal = "unsupported_afdb_fragment"
        elif "skipped_preflight" in values:
            terminal = "skipped_preflight"
        elif failed is not None:
            terminal = failed
        elif "in_progress" in values:
            terminal = "in_progress"
        elif all(
            stage_statuses[stage] in {"complete", "skipped_complete"}
            for stage in (PAIR, GEOMETRY, ROBUST)
        ) and stage_statuses[SEGMENT_CONTEXT] in {
            "complete",
            "skipped_complete",
            "successful_no_segments",
        }:
            terminal = "complete"
        else:
            terminal = "not_started"

        latest_error = last_errors.get(index, {})
        rows.append(
            {
                **metadata[index],
                "pair_status": stage_statuses[PAIR],
                "geometry_status": stage_statuses[GEOMETRY],
                "robust_status": stage_statuses[ROBUST],
                "segment_context_status": stage_statuses[SEGMENT_CONTEXT],
                "candidate_terminal_status": terminal,
                "last_error_type": latest_error.get("error_type"),
                "last_error_message": latest_error.get("error_message"),
                "total_runtime_seconds": total_runtime.get(index, 0.0),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        summary = pd.DataFrame(
            columns=[
                "screening_index",
                "pair_name",
                "preflight_status",
                "pair_status",
                "geometry_status",
                "robust_status",
                "segment_context_status",
                "candidate_terminal_status",
                "last_error_type",
                "last_error_message",
                "total_runtime_seconds",
            ]
        )
    terminal_counts = (
        summary["candidate_terminal_status"].value_counts().to_dict()
        if not summary.empty
        else {}
    )
    failed_by_stage = {
        stage: int(summary[f"{stage}_status"].astype(str).str.startswith("failed_").sum())
        for stage in STAGES
    }
    statistics = {
        "eligible_candidates": int(
            (
                (summary["preflight_status"] == "pass_full_length")
                | (
                    (summary["preflight_status"] == "warn_construct_difference")
                    & (
                        summary["candidate_terminal_status"]
                        != "skipped_preflight"
                    )
                )
            ).sum()
        ),
        "complete_candidates": int(terminal_counts.get("complete", 0)),
        "quality_preflight_skipped": int(
            terminal_counts.get("skipped_preflight", 0)
        ),
        "unsupported_afdb_fragments": int(
            terminal_counts.get("unsupported_afdb_fragment", 0)
        ),
        "failed_by_stage": failed_by_stage,
        "interrupted_attempts": interrupted_count,
        "successful_no_segment_cases": int(
            (summary["segment_context_status"] == "successful_no_segments").sum()
        ),
    }
    return summary, statistics


def execute_stage(
    stage_plan: StagePlan,
    *,
    project_root: str | Path,
    log_path: str | Path,
    status_path: str | Path,
    run_id: str,
    attempt: int,
) -> StatusRecord:
    root = Path(project_root)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    running = StatusRecord(
        run_id=run_id,
        screening_index=stage_plan.screening_index,
        pdb_id=stage_plan.pdb_id,
        chain_id=stage_plan.chain_id,
        uniprot_id=stage_plan.uniprot_id,
        pair_name=stage_plan.pair_name,
        provisional_stratum=stage_plan.provisional_stratum,
        preflight_status=stage_plan.preflight_status,
        stage=stage_plan.stage,
        attempt=attempt,
        status="running",
        reason=stage_plan.reason,
        started_at=started.isoformat(),
        finished_at=None,
        runtime_seconds=None,
        seed=stage_plan.seed,
        command=stage_plan.command,
        return_code=None,
        validated_outputs=(),
        error_type=None,
        error_message=None,
    )
    append_status_record(status_path, running)
    return_code = 1
    error_type = None
    error_message = None
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                list(stage_plan.command),
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return_code = process.returncode
    except Exception as exc:  # noqa: BLE001 - adapter must always terminalize attempts
        error_type = type(exc).__name__
        error_message = str(exc)
    candidate = {
        "screening_index": stage_plan.screening_index,
        "pdb_id": stage_plan.pdb_id,
        "chain_id": stage_plan.chain_id,
        "uniprot_id": stage_plan.uniprot_id,
        "pair_name": stage_plan.pair_name,
    }
    try:
        validation = validate_stage_outputs(stage_plan.stage, candidate, root)
    except Exception as exc:  # noqa: BLE001 - adapter must terminalize validator faults
        validation = OutputValidation(
            valid=False,
            details={"validator_exception": f"{type(exc).__name__}: {exc}"},
        )
        error_type = type(exc).__name__
        error_message = str(exc)
    finished = datetime.now(timezone.utc)
    terminal = make_terminal_status_record(
        stage_plan,
        run_id=run_id,
        attempt=attempt,
        return_code=return_code,
        runtime_seconds=(finished - started).total_seconds(),
        validation=validation,
        error_type=error_type,
        error_message=error_message,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
    )
    append_status_record(status_path, terminal)
    return terminal
