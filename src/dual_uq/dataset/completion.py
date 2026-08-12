"""Scale-1 full-frame acquisition, readiness, and admission orchestration."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.admission import (
    FormalAdmissionConfig,
    FormalAdmissionError,
    build_not_evaluated_common_mask,
    evaluate_formal_admission_frame,
    load_formal_candidate_mapping,
    validate_stage0_admission_ledger,
)
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.policies.fragments import resolve_exact_fragment
from dual_uq.dataset.policies.identity import (
    extract_canonical_sequence_from_source,
    metadata_records,
)
from dual_uq.dataset.sampling_frame import classify_local_availability
from dual_uq.dataset.stages.acquisition import (
    AFDB_PREDICTION_API,
    PDB_MMCIF_URL,
    SIFTS_URL,
    AcquisitionError,
    AssetSpec,
    TransportResponse,
    acquire_asset,
    default_transport,
    utc_now,
    validate_payload,
)

BLOCKED_UPSTREAM_INTEGRITY = "SCALE1A2_BLOCKED_UPSTREAM_INTEGRITY"
BLOCKED_REGRESSION = "SCALE1A2_BLOCKED_REGRESSION"
FULL_FRAME_COMPLETE = "SCALE1A2_FULL_FRAME_CENSUS_COMPLETE"
COMPLETE_WITH_RESIDUAL_FAILURES = (
    "SCALE1A2_CENSUS_COMPLETE_WITH_RESIDUAL_ACQUISITION_FAILURES"
)


class FullFrameError(RuntimeError):
    """One structured Scale-1A2 blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class FullFrameConfig:
    inventory_ref: str = "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    scale1a0_table_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_audit.parquet"
    )
    scale1a0_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_local_availability_summary.json"
    )
    scale1a0_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_manifest.json"
    )
    scale1a1_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_census.parquet"
    )
    scale1a1_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/scale1_common_masks.parquet"
    )
    scale1a1_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_summary.json"
    )
    scale1a1_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_manifest.json"
    )
    stage0_admission_ref: str = (
        "experiments/p2_design_baseline/stage0/"
        "stage0_intervention_admission_v1.jsonl"
    )
    scale1b_root_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol"
    )
    output_root_ref: str = "experiments/p2_design_baseline/scale1/scale1a2"
    expected_inventory_sha256: str = (
        "05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf"
    )
    expected_scale1a0_table_sha256: str = (
        "695f3587dae24b5945f975a7611c677fca9a5e89f728d2f3802f77b5a31a2d4d"
    )
    expected_scale1a0_summary_sha256: str = (
        "afa01243db4c5833d040af3585ca6d998a924b9fd351d8d083b7e6b6f5253f10"
    )
    expected_scale1a0_manifest_sha256: str = (
        "de4baae228b15adda0ad41bba674d6a45470a728be92b374f20aa30b72ef4fca"
    )
    expected_scale1a1_census_sha256: str = (
        "ca01e17f55637d0f4584aa7bbb8940e50ea4aaf25b9c7ee7f741bb51a4095a7f"
    )
    expected_scale1a1_masks_sha256: str = (
        "face3d6f211585d452f474b090adb618d121aada849a4cbd492e92e9b60a03b9"
    )
    expected_scale1a1_summary_sha256: str = (
        "50afd84f6ac4a3104fc817015c89b9e1127b16bb8b5c8f688fb385cee02ecc0f"
    )
    expected_scale1a1_manifest_sha256: str = (
        "54227ee11e67fdb6da12cbf9001b603875b1fd3c28d603608d384f0c50988223"
    )
    expected_stage0_admission_sha256: str = (
        "1fa86cde49cdac68433572d929a99c964574c8edc7f9f62832e0c2b3d4a529c0"
    )
    expected_source_count: int = 213
    expected_preexisting_ready: int = 72
    expected_acquisition_targets: int = 141
    expected_initial_status_counts: tuple[tuple[str, int], ...] = (
        ("LOCAL_READY_FOR_SCALE1A", 72),
        ("MISSING_PDB", 68),
        ("MISSING_PDB_AND_AFDB", 68),
        ("MISSING_AFDB", 3),
        ("MISSING_CANONICAL_SEQUENCE", 2),
    )


SCALE1B_V1_OUTPUTS = (
    (
        "scale1b_fixed_probe_candidates.parquet",
        "af405b7dafae84d27e92d3654d173eef29f268b68ad241c9934c6920fcd99126",
    ),
    (
        "scale1b_probe_manifest.json",
        "5701462be2ebe30b713723bd09bae2a70a980f8cc47eb6d0772884b0bea1fa13",
    ),
    (
        "scale1b_scoring_plan.parquet",
        "7d0f2aa7353f68d4c78736fafc40aa1ffe46269af7e4f5bfe8d86ee718898cea",
    ),
    (
        "scale1b_scoring_protocol.json",
        "e5ede85840192ae2191ba2c4523a00c1464f11759492a87e77ae2c2a7595039a",
    ),
    (
        "scale1b_protocol_freeze_manifest.json",
        "485a22a89d0fc532cdee72c09ed7faa36c5fdc54079aa2a971479d41c53af918",
    ),
)


@dataclass(frozen=True)
class CandidateIdentity:
    sampling_frame_index: int
    candidate_id: str
    polymer_entity_id: str
    pdb_id: str
    pdb_chain: str
    canonical_accession: str


@dataclass(frozen=True)
class AssetRequirement:
    sampling_frame_index: int
    candidate_id: str
    polymer_entity_id: str
    asset_type: str
    canonical_identifier: str
    source_authority: str
    source_record_identifier: str
    source_url: str
    local_path_relative: str
    dependency_asset_type: str | None = None
    source_metadata_version_if_available: str | None = None


@dataclass(frozen=True)
class AssetResolution:
    requirement: AssetRequirement | None
    failure_code: str | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionRecord:
    sampling_frame_index: int
    candidate_id: str
    asset_type: str
    canonical_identifier: str
    source_authority: str
    source_record_identifier: str
    source_url: str
    retrieval_status: str
    retrieval_timestamp: str
    retrieval_method: str
    attempt_count: int
    local_path_relative: str
    file_sha256: str | None
    file_size: int | None
    source_metadata_version_if_available: str | None
    validation_status: str
    failure_code: str | None
    http_status: int | None
    dependency_asset_type: str | None
    rejected_payload_path: str | None = None
    rejected_payload_sha256: str | None = None


@dataclass(frozen=True)
class FullFrameInputs:
    inventory: dict[str, Any]
    source_frame: pd.DataFrame
    preexisting_local_ready: pd.DataFrame
    acquisition_targets: pd.DataFrame
    initial_status_counts: dict[str, int]
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    scale1b_v1_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FullFrameResult:
    """Single structured source for every Scale-1A2 renderer."""

    status: str
    acquisition_ledger: pd.DataFrame
    census: pd.DataFrame
    common_masks: pd.DataFrame
    attrition_summary: dict[str, Any]


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "nonportable_input_path",
            f"Invalid logical path: {logical_ref}",
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected_sha256: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            expected_sha256=expected_sha256,
            observed_sha256=observed,
        )
    return path, {
        "path": logical_ref,
        "sha256": observed,
        "label": label,
    }


def _inventory_identity(record: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(record["candidate_index"]),
        str(record["pair_id"]),
        str(record["polymer_entity_id"]),
    )


def validate_full_frame_inputs(
    paths: ProjectPaths, config: FullFrameConfig
) -> FullFrameInputs:
    """Validate the frozen source frame, regression oracles, and Scale-1B bytes."""
    bindings = (
        (config.inventory_ref, config.expected_inventory_sha256, "candidate inventory"),
        (
            config.scale1a0_table_ref,
            config.expected_scale1a0_table_sha256,
            "Scale-1A0 sampling-frame table",
        ),
        (
            config.scale1a0_summary_ref,
            config.expected_scale1a0_summary_sha256,
            "Scale-1A0 availability summary",
        ),
        (
            config.scale1a0_manifest_ref,
            config.expected_scale1a0_manifest_sha256,
            "Scale-1A0 manifest",
        ),
        (
            config.scale1a1_census_ref,
            config.expected_scale1a1_census_sha256,
            "Scale-1A1 frozen census",
        ),
        (
            config.scale1a1_masks_ref,
            config.expected_scale1a1_masks_sha256,
            "Scale-1A1 frozen common masks",
        ),
        (
            config.scale1a1_summary_ref,
            config.expected_scale1a1_summary_sha256,
            "Scale-1A1 frozen summary",
        ),
        (
            config.scale1a1_manifest_ref,
            config.expected_scale1a1_manifest_sha256,
            "Scale-1A1 frozen manifest",
        ),
        (
            config.stage0_admission_ref,
            config.expected_stage0_admission_sha256,
            "Stage-0 frozen admission ledger",
        ),
    )
    artifacts: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    for logical_ref, expected, label in bindings:
        path, record = _require_hash(paths, logical_ref, expected, label)
        resolved[logical_ref] = path
        artifacts.append(record)

    try:
        inventory = json.loads(
            resolved[config.inventory_ref].read_text(encoding="utf-8")
        )
        source = pd.read_parquet(resolved[config.scale1a0_table_ref])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "unreadable_upstream_input",
            "Unable to read the frozen source frame",
        ) from exc
    records = inventory.get("candidates") if isinstance(inventory, dict) else None
    if not isinstance(records, list) or len(records) != config.expected_source_count:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "source_frame_count_mismatch",
            "Candidate inventory count changed",
        )
    if len(source) != config.expected_source_count:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "source_frame_count_mismatch",
            "Scale-1A0 source-frame count changed",
        )
    expected_order = list(range(1, config.expected_source_count + 1))
    if source["sampling_frame_index"].astype(int).tolist() != expected_order:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "source_frame_order_mismatch",
            "Scale-1A0 source ordering changed",
        )
    inventory_identities = [_inventory_identity(record) for record in records]
    source_identities = list(
        zip(
            source["sampling_frame_index"].astype(int),
            source["pair_id"].astype(str),
            source["polymer_entity_id"].astype(str),
            strict=True,
        )
    )
    if inventory_identities != source_identities or source["candidate_id"].duplicated().any():
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "source_frame_identity_mismatch",
            "Frozen inventory and Scale-1A0 identities disagree",
        )
    counts = {
        str(key): int(value)
        for key, value in Counter(source["local_availability_status"]).items()
    }
    expected_counts = dict(config.expected_initial_status_counts)
    if counts != expected_counts:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "initial_readiness_regression",
            "Scale-1A0 readiness counts changed",
            expected=expected_counts,
            observed=counts,
        )
    ready = source.loc[
        source["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ].copy()
    targets = source.loc[
        ~source["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ].copy()
    if (
        len(ready) != config.expected_preexisting_ready
        or len(targets) != config.expected_acquisition_targets
    ):
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "initial_readiness_regression",
            "Initial 72/141 split changed",
        )

    scale1b_records: list[dict[str, Any]] = []
    scale1b_root = _resolve(paths, config.scale1b_root_ref)
    for filename, expected in SCALE1B_V1_OUTPUTS:
        logical_ref = f"{config.scale1b_root_ref}/{filename}"
        path = scale1b_root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise FullFrameError(
                BLOCKED_UPSTREAM_INTEGRITY,
                "scale1b_v1_sha256_mismatch",
                f"Frozen Scale-1B-v1 artifact changed: {filename}",
            )
        record = {
            "path": logical_ref,
            "sha256": expected,
            "label": "Frozen Scale-1B-v1 artifact",
        }
        artifacts.append(record)
        scale1b_records.append(record)
    return FullFrameInputs(
        inventory=inventory,
        source_frame=source.reset_index(drop=True),
        preexisting_local_ready=ready.reset_index(drop=True),
        acquisition_targets=targets.reset_index(drop=True),
        initial_status_counts=counts,
        input_artifacts=tuple(artifacts),
        scale1b_v1_artifacts=tuple(scale1b_records),
    )


def resolve_asset_requirements(
    inputs: FullFrameInputs, paths: ProjectPaths
) -> tuple[AssetRequirement, ...]:
    """Derive independent missing-asset requirements in frozen source order."""
    del paths  # destinations are canonical repository-relative logical paths
    requirements: list[AssetRequirement] = []
    for row in inputs.acquisition_targets.to_dict(orient="records"):
        index = int(row["sampling_frame_index"])
        pair_id = str(row["pair_id"])
        polymer_entity = str(row["polymer_entity_id"])
        pdb_id = str(row["pdb_id"]).lower()
        accession = str(row["canonical_accession"])
        if bool(row["missing_pdb"]):
            requirements.append(
                AssetRequirement(
                    sampling_frame_index=index,
                    candidate_id=pair_id,
                    polymer_entity_id=polymer_entity,
                    asset_type="pdb_mmcif",
                    canonical_identifier=pdb_id,
                    source_authority="RCSB_PDB",
                    source_record_identifier=pdb_id,
                    source_url=PDB_MMCIF_URL.format(pdb_id=pdb_id),
                    local_path_relative=f"data/raw/pdb/{pdb_id}.cif",
                )
            )
        if bool(row["missing_mapping_metadata"]):
            requirements.append(
                AssetRequirement(
                    sampling_frame_index=index,
                    candidate_id=pair_id,
                    polymer_entity_id=polymer_entity,
                    asset_type="sifts",
                    canonical_identifier=pdb_id,
                    source_authority="PDBe_SIFTS",
                    source_record_identifier=pdb_id,
                    source_url=SIFTS_URL.format(pdb_id=pdb_id),
                    local_path_relative=f"data/raw/mappings/{pdb_id}.xml.gz",
                )
            )
        if bool(row["missing_afdb"] or row["missing_canonical_sequence"]):
            requirements.append(
                AssetRequirement(
                    sampling_frame_index=index,
                    candidate_id=pair_id,
                    polymer_entity_id=polymer_entity,
                    asset_type="afdb_metadata_collection",
                    canonical_identifier=accession,
                    source_authority="AlphaFold_DB",
                    source_record_identifier=accession,
                    source_url=AFDB_PREDICTION_API.format(accession=accession),
                    local_path_relative=f"data/raw/afdb/{accession}/metadata.json",
                )
            )
    return tuple(requirements)


_RETRIEVAL_STATUS_BY_FAILURE = {
    "http_not_found": "SOURCE_NOT_FOUND",
    "transport_failure": "HTTP_OR_REMOTE_FAILURE",
    "rate_limited_exhausted": "HTTP_OR_REMOTE_FAILURE",
    "identity_mismatch": "IDENTIFIER_BINDING_FAIL",
    "exact_identity_mismatch": "IDENTIFIER_BINDING_FAIL",
    "model_identity_mismatch": "IDENTIFIER_BINDING_FAIL",
    "ambiguous_exact_record_set": "AMBIGUOUS_REMOTE_RECORD",
    "malformed_payload": "DOWNLOADED_FILE_INVALID",
    "invalid_metadata": "DOWNLOADED_FILE_INVALID",
    "missing_canonical_sequence_provenance": "CANONICAL_SEQUENCE_UNAVAILABLE",
    "existing_file_conflict": "REQUIRES_MANUAL_DATA_REVIEW",
    "atomic_write_failure": "OTHER_STRUCTURED_ACQUISITION_FAILURE",
}


def _normalized_acquisition_record(
    requirement: AssetRequirement, raw: Mapping[str, Any]
) -> AcquisitionRecord:
    raw_status = str(raw.get("status", "failed"))
    failure_code = (
        str(raw["failure_code"])
        if raw.get("failure_code") is not None
        else None
    )
    if raw_status == "reused_valid":
        retrieval_status = "ALREADY_PRESENT_FROZEN"
        retrieval_method = "existing_file_validation"
        validation_status = "VALID"
    elif raw_status == "downloaded_new":
        retrieval_status = "ACQUIRED"
        retrieval_method = "https_get"
        validation_status = "VALID"
    else:
        retrieval_status = _RETRIEVAL_STATUS_BY_FAILURE.get(
            failure_code or "", "OTHER_STRUCTURED_ACQUISITION_FAILURE"
        )
        retrieval_method = "https_get"
        validation_status = (
            "INVALID"
            if failure_code
            in {
                "identity_mismatch",
                "exact_identity_mismatch",
                "model_identity_mismatch",
                "ambiguous_exact_record_set",
                "malformed_payload",
                "invalid_metadata",
                "missing_canonical_sequence_provenance",
                "existing_file_conflict",
            }
            else "NOT_VALIDATED"
        )
    stored = raw_status in {"reused_valid", "downloaded_new"}
    return AcquisitionRecord(
        sampling_frame_index=requirement.sampling_frame_index,
        candidate_id=requirement.candidate_id,
        asset_type=requirement.asset_type,
        canonical_identifier=requirement.canonical_identifier,
        source_authority=requirement.source_authority,
        source_record_identifier=requirement.source_record_identifier,
        source_url=requirement.source_url,
        retrieval_status=retrieval_status,
        retrieval_timestamp=str(raw.get("timestamp", "")),
        retrieval_method=retrieval_method,
        attempt_count=int(raw.get("attempt_count", 0)),
        local_path_relative=requirement.local_path_relative,
        file_sha256=str(raw["SHA256"]) if stored and raw.get("SHA256") else None,
        file_size=int(raw["byte_count"]) if stored and raw.get("byte_count") is not None else None,
        source_metadata_version_if_available=(
            requirement.source_metadata_version_if_available
            or (
                str(raw["source_metadata_version_if_available"])
                if raw.get("source_metadata_version_if_available") is not None
                else None
            )
        ),
        validation_status=validation_status,
        failure_code=failure_code,
        http_status=(
            int(raw["HTTP_status"])
            if raw.get("HTTP_status") is not None
            else None
        ),
        dependency_asset_type=requirement.dependency_asset_type,
        rejected_payload_path=(
            str(raw["rejected_payload_path"])
            if raw.get("rejected_payload_path")
            else None
        ),
        rejected_payload_sha256=(
            str(raw["rejected_payload_SHA256"])
            if raw.get("rejected_payload_SHA256")
            else None
        ),
    )


def execute_initial_acquisition(
    requirements: Sequence[AssetRequirement],
    paths: ProjectPaths,
    *,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    prior_records: Sequence[AcquisitionRecord] = (),
    rejected_evidence_ref: str = (
        "artifacts/dataset/audits/acquisition_rejected/scale1a2"
    ),
) -> tuple[AcquisitionRecord, ...]:
    """Acquire one deterministic independent-asset sequence without fallback."""
    records: list[AcquisitionRecord] = []
    prior = {
        (record.sampling_frame_index, record.asset_type): record
        for record in prior_records
    }
    rejected_dir = _resolve(paths, rejected_evidence_ref)
    for requirement in requirements:
        previous = prior.get(
            (requirement.sampling_frame_index, requirement.asset_type)
        )
        if previous is not None:
            if (
                previous.canonical_identifier != requirement.canonical_identifier
                or previous.local_path_relative != requirement.local_path_relative
            ):
                raise FullFrameError(
                    BLOCKED_UPSTREAM_INTEGRITY,
                    "prior_acquisition_binding_mismatch",
                    f"Prior acquisition binding changed for {requirement.candidate_id}",
                )
            if previous.retrieval_status in _SUCCESSFUL_ACQUISITION_STATUSES:
                path = _resolve(paths, requirement.local_path_relative)
                try:
                    data = path.read_bytes()
                    validation_result = validate_payload(
                        requirement.asset_type,
                        data,
                        requirement.canonical_identifier,
                        None,
                    )
                except (OSError, AcquisitionError) as exc:
                    raise FullFrameError(
                        BLOCKED_UPSTREAM_INTEGRITY,
                        "prior_acquisition_asset_drift",
                        f"Prior acquired asset no longer validates: {requirement.local_path_relative}",
                    ) from exc
                if previous.file_sha256 != sha256_file(path):
                    raise FullFrameError(
                        BLOCKED_UPSTREAM_INTEGRITY,
                        "prior_acquisition_asset_drift",
                        f"Prior acquired asset SHA changed: {requirement.local_path_relative}",
                    )
                source_version = requirement.source_metadata_version_if_available
                if (
                    source_version is None
                    and validation_result
                    and validation_result.get("source_metadata_versions")
                ):
                    source_version = json.dumps(
                        validation_result["source_metadata_versions"],
                        separators=(",", ":"),
                    )
                if (
                    previous.source_metadata_version_if_available is None
                    and source_version is not None
                ):
                    previous = replace(
                        previous,
                        source_metadata_version_if_available=source_version,
                    )
            records.append(previous)
            continue
        spec = AssetSpec(
            candidate_index=requirement.sampling_frame_index,
            polymer_entity_id=requirement.polymer_entity_id,
            pair_id=requirement.candidate_id,
            asset_type=requirement.asset_type,
            source_url=requirement.source_url,
            local_path=_resolve(paths, requirement.local_path_relative),
            expected_identity=requirement.canonical_identifier,
            display_path=requirement.local_path_relative,
        )
        raw = acquire_asset(
            spec,
            transport=transport,
            sleeper=sleeper,
            clock=clock,
            max_attempts=max_attempts,
            rejected_evidence_dir=rejected_dir,
        )
        record = _normalized_acquisition_record(requirement, raw)
        if record.rejected_payload_path is not None:
            try:
                portable_rejected_path = paths.logical_ref(
                    Path(record.rejected_payload_path)
                )
            except ProjectPathError as exc:
                raise FullFrameError(
                    BLOCKED_UPSTREAM_INTEGRITY,
                    "nonportable_rejected_payload_path",
                    "Rejected acquisition evidence is outside declared roots",
                ) from exc
            record = replace(
                record, rejected_payload_path=portable_rejected_path
            )
        records.append(record)
    return tuple(records)


def resolve_afdb_structure_requirement(
    identity: CandidateIdentity,
    metadata_payload: bytes,
    *,
    mapped_interval: tuple[int, int],
) -> AssetResolution:
    """Resolve exactly one interval-covering AFDB model without silent fallback."""
    try:
        records = metadata_records(metadata_payload)
        resolution = resolve_exact_fragment(
            records, identity.canonical_accession, mapped_interval
        )
    except DerivationError as exc:
        return AssetResolution(None, "PROVENANCE_UNRESOLVED", {"cause": exc.code})
    selected = resolution["selected_fragment"]
    if selected is None:
        code = (
            "AMBIGUOUS_REMOTE_RECORD"
            if resolution["fragment_resolution_status"]
            == "ambiguous_full_covering_fragments"
            else "AFDB_ASSET_UNAVAILABLE"
        )
        return AssetResolution(
            None,
            code,
            {
                "fragment_resolution_status": resolution[
                    "fragment_resolution_status"
                ],
                "full_cover_count": resolution["full_cover_count"],
            },
        )
    matches = [
        record
        for record in records
        if str(record.get("uniprotAccession", ""))
        == identity.canonical_accession
        and str(record.get("modelEntityId") or record.get("entryId") or "")
        == selected.model_entity_id
    ]
    if len(matches) != 1:
        return AssetResolution(
            None,
            "AMBIGUOUS_REMOTE_RECORD",
            {"selected_model_record_count": len(matches)},
        )
    url = matches[0].get("cifUrl")
    parsed = urlparse(str(url))
    if not isinstance(url, str) or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return AssetResolution(
            None, "AFDB_ASSET_UNAVAILABLE", {"cause": "missing_cif_url"}
        )
    model_id = selected.model_entity_id
    requirement = AssetRequirement(
        sampling_frame_index=identity.sampling_frame_index,
        candidate_id=identity.candidate_id,
        polymer_entity_id=identity.polymer_entity_id,
        asset_type="afdb_structure",
        canonical_identifier=model_id,
        source_authority="AlphaFold_DB",
        source_record_identifier=model_id,
        source_url=url,
        local_path_relative=(
            f"data/raw/afdb/{identity.canonical_accession}/{model_id}/model.cif"
        ),
        dependency_asset_type="afdb_metadata_collection",
        source_metadata_version_if_available=(
            str(matches[0]["latestVersion"])
            if isinstance(matches[0].get("latestVersion"), int)
            and not isinstance(matches[0].get("latestVersion"), bool)
            else None
        ),
    )
    return AssetResolution(
        requirement,
        None,
        {
            "mapped_interval": list(mapped_interval),
            "selected_model_entity_id": model_id,
        },
    )


def apply_canonical_sequence_overrides(
    frame: pd.DataFrame,
    paths: ProjectPaths,
    source_by_index: Mapping[int, str],
) -> pd.DataFrame:
    """Bind exact UniProtKB canonical records and reapply frozen readiness precedence."""
    output = frame.copy(deep=True)
    for index, logical_ref in sorted(source_by_index.items()):
        matches = output.index[
            output["sampling_frame_index"].astype(int).eq(int(index))
        ].tolist()
        if len(matches) != 1:
            raise FullFrameError(
                BLOCKED_UPSTREAM_INTEGRITY,
                "canonical_override_identity_mismatch",
                f"Canonical override index is not unique: {index}",
            )
        row_index = matches[0]
        accession = str(output.at[row_index, "canonical_accession"])
        path = _resolve(paths, logical_ref)
        try:
            payload = path.read_bytes()
            canonical = extract_canonical_sequence_from_source(payload, accession)
        except (OSError, DerivationError) as exc:
            raise FullFrameError(
                BLOCKED_UPSTREAM_INTEGRITY,
                "invalid_canonical_sequence_override",
                f"Canonical sequence source is invalid for {accession}",
            ) from exc
        output.at[row_index, "canonical_sequence_present"] = True
        output.at[row_index, "canonical_sequence_source_relative"] = logical_ref
        output.at[row_index, "canonical_sequence_sha256"] = canonical[
            "sequence_sha256"
        ]
        output.at[row_index, "missing_canonical_sequence"] = False

    flag_columns = [
        "missing_pdb",
        "missing_afdb",
        "missing_canonical_sequence",
        "missing_identity_metadata",
        "missing_mapping_metadata",
        "missing_provenance_metadata",
        "ambiguous_local_source",
        "unreadable_local_file",
        "other_structured_local_blocker",
    ]
    for row_index in output.index:
        flags = {
            column: bool(output.at[row_index, column]) for column in flag_columns
        }
        status, reason = classify_local_availability(flags)
        output.at[row_index, "local_availability_status"] = status
        output.at[row_index, "terminal_local_missing_reason"] = reason
    return output


def resolve_uniprot_requirements(
    frame: pd.DataFrame,
    target_indices: set[int],
    paths: ProjectPaths,
) -> tuple[AssetRequirement, ...]:
    """Request UniProt only when existing exact metadata cannot prove canonical span."""
    requirements: list[AssetRequirement] = []
    for row in frame.to_dict(orient="records"):
        index = int(row["sampling_frame_index"])
        if index not in target_indices or not bool(row["missing_canonical_sequence"]):
            continue
        accession = str(row["canonical_accession"])
        metadata_ref = f"data/raw/afdb/{accession}/metadata.json"
        metadata_path = _resolve(paths, metadata_ref)
        if metadata_path.is_file():
            try:
                extract_canonical_sequence_from_source(
                    metadata_path.read_bytes(), accession
                )
            except (OSError, DerivationError):
                pass
            else:
                continue
        requirements.append(
            AssetRequirement(
                sampling_frame_index=index,
                candidate_id=str(row["pair_id"]),
                polymer_entity_id=str(row["polymer_entity_id"]),
                asset_type="uniprot_canonical",
                canonical_identifier=accession,
                source_authority="UniProtKB",
                source_record_identifier=accession,
                source_url=f"https://rest.uniprot.org/uniprotkb/{accession}.json",
                local_path_relative=f"data/raw/uniprot/{accession}.json",
            )
        )
    return tuple(requirements)


def _successful_record_map(
    records: Sequence[AcquisitionRecord],
) -> dict[tuple[int, str], AcquisitionRecord]:
    output: dict[tuple[int, str], AcquisitionRecord] = {}
    for record in records:
        if record.retrieval_status in _SUCCESSFUL_ACQUISITION_STATUSES:
            output[(record.sampling_frame_index, record.asset_type)] = record
    return output


def recompute_full_frame_readiness(
    source_frame: pd.DataFrame,
    acquisition_records: Sequence[AcquisitionRecord],
    paths: ProjectPaths,
) -> pd.DataFrame:
    """Bind validated acquired assets and reapply the frozen A0 precedence."""
    output = source_frame.copy(deep=True)
    output["pre_scale1a2_local_status"] = output["local_availability_status"]
    successful = _successful_record_map(acquisition_records)
    for row_index, row in output.iterrows():
        index = int(row["sampling_frame_index"])
        accession = str(row["canonical_accession"])
        pdb = successful.get((index, "pdb_mmcif"))
        if pdb is not None:
            output.at[row_index, "pdb_file_present"] = True
            output.at[row_index, "pdb_file_readable"] = True
            output.at[row_index, "pdb_chain_identifiable"] = True
            output.at[row_index, "pdb_entity_identifiable"] = True
            output.at[row_index, "pdb_file_path_relative"] = pdb.local_path_relative
            output.at[row_index, "missing_pdb"] = False
        sifts = successful.get((index, "sifts"))
        if sifts is not None:
            output.at[row_index, "mapping_metadata_present"] = True
            output.at[row_index, "mapping_metadata_source_relative"] = (
                sifts.local_path_relative
            )
            output.at[row_index, "missing_mapping_metadata"] = False

        metadata = successful.get((index, "afdb_metadata_collection"))
        metadata_ref = (
            metadata.local_path_relative
            if metadata is not None
            else f"data/raw/afdb/{accession}/metadata.json"
        )
        metadata_path = _resolve(paths, metadata_ref)
        if metadata_path.is_file():
            output.at[row_index, "afdb_metadata_source_relative"] = metadata_ref

        if bool(output.at[row_index, "missing_canonical_sequence"]):
            canonical_candidates = []
            uniprot = successful.get((index, "uniprot_canonical"))
            if uniprot is not None:
                canonical_candidates.append(uniprot.local_path_relative)
            if metadata_path.is_file():
                canonical_candidates.append(metadata_ref)
            for canonical_ref in canonical_candidates:
                canonical_path = _resolve(paths, canonical_ref)
                try:
                    canonical = extract_canonical_sequence_from_source(
                        canonical_path.read_bytes(), accession
                    )
                except (OSError, DerivationError):
                    continue
                output.at[row_index, "canonical_sequence_present"] = True
                output.at[row_index, "canonical_sequence_source_relative"] = (
                    canonical_ref
                )
                output.at[row_index, "canonical_sequence_sha256"] = canonical[
                    "sequence_sha256"
                ]
                output.at[row_index, "missing_canonical_sequence"] = False
                break

        structure = successful.get((index, "afdb_structure"))
        if structure is not None:
            output.at[row_index, "afdb_file_present"] = True
            output.at[row_index, "afdb_file_readable"] = True
            output.at[row_index, "afdb_accession_identifiable"] = True
            output.at[row_index, "afdb_file_path_relative"] = (
                structure.local_path_relative
            )
            output.at[row_index, "afdb_id"] = structure.canonical_identifier
            output.at[row_index, "missing_afdb"] = False

        output.at[row_index, "local_pair_complete"] = bool(
            not output.at[row_index, "missing_pdb"]
            and not output.at[row_index, "missing_afdb"]
        )
        output.at[row_index, "local_metadata_complete"] = bool(
            not output.at[row_index, "missing_canonical_sequence"]
            and not output.at[row_index, "missing_identity_metadata"]
            and not output.at[row_index, "missing_mapping_metadata"]
            and not output.at[row_index, "missing_provenance_metadata"]
        )

    flag_columns = [
        "missing_pdb",
        "missing_afdb",
        "missing_canonical_sequence",
        "missing_identity_metadata",
        "missing_mapping_metadata",
        "missing_provenance_metadata",
        "ambiguous_local_source",
        "unreadable_local_file",
        "other_structured_local_blocker",
    ]
    for row_index in output.index:
        status, reason = classify_local_availability(
            {column: bool(output.at[row_index, column]) for column in flag_columns}
        )
        output.at[row_index, "local_availability_status"] = status
        output.at[row_index, "terminal_local_missing_reason"] = reason
    return output


def _dependency_failure_record(
    identity: CandidateIdentity,
    *,
    status: str,
    failure_code: str,
    dependency: str,
    timestamp: str,
) -> AcquisitionRecord:
    return AcquisitionRecord(
        sampling_frame_index=identity.sampling_frame_index,
        candidate_id=identity.candidate_id,
        asset_type="afdb_structure",
        canonical_identifier=identity.canonical_accession,
        source_authority="AlphaFold_DB",
        source_record_identifier=identity.canonical_accession,
        source_url=AFDB_PREDICTION_API.format(
            accession=identity.canonical_accession
        ),
        retrieval_status=status,
        retrieval_timestamp=timestamp,
        retrieval_method="not_attempted_dependency_blocked",
        attempt_count=0,
        local_path_relative="",
        file_sha256=None,
        file_size=None,
        source_metadata_version_if_available=None,
        validation_status="NOT_VALIDATED",
        failure_code=failure_code,
        http_status=None,
        dependency_asset_type=dependency,
    )


def resolve_afdb_structure_requirements(
    frame: pd.DataFrame,
    target_indices: set[int],
    paths: ProjectPaths,
    *,
    clock: Callable[[], str] = utc_now,
    prior_records: Sequence[AcquisitionRecord] = (),
) -> tuple[tuple[AssetRequirement, ...], tuple[AcquisitionRecord, ...]]:
    """Resolve missing AFDB structures only after exact mapping is available."""
    requirements: list[AssetRequirement] = []
    failures: list[AcquisitionRecord] = []
    prior_failures = {
        record.sampling_frame_index: record
        for record in prior_records
        if record.asset_type == "afdb_structure"
        and record.retrieval_status not in _SUCCESSFUL_ACQUISITION_STATUSES
    }
    for row in frame.to_dict(orient="records"):
        index = int(row["sampling_frame_index"])
        if index not in target_indices or not bool(row["missing_afdb"]):
            continue
        if (previous := prior_failures.get(index)) is not None:
            if previous.candidate_id != str(row["pair_id"]):
                raise FullFrameError(
                    BLOCKED_UPSTREAM_INTEGRITY,
                    "prior_acquisition_binding_mismatch",
                    f"Prior dependency failure identity changed at index {index}",
                )
            failures.append(previous)
            continue
        identity = CandidateIdentity(
            sampling_frame_index=index,
            candidate_id=str(row["pair_id"]),
            polymer_entity_id=str(row["polymer_entity_id"]),
            pdb_id=str(row["pdb_id"]),
            pdb_chain=str(row["pdb_chain"]),
            canonical_accession=str(row["canonical_accession"]),
        )
        metadata_ref = row.get("afdb_metadata_source_relative")
        if not isinstance(metadata_ref, str) or not metadata_ref:
            failures.append(
                _dependency_failure_record(
                    identity,
                    status="AFDB_ASSET_UNAVAILABLE",
                    failure_code="metadata_asset_missing",
                    dependency="afdb_metadata_collection",
                    timestamp=clock(),
                )
            )
            continue
        try:
            mapping = load_formal_candidate_mapping(row, paths)
            positions = mapping["uniprot_position"].astype(int)
            if positions.empty:
                raise ValueError("mapping is empty")
            metadata_payload = _resolve(paths, metadata_ref).read_bytes()
        except (OSError, ValueError, FormalAdmissionError) as exc:
            failures.append(
                _dependency_failure_record(
                    identity,
                    status="PROVENANCE_UNRESOLVED",
                    failure_code=getattr(
                        exc, "code", "mapping_dependency_unresolved"
                    ),
                    dependency="sifts",
                    timestamp=clock(),
                )
            )
            continue
        resolution = resolve_afdb_structure_requirement(
            identity,
            metadata_payload,
            mapped_interval=(int(positions.min()), int(positions.max())),
        )
        if resolution.requirement is None:
            failures.append(
                _dependency_failure_record(
                    identity,
                    status=resolution.failure_code or "AFDB_ASSET_UNAVAILABLE",
                    failure_code=str(
                        resolution.details.get(
                            "fragment_resolution_status",
                            resolution.details.get("cause", "fragment_unresolved"),
                        )
                    ),
                    dependency="afdb_metadata_collection",
                    timestamp=clock(),
                )
            )
        else:
            requirements.append(resolution.requirement)
    return tuple(requirements), tuple(failures)


_SUCCESSFUL_ACQUISITION_STATUSES = {"ACQUIRED", "ALREADY_PRESENT_FROZEN"}
_ACQUISITION_STATUS_PRECEDENCE = {
    "REQUIRES_MANUAL_DATA_REVIEW": 100,
    "IDENTIFIER_BINDING_FAIL": 90,
    "AMBIGUOUS_REMOTE_RECORD": 85,
    "DOWNLOADED_FILE_INVALID": 80,
    "PROVENANCE_UNRESOLVED": 75,
    "CANONICAL_SEQUENCE_UNAVAILABLE": 70,
    "PDB_ASSET_UNAVAILABLE": 70,
    "AFDB_ASSET_UNAVAILABLE": 70,
    "SOURCE_NOT_FOUND": 60,
    "HTTP_OR_REMOTE_FAILURE": 50,
    "OTHER_STRUCTURED_ACQUISITION_FAILURE": 40,
    "ACQUIRED": 10,
    "ALREADY_PRESENT_FROZEN": 0,
}


def _aggregate_acquisition_status(
    records: Sequence[AcquisitionRecord],
    *,
    default: str | None,
) -> str | None:
    statuses = [record.retrieval_status for record in records]
    if not statuses:
        return default
    return max(
        statuses,
        key=lambda status: (_ACQUISITION_STATUS_PRECEDENCE.get(status, 30), status),
    )


def build_full_frame_census(
    readiness: pd.DataFrame,
    admission: pd.DataFrame,
    acquisition_records: Sequence[AcquisitionRecord],
    *,
    preexisting_indices: set[int],
    scale1b_v1_ids: set[str],
) -> pd.DataFrame:
    """Combine readiness, transport provenance, and evaluated scientific outcomes."""
    if readiness["sampling_frame_index"].duplicated().any():
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "full_frame_identity_collision",
            "Readiness frame identities are not unique",
        )
    admission_by_index = {
        int(row["sampling_frame_index"]): row
        for row in admission.to_dict(orient="records")
    }
    by_index: dict[int, list[AcquisitionRecord]] = {}
    for record in acquisition_records:
        by_index.setdefault(record.sampling_frame_index, []).append(record)
    output: list[dict[str, Any]] = []
    for ready_row in readiness.to_dict(orient="records"):
        index = int(ready_row["sampling_frame_index"])
        candidate_id = str(ready_row["candidate_id"])
        local_ready = (
            str(ready_row["local_availability_status"])
            == "LOCAL_READY_FOR_SCALE1A"
        )
        scientific = admission_by_index.get(index)
        evaluated = local_ready and scientific is not None
        if local_ready != (scientific is not None):
            raise FullFrameError(
                BLOCKED_REGRESSION,
                "readiness_admission_domain_mismatch",
                f"Readiness/admission domain mismatch for {candidate_id}",
            )
        records = by_index.get(index, [])
        default = "ALREADY_PRESENT_FROZEN" if index in preexisting_indices else None
        pdb_records = [record for record in records if record.asset_type == "pdb_mmcif"]
        afdb_records = [
            record
            for record in records
            if record.asset_type in {"afdb_metadata_collection", "afdb_structure"}
        ]
        canonical_records = [
            record
            for record in records
            if record.asset_type
            in {"afdb_metadata_collection", "uniprot_canonical"}
        ]
        metadata_records_for_candidate = [
            record
            for record in records
            if record.asset_type in {"sifts", "afdb_metadata_collection"}
        ]
        status = str(scientific["admission_status"]) if evaluated else None
        output.append(
            {
                "sampling_frame_index": index,
                "candidate_id": candidate_id,
                "canonical_accession": str(ready_row["canonical_accession"]),
                "pdb_id": str(ready_row["pdb_id"]),
                "pdb_chain": str(ready_row["pdb_chain"]),
                "afdb_id": ready_row.get("afdb_id"),
                "pre_scale1a2_local_status": ready_row.get(
                    "pre_scale1a2_local_status",
                    ready_row["local_availability_status"],
                ),
                "pdb_acquisition_status": _aggregate_acquisition_status(
                    pdb_records, default=default
                ),
                "afdb_acquisition_status": _aggregate_acquisition_status(
                    afdb_records, default=default
                ),
                "canonical_sequence_acquisition_status": (
                    _aggregate_acquisition_status(canonical_records, default=default)
                ),
                "metadata_acquisition_status": _aggregate_acquisition_status(
                    metadata_records_for_candidate, default=default
                ),
                "post_acquisition_local_status": ready_row[
                    "local_availability_status"
                ],
                "terminal_local_missing_reason": ready_row.get(
                    "terminal_local_missing_reason"
                ),
                "formal_admission_evaluated": evaluated,
                "formal_admission_status": status,
                "terminal_scientific_reason": (
                    scientific.get("terminal_reason_code") if evaluated else None
                ),
                "paired_identity": (
                    scientific.get("paired_sequence_identity") if evaluated else None
                ),
                "mismatch_count": (
                    scientific.get("mismatch_count") if evaluated else None
                ),
                "variant_review_required": (
                    status == "PENDING_HUMAN_VARIANT_REVIEW" if evaluated else None
                ),
                "mapping_valid": (
                    bool(pd.notna(scientific.get("mapped_residue_count")))
                    if evaluated
                    else None
                ),
                "common_mask_count": (
                    scientific.get("common_mask_count") if evaluated else None
                ),
                "common_mask_fraction": (
                    scientific.get("common_mask_fraction_of_mapped")
                    if evaluated
                    else None
                ),
                "scale1a1_frozen_member": index in preexisting_indices,
                "scale1b_v1_member": candidate_id in scale1b_v1_ids,
            }
        )
    census = pd.DataFrame(output)
    if len(census) != len(readiness) or census["candidate_id"].duplicated().any():
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "full_frame_identity_collision",
            "Full-frame census identities are not exact",
        )
    return census


def build_attrition_summary(
    census: pd.DataFrame,
    acquisition_records: Sequence[AcquisitionRecord],
    *,
    n_preexisting_ready: int,
) -> dict[str, Any]:
    """Build readiness and admission denominators without conflating failures."""
    source_count = len(census)
    ready = int(
        census["post_acquisition_local_status"]
        .eq("LOCAL_READY_FOR_SCALE1A")
        .sum()
    )
    evaluated = int(census["formal_admission_evaluated"].sum())
    status_counts = Counter(
        census.loc[census["formal_admission_evaluated"], "formal_admission_status"]
    )
    known = {
        "FORMALLY_ADMITTED",
        "PENDING_HUMAN_VARIANT_REVIEW",
        "IDENTITY_CONTRACT_FAIL",
        "MAPPING_FAIL",
        "PROVENANCE_FAIL",
    }
    failure_counts = Counter(
        record.failure_code
        for record in acquisition_records
        if record.retrieval_status not in _SUCCESSFUL_ACQUISITION_STATUSES
        and record.failure_code is not None
    )
    admitted = int(status_counts.get("FORMALLY_ADMITTED", 0))
    return {
        "n_source_frame": source_count,
        "n_preexisting_local_ready": n_preexisting_ready,
        "n_acquisition_targets": source_count - n_preexisting_ready,
        "n_newly_local_ready": ready - n_preexisting_ready,
        "n_remaining_local_incomplete": source_count - ready,
        "n_total_scientifically_evaluated": evaluated,
        "n_formally_admitted_total": admitted,
        "n_pending_human_review_total": int(
            status_counts.get("PENDING_HUMAN_VARIANT_REVIEW", 0)
        ),
        "n_identity_contract_fail_total": int(
            status_counts.get("IDENTITY_CONTRACT_FAIL", 0)
        ),
        "n_mapping_fail_total": int(status_counts.get("MAPPING_FAIL", 0)),
        "n_provenance_fail_total": int(
            status_counts.get("PROVENANCE_FAIL", 0)
        ),
        "n_other_scientific_fail_total": int(
            sum(value for status, value in status_counts.items() if status not in known)
        ),
        "local_readiness_yield": ready / source_count if source_count else None,
        "conditional_scientific_admission_yield": (
            admitted / evaluated if evaluated else None
        ),
        "confirmed_admitted_fraction_of_frozen_frame": (
            admitted / source_count if source_count else None
        ),
        "residual_missingness_present": ready != source_count,
        "acquisition_failure_reason_counts": dict(sorted(failure_counts.items())),
    }


def validate_scale1a1_regression(
    census: pd.DataFrame,
    common_masks: pd.DataFrame,
    paths: ProjectPaths,
    config: FullFrameConfig,
) -> dict[str, Any]:
    """Require candidate and mask equality for the immutable original 72."""
    census_path, _ = _require_hash(
        paths,
        config.scale1a1_census_ref,
        config.expected_scale1a1_census_sha256,
        "Scale-1A1 frozen census",
    )
    masks_path, _ = _require_hash(
        paths,
        config.scale1a1_masks_ref,
        config.expected_scale1a1_masks_sha256,
        "Scale-1A1 frozen common masks",
    )
    frozen_census = pd.read_parquet(census_path).sort_values(
        "sampling_frame_index", kind="stable"
    ).reset_index(drop=True)
    frozen_masks = pd.read_parquet(masks_path).sort_values(
        ["sampling_frame_index", "canonical_position"], kind="stable"
    ).reset_index(drop=True)
    indices = set(frozen_census["sampling_frame_index"].astype(int))
    observed_census = census.loc[
        census["sampling_frame_index"].astype(int).isin(indices),
        frozen_census.columns,
    ].sort_values("sampling_frame_index", kind="stable").reset_index(drop=True)
    observed_masks = common_masks.loc[
        common_masks["sampling_frame_index"].astype(int).isin(indices),
        frozen_masks.columns,
    ].sort_values(
        ["sampling_frame_index", "canonical_position"], kind="stable"
    ).reset_index(drop=True)
    normalized_observed_census = observed_census.astype(object).where(
        observed_census.notna(), None
    )
    normalized_frozen_census = frozen_census.astype(object).where(
        frozen_census.notna(), None
    )
    normalized_observed_masks = observed_masks.astype(object).where(
        observed_masks.notna(), None
    )
    normalized_frozen_masks = frozen_masks.astype(object).where(
        frozen_masks.notna(), None
    )
    try:
        pd.testing.assert_frame_equal(
            normalized_observed_census,
            normalized_frozen_census,
            check_dtype=False,
            check_like=False,
        )
        pd.testing.assert_frame_equal(
            normalized_observed_masks,
            normalized_frozen_masks,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "scale1a1_candidate_regression",
            "Original Scale-1A1 candidate or common-mask facts changed",
        ) from exc
    counts = Counter(frozen_census["admission_status"])
    return {
        "status": "PASS",
        "candidate_count": len(frozen_census),
        "common_mask_row_count": len(frozen_masks),
        "terminal_status_counts": {
            status: int(counts.get(status, 0))
            for status in (
                "FORMALLY_ADMITTED",
                "PENDING_HUMAN_VARIANT_REVIEW",
                "IDENTITY_CONTRACT_FAIL",
                "MAPPING_FAIL",
                "PROVENANCE_FAIL",
            )
        },
    }


_LEDGER_COLUMNS = tuple(AcquisitionRecord.__dataclass_fields__)
_ASSET_SORT_ORDER = {
    "pdb_mmcif": 0,
    "sifts": 1,
    "afdb_metadata_collection": 2,
    "uniprot_canonical": 3,
    "afdb_structure": 4,
}


def acquisition_records_frame(
    records: Sequence[AcquisitionRecord],
) -> pd.DataFrame:
    """Render the canonical acquisition ledger in deterministic source order."""
    frame = pd.DataFrame(
        [
            {
                field_name: getattr(record, field_name)
                for field_name in _LEDGER_COLUMNS
            }
            for record in records
        ],
        columns=_LEDGER_COLUMNS,
    )
    if frame.empty:
        return frame
    frame["_asset_order"] = frame["asset_type"].map(_ASSET_SORT_ORDER)
    return frame.sort_values(
        ["sampling_frame_index", "_asset_order"], kind="stable"
    ).drop(columns="_asset_order").reset_index(drop=True)


def _optional(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def acquisition_records_from_frame(frame: pd.DataFrame) -> tuple[AcquisitionRecord, ...]:
    """Load a prior canonical ledger without changing its immutable provenance."""
    missing = set(_LEDGER_COLUMNS) - set(frame.columns)
    if missing:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "prior_acquisition_ledger_schema_mismatch",
            f"Prior acquisition ledger lacks columns: {sorted(missing)}",
        )
    records: list[AcquisitionRecord] = []
    for row in frame.loc[:, _LEDGER_COLUMNS].to_dict(orient="records"):
        normalized = {key: _optional(value) for key, value in row.items()}
        normalized["sampling_frame_index"] = int(
            normalized["sampling_frame_index"]
        )
        normalized["attempt_count"] = int(normalized["attempt_count"])
        if normalized["file_size"] is not None:
            normalized["file_size"] = int(normalized["file_size"])
        if normalized["http_status"] is not None:
            normalized["http_status"] = int(normalized["http_status"])
        records.append(AcquisitionRecord(**normalized))
    return tuple(records)


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise FullFrameError(
            BLOCKED_UPSTREAM_INTEGRITY,
            "nonportable_output_path",
            f"Output is outside the project root: {path}",
        ) from exc


def _render_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "immutable_scale1a2_output_conflict",
            f"Immutable Scale-1A2 output differs: {path.name}",
        )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False)
        if path.exists():
            if sha256_file(path) == sha256_file(temporary):
                return "reused_identical"
            raise FullFrameError(
                BLOCKED_REGRESSION,
                "immutable_scale1a2_output_conflict",
                f"Immutable Scale-1A2 output differs: {path.name}",
            )
        os.link(temporary, path)
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_frozen_inputs(
    paths: ProjectPaths, inputs: FullFrameInputs
) -> None:
    for artifact in inputs.input_artifacts:
        path = _resolve(paths, str(artifact["path"]))
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise FullFrameError(
                BLOCKED_REGRESSION,
                "frozen_input_drift_during_run",
                f"Frozen input changed during Scale-1A2: {artifact['path']}",
            )


def _not_evaluated_masks(
    readiness: pd.DataFrame,
    evaluated_indices: set[int],
    paths: ProjectPaths,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in readiness.to_dict(orient="records"):
        index = int(row["sampling_frame_index"])
        if index in evaluated_indices or bool(row["missing_canonical_sequence"]):
            continue
        try:
            frames.append(build_not_evaluated_common_mask(row, paths))
        except (OSError, DerivationError, FormalAdmissionError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _scale1b_v1_protein_ids(paths: ProjectPaths, config: FullFrameConfig) -> set[str]:
    table = pd.read_parquet(
        _resolve(
            paths,
            f"{config.scale1b_root_ref}/scale1b_fixed_probe_candidates.parquet",
        ),
        columns=["protein_id"],
    )
    return set(table["protein_id"].astype(str))


def _prior_ledger(
    paths: ProjectPaths, config: FullFrameConfig
) -> tuple[AcquisitionRecord, ...]:
    path = _resolve(
        paths,
        f"{config.output_root_ref}/scale1a2_acquisition_ledger.parquet",
    )
    if not path.is_file():
        return ()
    return acquisition_records_from_frame(pd.read_parquet(path))


def _common_mask_summary(
    census: pd.DataFrame,
) -> dict[str, Any]:
    admitted = census.loc[census["formal_admission_status"].eq("FORMALLY_ADMITTED")]
    result: dict[str, Any] = {"n": len(admitted)}
    for source, target in (
        ("common_mask_count", "common_mask_count"),
        ("common_mask_fraction", "common_mask_fraction"),
    ):
        values = pd.to_numeric(admitted[source], errors="coerce").dropna()
        result[target] = (
            {
                "min": float(values.min()),
                "q10": float(values.quantile(0.10, interpolation="linear")),
                "q25": float(values.quantile(0.25, interpolation="linear")),
                "median": float(values.quantile(0.50, interpolation="linear")),
                "q75": float(values.quantile(0.75, interpolation="linear")),
                "q90": float(values.quantile(0.90, interpolation="linear")),
                "max": float(values.max()),
            }
            if not values.empty
            else None
        )
    return result


def audit_full_frame_completion(
    paths: ProjectPaths,
    *,
    config: FullFrameConfig | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    prior_records: Sequence[AcquisitionRecord] | None = None,
) -> FullFrameResult:
    """Acquire the frozen missing inputs and apply the unchanged A1 contract."""
    config = config or FullFrameConfig()
    inputs = validate_full_frame_inputs(paths, config)
    target_indices = set(
        inputs.acquisition_targets["sampling_frame_index"].astype(int)
    )
    prior = (
        _prior_ledger(paths, config)
        if prior_records is None
        else tuple(prior_records)
    )

    initial_requirements = resolve_asset_requirements(inputs, paths)
    initial_records = execute_initial_acquisition(
        initial_requirements,
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
    )
    interim = recompute_full_frame_readiness(
        inputs.source_frame, initial_records, paths
    )
    uniprot_requirements = resolve_uniprot_requirements(
        interim, target_indices, paths
    )
    uniprot_records = execute_initial_acquisition(
        uniprot_requirements,
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
    )
    interim_records = (*initial_records, *uniprot_records)
    interim = recompute_full_frame_readiness(
        inputs.source_frame, interim_records, paths
    )
    structure_requirements, dependency_failures = (
        resolve_afdb_structure_requirements(
            interim,
            target_indices,
            paths,
            clock=clock,
            prior_records=prior,
        )
    )
    structure_records = execute_initial_acquisition(
        structure_requirements,
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
    )
    records = (*interim_records, *structure_records, *dependency_failures)
    readiness = recompute_full_frame_readiness(inputs.source_frame, records, paths)
    ready = readiness.loc[
        readiness["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ].copy()

    admission_config = FormalAdmissionConfig()
    stage0_gate = validate_stage0_admission_ledger(paths, admission_config)
    try:
        evaluated = evaluate_formal_admission_frame(
            ready,
            paths=paths,
            config=admission_config,
            stage0_gate=stage0_gate,
        )
    except FormalAdmissionError as exc:
        raise FullFrameError(
            BLOCKED_REGRESSION,
            getattr(exc, "code", "formal_admission_regression"),
            str(exc),
        ) from exc
    if evaluated.stage0_regression["status"] != "PASS":
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "stage0_admission_regression",
            "Scale-1A2 disagrees with the frozen Stage-0 ledger",
            disagreements=evaluated.stage0_regression["disagreements"],
        )
    scale1a1_regression = validate_scale1a1_regression(
        evaluated.census, evaluated.common_masks, paths, config
    )

    evaluated_indices = set(evaluated.census["sampling_frame_index"].astype(int))
    extra_masks = _not_evaluated_masks(readiness, evaluated_indices, paths)
    mask_frames = [evaluated.common_masks]
    if not extra_masks.empty:
        mask_frames.append(extra_masks)
    common_masks = pd.concat(mask_frames, ignore_index=True).sort_values(
        ["sampling_frame_index", "canonical_position"], kind="stable"
    ).reset_index(drop=True)
    if common_masks.duplicated(
        ["sampling_frame_index", "canonical_position"]
    ).any():
        raise FullFrameError(
            BLOCKED_REGRESSION,
            "common_mask_identity_collision",
            "Full-frame common masks are not unique",
        )

    census = build_full_frame_census(
        readiness,
        evaluated.census,
        records,
        preexisting_indices=set(
            inputs.preexisting_local_ready["sampling_frame_index"].astype(int)
        ),
        scale1b_v1_ids=_scale1b_v1_protein_ids(paths, config),
    )
    ledger = acquisition_records_frame(records)
    summary = build_attrition_summary(
        census,
        records,
        n_preexisting_ready=config.expected_preexisting_ready,
    )
    status = (
        FULL_FRAME_COMPLETE
        if summary["n_remaining_local_incomplete"] == 0
        else COMPLETE_WITH_RESIDUAL_FAILURES
    )
    summary.update(
        {
            "schema_version": "scale1a2-full-frame-v1",
            "scale1a2_status": status,
            "initial_status_counts": inputs.initial_status_counts,
            "final_local_status_counts": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(census["post_acquisition_local_status"]).items()
                )
            },
            "formal_admission_status_counts": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(
                        census.loc[
                            census["formal_admission_evaluated"],
                            "formal_admission_status",
                        ]
                    ).items()
                )
            },
            "scale1a1_regression": scale1a1_regression,
            "stage0_regression": evaluated.stage0_regression,
            "common_mask_summary_formally_admitted": _common_mask_summary(census),
            "residual_missingness_interpretation": (
                "Confirmed admitted fraction is not a biological admission probability "
                "while residual acquisition missingness remains."
                if status == COMPLETE_WITH_RESIDUAL_FAILURES
                else "No residual local acquisition missingness remains."
            ),
            "input_artifacts": list(inputs.input_artifacts),
            "forbidden_execution": {
                "proteinmpnn_forward_executions": 0,
                "scale1b_v1_modified": False,
                "scale1a3_started": False,
            },
        }
    )
    _rehash_frozen_inputs(paths, inputs)
    return FullFrameResult(
        status=status,
        acquisition_ledger=ledger,
        census=census,
        common_masks=common_masks,
        attrition_summary=summary,
    )


def materialize_full_frame_completion(
    result: FullFrameResult,
    paths: ProjectPaths,
    *,
    config: FullFrameConfig | None = None,
) -> dict[str, Any]:
    """Materialize the three tables, then the canonical summary, immutably."""
    config = config or FullFrameConfig()
    root = _resolve(paths, config.output_root_ref)
    root.mkdir(parents=True, exist_ok=True)
    targets = {
        "acquisition_ledger": root / "scale1a2_acquisition_ledger.parquet",
        "full_frame_census": root / "scale1_full_frame_census.parquet",
        "full_frame_common_masks": root / "scale1_full_frame_common_masks.parquet",
        "attrition_summary": root / "scale1_full_frame_attrition_summary.json",
    }
    write_status = {
        "acquisition_ledger": _write_immutable_parquet(
            targets["acquisition_ledger"], result.acquisition_ledger
        ),
        "full_frame_census": _write_immutable_parquet(
            targets["full_frame_census"], result.census
        ),
        "full_frame_common_masks": _write_immutable_parquet(
            targets["full_frame_common_masks"], result.common_masks
        ),
    }
    output_artifacts = {
        key: {
            "path": _logical(paths, targets[key]),
            "rows": len(frame),
            "sha256": sha256_file(targets[key]),
        }
        for key, frame in (
            ("acquisition_ledger", result.acquisition_ledger),
            ("full_frame_census", result.census),
            ("full_frame_common_masks", result.common_masks),
        )
    }
    summary = {
        **result.attrition_summary,
        "output_artifacts": output_artifacts,
        "summary_path": _logical(paths, targets["attrition_summary"]),
    }
    write_status["attrition_summary"] = _write_immutable_bytes(
        targets["attrition_summary"], _render_json(summary)
    )
    return {
        "scale1a2_status": result.status,
        "write_status": write_status,
        "outputs": {
            **output_artifacts,
            "attrition_summary": {
                "path": _logical(paths, targets["attrition_summary"]),
                "sha256": sha256_file(targets["attrition_summary"]),
            },
        },
    }


def run_full_frame_completion(
    paths: ProjectPaths,
    *,
    config: FullFrameConfig | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run the frozen Scale-1A2 acquisition and census workflow."""
    config = config or FullFrameConfig()
    result = audit_full_frame_completion(
        paths,
        config=config,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
    )
    return materialize_full_frame_completion(
        result, paths, config=config
    )
