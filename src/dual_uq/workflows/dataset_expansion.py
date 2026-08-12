"""Prospective cluster-coverage expansion workflow for a frozen dataset release."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.admission import (
    FORMALLY_ADMITTED,
    IDENTITY_CONTRACT_FAIL,
    PENDING_HUMAN_VARIANT_REVIEW,
    FormalAdmissionConfig,
    FormalAdmissionError,
    evaluate_admission_candidates,
    validate_stage0_admission_ledger,
)
from dual_uq.dataset.completion import (
    AcquisitionRecord,
    AssetRequirement,
    FullFrameInputs,
    acquisition_records_frame,
    acquisition_records_from_frame,
    execute_initial_acquisition,
    recompute_full_frame_readiness,
    resolve_afdb_structure_requirements,
    resolve_asset_requirements,
    resolve_uniprot_requirements,
)
from dual_uq.dataset.stages.acquisition import (
    TransportResponse,
    default_transport,
    utc_now,
)

BLOCKED_INPUT_INTEGRITY = "BLOCKED_INPUT_INTEGRITY"
PRIMARY_CONFIRMATORY_CAPACITY_REACHED = "PRIMARY_CONFIRMATORY_CAPACITY_REACHED"
CORE_REACHED_PRIMARY_TARGET_PENDING = "CORE_REACHED_PRIMARY_TARGET_PENDING"
WAVE2_REQUIRED_CORE_NOT_REACHED = "WAVE2_REQUIRED_CORE_NOT_REACHED"
PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED = (
    "PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED"
)


class DatasetExpansionError(RuntimeError):
    """One structured expansion-workflow blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class DatasetExpansionConfig:
    """Portable bindings for the frozen Scale-1 Wave-1 execution."""

    e0_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_design/"
        "scale1_expansion_design_manifest.json"
    )
    expected_e0_manifest_sha256: str = (
        "ead5184664f908c531e6ae316db923f9615bf3794f42734e24685a2032c23e99"
    )
    e0_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_design/"
        "scale1_expansion_design_summary.json"
    )
    expected_e0_summary_sha256: str = (
        "efcc39e9eef3bc04609777de27d9a59bfa50c10b887d2be71a621a1272e9daab"
    )
    e0_targets_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_design/"
        "scale1_expansion_cluster_targets.parquet"
    )
    expected_e0_targets_sha256: str = (
        "d036f64b96a1f793d6d9ee8a8098ffbe4b8b085c9bdf46a2a902455db6595218"
    )
    scale1a2_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_attrition_summary.json"
    )
    expected_scale1a2_summary_sha256: str = (
        "36ee351614ed3e0f9ba57161bbdc2a4410acc3ca5cb5ad8d1b2de54427ae7002"
    )
    scale1a2_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_census.parquet"
    )
    expected_scale1a2_census_sha256: str = (
        "fc69e13d3d94def1a751b5528230fa5f15dd1ae741d8361b4187236167120793"
    )
    scale1a2_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_common_masks.parquet"
    )
    expected_scale1a2_masks_sha256: str = (
        "4a6d89a9368c3349f0598616e3dda670b17164fe6dff7fd37e131a71e4acaae0"
    )
    scale1a2_ledger_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1a2_acquisition_ledger.parquet"
    )
    expected_scale1a2_ledger_sha256: str = (
        "a6310505eb8ba397b835ccaa9fb5fece2ed6309792d6af07368bfa517959869f"
    )
    scale1a3_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/scale1a3_manifest.json"
    )
    expected_scale1a3_manifest_sha256: str = (
        "eac4e19e6645d3aa6935e17bf8f25d1116d32a67dd8d3a4c78e68bc93d680d56"
    )
    scale1a3_capacity_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/"
        "scale1a3_nonredundant_capacity.parquet"
    )
    expected_scale1a3_capacity_sha256: str = (
        "203125ef114368c015ad86edf7f3802c05fa33085fde280befe955df537e91d9"
    )
    scale1b_cohort_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_freeze_manifest.json"
    )
    expected_scale1b_cohort_manifest_sha256: str = (
        "275f7373f4f5de9b0a1f12c2b5585bc7ef91e88d11dc2b61b387326798e6e730"
    )
    scale1b_protocol_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_protocol_freeze_manifest.json"
    )
    expected_scale1b_protocol_manifest_sha256: str = (
        "485a22a89d0fc532cdee72c09ed7faa36c5fdc54079aa2a971479d41c53af918"
    )
    output_root_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1"
    )
    original_recommended_wave1_n: int = 83
    frozen_wave1_primary_n: int = 100
    starting_admitted_clusters: int = 63
    minimum_core_gate: int = 100
    primary_confirmatory_target: int = 120


@dataclass(frozen=True)
class DatasetExpansionInputs:
    e0_manifest: dict[str, Any]
    e0_summary: dict[str, Any]
    targets: pd.DataFrame
    original_admitted_clusters: frozenset[str]
    input_artifacts: tuple[dict[str, Any], ...]
    config: DatasetExpansionConfig


@dataclass(frozen=True)
class DatasetExpansionFreeze:
    amendment: dict[str, Any]
    amendment_sha256: str
    declaration: tuple[dict[str, Any], ...]
    declaration_sha256: str
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetExpansionRunResult:
    """Single canonical result consumed by every Wave-1 renderer."""

    verdict: str
    capacity_status: str
    acquisition_ledger: pd.DataFrame
    readiness: pd.DataFrame
    admission: pd.DataFrame
    common_masks: pd.DataFrame
    cluster_capacity: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    frozen: DatasetExpansionFreeze
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetExpansionReadinessResult:
    records: tuple[AcquisitionRecord, ...]
    acquisition_ledger: pd.DataFrame
    readiness: pd.DataFrame


@dataclass(frozen=True)
class DatasetExpansionWave2Config:
    """Portable frozen bindings for Scale-1 expansion Wave-2."""

    base_config: DatasetExpansionConfig = field(default_factory=DatasetExpansionConfig)
    wave1_amendment_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1e0a_wave1_amendment.json"
    )
    expected_wave1_amendment_sha256: str = (
        "db2ea53d5ad6cffa72b4d9c5cb253aab6cf3d3f1c8f59d8fdecce0a26b4ea576"
    )
    wave1_declaration_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_candidates.jsonl"
    )
    expected_wave1_declaration_sha256: str = (
        "056bd17536b4538ee03088a4fe3993958f8774d4b0dc7fb7e4dc23ad7e5b0372"
    )
    wave1_acquisition_ledger_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_acquisition_ledger.parquet"
    )
    expected_wave1_acquisition_ledger_sha256: str = (
        "e2f1f5b234845a32261b29f3709a52e7ac796e2f6e690eb8e9117b8e85541fb4"
    )
    wave1_readiness_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_readiness.parquet"
    )
    expected_wave1_readiness_sha256: str = (
        "ac55fc0a3d9e8c9473e9aa03355590a332d48f4aab101bcb204658e8429ea7de"
    )
    wave1_admission_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_admission.parquet"
    )
    expected_wave1_admission_sha256: str = (
        "2d627182e1342bb84d66317fdfe556edf2cf115a790147ffe884cb5d0ddb13de"
    )
    wave1_common_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_common_masks.parquet"
    )
    expected_wave1_common_masks_sha256: str = (
        "56a66bd8ff1185876846918552bfa174c5b31467b268b32f814e18f2e8aa7a5a"
    )
    wave1_cluster_capacity_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_cluster_capacity.parquet"
    )
    expected_wave1_cluster_capacity_sha256: str = (
        "e29767e4fad7d4d7775f698a276d662b79ed23a1d41daa49264efd5ab11438bf"
    )
    wave1_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_summary.json"
    )
    expected_wave1_summary_sha256: str = (
        "784d15487928d6a35e9ce6dbd015c26dc0f2ce87b316e2d0a254c61df9426308"
    )
    wave1_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1_expansion_wave1_manifest.json"
    )
    expected_wave1_manifest_sha256: str = (
        "cb1ea3e1cd42a1cdc79fd972e2305570a76afef65f9ec18b6530f017286f3ce2"
    )
    output_root_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave2"
    )
    wave2_target_cluster_n: int = 24
    starting_admitted_clusters: int = 112
    primary_confirmatory_target: int = 120


@dataclass(frozen=True)
class DatasetExpansionWave2Inputs:
    """SHA-gated trust roots and exact remaining target set for Wave-2."""

    base_inputs: DatasetExpansionInputs
    wave1_summary: dict[str, Any]
    remaining_targets: pd.DataFrame
    clusters_before: frozenset[str]
    input_artifacts: tuple[dict[str, Any], ...]
    config: DatasetExpansionWave2Config


@dataclass(frozen=True)
class DatasetExpansionWave2Freeze:
    """Prospectively frozen 24-primary Wave-2 declaration."""

    declaration: tuple[dict[str, Any], ...]
    declaration_sha256: str
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetExpansionWave2ReadinessResult:
    """Declaration-bound acquisition records and readiness outcomes."""

    records: tuple[AcquisitionRecord, ...]
    acquisition_ledger: pd.DataFrame
    readiness: pd.DataFrame


@dataclass(frozen=True)
class DatasetExpansionWave2RunResult:
    """Single canonical source consumed by every Wave-2 renderer."""

    verdict: str
    capacity_status: str
    acquisition_ledger: pd.DataFrame
    readiness: pd.DataFrame
    admission: pd.DataFrame
    common_masks: pd.DataFrame
    cluster_capacity: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    frozen: DatasetExpansionWave2Freeze
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_path",
            f"Invalid portable path: {logical_ref}",
        ) from exc


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_path",
            f"Output is outside declared roots: {path.name}",
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            path=logical_ref,
            expected_sha256=expected,
            observed_sha256=observed,
        )
    return path, {"label": label, "path": logical_ref, "sha256": observed}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must be a JSON object",
        )
    return payload


def validate_expansion_inputs(
    paths: ProjectPaths, config: DatasetExpansionConfig | None = None
) -> DatasetExpansionInputs:
    """SHA-gate all frozen E0/A2/A3/B-v1 inputs before declaration."""
    config = config or DatasetExpansionConfig()
    bindings = (
        (config.e0_manifest_ref, config.expected_e0_manifest_sha256, "Scale-1E0 manifest"),
        (config.e0_summary_ref, config.expected_e0_summary_sha256, "Scale-1E0 summary"),
        (config.e0_targets_ref, config.expected_e0_targets_sha256, "Scale-1E0 targets"),
        (config.scale1a2_summary_ref, config.expected_scale1a2_summary_sha256, "Scale-1A2 summary"),
        (config.scale1a2_census_ref, config.expected_scale1a2_census_sha256, "Scale-1A2 census"),
        (config.scale1a2_masks_ref, config.expected_scale1a2_masks_sha256, "Scale-1A2 masks"),
        (config.scale1a2_ledger_ref, config.expected_scale1a2_ledger_sha256, "Scale-1A2 acquisition ledger"),
        (config.scale1a3_manifest_ref, config.expected_scale1a3_manifest_sha256, "Scale-1A3 manifest"),
        (config.scale1a3_capacity_ref, config.expected_scale1a3_capacity_sha256, "Scale-1A3 capacity"),
        (config.scale1b_cohort_manifest_ref, config.expected_scale1b_cohort_manifest_sha256, "Scale-1B-v1 cohort manifest"),
        (config.scale1b_protocol_manifest_ref, config.expected_scale1b_protocol_manifest_sha256, "Scale-1B-v1 protocol manifest"),
    )
    paths_by_ref: dict[str, Path] = {}
    artifacts: list[dict[str, Any]] = []
    for logical_ref, expected, label in bindings:
        path, artifact = _require_hash(paths, logical_ref, expected, label)
        paths_by_ref[logical_ref] = path
        artifacts.append(artifact)

    e0_manifest = _read_json(paths_by_ref[config.e0_manifest_ref], "Scale-1E0 manifest")
    e0_summary = _read_json(paths_by_ref[config.e0_summary_ref], "Scale-1E0 summary")
    a2_summary = _read_json(paths_by_ref[config.scale1a2_summary_ref], "Scale-1A2 summary")
    a3_manifest = _read_json(paths_by_ref[config.scale1a3_manifest_ref], "Scale-1A3 manifest")
    b_cohort = _read_json(paths_by_ref[config.scale1b_cohort_manifest_ref], "Scale-1B-v1 cohort manifest")
    b_protocol = _read_json(paths_by_ref[config.scale1b_protocol_manifest_ref], "Scale-1B-v1 protocol manifest")
    try:
        targets = pd.read_parquet(paths_by_ref[config.e0_targets_ref])
        capacity = pd.read_parquet(paths_by_ref[config.scale1a3_capacity_ref])
    except (OSError, ValueError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            "Unable to read frozen E0/A3 Parquet",
        ) from exc

    current = e0_summary.get("current_capacity", {})
    universe = e0_summary.get("expansion_universe", {})
    expected_state = {
        "formally_admitted_proteins": 135,
        "admitted_30pct_clusters": 63,
        "pending_proteins": 51,
        "pending_new_clusters": 15,
    }
    if (
        e0_manifest.get("scale1_expansion_design_status")
        != "SCALE1_EXPANSION_DESIGN_READY"
        or e0_summary.get("scale1_expansion_design_status")
        != "SCALE1_EXPANSION_DESIGN_READY"
        or any(current.get(key) != value for key, value in expected_state.items())
        or universe.get("n_actionable_target_clusters") != 124
        or universe.get("n_new_external_clusters") != 100
        or universe.get("n_unadmitted_existing_clusters") != 24
        or universe.get("n_primary_candidates") != 124
        or universe.get("n_total_reserve_candidates") != 123
        or a2_summary.get("n_source_frame") != 213
        or a2_summary.get("n_formally_admitted_total") != 135
        or a3_manifest.get("capacity_statistics", {}).get("n_nr_admitted") != 63
        or b_cohort.get("freeze_status") != "SCALE1B_COHORT_FREEZE_COMPLETE"
        or b_protocol.get("protocol_status") != "SCALE1B_PROTOCOL_FREEZE_COMPLETE"
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_state_mismatch",
            "Frozen E0/A2/A3/B-v1 state does not match the approved Wave-1 input",
        )
    required = {
        "sequence_cluster",
        "primary_pair_id",
        "primary_candidate_id",
        "primary_polymer_entity_id",
        "primary_canonical_accession",
        "primary_pdb_id",
        "primary_pdb_chain",
        "primary_candidate_origin",
        "prospective_global_primary_rank",
        "selection_provenance",
        "reserve_candidate_count",
        "target_source_category",
        "target_class",
    }
    ranks = targets.get("prospective_global_primary_rank")
    if (
        not required.issubset(targets.columns)
        or len(targets) != 124
        or targets["sequence_cluster"].duplicated().any()
        or targets["primary_pair_id"].duplicated().any()
        or ranks is None
        or ranks.astype(int).tolist() != list(range(1, 125))
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_target_order_mismatch",
            "Persisted E0 target identities/order are not exact",
        )
    original_clusters = frozenset(capacity["redundancy_cluster_id"].astype(str))
    if len(original_clusters) != 63 or set(targets["sequence_cluster"]).intersection(
        original_clusters
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "target_cluster_overlap",
            "E0 targets overlap the frozen original admitted clusters",
        )
    manifest_target = e0_manifest.get("outputs", {}).get("cluster_targets", {})
    if (
        manifest_target.get("path") != config.e0_targets_ref
        or manifest_target.get("sha256") != config.expected_e0_targets_sha256
        or manifest_target.get("rows") != 124
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "manifest_binding_mismatch",
            "Scale-1E0 manifest does not bind the frozen target table",
        )
    return DatasetExpansionInputs(
        e0_manifest=e0_manifest,
        e0_summary=e0_summary,
        targets=targets.reset_index(drop=True),
        original_admitted_clusters=original_clusters,
        input_artifacts=tuple(artifacts),
        config=config,
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )
    ).encode("utf-8")


def build_expansion_freeze(inputs: DatasetExpansionInputs) -> DatasetExpansionFreeze:
    """Build the size amendment and first-N persisted-rank declaration in memory."""
    config = inputs.config
    amendment = {
        "schema_version": "dual-uq.scale1e0a-wave1-amendment.v1",
        "amendment_status": "FROZEN",
        "original_recommended_wave1_n": config.original_recommended_wave1_n,
        "frozen_wave1_primary_n": config.frozen_wave1_primary_n,
        "reason": "ADD_PRE_OUTCOME_ACQUISITION_AND_ADMISSION_ATTRITION_BUFFER",
        "changes_wave1_size_only": True,
        "target_classes_unchanged": True,
        "target_ordering_unchanged": True,
        "candidate_ranking_unchanged": True,
        "admission_contract_unchanged": True,
        "redundancy_contract_unchanged": True,
        "wave1_reserve_activation_authorized": False,
        "e0_manifest_sha256": config.expected_e0_manifest_sha256,
    }
    amendment_sha = hashlib.sha256(_json_bytes(amendment)).hexdigest()
    selected = inputs.targets.iloc[: config.frozen_wave1_primary_n]
    records: list[dict[str, Any]] = []
    for wave_index, row in enumerate(selected.to_dict(orient="records"), 1):
        records.append(
            {
                "schema_version": "dual-uq.scale1-expansion-wave1-candidate.v1",
                "wave1_candidate_index": wave_index,
                "sequence_cluster": str(row["sequence_cluster"]),
                "target_class": str(row["target_class"]),
                "pair_id": str(row["primary_pair_id"]),
                "candidate_id": str(row["primary_candidate_id"]),
                "polymer_entity_id": str(row["primary_polymer_entity_id"]),
                "canonical_accession": str(row["primary_canonical_accession"]),
                "pdb_id": str(row["primary_pdb_id"]),
                "pdb_chain": str(row["primary_pdb_chain"]),
                "candidate_origin": str(row["primary_candidate_origin"]),
                "prospective_global_primary_rank": int(
                    row["prospective_global_primary_rank"]
                ),
                "selection_provenance": str(row["selection_provenance"]),
                "target_source_category": str(row["target_source_category"]),
                "reserve_candidate_count": int(row["reserve_candidate_count"]),
                "selection_role": "PRIMARY_CANDIDATE",
                "reserve_activated": False,
                "amendment_sha256": amendment_sha,
            }
        )
    if (
        len(records) != config.frozen_wave1_primary_n
        or len({record["sequence_cluster"] for record in records}) != len(records)
        or len({record["pair_id"] for record in records}) != len(records)
        or [record["prospective_global_primary_rank"] for record in records]
        != list(range(1, config.frozen_wave1_primary_n + 1))
        or {record["sequence_cluster"] for record in records}.intersection(
            inputs.original_admitted_clusters
        )
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_wave1_declaration",
            "Wave-1 declaration is not the first 100 unique persisted E0 targets",
        )
    declaration = tuple(records)
    return DatasetExpansionFreeze(
        amendment=amendment,
        amendment_sha256=amendment_sha,
        declaration=declaration,
        declaration_sha256=hashlib.sha256(_jsonl_bytes(declaration)).hexdigest(),
        input_artifacts=inputs.input_artifacts,
    )


def _write_immutable(path: Path, payload: bytes) -> str:
    if path.is_file():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "immutable_output_conflict",
            f"Immutable expansion output differs: {path.name}",
        )
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "immutable_output_conflict",
            f"Expansion output appeared concurrently: {path.name}",
        ) from exc
    return "created"


def materialize_expansion_freeze(
    result: DatasetExpansionFreeze,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionConfig | None = None,
) -> dict[str, Any]:
    """Write amendment before declaration and verify the declaration hash."""
    config = config or DatasetExpansionConfig()
    root = _resolve(paths, config.output_root_ref)
    amendment_path = root / "scale1e0a_wave1_amendment.json"
    declaration_path = root / "scale1_expansion_wave1_candidates.jsonl"
    amendment_payload = _json_bytes(result.amendment)
    declaration_payload = _jsonl_bytes(result.declaration)
    if hashlib.sha256(amendment_payload).hexdigest() != result.amendment_sha256:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "amendment_hash_mismatch",
            "In-memory E0a amendment changed before materialization",
        )
    amendment_status = _write_immutable(amendment_path, amendment_payload)
    declaration_status = _write_immutable(declaration_path, declaration_payload)
    observed = sha256_file(declaration_path)
    if observed != result.declaration_sha256:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "declaration_hash_mismatch",
            "Frozen Wave-1 declaration SHA256 does not match",
        )
    return {
        "write_status": {
            "amendment": amendment_status,
            "declaration": declaration_status,
        },
        "amendment": {
            "path": _logical(paths, amendment_path),
            "sha256": result.amendment_sha256,
        },
        "declaration": {
            "path": _logical(paths, declaration_path),
            "rows": len(result.declaration),
            "sha256": observed,
        },
    }


def validate_frozen_declaration(
    frozen: DatasetExpansionFreeze,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionConfig | None = None,
) -> None:
    """Fail before acquisition if either prospectively frozen file drifted."""
    config = config or DatasetExpansionConfig()
    root = _resolve(paths, config.output_root_ref)
    checks = (
        (
            root / "scale1e0a_wave1_amendment.json",
            frozen.amendment_sha256,
            "amendment_hash_mismatch",
        ),
        (
            root / "scale1_expansion_wave1_candidates.jsonl",
            frozen.declaration_sha256,
            "declaration_hash_mismatch",
        ),
    )
    for path, expected, code in checks:
        if not path.is_file() or sha256_file(path) != expected:
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                code,
                f"Frozen prospective artifact changed: {path.name}",
            )


def build_initial_readiness_frame(
    declaration: Sequence[Mapping[str, Any]],
    *,
    declaration_ref: str,
    declaration_sha256: str,
) -> pd.DataFrame:
    """Create a conservative pre-acquisition frame without claiming absent evidence."""
    rows: list[dict[str, Any]] = []
    for record in declaration:
        index = int(record["wave1_candidate_index"])
        rows.append(
            {
                "candidate_id": str(record["candidate_id"]),
                "sampling_frame_index": index,
                "canonical_source_row": index,
                "polymer_entity_id": str(record["polymer_entity_id"]),
                "pair_id": str(record["pair_id"]),
                "canonical_accession": str(record["canonical_accession"]),
                "pdb_id": str(record["pdb_id"]).lower(),
                "pdb_chain": str(record["pdb_chain"]),
                "afdb_id": None,
                "sampling_frame_source": declaration_ref,
                "pdb_file_present": False,
                "pdb_file_path_relative": None,
                "pdb_file_readable": False,
                "pdb_chain_identifiable": False,
                "pdb_entity_identifiable": False,
                "afdb_file_present": False,
                "afdb_file_path_relative": None,
                "afdb_file_readable": False,
                "afdb_accession_identifiable": False,
                "canonical_accession_present": True,
                "canonical_sequence_present": False,
                "canonical_sequence_source_relative": None,
                "canonical_sequence_sha256": None,
                "identity_metadata_present": True,
                "identity_metadata_source_relative": declaration_ref,
                "mapping_metadata_present": False,
                "mapping_metadata_source_relative": None,
                "provenance_metadata_present": True,
                "provenance_metadata_source_relative": declaration_ref,
                "local_pair_complete": False,
                "local_metadata_complete": False,
                "missing_pdb": True,
                "missing_afdb": True,
                "missing_canonical_sequence": True,
                "missing_identity_metadata": False,
                "missing_mapping_metadata": True,
                "missing_provenance_metadata": False,
                "ambiguous_local_source": False,
                "unreadable_local_file": False,
                "other_structured_local_blocker": False,
                "local_availability_status": "MISSING_PDB_AND_AFDB",
                "terminal_local_missing_reason": "pdb_and_afdb_structures_absent",
                "stage0_declared_member": False,
                "stage0_admitted_member": False,
                "sequence_cluster": str(record["sequence_cluster"]),
                "target_class": str(record["target_class"]),
                "candidate_origin": str(record["candidate_origin"]),
                "acquisition_bound_declaration_sha256": declaration_sha256,
            }
        )
    frame = pd.DataFrame(rows)
    if (
        len(frame) != len(declaration)
        or frame["candidate_id"].duplicated().any()
        or frame["sampling_frame_index"].tolist()
        != list(range(1, len(declaration) + 1))
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "declaration_identity_mismatch",
            "Wave-1 readiness frame does not preserve declaration identity/order",
        )
    return frame


def build_initial_asset_requirements(
    frame: pd.DataFrame, paths: ProjectPaths
) -> tuple[AssetRequirement, ...]:
    """Reuse the canonical acquisition planner for the frozen expansion frame."""
    inputs = FullFrameInputs(
        inventory={},
        source_frame=frame,
        preexisting_local_ready=frame.iloc[0:0].copy(),
        acquisition_targets=frame,
        initial_status_counts={"MISSING_PDB_AND_AFDB": len(frame)},
    )
    return resolve_asset_requirements(inputs, paths)


def execute_expansion_readiness(
    *,
    inputs: DatasetExpansionInputs,
    frozen: DatasetExpansionFreeze,
    paths: ProjectPaths,
    config: DatasetExpansionConfig | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    prior_records: Sequence[AcquisitionRecord] | None = None,
) -> DatasetExpansionReadinessResult:
    """Acquire/reuse declaration-bound inputs, then resolve readiness exactly once."""
    config = config or inputs.config
    validate_frozen_declaration(frozen, paths, config=config)
    _rehash_inputs(paths, inputs.input_artifacts)
    prior = (
        load_prior_acquisition_records(
            paths, frozen.declaration_sha256, config=config
        )
        if prior_records is None
        else tuple(prior_records)
    )
    declaration_ref = (
        f"{config.output_root_ref}/scale1_expansion_wave1_candidates.jsonl"
    )
    source = build_initial_readiness_frame(
        frozen.declaration,
        declaration_ref=declaration_ref,
        declaration_sha256=frozen.declaration_sha256,
    )
    target_indices = set(source["sampling_frame_index"].astype(int))
    rejected_ref = (
        "artifacts/dataset/audits/acquisition_rejected/"
        f"scale1-expansion-wave1/{frozen.declaration_sha256}"
    )
    initial = execute_initial_acquisition(
        build_initial_asset_requirements(source, paths),
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    interim = recompute_full_frame_readiness(source, initial, paths)
    canonical = execute_initial_acquisition(
        resolve_uniprot_requirements(interim, target_indices, paths),
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    interim_records = (*initial, *canonical)
    interim = recompute_full_frame_readiness(source, interim_records, paths)
    structure_requirements, dependency_failures = (
        resolve_afdb_structure_requirements(
            interim,
            target_indices,
            paths,
            clock=clock,
            prior_records=prior,
        )
    )
    structures = execute_initial_acquisition(
        structure_requirements,
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    records = (*interim_records, *structures, *dependency_failures)
    readiness = recompute_full_frame_readiness(source, records, paths)
    if (
        len(readiness) != len(frozen.declaration)
        or readiness["candidate_id"].duplicated().any()
        or set(readiness["candidate_id"])
        != {str(record["candidate_id"]) for record in frozen.declaration}
        or readiness["local_availability_status"].isna().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_identity_mismatch",
            "Wave-1 readiness outcomes are not declaration-complete",
        )
    ledger = bind_acquisition_ledger(
        acquisition_records_frame(records), frozen.declaration_sha256
    )
    _rehash_inputs(paths, inputs.input_artifacts)
    validate_frozen_declaration(frozen, paths, config=config)
    return DatasetExpansionReadinessResult(
        records=tuple(records),
        acquisition_ledger=ledger,
        readiness=readiness.reset_index(drop=True),
    )


def load_prior_acquisition_records(
    paths: ProjectPaths,
    declaration_sha256: str,
    *,
    config: DatasetExpansionConfig | None = None,
) -> tuple[AcquisitionRecord, ...]:
    """Load an immutable prior ledger only when its declaration binding is exact."""
    config = config or DatasetExpansionConfig()
    path = _resolve(
        paths,
        f"{config.output_root_ref}/"
        "scale1_expansion_wave1_acquisition_ledger.parquet",
    )
    if not path.is_file():
        return ()
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_ledger_unreadable",
            "Prior Wave-1 acquisition ledger is unreadable",
        ) from exc
    binding = frame.get("acquisition_bound_declaration_sha256")
    if (
        binding is None
        or len(binding) != len(frame)
        or not binding.eq(declaration_sha256).all()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_declaration_mismatch",
            "Prior acquisition ledger belongs to another declaration",
        )
    try:
        return acquisition_records_from_frame(frame)
    except Exception as exc:  # canonical loader provides structured detail
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_ledger_schema_mismatch",
            "Prior Wave-1 acquisition ledger schema is invalid",
        ) from exc


def evaluate_expansion_admission(
    readiness: pd.DataFrame,
    *,
    paths: ProjectPaths,
    config: FormalAdmissionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the unchanged candidate evaluator only to locally ready records."""
    config = config or FormalAdmissionConfig()
    ready = readiness.loc[
        readiness["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ].copy()
    gate = validate_stage0_admission_ledger(paths, config)
    try:
        evaluated = evaluate_admission_candidates(
            ready, paths=paths, config=config, stage0_gate=gate
        )
    except FormalAdmissionError as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            getattr(exc, "code", "formal_admission_failure"),
            str(exc),
        ) from exc
    expected_stage0 = {
        str(record["protein_id"]): {
            "ADMITTED": FORMALLY_ADMITTED,
            "PENDING_HUMAN_VARIANT_REVIEW": PENDING_HUMAN_VARIANT_REVIEW,
            "IDENTITY_CONTRACT_FAIL": IDENTITY_CONTRACT_FAIL,
        }[str(record["intervention_admission_status"])]
        for record in gate.records
    }
    observed = evaluated.census.set_index("pair_id")["admission_status"].to_dict()
    disagreements = [
        {"pair_id": pair_id, "expected": expected, "observed": observed[pair_id]}
        for pair_id, expected in expected_stage0.items()
        if pair_id in observed and observed[pair_id] != expected
    ]
    if disagreements:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "stage0_overlap_admission_regression",
            "Wave-1 overlap disagrees with the frozen Stage-0 admission ledger",
            disagreements=disagreements,
        )
    admitted_indices = set(
        evaluated.census.loc[
            evaluated.census["admission_status"].eq(FORMALLY_ADMITTED),
            "sampling_frame_index",
        ].astype(int)
    )
    masks = evaluated.common_masks.loc[
        evaluated.common_masks["sampling_frame_index"]
        .astype(int)
        .isin(admitted_indices)
    ].reset_index(drop=True)
    return evaluated.census.reset_index(drop=True), masks


def bind_acquisition_ledger(
    ledger: pd.DataFrame, declaration_sha256: str
) -> pd.DataFrame:
    """Bind every acquisition/reuse operation to the prospective declaration."""
    if len(declaration_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in declaration_sha256
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_declaration_sha256",
            "Acquisition binding requires one lowercase SHA256",
        )
    output = ledger.copy(deep=True)
    output["acquisition_bound_declaration_sha256"] = declaration_sha256
    return output


def build_cluster_capacity(
    declaration: Sequence[Mapping[str, Any]],
    readiness: pd.DataFrame,
    admission: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile one terminal outcome and zero-or-one capacity per target cluster."""
    if (
        len(readiness) != len(declaration)
        or readiness["candidate_id"].duplicated().any()
        or admission.get("pair_id", pd.Series(dtype=str)).duplicated().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave1_result_identity_mismatch",
            "Readiness/admission identities are not unique and declaration-complete",
        )
    readiness_by_id = {
        str(row["candidate_id"]): row for row in readiness.to_dict(orient="records")
    }
    admission_by_id = {
        str(row["pair_id"]): row for row in admission.to_dict(orient="records")
    }
    declared_ids = {str(record["pair_id"]) for record in declaration}
    ready_ids = {
        pair_id
        for pair_id, row in readiness_by_id.items()
        if str(row["local_availability_status"]) == "LOCAL_READY_FOR_SCALE1A"
    }
    if set(readiness_by_id) != declared_ids or set(admission_by_id) != ready_ids:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_admission_domain_mismatch",
            "Only and all ready Wave-1 candidates must enter formal admission",
        )
    rows: list[dict[str, Any]] = []
    for record in declaration:
        pair_id = str(record["pair_id"])
        local = readiness_by_id[pair_id]
        scientific = admission_by_id.get(pair_id)
        if scientific is None:
            outcome = "READINESS_FAILED"
            reason = local.get("terminal_local_missing_reason")
            contribution = 0
        else:
            outcome = str(scientific["admission_status"])
            reason = scientific.get("terminal_reason_code")
            contribution = int(outcome == "FORMALLY_ADMITTED")
        rows.append(
            {
                "wave1_candidate_index": int(record["wave1_candidate_index"]),
                "pair_id": pair_id,
                "sequence_cluster": str(record["sequence_cluster"]),
                "target_class": str(record["target_class"]),
                "candidate_origin": str(record["candidate_origin"]),
                "readiness_status": str(local["local_availability_status"]),
                "scientifically_evaluated": scientific is not None,
                "terminal_wave1_cluster_outcome": outcome,
                "terminal_reason_code": reason,
                "cluster_contribution": contribution,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        "wave1_candidate_index", kind="stable"
    ).reset_index(drop=True)
    if (
        len(output) != len(declaration)
        or output["sequence_cluster"].duplicated().any()
        or not output["cluster_contribution"].isin([0, 1]).all()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "cluster_capacity_invariant_failed",
            "Wave-1 target clusters do not have exact binary capacity outcomes",
        )
    return output


def capacity_status(*, starting_capacity: int, new_clusters: int) -> str:
    """Apply the frozen 100/120 independent-cluster capacity gates."""
    combined = starting_capacity + new_clusters
    if combined >= 120:
        return PRIMARY_CONFIRMATORY_CAPACITY_REACHED
    if combined >= 100:
        return CORE_REACHED_PRIMARY_TARGET_PENDING
    return WAVE2_REQUIRED_CORE_NOT_REACHED


def _rehash_inputs(
    paths: ProjectPaths, artifacts: Sequence[Mapping[str, Any]]
) -> None:
    for artifact in artifacts:
        path = _resolve(paths, str(artifact["path"]))
        if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "frozen_input_drift_during_run",
                f"Frozen input changed during Wave-1: {artifact['path']}",
            )


def assemble_expansion_result(
    *,
    inputs: DatasetExpansionInputs,
    frozen: DatasetExpansionFreeze,
    acquisition_ledger: pd.DataFrame,
    readiness: pd.DataFrame,
    admission: pd.DataFrame,
    common_masks: pd.DataFrame,
    cluster_capacity: pd.DataFrame,
) -> DatasetExpansionRunResult:
    """Validate and assemble the sole structured source for formal outputs."""
    config = inputs.config
    if len(readiness) != config.frozen_wave1_primary_n:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_count_mismatch",
            "Wave-1 requires exactly 100 readiness outcomes",
        )
    ready = readiness.loc[
        readiness["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ]
    if (
        len(admission) != len(ready)
        or admission.get("admission_status", pd.Series(dtype=object)).isna().any()
        or len(cluster_capacity) != config.frozen_wave1_primary_n
        or cluster_capacity["sequence_cluster"].duplicated().any()
        or not cluster_capacity["cluster_contribution"].isin([0, 1]).all()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave1_terminal_outcome_mismatch",
            "Readiness, admission, and cluster outcomes do not reconcile",
        )
    bound = acquisition_ledger.get(
        "acquisition_bound_declaration_sha256", pd.Series(dtype=str)
    )
    if not bound.empty and not bound.eq(frozen.declaration_sha256).all():
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "acquisition_declaration_binding_mismatch",
            "Acquisition ledger is not bound to the frozen declaration",
        )
    admitted = admission.loc[
        admission.get("admission_status", pd.Series(dtype=str)).eq(
            "FORMALLY_ADMITTED"
        )
    ]
    admitted_indices = set(admitted.get("sampling_frame_index", pd.Series(dtype=int)))
    if not common_masks.empty:
        mask_indices = set(common_masks["sampling_frame_index"].astype(int))
        if mask_indices != {int(value) for value in admitted_indices}:
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "common_mask_domain_mismatch",
                "Common masks must contain exactly newly admitted realizations",
            )
    elif admitted_indices:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "common_mask_domain_mismatch",
            "Newly admitted realizations require canonical common-mask rows",
        )
    new_clusters = int(cluster_capacity["cluster_contribution"].sum())
    if new_clusters != len(admitted):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "cluster_contribution_mismatch",
            "Each newly admitted primary must contribute exactly one new cluster",
        )
    combined = config.starting_admitted_clusters + new_clusters
    status = capacity_status(
        starting_capacity=config.starting_admitted_clusters,
        new_clusters=new_clusters,
    )
    admission_counts = {
        str(key): int(value)
        for key, value in sorted(Counter(admission["admission_status"]).items())
    }
    readiness_failures = int(len(readiness) - len(ready))
    summary = {
        "schema_version": "dual-uq.scale1-expansion-wave1-summary.v1",
        "verdict": "PASS",
        "capacity_status": status,
        "expansion_stratum": "PROSPECTIVE_CLUSTER_COVERAGE_ENRICHMENT",
        "n_declared": len(frozen.declaration),
        "n_ready": len(ready),
        "n_readiness_failed": readiness_failures,
        "n_scientifically_evaluated": len(admission),
        "admission_status_counts": admission_counts,
        "n_wave1_new_admitted_clusters": new_clusters,
        "n_original_admitted_clusters": config.starting_admitted_clusters,
        "n_nr_combined": combined,
        "minimum_core_gate": config.minimum_core_gate,
        "primary_confirmatory_target": config.primary_confirmatory_target,
        "wave1_operational_admission_yield": (
            new_clusters / len(admission) if len(admission) else None
        ),
        "combined_prevalence_estimation_forbidden": True,
    }
    manifest = {
        "schema_version": "dual-uq.scale1-expansion-wave1-manifest.v1",
        "verdict": "PASS",
        "capacity_status": status,
        "upstream_artifacts": list(inputs.input_artifacts),
        "e0a_amendment_sha256": frozen.amendment_sha256,
        "wave1_declaration_sha256": frozen.declaration_sha256,
        "acquisition_bound_declaration_sha256": frozen.declaration_sha256,
        "scale1a1_admission_contract": {
            "implementation": (
                "dual_uq.dataset.admission."
                "evaluate_admission_candidates"
            ),
            "candidate_evaluator": (
                "dual_uq.dataset.admission._evaluate_candidate"
            ),
            "changed": False,
        },
        "acquisition_contract": {
            "authorities": [
                "RCSB_PDB",
                "PDBe_SIFTS",
                "AlphaFold_DB",
                "UniProtKB",
            ],
            "candidate_identity": "frozen_declaration_exact_identity",
            "afdb_record_identity": "exact_uniprot_accession",
            "fragment_selection": (
                "unique_full_cover_of_sifts_mapped_interval"
            ),
            "readiness_failure_is_scientific_rejection": False,
            "forbidden_fallbacks": [
                "F1_fallback",
                "nearest_fragment",
                "first_record",
                "manual_choice",
                "offset_inference",
                "closest_coverage_heuristic",
            ],
        },
        "redundancy_convention": {
            "provider": "RCSB rcsb_cluster_membership",
            "identity_threshold_percent": 30,
            "cluster_id_serialization": "30:<cluster_id>",
            "changed": False,
        },
        "common_mask_serialization": {
            "unit": "canonical_residue_row",
            "realization_count": len(admitted),
            "row_count": len(common_masks),
            "admitted_realizations_only": True,
        },
        "capacity_gates": {
            "minimum_core_gate": config.minimum_core_gate,
            "primary_confirmatory_target": config.primary_confirmatory_target,
        },
        "ORIGINAL_213_FRAME_IMMUTABLE": True,
        "EXPANSION_STRATUM_SEPARATE": True,
        "COMBINED_PREVALENCE_ESTIMATION_FORBIDDEN": True,
        "ADMISSION_CONTRACT_UNCHANGED": True,
        "REDUNDANCY_CONVENTION_UNCHANGED": True,
        "WAVE1_RESERVE_ACTIVATION_AUTHORIZED": False,
        "WAVE1_DECLARATION_FROZEN_BEFORE_ACQUISITION": True,
        "NO_SILENT_SUBSTITUTION": True,
        "NO_PROTEINMPNN": True,
        "NO_FIXED_PROBES": True,
        "NO_SCALE1B_V2_SELECTION": True,
        "NO_MODEL_OUTCOME_DEPENDENT_SELECTION": True,
    }
    return DatasetExpansionRunResult(
        verdict="PASS",
        capacity_status=status,
        acquisition_ledger=acquisition_ledger.reset_index(drop=True),
        readiness=readiness.reset_index(drop=True),
        admission=admission.reset_index(drop=True),
        common_masks=common_masks.reset_index(drop=True),
        cluster_capacity=cluster_capacity.reset_index(drop=True),
        summary=summary,
        manifest=manifest,
        frozen=frozen,
        input_artifacts=inputs.input_artifacts,
    )


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
        if path.is_file():
            if sha256_file(path) == sha256_file(temporary):
                return "reused_identical"
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_output_conflict",
                f"Immutable expansion output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_output_conflict",
                f"Expansion output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_expansion_result(
    result: DatasetExpansionRunResult,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionConfig | None = None,
) -> dict[str, Any]:
    """Render five tables and summary, then write the manifest last."""
    config = config or DatasetExpansionConfig()
    validate_frozen_declaration(result.frozen, paths, config=config)
    _rehash_inputs(paths, result.input_artifacts)
    root = _resolve(paths, config.output_root_ref)
    targets = {
        "acquisition_ledger": root / "scale1_expansion_wave1_acquisition_ledger.parquet",
        "readiness": root / "scale1_expansion_wave1_readiness.parquet",
        "admission": root / "scale1_expansion_wave1_admission.parquet",
        "common_masks": root / "scale1_expansion_wave1_common_masks.parquet",
        "cluster_capacity": root / "scale1_expansion_wave1_cluster_capacity.parquet",
        "summary": root / "scale1_expansion_wave1_summary.json",
        "manifest": root / "scale1_expansion_wave1_manifest.json",
    }
    tables = {
        "acquisition_ledger": result.acquisition_ledger,
        "readiness": result.readiness,
        "admission": result.admission,
        "common_masks": result.common_masks,
        "cluster_capacity": result.cluster_capacity,
    }
    write_status = {
        name: _write_immutable_parquet(targets[name], frame)
        for name, frame in tables.items()
    }
    outputs: dict[str, dict[str, Any]] = {
        "amendment": {
            "path": _logical(paths, root / "scale1e0a_wave1_amendment.json"),
            "sha256": result.frozen.amendment_sha256,
        },
        "declaration": {
            "path": _logical(
                paths, root / "scale1_expansion_wave1_candidates.jsonl"
            ),
            "rows": len(result.frozen.declaration),
            "sha256": result.frozen.declaration_sha256,
        },
    }
    outputs.update(
        {
            name: {
                "path": _logical(paths, targets[name]),
                "rows": len(frame),
                "sha256": sha256_file(targets[name]),
            }
            for name, frame in tables.items()
        }
    )
    summary = {**result.summary, "output_artifacts": outputs}
    write_status["summary"] = _write_immutable(
        targets["summary"], _json_bytes(summary)
    )
    outputs["summary"] = {
        "path": _logical(paths, targets["summary"]),
        "sha256": sha256_file(targets["summary"]),
    }
    outputs["manifest"] = {
        "path": _logical(paths, targets["manifest"]),
        "self_hash_policy": "reported_externally_to_avoid_recursive_self_hash",
    }
    manifest = {**result.manifest, "outputs": outputs}
    _rehash_inputs(paths, result.input_artifacts)
    validate_frozen_declaration(result.frozen, paths, config=config)
    write_status["manifest"] = _write_immutable(
        targets["manifest"], _json_bytes(manifest)
    )
    return {
        "verdict": result.verdict,
        "capacity_status": result.capacity_status,
        "write_status": write_status,
        "outputs": {
            **outputs,
            "manifest": {
                "path": _logical(paths, targets["manifest"]),
                "sha256": sha256_file(targets["manifest"]),
            },
        },
    }


def run_dataset_expansion(
    paths: ProjectPaths,
    *,
    config: DatasetExpansionConfig | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Execute the prospectively frozen expansion declaration end to end."""
    config = config or DatasetExpansionConfig()
    inputs = validate_expansion_inputs(paths, config)
    frozen = build_expansion_freeze(inputs)
    materialize_expansion_freeze(frozen, paths, config=config)
    readiness_result = execute_expansion_readiness(
        inputs=inputs,
        frozen=frozen,
        paths=paths,
        config=config,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
    )
    admission, masks = evaluate_expansion_admission(
        readiness_result.readiness, paths=paths
    )
    cluster_capacity = build_cluster_capacity(
        frozen.declaration,
        readiness_result.readiness,
        admission,
    )
    result = assemble_expansion_result(
        inputs=inputs,
        frozen=frozen,
        acquisition_ledger=readiness_result.acquisition_ledger,
        readiness=readiness_result.readiness,
        admission=admission,
        common_masks=masks,
        cluster_capacity=cluster_capacity,
    )
    return materialize_expansion_result(result, paths, config=config)


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    try:
        records = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not all(isinstance(record, dict) for record in records):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must contain JSON objects",
        )
    return records


def validate_wave2_inputs(
    paths: ProjectPaths,
    config: DatasetExpansionWave2Config | None = None,
) -> DatasetExpansionWave2Inputs:
    """SHA-gate Wave-2 trust roots and derive the exact E0 minus Wave-1 set."""
    config = config or DatasetExpansionWave2Config()
    base = validate_expansion_inputs(paths, config.base_config)
    bindings = (
        (
            config.wave1_amendment_ref,
            config.expected_wave1_amendment_sha256,
            "Wave-1 E0a amendment",
        ),
        (
            config.wave1_declaration_ref,
            config.expected_wave1_declaration_sha256,
            "Wave-1 declaration",
        ),
        (
            config.wave1_acquisition_ledger_ref,
            config.expected_wave1_acquisition_ledger_sha256,
            "Wave-1 acquisition ledger",
        ),
        (
            config.wave1_readiness_ref,
            config.expected_wave1_readiness_sha256,
            "Wave-1 readiness",
        ),
        (
            config.wave1_admission_ref,
            config.expected_wave1_admission_sha256,
            "Wave-1 admission",
        ),
        (
            config.wave1_common_masks_ref,
            config.expected_wave1_common_masks_sha256,
            "Wave-1 common masks",
        ),
        (
            config.wave1_cluster_capacity_ref,
            config.expected_wave1_cluster_capacity_sha256,
            "Wave-1 cluster capacity",
        ),
        (
            config.wave1_summary_ref,
            config.expected_wave1_summary_sha256,
            "Wave-1 summary",
        ),
        (
            config.wave1_manifest_ref,
            config.expected_wave1_manifest_sha256,
            "Wave-1 manifest",
        ),
    )
    artifacts = list(base.input_artifacts)
    wave_paths: dict[str, Path] = {}
    for logical_ref, expected, label in bindings:
        path, artifact = _require_hash(paths, logical_ref, expected, label)
        wave_paths[logical_ref] = path
        artifacts.append(artifact)
    wave1_summary = _read_json(
        wave_paths[config.wave1_summary_ref], "Wave-1 summary"
    )
    wave1_manifest = _read_json(
        wave_paths[config.wave1_manifest_ref], "Wave-1 manifest"
    )
    declaration = _read_jsonl(
        wave_paths[config.wave1_declaration_ref], "Wave-1 declaration"
    )
    try:
        readiness = pd.read_parquet(wave_paths[config.wave1_readiness_ref])
        admission = pd.read_parquet(wave_paths[config.wave1_admission_ref])
        capacity = pd.read_parquet(wave_paths[config.wave1_cluster_capacity_ref])
    except (OSError, ValueError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            "Unable to read frozen Wave-1 tables",
        ) from exc
    admission_counts = Counter(admission.get("admission_status", pd.Series(dtype=str)))
    expected_counts = Counter(
        {
            "FORMALLY_ADMITTED": 49,
            "PENDING_HUMAN_VARIANT_REVIEW": 33,
            "IDENTITY_CONTRACT_FAIL": 8,
            "PROVENANCE_FAIL": 3,
            "MAPPING_FAIL": 2,
        }
    )
    declared_clusters = {str(record.get("sequence_cluster")) for record in declaration}
    expected_first = set(base.targets.iloc[:100]["sequence_cluster"].astype(str))
    if (
        wave1_summary.get("capacity_status")
        != CORE_REACHED_PRIMARY_TARGET_PENDING
        or wave1_summary.get("n_declared") != 100
        or wave1_summary.get("n_ready") != 95
        or wave1_summary.get("n_readiness_failed") != 5
        or wave1_summary.get("n_scientifically_evaluated") != 95
        or wave1_summary.get("n_wave1_new_admitted_clusters") != 49
        or wave1_summary.get("n_nr_combined") != config.starting_admitted_clusters
        or len(declaration) != 100
        or len(readiness) != 100
        or len(admission) != 95
        or admission_counts != expected_counts
        or len(capacity) != 100
        or int(capacity["cluster_contribution"].sum()) != 49
        or declared_clusters != expected_first
        or wave1_manifest.get("wave1_declaration_sha256")
        != config.expected_wave1_declaration_sha256
        or wave1_manifest.get("capacity_status")
        != CORE_REACHED_PRIMARY_TARGET_PENDING
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave1_frozen_state_mismatch",
            "Frozen Wave-1 state does not match the approved Wave-2 start",
        )
    wave1_admitted = set(
        capacity.loc[
            capacity["cluster_contribution"].eq(1), "sequence_cluster"
        ].astype(str)
    )
    clusters_before = frozenset((*base.original_admitted_clusters, *wave1_admitted))
    remaining = base.targets.loc[
        ~base.targets["sequence_cluster"].astype(str).isin(declared_clusters)
    ].copy()
    if (
        len(clusters_before) != config.starting_admitted_clusters
        or len(remaining) != config.wave2_target_cluster_n
        or remaining["prospective_global_primary_rank"].astype(int).tolist()
        != list(range(101, 125))
        or remaining["sequence_cluster"].duplicated().any()
        or set(remaining["sequence_cluster"].astype(str)).intersection(
            clusters_before
        )
        or len(base.targets) != 124
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave2_set_difference_mismatch",
            "E0 actionable minus Wave-1 declaration is not the exact 24-cluster set",
        )
    return DatasetExpansionWave2Inputs(
        base_inputs=base,
        wave1_summary=wave1_summary,
        remaining_targets=remaining.reset_index(drop=True),
        clusters_before=clusters_before,
        input_artifacts=tuple(artifacts),
        config=config,
    )


def build_wave2_freeze(
    inputs: DatasetExpansionWave2Inputs,
) -> DatasetExpansionWave2Freeze:
    """Build the exact remaining-primary declaration without reranking."""
    records: list[dict[str, Any]] = []
    for wave_index, row in enumerate(
        inputs.remaining_targets.to_dict(orient="records"), 1
    ):
        records.append(
            {
                "schema_version": "dual-uq.scale1-expansion-wave2-candidate.v1",
                "wave2_candidate_index": wave_index,
                "sequence_cluster": str(row["sequence_cluster"]),
                "target_class": str(row["target_class"]),
                "pair_id": str(row["primary_pair_id"]),
                "candidate_id": str(row["primary_candidate_id"]),
                "polymer_entity_id": str(row["primary_polymer_entity_id"]),
                "canonical_accession": str(row["primary_canonical_accession"]),
                "pdb_id": str(row["primary_pdb_id"]),
                "pdb_chain": str(row["primary_pdb_chain"]),
                "candidate_origin": str(row["primary_candidate_origin"]),
                "prospective_global_primary_rank": int(
                    row["prospective_global_primary_rank"]
                ),
                "selection_provenance": str(row["selection_provenance"]),
                "target_source_category": str(row["target_source_category"]),
                "reserve_candidate_count": int(row["reserve_candidate_count"]),
                "selection_role": "PRIMARY_CANDIDATE",
                "reserve_activated": False,
            }
        )
    declaration = tuple(records)
    if (
        len(declaration) != inputs.config.wave2_target_cluster_n
        or len({record["sequence_cluster"] for record in declaration}) != 24
        or len({record["pair_id"] for record in declaration}) != 24
        or [record["prospective_global_primary_rank"] for record in declaration]
        != list(range(101, 125))
        or {record["sequence_cluster"] for record in declaration}.intersection(
            inputs.clusters_before
        )
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_wave2_declaration",
            "Wave-2 declaration is not the exact remaining 24 E0 targets",
        )
    return DatasetExpansionWave2Freeze(
        declaration=declaration,
        declaration_sha256=hashlib.sha256(_jsonl_bytes(declaration)).hexdigest(),
        input_artifacts=inputs.input_artifacts,
    )


def materialize_wave2_declaration(
    frozen: DatasetExpansionWave2Freeze,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionWave2Config | None = None,
) -> dict[str, Any]:
    """Write and revalidate the immutable Wave-2 declaration."""
    config = config or DatasetExpansionWave2Config()
    path = _resolve(
        paths,
        f"{config.output_root_ref}/scale1_expansion_wave2_candidates.jsonl",
    )
    payload = _jsonl_bytes(frozen.declaration)
    if hashlib.sha256(payload).hexdigest() != frozen.declaration_sha256:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "declaration_hash_mismatch",
            "In-memory Wave-2 declaration changed before materialization",
        )
    status = _write_immutable(path, payload)
    validate_wave2_declaration(frozen, paths, config=config)
    return {
        "write_status": status,
        "path": _logical(paths, path),
        "rows": len(frozen.declaration),
        "sha256": frozen.declaration_sha256,
    }


def validate_wave2_declaration(
    frozen: DatasetExpansionWave2Freeze,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionWave2Config | None = None,
) -> None:
    """Block realization when the prospectively frozen declaration drifts."""
    config = config or DatasetExpansionWave2Config()
    path = _resolve(
        paths,
        f"{config.output_root_ref}/scale1_expansion_wave2_candidates.jsonl",
    )
    if not path.is_file() or sha256_file(path) != frozen.declaration_sha256:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "declaration_hash_mismatch",
            "Frozen Wave-2 declaration SHA256 does not match",
        )


def wave2_capacity_status(
    *, starting_capacity: int, new_clusters: int, target: int
) -> tuple[str, int]:
    """Apply the frozen final-capacity gate after the complete Wave-2 cohort."""
    final_capacity = starting_capacity + new_clusters
    remaining_gap = max(0, target - final_capacity)
    status = (
        PRIMARY_CONFIRMATORY_CAPACITY_REACHED
        if final_capacity >= target
        else PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED
    )
    return status, remaining_gap


def _wave2_readiness_source(
    frozen: DatasetExpansionWave2Freeze,
    *,
    config: DatasetExpansionWave2Config,
) -> pd.DataFrame:
    adapted = tuple(
        {
            **record,
            "wave1_candidate_index": record["wave2_candidate_index"],
        }
        for record in frozen.declaration
    )
    return build_initial_readiness_frame(
        adapted,
        declaration_ref=(
            f"{config.output_root_ref}/scale1_expansion_wave2_candidates.jsonl"
        ),
        declaration_sha256=frozen.declaration_sha256,
    )


def load_wave2_prior_acquisition_records(
    paths: ProjectPaths,
    declaration_sha256: str,
    *,
    config: DatasetExpansionWave2Config | None = None,
) -> tuple[AcquisitionRecord, ...]:
    """Load prior Wave-2 operations only under the exact declaration binding."""
    config = config or DatasetExpansionWave2Config()
    path = _resolve(
        paths,
        f"{config.output_root_ref}/"
        "scale1_expansion_wave2_acquisition_ledger.parquet",
    )
    if not path.is_file():
        return ()
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_ledger_unreadable",
            "Prior Wave-2 acquisition ledger is unreadable",
        ) from exc
    binding = frame.get("acquisition_bound_declaration_sha256")
    if (
        binding is None
        or len(binding) != len(frame)
        or not binding.eq(declaration_sha256).all()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_declaration_mismatch",
            "Prior acquisition ledger belongs to another declaration",
        )
    try:
        return acquisition_records_from_frame(frame)
    except Exception as exc:  # canonical loader provides structured detail
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "prior_acquisition_ledger_schema_mismatch",
            "Prior Wave-2 acquisition ledger schema is invalid",
        ) from exc


def execute_wave2_readiness(
    *,
    inputs: DatasetExpansionWave2Inputs,
    frozen: DatasetExpansionWave2Freeze,
    paths: ProjectPaths,
    config: DatasetExpansionWave2Config | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    prior_records: Sequence[AcquisitionRecord] | None = None,
) -> DatasetExpansionWave2ReadinessResult:
    """Acquire/reuse all 24 declared primaries without early stopping."""
    config = config or inputs.config
    validate_wave2_declaration(frozen, paths, config=config)
    _rehash_inputs(paths, inputs.input_artifacts)
    prior = (
        load_wave2_prior_acquisition_records(
            paths, frozen.declaration_sha256, config=config
        )
        if prior_records is None
        else tuple(prior_records)
    )
    source = _wave2_readiness_source(frozen, config=config)
    target_indices = set(source["sampling_frame_index"].astype(int))
    rejected_ref = (
        "artifacts/dataset/audits/acquisition_rejected/"
        f"scale1-expansion-wave2/{frozen.declaration_sha256}"
    )
    initial = execute_initial_acquisition(
        build_initial_asset_requirements(source, paths),
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    interim = recompute_full_frame_readiness(source, initial, paths)
    canonical = execute_initial_acquisition(
        resolve_uniprot_requirements(interim, target_indices, paths),
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    interim_records = (*initial, *canonical)
    interim = recompute_full_frame_readiness(source, interim_records, paths)
    structure_requirements, dependency_failures = (
        resolve_afdb_structure_requirements(
            interim,
            target_indices,
            paths,
            clock=clock,
            prior_records=prior,
        )
    )
    structures = execute_initial_acquisition(
        structure_requirements,
        paths,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
        prior_records=prior,
        rejected_evidence_ref=rejected_ref,
    )
    records = (*interim_records, *structures, *dependency_failures)
    readiness = recompute_full_frame_readiness(source, records, paths)
    declared_ids = {str(record["candidate_id"]) for record in frozen.declaration}
    if (
        len(readiness) != config.wave2_target_cluster_n
        or readiness["candidate_id"].duplicated().any()
        or set(readiness["candidate_id"]) != declared_ids
        or readiness["local_availability_status"].isna().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_identity_mismatch",
            "Wave-2 readiness outcomes are not declaration-complete",
        )
    ledger = bind_acquisition_ledger(
        acquisition_records_frame(records), frozen.declaration_sha256
    )
    _rehash_inputs(paths, inputs.input_artifacts)
    validate_wave2_declaration(frozen, paths, config=config)
    return DatasetExpansionWave2ReadinessResult(
        records=tuple(records),
        acquisition_ledger=ledger,
        readiness=readiness.reset_index(drop=True),
    )


def build_wave2_cluster_capacity(
    declaration: Sequence[Mapping[str, Any]],
    readiness: pd.DataFrame,
    admission: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile all 24 targets; reaching 120 never truncates evaluation."""
    if (
        len(readiness) != len(declaration)
        or readiness["candidate_id"].duplicated().any()
        or admission.get("pair_id", pd.Series(dtype=str)).duplicated().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave2_result_identity_mismatch",
            "Readiness/admission identities are not unique and declaration-complete",
        )
    readiness_by_id = {
        str(row["candidate_id"]): row
        for row in readiness.to_dict(orient="records")
    }
    admission_by_id = {
        str(row["pair_id"]): row for row in admission.to_dict(orient="records")
    }
    declared_ids = {str(record["pair_id"]) for record in declaration}
    ready_ids = {
        pair_id
        for pair_id, row in readiness_by_id.items()
        if str(row["local_availability_status"]) == "LOCAL_READY_FOR_SCALE1A"
    }
    if set(readiness_by_id) != declared_ids or set(admission_by_id) != ready_ids:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_admission_domain_mismatch",
            "Only and all ready Wave-2 candidates must enter formal admission",
        )
    rows: list[dict[str, Any]] = []
    for record in declaration:
        pair_id = str(record["pair_id"])
        local = readiness_by_id[pair_id]
        scientific = admission_by_id.get(pair_id)
        if scientific is None:
            outcome = "READINESS_FAILED"
            reason = local.get("terminal_local_missing_reason")
            contribution = 0
        else:
            outcome = str(scientific["admission_status"])
            reason = scientific.get("terminal_reason_code")
            contribution = int(outcome == FORMALLY_ADMITTED)
        rows.append(
            {
                "wave2_candidate_index": int(record["wave2_candidate_index"]),
                "prospective_global_primary_rank": int(
                    record["prospective_global_primary_rank"]
                ),
                "pair_id": pair_id,
                "sequence_cluster": str(record["sequence_cluster"]),
                "target_class": str(record["target_class"]),
                "candidate_origin": str(record["candidate_origin"]),
                "readiness_status": str(local["local_availability_status"]),
                "scientifically_evaluated": scientific is not None,
                "terminal_wave2_cluster_outcome": outcome,
                "terminal_reason_code": reason,
                "capacity_contribution": contribution,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        "wave2_candidate_index", kind="stable"
    ).reset_index(drop=True)
    if (
        len(output) != len(declaration)
        or output["sequence_cluster"].duplicated().any()
        or not output["capacity_contribution"].isin([0, 1]).all()
        or output["terminal_wave2_cluster_outcome"].isna().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "cluster_capacity_invariant_failed",
            "Wave-2 targets lack exact binary terminal capacity outcomes",
        )
    return output


def assemble_wave2_result(
    *,
    inputs: DatasetExpansionWave2Inputs,
    frozen: DatasetExpansionWave2Freeze,
    acquisition_ledger: pd.DataFrame,
    readiness: pd.DataFrame,
    admission: pd.DataFrame,
    common_masks: pd.DataFrame,
    cluster_capacity: pd.DataFrame,
) -> DatasetExpansionWave2RunResult:
    """Validate and assemble the sole structured source for Wave-2 outputs."""
    config = inputs.config
    if len(readiness) != config.wave2_target_cluster_n:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "readiness_count_mismatch",
            "Wave-2 requires exactly 24 readiness outcomes",
        )
    ready = readiness.loc[
        readiness["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ]
    if (
        len(admission) != len(ready)
        or admission.get("admission_status", pd.Series(dtype=object)).isna().any()
        or len(cluster_capacity) != config.wave2_target_cluster_n
        or cluster_capacity["sequence_cluster"].duplicated().any()
        or not cluster_capacity["capacity_contribution"].isin([0, 1]).all()
        or cluster_capacity["terminal_wave2_cluster_outcome"].isna().any()
    ):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave2_terminal_outcome_mismatch",
            "Readiness, admission, and all 24 cluster outcomes must reconcile",
        )
    bound = acquisition_ledger.get(
        "acquisition_bound_declaration_sha256", pd.Series(dtype=str)
    )
    if not bound.empty and not bound.eq(frozen.declaration_sha256).all():
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "acquisition_declaration_binding_mismatch",
            "Acquisition ledger is not bound to the Wave-2 declaration",
        )
    declared_clusters = {
        str(record["sequence_cluster"]) for record in frozen.declaration
    }
    if declared_clusters.intersection(inputs.clusters_before):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "wave2_cluster_overlap",
            "Wave-2 target clusters overlap the frozen 112-cluster start",
        )
    admitted = admission.loc[admission["admission_status"].eq(FORMALLY_ADMITTED)]
    admitted_indices = set(admitted["sampling_frame_index"].astype(int))
    if not common_masks.empty:
        mask_indices = set(common_masks["sampling_frame_index"].astype(int))
        if mask_indices != admitted_indices:
            raise DatasetExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "common_mask_domain_mismatch",
                "Wave-2 masks must contain exactly newly admitted realizations",
            )
    elif admitted_indices:
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "common_mask_domain_mismatch",
            "Newly admitted Wave-2 realizations require canonical mask rows",
        )
    new_clusters = int(cluster_capacity["capacity_contribution"].sum())
    if new_clusters != len(admitted):
        raise DatasetExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "cluster_contribution_mismatch",
            "Each Wave-2 admitted primary must contribute exactly one cluster",
        )
    final_capacity = config.starting_admitted_clusters + new_clusters
    status, remaining_gap = wave2_capacity_status(
        starting_capacity=config.starting_admitted_clusters,
        new_clusters=new_clusters,
        target=config.primary_confirmatory_target,
    )
    counts = {
        str(key): int(value)
        for key, value in sorted(Counter(admission["admission_status"]).items())
    }
    summary = {
        "schema_version": "dual-uq.scale1-expansion-wave2-summary.v1",
        "verdict": "PASS",
        "capacity_status": status,
        "declared_clusters": len(frozen.declaration),
        "ready": len(ready),
        "scientifically_evaluated": len(admission),
        "admission_status_counts": counts,
        "readiness_failures": len(readiness) - len(ready),
        "starting_N_NR": config.starting_admitted_clusters,
        "new_admitted_clusters": new_clusters,
        "final_N_NR": final_capacity,
        "remaining_gap_to_120": remaining_gap,
        "expansion_stratum": "PROSPECTIVE_CLUSTER_COVERAGE_ENRICHMENT_WAVE2",
        "wave2_operational_admission_yield": (
            new_clusters / len(admission) if len(admission) else None
        ),
        "combined_prevalence_estimation_forbidden": True,
    }
    manifest = {
        "schema_version": "dual-uq.scale1-expansion-wave2-manifest.v1",
        "verdict": "PASS",
        "capacity_status": status,
        "upstream_artifacts": list(inputs.input_artifacts),
        "wave2_declaration_sha256": frozen.declaration_sha256,
        "acquisition_bound_declaration_sha256": frozen.declaration_sha256,
        "scale1a1_admission_contract": {
            "implementation": (
                "dual_uq.dataset.admission."
                "evaluate_admission_candidates"
            ),
            "candidate_evaluator": (
                "dual_uq.dataset.admission._evaluate_candidate"
            ),
            "changed": False,
        },
        "acquisition_contract": {
            "authorities": [
                "RCSB_PDB",
                "PDBe_SIFTS",
                "AlphaFold_DB",
                "UniProtKB",
            ],
            "sifts_role": "MAPPING_INPUT_ONLY",
            "candidate_identity": "frozen_declaration_exact_identity",
            "afdb_record_identity": "exact_uniprot_accession",
            "fragment_selection": "unique_full_cover_of_sifts_mapped_interval",
            "readiness_failure_is_scientific_rejection": False,
            "forbidden_fallbacks": [
                "F1_fallback",
                "nearest_fragment",
                "first_record",
                "manual_choice",
                "offset_inference",
                "closest_coverage_heuristic",
            ],
        },
        "redundancy_convention": {
            "provider": "RCSB rcsb_cluster_membership",
            "identity_threshold_percent": 30,
            "cluster_id_serialization": "30:<cluster_id>",
            "changed": False,
        },
        "common_mask_serialization": {
            "unit": "canonical_residue_row",
            "realization_count": len(admitted),
            "row_count": len(common_masks),
            "admitted_realizations_only": True,
        },
        "STARTING_NR_ADMITTED_CLUSTERS": 112,
        "WAVE2_TARGET_CLUSTER_N": 24,
        "ONE_PRIMARY_PER_CLUSTER": True,
        "WAVE2_RESERVE_ACTIVATION_AUTHORIZED": False,
        "ADMISSION_CONTRACT_UNCHANGED": True,
        "REDUNDANCY_CONVENTION_UNCHANGED": True,
        "ORIGINAL_213_FRAME_IMMUTABLE": True,
        "EXPANSION_STRATA_PRESERVED": True,
        "COMBINED_PREVALENCE_ESTIMATION_FORBIDDEN": True,
        "NO_OUTCOME_DEPENDENT_SELECTION": True,
        "NO_EARLY_STOP_AT_120": True,
        "NO_PROTEINMPNN": True,
        "NO_FIXED_PROBES": True,
        "NO_SCALE1B_V2_FREEZE": True,
    }
    return DatasetExpansionWave2RunResult(
        verdict="PASS",
        capacity_status=status,
        acquisition_ledger=acquisition_ledger.reset_index(drop=True),
        readiness=readiness.reset_index(drop=True),
        admission=admission.reset_index(drop=True),
        common_masks=common_masks.reset_index(drop=True),
        cluster_capacity=cluster_capacity.reset_index(drop=True),
        summary=summary,
        manifest=manifest,
        frozen=frozen,
        input_artifacts=inputs.input_artifacts,
    )


def materialize_wave2_result(
    result: DatasetExpansionWave2RunResult,
    paths: ProjectPaths,
    *,
    config: DatasetExpansionWave2Config | None = None,
) -> dict[str, Any]:
    """Render canonical Wave-2 tables and summary, then manifest last."""
    config = config or DatasetExpansionWave2Config()
    validate_wave2_declaration(result.frozen, paths, config=config)
    _rehash_inputs(paths, result.input_artifacts)
    root = _resolve(paths, config.output_root_ref)
    targets = {
        "acquisition_ledger": root
        / "scale1_expansion_wave2_acquisition_ledger.parquet",
        "readiness": root / "scale1_expansion_wave2_readiness.parquet",
        "admission": root / "scale1_expansion_wave2_admission.parquet",
        "common_masks": root / "scale1_expansion_wave2_common_masks.parquet",
        "cluster_capacity": root
        / "scale1_expansion_wave2_cluster_capacity.parquet",
        "summary": root / "scale1_expansion_wave2_summary.json",
        "manifest": root / "scale1_expansion_wave2_manifest.json",
    }
    tables = {
        "acquisition_ledger": result.acquisition_ledger,
        "readiness": result.readiness,
        "admission": result.admission,
        "common_masks": result.common_masks,
        "cluster_capacity": result.cluster_capacity,
    }
    write_status = {
        name: _write_immutable_parquet(targets[name], frame)
        for name, frame in tables.items()
    }
    declaration_path = root / "scale1_expansion_wave2_candidates.jsonl"
    outputs: dict[str, dict[str, Any]] = {
        "declaration": {
            "path": _logical(paths, declaration_path),
            "rows": len(result.frozen.declaration),
            "sha256": result.frozen.declaration_sha256,
        }
    }
    outputs.update(
        {
            name: {
                "path": _logical(paths, targets[name]),
                "rows": len(frame),
                "sha256": sha256_file(targets[name]),
            }
            for name, frame in tables.items()
        }
    )
    summary = {**result.summary, "output_artifacts": outputs}
    write_status["summary"] = _write_immutable(
        targets["summary"], _json_bytes(summary)
    )
    outputs["summary"] = {
        "path": _logical(paths, targets["summary"]),
        "sha256": sha256_file(targets["summary"]),
    }
    outputs["manifest"] = {
        "path": _logical(paths, targets["manifest"]),
        "self_hash_policy": "reported_externally_to_avoid_recursive_self_hash",
    }
    manifest = {**result.manifest, "outputs": outputs}
    _rehash_inputs(paths, result.input_artifacts)
    validate_wave2_declaration(result.frozen, paths, config=config)
    write_status["manifest"] = _write_immutable(
        targets["manifest"], _json_bytes(manifest)
    )
    return {
        "verdict": result.verdict,
        "capacity_status": result.capacity_status,
        "write_status": write_status,
        "outputs": {
            **outputs,
            "manifest": {
                "path": _logical(paths, targets["manifest"]),
                "sha256": sha256_file(targets["manifest"]),
            },
        },
    }


def run_dataset_expansion_wave2(
    paths: ProjectPaths,
    *,
    config: DatasetExpansionWave2Config | None = None,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Execute all 24 frozen Wave-2 primaries without early stopping."""
    config = config or DatasetExpansionWave2Config()
    inputs = validate_wave2_inputs(paths, config)
    frozen = build_wave2_freeze(inputs)
    materialize_wave2_declaration(frozen, paths, config=config)
    readiness_result = execute_wave2_readiness(
        inputs=inputs,
        frozen=frozen,
        paths=paths,
        config=config,
        transport=transport,
        sleeper=sleeper,
        clock=clock,
        max_attempts=max_attempts,
    )
    admission, masks = evaluate_expansion_admission(
        readiness_result.readiness, paths=paths
    )
    cluster_capacity = build_wave2_cluster_capacity(
        frozen.declaration,
        readiness_result.readiness,
        admission,
    )
    result = assemble_wave2_result(
        inputs=inputs,
        frozen=frozen,
        acquisition_ledger=readiness_result.acquisition_ledger,
        readiness=readiness_result.readiness,
        admission=admission,
        common_masks=masks,
        cluster_capacity=cluster_capacity,
    )
    return materialize_wave2_result(result, paths, config=config)
