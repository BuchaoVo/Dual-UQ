"""Offline Scale-1A1 formal intervention-admission census.

The module composes the frozen Dataset/Stage-0 identity, SIFTS, fragment and
P1 backbone primitives.  It deliberately stops before any geometry,
confidence, scoring, probe construction or Scale-1B eligibility decision.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.schema import AmbiguousLegacyResidueIdentifier, normalize_residue_mapping
from dual_uq.structure_io import load_chain_ca_table, residue_name_to_one_letter

from .models import AFDBFragment, DerivationError, ResidueKey, group_residue_records
from .policies.fragments import (
    resolve_exact_fragment,
    validate_frozen_model_artifacts,
)
from .policies.identity import (
    extract_canonical_sequence_from_source,
    metadata_records,
)
from .services.backbone import paired_common_backbone_positions
from .services.mapping import (
    mapping_with_provenance,
    parse_sifts_mapping_with_explicit_labels,
)
from .stages.derivation import P1ValidationError, _load_atom_records
from .stages.resolution import P0ValidationError, _validate_pdb_mmcif

SCHEMA_VERSION = "dual-uq.scale1-formal-admission.v1"
DEFAULT_OUTPUT_ROOT = "experiments/p2_design_baseline/scale1/scale1a1"

FORMALLY_ADMITTED = "FORMALLY_ADMITTED"
PENDING_HUMAN_VARIANT_REVIEW = "PENDING_HUMAN_VARIANT_REVIEW"
IDENTITY_CONTRACT_FAIL = "IDENTITY_CONTRACT_FAIL"
MAPPING_FAIL = "MAPPING_FAIL"
PROVENANCE_FAIL = "PROVENANCE_FAIL"
NOT_EVALUATED = "NOT_EVALUATED_LOCAL_DATA_INCOMPLETE"

STATUS_VOCABULARY = (
    FORMALLY_ADMITTED,
    PENDING_HUMAN_VARIANT_REVIEW,
    IDENTITY_CONTRACT_FAIL,
    MAPPING_FAIL,
    PROVENANCE_FAIL,
)

_LOCAL_READY_PATH_COLUMNS = (
    "canonical_sequence_source_relative",
    "mapping_metadata_source_relative",
    "pdb_file_path_relative",
    "afdb_file_path_relative",
)

_EXPECTED_STAGE0_COUNTS = {
    FORMALLY_ADMITTED: 8,
    PENDING_HUMAN_VARIANT_REVIEW: 2,
    IDENTITY_CONTRACT_FAIL: 1,
}

_EXPECTED_STAGE0_STATUS_BY_ID = {
    "5gv8_A__P83686": "ADMITTED",
    "6jgj_A__P42212": "IDENTITY_CONTRACT_FAIL",
    "5mn1_A__P00760": "ADMITTED",
    "1fn8_A__P35049": "ADMITTED",
    "2ykz_A__P00138": "PENDING_HUMAN_VARIANT_REVIEW",
    "1pjx_A__Q7SIG4": "ADMITTED",
    "3pyp_A__P16113": "ADMITTED",
    "6s2s_A__P02689": "ADMITTED",
    "1ix9_A__P00448": "PENDING_HUMAN_VARIANT_REVIEW",
    "4ce8_A__Q9HYN5": "ADMITTED",
    "5avh_A__P24300": "ADMITTED",
}


class FormalAdmissionError(RuntimeError):
    """One structured input, scientific-regression, or output failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class FormalAdmissionConfig:
    """Frozen input bindings and portable output location."""

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
    inventory_ref: str = "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    stage0_admission_ref: str = (
        "experiments/p2_design_baseline/stage0/"
        "stage0_intervention_admission_v1.jsonl"
    )
    output_root_ref: str = DEFAULT_OUTPUT_ROOT
    expected_scale1a0_table_sha256: str = (
        "695f3587dae24b5945f975a7611c677fca9a5e89f728d2f3802f77b5a31a2d4d"
    )
    expected_scale1a0_summary_sha256: str = (
        "afa01243db4c5833d040af3585ca6d998a924b9fd351d8d083b7e6b6f5253f10"
    )
    expected_scale1a0_manifest_sha256: str = (
        "de4baae228b15adda0ad41bba674d6a45470a728be92b374f20aa30b72ef4fca"
    )
    expected_inventory_sha256: str = (
        "05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf"
    )
    expected_stage0_admission_sha256: str = (
        "1fa86cde49cdac68433572d929a99c964574c8edc7f9f62832e0c2b3d4a529c0"
    )
    expected_source_count: int = 213
    expected_evaluated_count: int = 72
    expected_unevaluated_count: int = 141
    mismatch_budget: int = 3
    paired_identity_threshold: float = 0.99


@dataclass(frozen=True)
class A0Gate:
    source_frame: pd.DataFrame
    evaluated_frame: pd.DataFrame
    unevaluated_count: int
    input_artifacts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class IdentityAdmissionDecision:
    status: str
    reason_code: str
    numerical_screen_pass: bool
    exact_variant_authorized: bool
    failed_predicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage0AdmissionGate:
    records: tuple[dict[str, Any], ...]
    input_artifact: dict[str, Any]


@dataclass(frozen=True)
class FormalAdmissionResult:
    census: pd.DataFrame
    common_masks: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FormalAdmissionFrameResult:
    """Scientific results for one validated local-ready candidate frame."""

    census: pd.DataFrame
    common_masks: pd.DataFrame
    stage0_regression: dict[str, Any]


@dataclass(frozen=True)
class AdmissionCandidateResult:
    """Canonical candidate-level evaluator output without release regression."""

    census: pd.DataFrame
    common_masks: pd.DataFrame


_ADMISSION_COLUMNS = (
    "candidate_id",
    "sampling_frame_index",
    "canonical_source_row",
    "polymer_entity_id",
    "pair_id",
    "pdb_id",
    "pdb_chain",
    "canonical_accession",
    "canonical_sequence_source_relative",
    "canonical_sequence_source_file_sha256",
    "mapping_metadata_source_relative",
    "mapping_metadata_source_sha256",
    "pdb_file_path_relative",
    "pdb_file_sha256",
    "afdb_file_path_relative",
    "afdb_file_sha256",
    "admission_status",
    "terminal_reason_code",
    "terminal_reason_details",
    "canonical_sequence_length",
    "canonical_sequence_sha256",
    "mapped_residue_count",
    "mapped_uniprot_start",
    "mapped_uniprot_end",
    "coordinate_bearing_pdb_residue_count",
    "coordinate_nonobservable_count",
    "mismatch_count",
    "observed_variants_json",
    "paired_sequence_identity",
    "identity_mismatch_budget",
    "identity_threshold",
    "identity_numerical_screen_pass",
    "exact_variant_authorized",
    "selected_model_entity_id",
    "selected_fragment_start",
    "selected_fragment_end",
    "common_mask_count",
    "common_mask_fraction_of_mapped",
    "new_common_mask_threshold_applied",
)

_COMMON_MASK_COLUMNS = (
    "candidate_id",
    "sampling_frame_index",
    "polymer_entity_id",
    "pair_id",
    "canonical_accession",
    "canonical_position",
    "canonical_aa",
    "mapping_present",
    "mapping_aa",
    "auth_asym_id",
    "auth_seq_id",
    "insertion_code",
    "label_asym_id",
    "label_seq_id",
    "pdb_observed_aa",
    "afdb_observed_aa",
    "observability_evaluation_status",
    "pdb_mapped",
    "afdb_mapped",
    "pdb_backbone_complete",
    "afdb_backbone_complete",
    "common_mask",
)


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise FormalAdmissionError(
            "nonportable_input_path", f"Invalid logical path: {logical_ref}"
        ) from exc


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise FormalAdmissionError(
            "nonportable_output_path", f"Path is outside declared roots: {path.name}"
        ) from exc


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalAdmissionError(
            "unreadable_upstream_input", f"Unable to read {label}: {path.name}"
        ) from exc


def _require_file_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FormalAdmissionError(
            "missing_upstream_input", f"Missing required {label}: {path.name}"
        )
    digest = sha256_file(path)
    if digest != expected:
        raise FormalAdmissionError(
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch: expected {expected}, found {digest}",
            expected_sha256=expected,
            observed_sha256=digest,
        )
    return {"path": path, "sha256": digest, "label": label}


def validate_a0_gate(
    paths: ProjectPaths, config: FormalAdmissionConfig
) -> A0Gate:
    """Validate byte identity, schema, counts and canonical evaluated ordering."""
    bindings = (
        (
            config.scale1a0_table_ref,
            config.expected_scale1a0_table_sha256,
            "Scale-1A0 candidate audit",
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
        (config.inventory_ref, config.expected_inventory_sha256, "candidate inventory"),
    )
    artifacts: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    for logical, expected, label in bindings:
        path = _resolve(paths, logical)
        record = _require_file_hash(path, expected, label)
        resolved[logical] = path
        artifacts.append(
            {"path": logical, "sha256": record["sha256"], "label": label}
        )
    try:
        source = pd.read_parquet(resolved[config.scale1a0_table_ref])
    except Exception as exc:
        raise FormalAdmissionError(
            "invalid_scale1a0_table", "Unable to read Scale-1A0 candidate audit"
        ) from exc
    required = {
        "candidate_id",
        "sampling_frame_index",
        "polymer_entity_id",
        "pair_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "canonical_sequence_source_relative",
        "canonical_sequence_sha256",
        "mapping_metadata_source_relative",
        "pdb_file_path_relative",
        "afdb_file_path_relative",
        "local_availability_status",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise FormalAdmissionError(
            "invalid_scale1a0_schema",
            f"Scale-1A0 table lacks required columns: {missing}",
        )
    if len(source) != config.expected_source_count:
        raise FormalAdmissionError(
            "scale1a0_count_mismatch", "Scale-1A0 source row count is not frozen"
        )
    expected_order = list(range(1, config.expected_source_count + 1))
    if source["sampling_frame_index"].astype(int).tolist() != expected_order:
        raise FormalAdmissionError(
            "scale1a0_order_mismatch", "Canonical sampling-frame order changed"
        )
    if source["pair_id"].duplicated().any() or source["candidate_id"].duplicated().any():
        raise FormalAdmissionError(
            "scale1a0_identity_collision", "Scale-1A0 candidate identities are not unique"
        )
    evaluated = source.loc[
        source["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A")
    ].copy()
    if len(evaluated) != config.expected_evaluated_count:
        raise FormalAdmissionError(
            "scale1a0_evaluated_count_mismatch", "Local-ready count is not frozen"
        )
    unevaluated = len(source) - len(evaluated)
    if unevaluated != config.expected_unevaluated_count:
        raise FormalAdmissionError(
            "scale1a0_unevaluated_count_mismatch", "Unevaluated count is not frozen"
        )
    validate_local_ready_files(evaluated, paths)
    summary = _read_json(resolved[config.scale1a0_summary_ref], "Scale-1A0 summary")
    manifest = _read_json(resolved[config.scale1a0_manifest_ref], "Scale-1A0 manifest")
    inventory = _read_json(resolved[config.inventory_ref], "candidate inventory")
    if (
        summary.get("n_sampling_frame") != config.expected_source_count
        or summary.get("n_local_ready") != config.expected_evaluated_count
        or manifest.get("scale1a0_status") != "SCALE1A0_PASS_SAMPLING_FRAME_READY"
        or len(inventory.get("candidates", [])) != config.expected_source_count
    ):
        raise FormalAdmissionError(
            "scale1a0_metadata_mismatch", "Scale-1A0 JSON contracts disagree with the table"
        )
    inventory_identities = [
        (
            int(record.get("candidate_index", -1)),
            str(record.get("pair_id", "")),
            str(record.get("polymer_entity_id", "")),
        )
        for record in inventory["candidates"]
    ]
    source_identities = list(
        zip(
            source["sampling_frame_index"].astype(int),
            source["pair_id"].astype(str),
            source["polymer_entity_id"].astype(str),
            strict=True,
        )
    )
    if inventory_identities != source_identities:
        raise FormalAdmissionError(
            "source_identity_drift",
            "Scale-1A0 identities/order disagree with the frozen candidate inventory",
        )
    return A0Gate(
        source_frame=source,
        evaluated_frame=evaluated.reset_index(drop=True),
        unevaluated_count=unevaluated,
        input_artifacts=tuple(artifacts),
    )


def validate_local_ready_files(frame: pd.DataFrame, paths: ProjectPaths) -> None:
    """Require every Scale-1A0 local-ready scientific input to remain readable."""
    missing_columns = sorted(set(_LOCAL_READY_PATH_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise FormalAdmissionError(
            "invalid_scale1a0_schema",
            f"Local-ready frame lacks path columns: {missing_columns}",
        )
    failures: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        for column in _LOCAL_READY_PATH_COLUMNS:
            logical = row.get(column)
            if not isinstance(logical, str) or not logical:
                failures.append(
                    {
                        "pair_id": row.get("pair_id"),
                        "field": column,
                        "reason": "missing_logical_reference",
                    }
                )
                continue
            try:
                path = _resolve(paths, logical)
                if not path.is_file():
                    raise OSError("not a regular file")
                with path.open("rb") as handle:
                    handle.read(1)
            except (OSError, ProjectPathError, FormalAdmissionError) as exc:
                failures.append(
                    {
                        "pair_id": row.get("pair_id"),
                        "field": column,
                        "logical_path": logical,
                        "reason": str(exc),
                    }
                )
    if failures:
        raise FormalAdmissionError(
            "blocked_local_input_regression",
            f"Scale-1A0 local-ready inputs are no longer readable ({len(failures)})",
            failures=failures,
        )


def classify_identity_admission(
    *,
    mismatch_count: int,
    paired_identity: float,
    observed_variants: Sequence[str],
    authorized_variants: Sequence[str] | None,
    mismatch_budget: int,
    identity_threshold: float,
) -> IdentityAdmissionDecision:
    """Apply frozen numerical gates, then the independent human-authorization gate."""
    failed_predicates = tuple(
        predicate
        for predicate, failed in (
            ("sequence_variant_budget_exceeded", mismatch_count > mismatch_budget),
            (
                "sequence_identity_below_threshold",
                paired_identity < identity_threshold,
            ),
        )
        if failed
    )
    if failed_predicates:
        return IdentityAdmissionDecision(
            IDENTITY_CONTRACT_FAIL,
            failed_predicates[0],
            False,
            False,
            failed_predicates,
        )
    if mismatch_count == 0:
        return IdentityAdmissionDecision(
            FORMALLY_ADMITTED, "exact_canonical_identity", True, False
        )
    observed = tuple(observed_variants)
    authorized = None if authorized_variants is None else tuple(authorized_variants)
    exact_authorized = (
        authorized is not None
        and len(set(authorized)) == len(authorized)
        and len(set(observed)) == len(observed)
        and set(authorized) == set(observed)
    )
    if exact_authorized:
        return IdentityAdmissionDecision(
            FORMALLY_ADMITTED, "exact_variant_authorization", True, True
        )
    return IdentityAdmissionDecision(
        PENDING_HUMAN_VARIANT_REVIEW,
        "pending_exact_variant_authorization",
        True,
        False,
    )


def validate_stage0_admission_ledger(
    paths: ProjectPaths, config: FormalAdmissionConfig
) -> Stage0AdmissionGate:
    """Gate the frozen Stage-0 ledger before it can affect authorization or regression."""
    path = _resolve(paths, config.stage0_admission_ref)
    binding = _require_file_hash(
        path,
        config.expected_stage0_admission_sha256,
        "frozen Stage-0 intervention admission ledger",
    )
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalAdmissionError(
            "unreadable_stage0_admission_ledger",
            "Unable to read frozen Stage-0 admission ledger",
        ) from exc
    if not all(isinstance(record, dict) for record in records):
        raise FormalAdmissionError(
            "invalid_stage0_admission_ledger", "Stage-0 ledger records must be objects"
        )
    protein_ids = [str(record.get("protein_id", "")) for record in records]
    observed_identities = set(protein_ids)
    expected_identities = set(_EXPECTED_STAGE0_STATUS_BY_ID)
    if (
        len(protein_ids) != len(expected_identities)
        or len(observed_identities) != len(protein_ids)
        or observed_identities != expected_identities
    ):
        raise FormalAdmissionError(
            "stage0_ledger_identity_mismatch",
            "Stage-0 ledger does not contain the exact unique frozen 11-member set",
            expected=sorted(expected_identities),
            observed=sorted(observed_identities),
        )
    observed_status = {
        protein_id: str(record.get("intervention_admission_status", ""))
        for protein_id, record in zip(protein_ids, records, strict=True)
    }
    if observed_status != _EXPECTED_STAGE0_STATUS_BY_ID:
        raise FormalAdmissionError(
            "stage0_ledger_status_mismatch",
            "Stage-0 ledger classifications differ from the frozen 8/2/1 contract",
        )
    counts = Counter(observed_status.values())
    expected_raw_counts = {"ADMITTED": 8, "PENDING_HUMAN_VARIANT_REVIEW": 2, "IDENTITY_CONTRACT_FAIL": 1}
    if dict(counts) != expected_raw_counts:
        raise FormalAdmissionError(
            "stage0_ledger_status_mismatch", "Stage-0 ledger counts are not 8/2/1"
        )
    for record in records:
        variants = record.get("authorized_sequence_variants")
        assigned = record.get("admission_basis", {}).get(
            "variant_authorization_assigned"
        )
        if variants is not None and (
            not isinstance(variants, list)
            or not all(isinstance(value, str) and value for value in variants)
        ):
            raise FormalAdmissionError(
                "invalid_variant_authorization", "Frozen authorization is malformed"
            )
        if not isinstance(assigned, bool) or assigned != (variants is not None):
            raise FormalAdmissionError(
                "invalid_variant_authorization",
                "Authorization declaration and explicit variant set disagree",
            )
    return Stage0AdmissionGate(
        records=tuple(records),
        input_artifact={
            "path": config.stage0_admission_ref,
            "sha256": binding["sha256"],
            "label": binding["label"],
        },
    )


def _load_authorizations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    authorizations: dict[str, tuple[str, ...]] = {}
    for record in records:
        variants = record.get("authorized_sequence_variants")
        if variants is None:
            continue
        authorizations[str(record["protein_id"])] = tuple(variants)
    return authorizations


def _load_mapping(row: Mapping[str, Any], paths: ProjectPaths) -> pd.DataFrame:
    mapping_path = _resolve(paths, str(row["mapping_metadata_source_relative"]))
    pdb_path = _resolve(paths, str(row["pdb_file_path_relative"]))
    accession = str(row["canonical_accession"])
    chain = str(row["pdb_chain"])
    try:
        if mapping_path.name.endswith(".xml.gz"):
            mapping = parse_sifts_mapping_with_explicit_labels(
                mapping_path,
                pdb_path,
                chain_id=chain,
                uniprot_id=accession,
            )
        elif mapping_path.suffix.lower() == ".parquet":
            mapping = normalize_residue_mapping(pd.read_parquet(mapping_path))
        else:
            raise FormalAdmissionError(
                "unsupported_mapping_artifact",
                f"Unsupported mapping artifact: {mapping_path.name}",
            )
    except FormalAdmissionError:
        raise
    except (AmbiguousLegacyResidueIdentifier, LookupError, OSError, ValueError) as exc:
        code = (
            "ambiguous_sifts_author_residue_identifier"
            if isinstance(exc, AmbiguousLegacyResidueIdentifier)
            else "invalid_or_missing_sifts_mapping"
        )
        raise FormalAdmissionError(code, str(exc)) from exc
    if mapping.empty:
        raise FormalAdmissionError("empty_mapping", "Residue mapping is empty")
    if "uniprot_id" in mapping and {
        str(value).strip() for value in mapping["uniprot_id"]
    } != {accession}:
        raise FormalAdmissionError(
            "mapping_uniprot_identity_mismatch", "Mapping accession is not exact"
        )
    try:
        pdb_ca = load_chain_ca_table(pdb_path, chain)
        mapping, _, _ = mapping_with_provenance(mapping, pdb_ca)
    except (AmbiguousLegacyResidueIdentifier, ValueError) as exc:
        raise FormalAdmissionError("invalid_mapping_provenance", str(exc)) from exc
    return mapping


def load_formal_candidate_mapping(
    row: Mapping[str, Any], paths: ProjectPaths
) -> pd.DataFrame:
    """Expose the frozen Scale-1A1 mapping loader for acquisition dependencies."""
    return _load_mapping(row, paths)


def _exact_selected_record(
    records: Sequence[Mapping[str, Any]], accession: str, model_id: str
) -> dict[str, Any]:
    matches = [
        dict(record)
        for record in records
        if str(record.get("uniprotAccession", "")) == accession
        and str(record.get("modelEntityId") or record.get("entryId") or "") == model_id
    ]
    if len(matches) != 1:
        raise FormalAdmissionError(
            "ambiguous_selected_model_record",
            f"Expected one exact selected model record; found {len(matches)}",
        )
    return matches[0]


def _base_mask_frame(row: Mapping[str, Any], sequence: str) -> pd.DataFrame:
    size = len(sequence)
    return pd.DataFrame(
        {
            "candidate_id": [str(row["candidate_id"])] * size,
            "sampling_frame_index": [int(row["sampling_frame_index"])] * size,
            "polymer_entity_id": [str(row["polymer_entity_id"])] * size,
            "pair_id": [str(row["pair_id"])] * size,
            "canonical_accession": [str(row["canonical_accession"])] * size,
            "canonical_position": np.arange(1, size + 1, dtype=int),
            "canonical_aa": list(sequence),
            "mapping_present": [False] * size,
            "mapping_aa": [None] * size,
            "auth_asym_id": [None] * size,
            "auth_seq_id": pd.Series([pd.NA] * size, dtype="Int64"),
            "insertion_code": [None] * size,
            "label_asym_id": [None] * size,
            "label_seq_id": pd.Series([pd.NA] * size, dtype="Int64"),
            "pdb_observed_aa": [None] * size,
            "afdb_observed_aa": [None] * size,
            "observability_evaluation_status": ["NOT_EVALUATED"] * size,
            "pdb_mapped": pd.Series([pd.NA] * size, dtype="boolean"),
            "afdb_mapped": pd.Series([pd.NA] * size, dtype="boolean"),
            "pdb_backbone_complete": pd.Series([pd.NA] * size, dtype="boolean"),
            "afdb_backbone_complete": pd.Series([pd.NA] * size, dtype="boolean"),
            "common_mask": pd.Series([pd.NA] * size, dtype="boolean"),
        }
    )


def build_not_evaluated_common_mask(
    row: Mapping[str, Any], paths: ProjectPaths
) -> pd.DataFrame:
    """Represent canonical positions whose scientific evaluation did not run."""
    source = _resolve(paths, str(row["canonical_sequence_source_relative"]))
    canonical = extract_canonical_sequence_from_source(
        source.read_bytes(), str(row["canonical_accession"])
    )
    return _base_mask_frame(row, str(canonical["sequence"]))


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def _clean_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise FormalAdmissionError(
            "invalid_residue_identity", "Residue identity is not an integer"
        )
    return int(numeric)


def _residue_groups_by_namespace(records: Sequence[Any]) -> tuple[dict[ResidueKey, Any], dict[tuple[str, int], Any]]:
    auth: dict[ResidueKey, Any] = {}
    label: dict[tuple[str, int], Any] = {}
    for group in group_residue_records(records):
        auth[group.residue.key] = group
        provenance = group.residue
        if provenance.label_chain_id is None or provenance.label_seq_id is None:
            continue
        key = (provenance.label_chain_id, provenance.label_seq_id)
        if key in label:
            raise FormalAdmissionError(
                "ambiguous_label_residue_mapping",
                "Structure contains duplicate requested label residue identities",
            )
        label[key] = group
    return auth, label


def _complete_and_aa(group: Any, source: str) -> tuple[bool, str | None]:
    if group is None:
        return False, None
    try:
        from .models import select_backbone_atoms

        selection = select_backbone_atoms(group.atoms)
    except ValueError as exc:
        raise FormalAdmissionError(
            "ambiguous_backbone_atom",
            f"Unable to select unique {source} backbone atoms",
        ) from exc
    return not selection.missing_atoms, selection.residue.canonical_aa


def _mask_with_mapping_provenance(
    row: Mapping[str, Any], mapping: pd.DataFrame, sequence: str
) -> pd.DataFrame:
    """Bind validated mapping facts before any structure observability work."""
    masks = _base_mask_frame(row, sequence)
    for mapping_row in mapping.itertuples(index=False):
        position = int(mapping_row.uniprot_position)
        mapping_aa = residue_name_to_one_letter(mapping_row.uniprot_residue_name)
        canonical_aa = sequence[position - 1]
        if mapping_aa != canonical_aa:
            raise FormalAdmissionError(
                "mapping_amino_acid_mismatch",
                f"Mapping conflicts with canonical sequence at UniProt {position}",
            )
        mask_index = position - 1
        masks.loc[mask_index, "mapping_present"] = True
        masks.loc[mask_index, "mapping_aa"] = mapping_aa
        masks.loc[mask_index, "auth_asym_id"] = _clean_text(mapping_row.auth_asym_id)
        masks.loc[mask_index, "auth_seq_id"] = _clean_int(mapping_row.auth_seq_id)
        masks.loc[mask_index, "insertion_code"] = (
            _clean_text(mapping_row.insertion_code) or ""
        )
        masks.loc[mask_index, "label_asym_id"] = _clean_text(
            mapping_row.label_asym_id
        )
        masks.loc[mask_index, "label_seq_id"] = _clean_int(mapping_row.label_seq_id)
    return masks


def _mapped_observability(
    *,
    row: Mapping[str, Any],
    mapping: pd.DataFrame,
    masks: pd.DataFrame,
    sequence: str,
    fragment: AFDBFragment,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, tuple[str, ...], float, int, int]:
    pair_id = str(row["pair_id"])
    pdb_path = _resolve(paths, str(row["pdb_file_path_relative"]))
    afdb_path = _resolve(paths, str(row["afdb_file_path_relative"]))
    try:
        pdb_records = _load_atom_records(pdb_path, pair_id)
        afdb_records = _load_atom_records(afdb_path, fragment.model_entity_id)
    except P1ValidationError as exc:
        raise FormalAdmissionError(exc.code, str(exc)) from exc
    pdb_auth, pdb_label = _residue_groups_by_namespace(pdb_records)
    afdb_auth, _ = _residue_groups_by_namespace(afdb_records)
    afdb_by_position = {
        key.auth_seq_id: group for key, group in afdb_auth.items() if not key.insertion_code
    }
    if set(afdb_by_position) != set(range(1, fragment.model_residue_count + 1)):
        raise FormalAdmissionError(
            "afdb_model_position_mismatch", "AFDB model positions are not exactly 1..N"
        )

    evaluated_masks = masks.copy(deep=True)
    observable_columns = [
        "pdb_mapped",
        "afdb_mapped",
        "pdb_backbone_complete",
        "afdb_backbone_complete",
        "common_mask",
    ]
    evaluated_masks.loc[:, observable_columns] = False
    evaluated_masks.loc[:, "observability_evaluation_status"] = "EVALUATED"
    observed_variants: list[str] = []
    afdb_mismatches: list[str] = []
    pdb_mapped_count = 0
    for mapping_row in mapping.itertuples(index=False):
        position = int(mapping_row.uniprot_position)
        mapping_aa = residue_name_to_one_letter(mapping_row.uniprot_residue_name)
        canonical_aa = sequence[position - 1]
        if mapping_aa != canonical_aa:
            raise FormalAdmissionError(
                "mapping_amino_acid_mismatch",
                f"Mapping conflicts with canonical sequence at UniProt {position}",
            )
        auth_chain = _clean_text(mapping_row.auth_asym_id)
        auth_seq = _clean_int(mapping_row.auth_seq_id)
        label_chain = _clean_text(mapping_row.label_asym_id)
        label_seq = _clean_int(mapping_row.label_seq_id)
        insertion = _clean_text(mapping_row.insertion_code) or ""
        if auth_chain is not None and auth_seq is not None:
            pdb_group = pdb_auth.get(
                ResidueKey(pair_id, auth_chain, auth_seq, insertion)
            )
        elif label_chain is not None and label_seq is not None:
            pdb_group = pdb_label.get((label_chain, label_seq))
        else:
            raise FormalAdmissionError(
                "invalid_mapping_provenance",
                "Mapping row has neither explicit auth nor label residue identity",
            )
        model_position = position - fragment.uniprot_start + 1
        afdb_group = afdb_by_position.get(model_position)
        pdb_complete, pdb_aa = _complete_and_aa(pdb_group, "pdb")
        afdb_complete, afdb_aa = _complete_and_aa(afdb_group, "afdb")
        if pdb_group is not None:
            pdb_mapped_count += 1
            if pdb_aa != canonical_aa:
                observed_variants.append(f"{canonical_aa}{position}{pdb_aa}")
        if afdb_group is not None and afdb_aa != canonical_aa:
            afdb_mismatches.append(f"{canonical_aa}{position}{afdb_aa}")
        mask_index = position - 1
        evaluated_masks.loc[mask_index, [
            "pdb_mapped",
            "afdb_mapped",
            "pdb_backbone_complete",
            "afdb_backbone_complete",
            "common_mask",
        ]] = [
            pdb_group is not None,
            afdb_group is not None,
            pdb_complete,
            afdb_complete,
            pdb_complete and afdb_complete,
        ]
        evaluated_masks.loc[mask_index, "pdb_observed_aa"] = pdb_aa
        evaluated_masks.loc[mask_index, "afdb_observed_aa"] = afdb_aa
    if afdb_mismatches:
        raise FormalAdmissionError(
            "afdb_amino_acid_mismatch",
            f"AFDB/model identity conflicts with canonical mapping: {afdb_mismatches[:5]}",
            afdb_mismatches=afdb_mismatches,
        )
    paired_identity = (
        1.0 - len(observed_variants) / pdb_mapped_count
        if pdb_mapped_count
        else 0.0
    )
    return (
        evaluated_masks,
        tuple(observed_variants),
        paired_identity,
        len(mapping) - pdb_mapped_count,
        int(evaluated_masks["common_mask"].sum()),
    )


def _candidate_base(
    row: Mapping[str, Any], config: FormalAdmissionConfig
) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "sampling_frame_index": int(row["sampling_frame_index"]),
        "canonical_source_row": int(row["canonical_source_row"]),
        "polymer_entity_id": str(row["polymer_entity_id"]),
        "pair_id": str(row["pair_id"]),
        "pdb_id": str(row["pdb_id"]),
        "pdb_chain": str(row["pdb_chain"]),
        "canonical_accession": str(row["canonical_accession"]),
        "canonical_sequence_source_relative": str(
            row["canonical_sequence_source_relative"]
        ),
        "canonical_sequence_source_file_sha256": None,
        "mapping_metadata_source_relative": str(
            row["mapping_metadata_source_relative"]
        ),
        "mapping_metadata_source_sha256": None,
        "pdb_file_path_relative": str(row["pdb_file_path_relative"]),
        "pdb_file_sha256": None,
        "afdb_file_path_relative": str(row["afdb_file_path_relative"]),
        "afdb_file_sha256": None,
        "admission_status": None,
        "terminal_reason_code": None,
        "terminal_reason_details": None,
        "canonical_sequence_length": None,
        "canonical_sequence_sha256": None,
        "mapped_residue_count": None,
        "mapped_uniprot_start": None,
        "mapped_uniprot_end": None,
        "coordinate_bearing_pdb_residue_count": None,
        "coordinate_nonobservable_count": None,
        "mismatch_count": None,
        "observed_variants_json": "[]",
        "paired_sequence_identity": None,
        "identity_mismatch_budget": config.mismatch_budget,
        "identity_threshold": config.paired_identity_threshold,
        "identity_numerical_screen_pass": None,
        "exact_variant_authorized": False,
        "selected_model_entity_id": None,
        "selected_fragment_start": None,
        "selected_fragment_end": None,
        "common_mask_count": None,
        "common_mask_fraction_of_mapped": None,
        "new_common_mask_threshold_applied": False,
    }


def _failed_candidate(
    base: dict[str, Any], status: str, error: FormalAdmissionError
) -> dict[str, Any]:
    return {
        **base,
        "admission_status": status,
        "terminal_reason_code": error.code,
        "terminal_reason_details": json.dumps(
            {"message": str(error), **error.details}, sort_keys=True
        ),
    }


def _evaluate_candidate(
    row: Mapping[str, Any],
    *,
    paths: ProjectPaths,
    config: FormalAdmissionConfig,
    authorizations: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    base = _candidate_base(row, config)
    sequence: str | None = None
    masks = pd.DataFrame()
    try:
        bound_paths = {
            "canonical_sequence_source_file_sha256": _resolve(
                paths, str(row["canonical_sequence_source_relative"])
            ),
            "mapping_metadata_source_sha256": _resolve(
                paths, str(row["mapping_metadata_source_relative"])
            ),
            "pdb_file_sha256": _resolve(
                paths, str(row["pdb_file_path_relative"])
            ),
            "afdb_file_sha256": _resolve(
                paths, str(row["afdb_file_path_relative"])
            ),
        }
        base.update(
            {field: sha256_file(path) for field, path in bound_paths.items()}
        )
        accession = str(row["canonical_accession"])
        pair_id = str(row["pair_id"])
        expected_pair = f"{str(row['pdb_id']).lower()}_{row['pdb_chain']}__{accession}"
        if pair_id.lower() != expected_pair.lower():
            raise FormalAdmissionError(
                "candidate_identity_mismatch", "PDB/chain/UniProt binding is inconsistent"
            )
        canonical_path = _resolve(
            paths, str(row["canonical_sequence_source_relative"])
        )
        canonical_bytes = canonical_path.read_bytes()
        canonical = extract_canonical_sequence_from_source(
            canonical_bytes, accession
        )
        sequence = str(canonical["sequence"])
        if canonical["sequence_sha256"] != str(row["canonical_sequence_sha256"]):
            raise FormalAdmissionError(
                "canonical_sequence_sha256_mismatch",
                "Canonical sequence digest disagrees with Scale-1A0 binding",
            )
        base.update(
            {
                "canonical_sequence_length": len(sequence),
                "canonical_sequence_sha256": canonical["sequence_sha256"],
            }
        )
        masks = _base_mask_frame(row, sequence)
        pdb_path = _resolve(paths, str(row["pdb_file_path_relative"]))
        try:
            _validate_pdb_mmcif(pdb_path, str(row["pdb_id"]))
        except P0ValidationError as exc:
            raise FormalAdmissionError(exc.code, str(exc)) from exc
        metadata_source = row.get("afdb_metadata_source_relative")
        metadata_path = (
            _resolve(paths, str(metadata_source))
            if isinstance(metadata_source, str) and metadata_source
            else canonical_path
        )
        records = metadata_records(metadata_path.read_bytes())
    except (OSError, UnicodeError, DerivationError, FormalAdmissionError) as exc:
        error = (
            exc
            if isinstance(exc, FormalAdmissionError)
            else FormalAdmissionError(
                getattr(exc, "code", "invalid_provenance_binding"), str(exc)
            )
        )
        if not masks.empty:
            masks.loc[:, "observability_evaluation_status"] = (
                "NOT_EVALUATED_AFTER_PROVENANCE_FAILURE"
            )
        return _failed_candidate(base, PROVENANCE_FAIL, error), masks

    failure_status = MAPPING_FAIL
    try:
        mapping = _load_mapping(row, paths)
        positions = mapping["uniprot_position"].astype(int)
        if positions.min() < 1 or positions.max() > len(sequence):
            raise FormalAdmissionError(
                "uniprot_position_mismatch", "Mapping exceeds canonical sequence"
            )
        for mapped in mapping.itertuples(index=False):
            position = int(mapped.uniprot_position)
            if residue_name_to_one_letter(mapped.uniprot_residue_name) != sequence[
                position - 1
            ]:
                raise FormalAdmissionError(
                    "mapping_amino_acid_mismatch",
                    f"Mapping conflicts with canonical sequence at UniProt {position}",
                )
        masks = _mask_with_mapping_provenance(row, mapping, sequence)
        base.update(
            {
                "mapped_residue_count": len(mapping),
                "mapped_uniprot_start": int(positions.min()),
                "mapped_uniprot_end": int(positions.max()),
            }
        )
        fragment_result = resolve_exact_fragment(
            records, accession, (int(positions.min()), int(positions.max()))
        )
        fragment = fragment_result["selected_fragment"]
        if fragment is None:
            raise FormalAdmissionError(
                str(fragment_result["fragment_resolution_status"]),
                "Exact-accession fragment resolution failed",
            )
        failure_status = PROVENANCE_FAIL
        selected_record = _exact_selected_record(
            records, accession, fragment.model_entity_id
        )
        version = selected_record.get("latestVersion")
        if isinstance(version, bool) or not isinstance(version, int):
            raise FormalAdmissionError(
                "invalid_model_version", "Selected model version is not explicit"
            )
        model_path = _resolve(paths, str(row["afdb_file_path_relative"]))
        validate_frozen_model_artifacts(
            selected_record,
            model_id=fragment.model_entity_id,
            version=version,
            model_path=model_path,
            expected_length=fragment.model_residue_count,
        )
        base.update(
            {
                "selected_model_entity_id": fragment.model_entity_id,
                "selected_fragment_start": fragment.uniprot_start,
                "selected_fragment_end": fragment.uniprot_end,
            }
        )
        masks, variants, paired_identity, missing_pdb, common_count = (
            _mapped_observability(
                row=row,
                mapping=mapping,
                masks=masks,
                sequence=sequence,
                fragment=fragment,
                paths=paths,
            )
        )
        # The frozen helper remains an executable regression oracle whenever the
        # sequence is exact (variant candidates intentionally fail its P1 gate).
        if not variants:
            expected_common = set(
                paired_common_backbone_positions(
                    mapping,
                    canonical_sequence=sequence,
                    pair_id=pair_id,
                    pdb_path=pdb_path,
                    afdb_path=model_path,
                    fragment=fragment,
                )
            )
            observed_common = set(
                masks.loc[masks["common_mask"], "canonical_position"].astype(int)
            )
            if observed_common != expected_common:
                raise FormalAdmissionError(
                    "common_mask_semantics_regression",
                    "Scale-1 common mask disagrees with frozen Stage-0 helper",
                )
    except (
        AmbiguousLegacyResidueIdentifier,
        DerivationError,
        P0ValidationError,
        P1ValidationError,
        FormalAdmissionError,
        ValueError,
    ) as exc:
        error = (
            exc
            if isinstance(exc, FormalAdmissionError)
            else FormalAdmissionError(
                getattr(exc, "code", "mapping_or_fragment_failure"), str(exc)
            )
        )
        if error.code == "common_mask_semantics_regression":
            raise FormalAdmissionError(
                "blocked_admission_regression",
                "Scale-1 common mask disagrees with the frozen Stage-0 helper",
                pair_id=str(row["pair_id"]),
                cause_code=error.code,
            ) from exc
        status = (
            MAPPING_FAIL
            if error.code == "afdb_amino_acid_mismatch"
            else failure_status
        )
        if not masks.empty:
            masks.loc[:, "observability_evaluation_status"] = (
                "NOT_EVALUATED_AFTER_PROVENANCE_FAILURE"
                if status == PROVENANCE_FAIL
                else "NOT_EVALUATED_AFTER_MAPPING_FAILURE"
            )
        return _failed_candidate(base, status, error), masks

    decision = classify_identity_admission(
        mismatch_count=len(variants),
        paired_identity=paired_identity,
        observed_variants=variants,
        authorized_variants=authorizations.get(pair_id),
        mismatch_budget=config.mismatch_budget,
        identity_threshold=config.paired_identity_threshold,
    )
    base.update(
        {
            "admission_status": decision.status,
            "terminal_reason_code": decision.reason_code,
            "terminal_reason_details": json.dumps(
                {"failed_predicates": list(decision.failed_predicates)},
                sort_keys=True,
            ),
            "coordinate_bearing_pdb_residue_count": len(mapping) - missing_pdb,
            "coordinate_nonobservable_count": missing_pdb,
            "mismatch_count": len(variants),
            "observed_variants_json": json.dumps(list(variants)),
            "paired_sequence_identity": paired_identity,
            "identity_mismatch_budget": config.mismatch_budget,
            "identity_threshold": config.paired_identity_threshold,
            "identity_numerical_screen_pass": decision.numerical_screen_pass,
            "exact_variant_authorized": decision.exact_variant_authorized,
            "common_mask_count": common_count,
            "common_mask_fraction_of_mapped": common_count / len(mapping),
        }
    )
    return base, masks


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=float)
    result = {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }
    if all(float(value).is_integer() for value in array):
        result["min"] = int(result["min"])
        result["max"] = int(result["max"])
    return result


def _stage0_regression(
    census: pd.DataFrame, gate: Stage0AdmissionGate
) -> dict[str, Any]:
    expected_by_id = {
        str(record["protein_id"]): {
            "ADMITTED": FORMALLY_ADMITTED,
            "PENDING_HUMAN_VARIANT_REVIEW": PENDING_HUMAN_VARIANT_REVIEW,
            "IDENTITY_CONTRACT_FAIL": IDENTITY_CONTRACT_FAIL,
        }[str(record["intervention_admission_status"])]
        for record in gate.records
    }
    observed = census.set_index("pair_id")["admission_status"].to_dict()
    disagreements = [
        {"pair_id": pair_id, "expected": expected, "observed": observed.get(pair_id)}
        for pair_id, expected in expected_by_id.items()
        if observed.get(pair_id) != expected
    ]
    observed_counts = dict(
        Counter(observed[pair_id] for pair_id in expected_by_id if pair_id in observed)
    )
    return {
        "status": "PASS" if not disagreements else "BLOCKED_ADMISSION_REGRESSION",
        "candidate_count": len(expected_by_id),
        "expected_counts": dict(_EXPECTED_STAGE0_COUNTS),
        "observed_counts": {
            status: int(observed_counts.get(status, 0))
            for status in _EXPECTED_STAGE0_COUNTS
        },
        "disagreements": disagreements,
        "source_path": gate.input_artifact["path"],
        "source_sha256": gate.input_artifact["sha256"],
    }


def evaluate_admission_candidates(
    frame: pd.DataFrame,
    *,
    paths: ProjectPaths,
    config: FormalAdmissionConfig,
    stage0_gate: Stage0AdmissionGate,
) -> AdmissionCandidateResult:
    """Apply the canonical evaluator without imposing a release-wide regression set."""
    authorizations = _load_authorizations(stage0_gate.records)
    census_rows: list[dict[str, Any]] = []
    mask_frames: list[pd.DataFrame] = []
    for row in frame.to_dict(orient="records"):
        census, masks = _evaluate_candidate(
            row, paths=paths, config=config, authorizations=authorizations
        )
        census_rows.append(census)
        if not masks.empty:
            mask_frames.append(masks)
    census = pd.DataFrame(census_rows, columns=_ADMISSION_COLUMNS)
    if len(census) != len(frame) or census["pair_id"].duplicated().any():
        raise FormalAdmissionError(
            "evaluated_identity_invariant_failed", "Evaluated census is not exact"
        )
    if census["admission_status"].isna().any() or not set(
        census["admission_status"]
    ).issubset(STATUS_VOCABULARY):
        raise FormalAdmissionError(
            "terminal_status_invariant_failed", "Every evaluated candidate needs one status"
        )
    masks = (
        pd.concat(mask_frames, ignore_index=True)
        if mask_frames
        else pd.DataFrame(columns=_COMMON_MASK_COLUMNS)
    )
    if not masks.empty:
        masks = masks.sort_values(
            ["sampling_frame_index", "canonical_position"], kind="stable"
        ).reset_index(drop=True)
        if masks.duplicated(["sampling_frame_index", "canonical_position"]).any():
            raise FormalAdmissionError(
                "common_mask_identity_collision", "Common-mask identities are not unique"
            )
        expected_common = masks["pdb_backbone_complete"] & masks[
            "afdb_backbone_complete"
        ]
        if not masks["common_mask"].equals(expected_common):
            raise FormalAdmissionError(
                "common_mask_semantics_regression", "Common-mask intersection changed"
            )
    return AdmissionCandidateResult(census=census, common_masks=masks)


def evaluate_formal_admission_frame(
    frame: pd.DataFrame,
    *,
    paths: ProjectPaths,
    config: FormalAdmissionConfig,
    stage0_gate: Stage0AdmissionGate,
) -> FormalAdmissionFrameResult:
    """Apply the unchanged candidate evaluator to one validated local-ready frame."""
    evaluated = evaluate_admission_candidates(
        frame, paths=paths, config=config, stage0_gate=stage0_gate
    )
    regression = _stage0_regression(evaluated.census, stage0_gate)
    return FormalAdmissionFrameResult(
        census=evaluated.census,
        common_masks=evaluated.common_masks,
        stage0_regression=regression,
    )


def audit_formal_admission(
    paths: ProjectPaths,
    *,
    config: FormalAdmissionConfig | None = None,
) -> FormalAdmissionResult:
    """Evaluate exactly the frozen 72-candidate local-ready subset."""
    config = config or FormalAdmissionConfig()
    gate = validate_a0_gate(paths, config)
    stage0_gate = validate_stage0_admission_ledger(paths, config)
    frame_result = evaluate_formal_admission_frame(
        gate.evaluated_frame,
        paths=paths,
        config=config,
        stage0_gate=stage0_gate,
    )
    census = frame_result.census
    masks = frame_result.common_masks
    regression = frame_result.stage0_regression
    if regression["status"] != "PASS":
        raise FormalAdmissionError(
            "blocked_admission_regression",
            "Scale-1A1 disagrees with frozen Stage-0 classifications",
            disagreements=regression["disagreements"],
        )
    counts = Counter(census["admission_status"])
    admitted = census.loc[census["admission_status"].eq(FORMALLY_ADMITTED)]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scale1a1_status": "SCALE1A1_FORMAL_CENSUS_COMPLETE",
        "sampling_frame_count": config.expected_source_count,
        "evaluated_count": len(census),
        "unevaluated_count": gate.unevaluated_count,
        "unevaluated_status": NOT_EVALUATED,
        "terminal_status_counts": {
            status: int(counts.get(status, 0)) for status in STATUS_VOCABULARY
        },
        "conditional_on_local_readiness_admission_yield": (
            int(counts.get(FORMALLY_ADMITTED, 0)) / len(census)
        ),
        "admission_yield_scope": "CONDITIONAL_ON_LOCAL_READINESS",
        "deterministic_admission_funnel": {
            "source_sampling_frame": config.expected_source_count,
            "local_ready_evaluated": len(census),
            "not_evaluated_local_data_incomplete": gate.unevaluated_count,
            "terminal_status_counts": {
                status: int(counts.get(status, 0)) for status in STATUS_VOCABULARY
            },
        },
        "formally_admitted_common_mask": {
            "candidate_count": len(admitted),
            "count_quantiles": _quantiles(
                admitted["common_mask_count"].dropna().astype(int).tolist()
            ),
            "fraction_of_mapped_quantiles": _quantiles(
                admitted["common_mask_fraction_of_mapped"].dropna().astype(float).tolist()
            ),
            "minimum_threshold_applied": None,
        },
        "stage0_regression": regression,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scale1a1_status": "SCALE1A1_FORMAL_CENSUS_COMPLETE",
        "ADMISSION_RATE_SCOPE": "CONDITIONAL_ON_LOCAL_READINESS",
        "SCALE1B_ELIGIBILITY_NOT_DEFINED": True,
        "SCALE1_SCORING_NOT_STARTED": True,
        "upstream_artifacts": [
            *gate.input_artifacts,
            stage0_gate.input_artifact,
        ],
        "candidate_input_bindings": [
            {
                "sampling_frame_index": int(row.sampling_frame_index),
                "pair_id": str(row.pair_id),
                "canonical_sequence_source": {
                    "path": str(row.canonical_sequence_source_relative),
                    "sha256": str(row.canonical_sequence_source_file_sha256),
                },
                "mapping_source": {
                    "path": str(row.mapping_metadata_source_relative),
                    "sha256": str(row.mapping_metadata_source_sha256),
                },
                "pdb_mmcif": {
                    "path": str(row.pdb_file_path_relative),
                    "sha256": str(row.pdb_file_sha256),
                },
                "afdb_structure": {
                    "path": str(row.afdb_file_path_relative),
                    "sha256": str(row.afdb_file_sha256),
                },
            }
            for row in census.itertuples(index=False)
        ],
        "evaluated_subset": {
            "predicate": "local_availability_status == LOCAL_READY_FOR_SCALE1A",
            "source_count": config.expected_source_count,
            "evaluated_count": len(census),
            "unevaluated_count": gate.unevaluated_count,
            "ordering": "Scale-1A0 canonical sampling_frame_index order",
            "unevaluated_status": NOT_EVALUATED,
        },
        "scientific_contract": {
            "identity_contract": {
                "mismatch_count_max": config.mismatch_budget,
                "paired_sequence_identity_min": config.paired_identity_threshold,
                "coordinate_nonobservability_is_not_sequence_mismatch": True,
                "afdb_mismatch_is_mapping_failure_not_authorizable_pdb_variant": True,
            },
            "variant_handling": (
                "numerical identity pass with an unapproved exact mismatch set remains "
                "PENDING_HUMAN_VARIANT_REVIEW"
            ),
            "mapping": "explicit auth identity with label fallback only when auth is absent",
            "fragment": "exact-accession, exactly one full mapped-interval covering record",
            "common_mask": "PDB complete N/CA/C/O intersect AFDB complete N/CA/C/O",
            "common_mask_fraction_denominator": "mapped_residue_count",
            "common_mask_minimum": None,
            "scientific_rule_change": False,
        },
        "status_vocabulary": list(STATUS_VOCABULARY),
        "stage0_regression": regression,
        "scope_confirmations": {
            "network_access_used": False,
            "downloads_performed": False,
            "proteinmpnn_executed": False,
            "fixed_probes_constructed": False,
            "geometry_or_confidence_used_for_admission": False,
            "outcome_dependent_admission_variable_used": False,
            "new_human_authorization_created": False,
            "scale1b_started": False,
        },
        "next_task": "SCALE1_PROTEIN_LEVEL_FEASIBILITY_AND_POWER_PLANNING",
    }
    return FormalAdmissionResult(
        census=census,
        common_masks=masks,
        summary=summary,
        manifest=manifest,
        input_artifacts=(*gate.input_artifacts, stage0_gate.input_artifact),
    )


def _render_json(payload: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise FormalAdmissionError(
            "invalid_admission_output", "Admission JSON cannot be serialized"
        ) from exc


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise FormalAdmissionError(
            "immutable_admission_artifact_conflict",
            f"immutable output differs: {path.name}",
        )
    atomic_write_new_bytes(path, payload)
    return "created"


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
            if path.stat().st_size == temporary.stat().st_size and sha256_file(
                path
            ) == sha256_file(temporary):
                return "reused_identical"
            raise FormalAdmissionError(
                "immutable_admission_artifact_conflict",
                f"immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FormalAdmissionError(
                "immutable_admission_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_formal_admission(
    result: FormalAdmissionResult,
    paths: ProjectPaths,
    *,
    config: FormalAdmissionConfig | None = None,
) -> dict[str, Any]:
    """Write two Parquets and summary, then the manifest, without overwrite."""
    config = config or FormalAdmissionConfig()
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    census_path = output_root / "scale1_formal_admission_census.parquet"
    masks_path = output_root / "scale1_common_masks.parquet"
    summary_path = output_root / "scale1_formal_admission_summary.json"
    manifest_path = output_root / "scale1_formal_admission_manifest.json"
    write_status = {
        "census": _write_immutable_parquet(census_path, result.census),
        "common_masks": _write_immutable_parquet(masks_path, result.common_masks),
        "summary": _write_immutable_bytes(summary_path, _render_json(result.summary)),
    }
    outputs = {
        "formal_admission_census": {
            "path": _logical(paths, census_path),
            "rows": len(result.census),
            "sha256": sha256_file(census_path),
        },
        "common_masks": {
            "path": _logical(paths, masks_path),
            "rows": len(result.common_masks),
            "sha256": sha256_file(masks_path),
        },
        "formal_admission_summary": {
            "path": _logical(paths, summary_path),
            "sha256": sha256_file(summary_path),
        },
        "manifest": {
            "path": _logical(paths, manifest_path),
            "self_hash_policy": "reported_externally_to_avoid_recursive_self-hash",
        },
    }
    manifest = {**result.manifest, "output_artifacts": outputs}
    write_status["manifest"] = _write_immutable_bytes(
        manifest_path, _render_json(manifest)
    )
    return {
        "scale1a1_status": result.manifest["scale1a1_status"],
        "write_status": write_status,
        "outputs": {
            **outputs,
            "manifest": {
                "path": _logical(paths, manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
    }


def run_formal_admission(
    paths: ProjectPaths,
    *,
    config: FormalAdmissionConfig | None = None,
) -> dict[str, Any]:
    config = config or FormalAdmissionConfig()
    result = audit_formal_admission(paths, config=config)
    return materialize_formal_admission(result, paths, config=config)
