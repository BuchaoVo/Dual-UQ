"""Offline sampling-frame and local-availability audit.

This module inventories existing evidence only.  It neither performs formal
candidate admission nor invokes any structure, sequence-generation, or scoring
workflow.
"""

from __future__ import annotations

import gzip
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.policies.identity import (
    exact_accession_records,
    extract_canonical_sequence,
    metadata_records,
)

SCHEMA_VERSION = "dual-uq.scale1-sampling-frame-audit.v1"
DEFAULT_OUTPUT_ROOT = "experiments/p2_design_baseline/scale1/scale1a0"

SAMPLING_FRAME_VALID = "SAMPLING_FRAME_VALID"
SAMPLING_FRAME_INVALID = "SAMPLING_FRAME_INVALID_OUTCOME_DEPENDENT"
SAMPLING_FRAME_INSUFFICIENT = "SAMPLING_FRAME_PROVENANCE_INSUFFICIENT"
SAMPLING_FRAME_AMBIGUOUS = "SAMPLING_FRAME_AMBIGUOUS"

PASS_STATUS = "SCALE1A0_PASS_SAMPLING_FRAME_READY"
BLOCKED_FRAME_STATUS = "SCALE1A0_BLOCKED_SAMPLING_FRAME"

STATUS_PRECEDENCE = (
    ("ambiguous_local_source", "MULTIPLE_LOCAL_SOURCE_AMBIGUITY"),
    ("unreadable_local_file", "UNREADABLE_LOCAL_FILE"),
    ("missing_both_structures", "MISSING_PDB_AND_AFDB"),
    ("missing_pdb", "MISSING_PDB"),
    ("missing_afdb", "MISSING_AFDB"),
    ("missing_canonical_sequence", "MISSING_CANONICAL_SEQUENCE"),
    ("missing_identity_metadata", "MISSING_IDENTITY_METADATA"),
    ("missing_mapping_metadata", "MISSING_MAPPING_METADATA"),
    ("missing_provenance_metadata", "MISSING_PROVENANCE_METADATA"),
    ("other_structured_local_blocker", "OTHER_STRUCTURED_LOCAL_BLOCKER"),
)

HISTORICAL_NEUTRALITY_HASHES = (
    (
        "scripts/13_discover_screening_pool.py",
        "b4706d39bb1d178d5acac82328df1a2ea7ad0c9b5c6894ee1f7eec02ede61327",
    ),
    (
        "src/dual_uq/rcsb_discovery.py",
        "db70dfcecf21588be1a8ea84da36cac1e06ea4de1985598a6e9221ef073960e8",
    ),
    (
        "configs/legacy/a0_screening/screening_pool.yaml",
        "2d8b2933e38212618a0e056f78608609c3a90e0fa6a643b4c4ab8b0974ff3694",
    ),
    (
        "reports/screening_pool_discovery.log",
        "93c98e3030c39450f7ef3743ef619224bcd2ae74d699abcbbd42194d4048f39e",
    ),
    (
        "reports/screening_pool_discovery_summary.json",
        "469e0eed7d49ce644c5202f8b7b3c06d31269302c445b8143d69c237e7b56aa2",
    ),
)


class SamplingFrameAuditError(RuntimeError):
    """Structured sampling-frame, census, or immutable-output failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class NeutralityEvidence:
    """Human-audited, source-bound facts used for a deterministic conclusion."""

    candidate_selection_precedes_stage0_scoring: bool
    source_derivation_reproducible: bool
    outcome_fields_used_in_frame_selection: tuple[str, ...]
    geometry_enrichment_used: bool
    uncertainty_enrichment_used: bool
    plddt_recorded_but_not_used_for_frame_inclusion: bool
    evidence_complete: bool


DEFAULT_NEUTRALITY_EVIDENCE = NeutralityEvidence(
    candidate_selection_precedes_stage0_scoring=True,
    source_derivation_reproducible=True,
    outcome_fields_used_in_frame_selection=(),
    geometry_enrichment_used=False,
    uncertainty_enrichment_used=False,
    plddt_recorded_but_not_used_for_frame_inclusion=True,
    evidence_complete=True,
)


@dataclass(frozen=True)
class SamplingFrameAuditConfig:
    """Portable bindings for the historical frame and read-only local census."""

    inventory_ref: str = "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    inventory_table_ref: str = "artifacts/dataset/reports/census/candidate_inventory_v1.tsv"
    discovery_ref: str = "data/processed/discovery/discovered_candidates.parquet"
    stage0_panel_ref: str = (
        "experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl"
    )
    stage0_admission_ref: str = (
        "experiments/p2_design_baseline/stage0/stage0_intervention_admission_v1.jsonl"
    )
    stage0_protein_manifest_ref: str = (
        "experiments/p2_design_baseline/stage0/protein_manifest.json"
    )
    output_root_ref: str = DEFAULT_OUTPUT_ROOT
    expected_total_discovered: int = 220
    expected_candidates: int = 213
    expected_stage0_declared: int = 11
    expected_stage0_admitted: int = 8
    neutrality_source_hashes: tuple[tuple[str, str], ...] = HISTORICAL_NEUTRALITY_HASHES
    neutrality_evidence: NeutralityEvidence = DEFAULT_NEUTRALITY_EVIDENCE


@dataclass(frozen=True)
class SamplingFrameAuditResult:
    """Single structured source for the table, summary, and manifest."""

    candidates: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise SamplingFrameAuditError(
            "nonportable_input_path", f"Invalid logical path: {logical_ref}"
        ) from exc


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise SamplingFrameAuditError(
            "nonportable_input_path", f"Input is outside declared roots: {path.name}"
        ) from exc


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SamplingFrameAuditError(
            "unreadable_audit_input", f"Unable to read {label}: {path.name}"
        ) from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SamplingFrameAuditError(
            "unreadable_audit_input", f"Unable to read {label}: {path.name}"
        ) from exc
    if not all(isinstance(value, dict) for value in values):
        raise SamplingFrameAuditError(
            "invalid_audit_input", f"{label} must contain JSON objects"
        )
    return values


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _candidate_identity(record: Mapping[str, Any]) -> tuple[int, str, str, str, str]:
    try:
        index = int(record["candidate_index"])
        pair_id = str(record["pair_id"])
        pdb_id = str(record["PDB"]).strip().lower()
        chain = str(record["chain"]).strip()
        accession = str(record["UniProt"]).strip().upper()
    except (KeyError, TypeError, ValueError) as exc:
        raise SamplingFrameAuditError(
            "invalid_sampling_frame_schema", "Sampling-frame identity is incomplete"
        ) from exc
    if not chain or pair_id != f"{pdb_id}_{chain}__{accession}":
        raise SamplingFrameAuditError(
            "invalid_sampling_frame_identity", f"Candidate identity disagrees: {pair_id}"
        )
    return index, pair_id, pdb_id, chain, accession


def _load_sampling_frame(
    paths: ProjectPaths, config: SamplingFrameAuditConfig
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[tuple[str, str], ...]]:
    inventory_path = _resolve(paths, config.inventory_ref)
    inventory_table_path = _resolve(paths, config.inventory_table_ref)
    discovery_path = _resolve(paths, config.discovery_ref)
    payload = _read_json(inventory_path, "candidate inventory")
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise SamplingFrameAuditError(
            "invalid_sampling_frame_schema", "Candidate inventory lacks candidates"
        )
    records = payload["candidates"]
    if len(records) != config.expected_candidates:
        raise SamplingFrameAuditError(
            "sampling_frame_count_mismatch",
            f"Expected {config.expected_candidates} candidates; observed {len(records)}",
        )
    identities = [_candidate_identity(record) for record in records]
    if [identity[0] for identity in identities] != list(
        range(1, config.expected_candidates + 1)
    ):
        raise SamplingFrameAuditError(
            "sampling_frame_order_mismatch", "Candidate indices are not source-order 1..N"
        )
    uniqueness = {
        "candidate_index": [identity[0] for identity in identities],
        "pair_id": [identity[1] for identity in identities],
        "polymer_entity_id": [str(record.get("polymer_entity_id", "")) for record in records],
        "biological_tuple": [identity[2:] for identity in identities],
    }
    duplicates = [name for name, values in uniqueness.items() if len(values) != len(set(values))]
    if duplicates or any(not value for value in uniqueness["polymer_entity_id"]):
        raise SamplingFrameAuditError(
            "sampling_frame_duplicate_identity",
            "Duplicate or empty canonical identity: " + ", ".join(duplicates),
        )

    try:
        discovery = pd.read_parquet(discovery_path).reset_index(drop=True)
    except (OSError, ValueError) as exc:
        raise SamplingFrameAuditError(
            "unreadable_sampling_frame_source", "Canonical discovery Parquet is unreadable"
        ) from exc
    if len(discovery) != config.expected_total_discovered:
        raise SamplingFrameAuditError(
            "sampling_frame_source_count_mismatch",
            f"Expected {config.expected_total_discovered} discovered rows; observed {len(discovery)}",
        )
    required = {"polymer_entity_id", "pdb_id", "chain_id", "uniprot_id", "discovery_status"}
    if not required.issubset(discovery.columns):
        raise SamplingFrameAuditError(
            "invalid_sampling_frame_source_schema", "Discovery source lacks identity columns"
        )
    eligible = discovery.loc[discovery["discovery_status"].eq("eligible")]
    if len(eligible) != len(records):
        raise SamplingFrameAuditError(
            "sampling_frame_source_count_mismatch", "Eligible discovery count differs from inventory"
        )
    observed = [
        (
            int(record["canonical_source_row"]),
            str(record["polymer_entity_id"]),
            str(record["PDB"]).lower(),
            str(record["chain"]),
            str(record["UniProt"]).upper(),
        )
        for record in records
    ]
    expected = [
        (
            int(index),
            str(row.polymer_entity_id),
            str(row.pdb_id).lower(),
            str(row.chain_id),
            str(row.uniprot_id).upper(),
        )
        for index, row in eligible.iterrows()
    ]
    if observed != expected:
        raise SamplingFrameAuditError(
            "sampling_frame_artifact_disagreement",
            "Inventory identities/order disagree with the canonical discovery source",
        )
    try:
        inventory_table = pd.read_csv(inventory_table_path, sep="\t")
    except (OSError, ValueError) as exc:
        raise SamplingFrameAuditError(
            "unreadable_sampling_frame_companion",
            "Canonical inventory TSV companion is unreadable",
        ) from exc
    companion_required = {
        "candidate_index",
        "polymer_entity_id",
        "pair_id",
        "PDB",
        "chain",
        "UniProt",
    }
    if len(inventory_table) != len(records) or not companion_required.issubset(
        inventory_table.columns
    ):
        raise SamplingFrameAuditError(
            "sampling_frame_companion_mismatch",
            "Inventory TSV row count or schema differs from the JSON artifact",
        )
    companion_identity = [
        (
            int(row.candidate_index),
            str(row.polymer_entity_id),
            str(row.pair_id),
            str(row.PDB).lower(),
            str(row.chain),
            str(row.UniProt).upper(),
        )
        for row in inventory_table.itertuples(index=False)
    ]
    json_identity = [
        (
            int(record["candidate_index"]),
            str(record["polymer_entity_id"]),
            str(record["pair_id"]),
            str(record["PDB"]).lower(),
            str(record["chain"]),
            str(record["UniProt"]).upper(),
        )
        for record in records
    ]
    if companion_identity != json_identity:
        raise SamplingFrameAuditError(
            "sampling_frame_companion_mismatch",
            "Inventory TSV identity/order differs from the JSON artifact",
        )
    source_metadata = payload.get("source_metadata")
    if not isinstance(source_metadata, dict):
        raise SamplingFrameAuditError(
            "sampling_frame_provenance_missing", "Inventory lacks source metadata"
        )
    if (
        source_metadata.get("canonical_source_path") != config.discovery_ref
        or source_metadata.get("source_sha256") != sha256_file(discovery_path)
        or int(source_metadata.get("total_row_count", -1)) != len(discovery)
        or int(source_metadata.get("eligible_row_count", -1)) != len(records)
    ):
        raise SamplingFrameAuditError(
            "sampling_frame_provenance_mismatch", "Inventory source binding is inconsistent"
        )
    return (
        [dict(record) for record in records],
        payload,
        (
            (config.inventory_ref, sha256_file(inventory_path)),
            (config.inventory_table_ref, sha256_file(inventory_table_path)),
            (config.discovery_ref, sha256_file(discovery_path)),
        ),
    )


def _neutrality_status(evidence: NeutralityEvidence) -> str:
    if not evidence.evidence_complete:
        return SAMPLING_FRAME_INSUFFICIENT
    if not evidence.source_derivation_reproducible:
        return SAMPLING_FRAME_AMBIGUOUS
    if (
        evidence.outcome_fields_used_in_frame_selection
        or evidence.geometry_enrichment_used
        or evidence.uncertainty_enrichment_used
    ):
        return SAMPLING_FRAME_INVALID
    if not evidence.candidate_selection_precedes_stage0_scoring:
        return SAMPLING_FRAME_INVALID
    return SAMPLING_FRAME_VALID


def _audit_neutrality(
    paths: ProjectPaths, config: SamplingFrameAuditConfig
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    expected = dict(config.neutrality_source_hashes)
    observed: list[tuple[str, str]] = []
    mismatches: list[dict[str, Any]] = []
    for logical_ref, digest in config.neutrality_source_hashes:
        path = _resolve(paths, logical_ref)
        if not path.is_file():
            mismatches.append(
                {"path": logical_ref, "expected_sha256": digest, "observed_sha256": None}
            )
            continue
        current = sha256_file(path)
        observed.append((logical_ref, current))
        if current != expected[logical_ref]:
            mismatches.append(
                {"path": logical_ref, "expected_sha256": digest, "observed_sha256": current}
            )
    evidence = config.neutrality_evidence
    if mismatches:
        evidence = replace(evidence, evidence_complete=False)
    status = _neutrality_status(evidence)
    payload = {
        "status": status,
        "candidate_selection_precedes_stage0_scoring": (
            evidence.candidate_selection_precedes_stage0_scoring
        ),
        "source_derivation_reproducible": evidence.source_derivation_reproducible,
        "outcome_fields_used_in_selection": list(
            evidence.outcome_fields_used_in_frame_selection
        ),
        "geometry_enrichment_used": evidence.geometry_enrichment_used,
        "uncertainty_enrichment_used": evidence.uncertainty_enrichment_used,
        "plddt_recorded_but_not_used_for_frame_inclusion": (
            evidence.plddt_recorded_but_not_used_for_frame_inclusion
        ),
        "source_hash_mismatches": mismatches,
        "allowed_upstream_filters": [
            "protein-only PDB polymer entity",
            "X-ray or electron-microscopy record",
            "resolution <= 3.0 A",
            "PDB entity length 100..500",
            "single exact UniProt reference",
            "AFDB metadata availability",
            "source-order duplicate removal by polymer_entity_id",
        ],
        "disallowed_stage0_outcomes_used": [],
        "audit_interpretation": (
            "The 213-row frame is the eligible discovery universe, not the downstream "
            "36-row pLDDT/length-stratified screening pool. AFDB global pLDDT was recorded "
            "during discovery but did not determine discovery_status eligibility."
        ),
        "limitations": (
            "The frame is source-neutral relative to Stage-0 outcomes, but it remains an "
            "upstream availability/quality-selected PDB-AFDB frame rather than a proteome-wide "
            "random sample."
        ),
        "historical_inventory_commit": "e58ffbe8756b06803ef1ddaed73330079c467992",
        "first_stage0_panel_commit": "9cf144656b96ef26a14c42d7a7ff6d400e2c8f15",
        "source_evidence": [
            {"path": logical_ref, "sha256": digest}
            for logical_ref, digest in sorted(observed)
        ],
    }
    return payload, tuple(observed)


def classify_local_availability(flags: Mapping[str, bool]) -> tuple[str, str | None]:
    """Apply the frozen deterministic terminal-status precedence."""
    normalized = dict(flags)
    normalized["missing_both_structures"] = bool(
        normalized.get("missing_both_structures")
        or normalized.get("missing_pdb") and normalized.get("missing_afdb")
    )
    reasons = {
        "MULTIPLE_LOCAL_SOURCE_AMBIGUITY": "multiple_distinct_local_sources",
        "UNREADABLE_LOCAL_FILE": "one_or_more_present_local_files_unreadable",
        "MISSING_PDB_AND_AFDB": "pdb_and_afdb_structures_absent",
        "MISSING_PDB": "pdb_structure_absent",
        "MISSING_AFDB": "afdb_structure_absent",
        "MISSING_CANONICAL_SEQUENCE": "canonical_sequence_not_locally_proven",
        "MISSING_IDENTITY_METADATA": "canonical_identity_metadata_absent",
        "MISSING_MAPPING_METADATA": "pair_or_mapping_metadata_absent",
        "MISSING_PROVENANCE_METADATA": "candidate_provenance_metadata_absent",
        "OTHER_STRUCTURED_LOCAL_BLOCKER": "local_identity_or_structure_binding_unresolved",
    }
    for field_name, status in STATUS_PRECEDENCE:
        if normalized.get(field_name, False):
            return status, reasons[status]
    return "LOCAL_READY_FOR_SCALE1A", None


def _forensic_metadata_sources(paths: ProjectPaths) -> dict[int, list[Path]]:
    """Index immutable rejected metadata by its candidate-index directory.

    Both canonical and historical layouts are discovered through the declared
    artifacts root.  Exact-accession validation still happens record-by-record;
    directory discovery never authorizes an alias or fallback.
    """
    index: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(paths.artifacts_root.rglob("afdb_metadata/*.json")):
        try:
            candidate_index = int(path.parent.parent.name)
        except ValueError:
            continue
        index[candidate_index].append(path)
    return dict(index)


def _metadata_evidence(
    paths: ProjectPaths,
    *,
    candidate_index: int,
    accession: str,
    resolution_sources: Mapping[int, list[Path]],
) -> dict[str, Any]:
    root = paths.raw_root / "afdb" / accession
    candidates = sorted(root.rglob("metadata.json")) if root.is_dir() else []
    candidates.extend(resolution_sources.get(candidate_index, []))
    unique_paths = sorted(set(candidates), key=lambda item: _logical(paths, item))
    valid: list[dict[str, Any]] = []
    unreadable = False
    input_files: list[Path] = []
    for path in unique_paths:
        input_files.append(path)
        try:
            raw = path.read_bytes()
            exact = exact_accession_records(metadata_records(raw), accession)
        except (OSError, DerivationError):
            unreadable = True
            continue
        if len(exact) != 1:
            unreadable = True
            continue
        record = exact[0]
        model_id = record.get("modelEntityId") or record.get("entryId")
        sequence = record.get("uniprotSequence") or record.get("sequence")
        if not isinstance(model_id, str) or not model_id.strip():
            unreadable = True
            continue
        signature = (
            model_id.strip(),
            sequence if isinstance(sequence, str) else None,
            record.get("sequenceStart"),
            record.get("sequenceEnd"),
        )
        canonical = None
        try:
            canonical = extract_canonical_sequence(raw, accession)
        except DerivationError:
            pass
        valid.append(
            {
                "path": path,
                "signature": signature,
                "model_id": model_id.strip(),
                "canonical": canonical,
            }
        )
    signatures = {item["signature"] for item in valid}
    ambiguous = len(signatures) > 1
    selected = None
    if valid and not ambiguous:
        selected = min(
            valid,
            key=lambda item: (
                0 if item["path"].name == "metadata.json" else 1,
                len(item["path"].parts),
                _logical(paths, item["path"]),
            ),
        )
    return {
        "present": bool(unique_paths),
        "identity_resolved": bool(selected),
        "model_id": selected["model_id"] if selected else None,
        "canonical_present": bool(selected and selected["canonical"]),
        "canonical_sequence_sha256": (
            selected["canonical"]["sequence_sha256"]
            if selected and selected["canonical"]
            else None
        ),
        "selected_path": selected["path"] if selected else None,
        "ambiguous": ambiguous,
        "unreadable": unreadable,
        "input_files": input_files,
    }


def _parse_mmcif(path: Path) -> dict[str, Any] | None:
    try:
        payload = MMCIF2Dict(str(path))
    except Exception:  # noqa: BLE001 - parser failures become audit evidence
        return None
    return payload if isinstance(payload, dict) and payload else None


def _pdb_evidence(
    paths: ProjectPaths, *, pdb_id: str, chain: str, polymer_entity_id: str
) -> dict[str, Any]:
    path = paths.raw_root / "pdb" / f"{pdb_id}.cif"
    if not path.is_file():
        return {
            "present": False,
            "path": None,
            "readable": False,
            "chain_identifiable": False,
            "entity_identifiable": False,
            "identity_resolved": False,
            "input_files": [],
        }
    parsed = _parse_mmcif(path)
    if parsed is None:
        return {
            "present": True,
            "path": path,
            "readable": False,
            "chain_identifiable": False,
            "entity_identifiable": False,
            "identity_resolved": False,
            "input_files": [path],
        }
    entries = {str(value).strip().lower() for value in _as_list(parsed.get("_entry.id"))}
    chains = {
        str(value).strip()
        for field_name in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        for value in _as_list(parsed.get(field_name))
    }
    entity_id = polymer_entity_id.split("_", 1)[1] if "_" in polymer_entity_id else None
    entities = {
        str(value).strip() for value in _as_list(parsed.get("_atom_site.label_entity_id"))
    }
    chain_ok = chain in chains
    entity_ok = entity_id is not None and entity_id in entities
    entry_ok = entries == {pdb_id.lower()}
    readable = bool(_as_list(parsed.get("_atom_site.label_atom_id")))
    return {
        "present": True,
        "path": path,
        "readable": readable,
        "chain_identifiable": chain_ok,
        "entity_identifiable": entity_ok,
        "identity_resolved": readable and entry_ok and chain_ok and entity_ok,
        "input_files": [path],
    }


def _afdb_evidence(
    paths: ProjectPaths, *, accession: str, expected_model_id: str | None
) -> dict[str, Any]:
    root = paths.raw_root / "afdb" / accession
    files = (
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (path.name == "model.cif" or path.name.endswith("-model_v6.cif"))
        )
        if root.is_dir()
        else []
    )
    parsed_records: list[dict[str, Any]] = []
    unreadable = False
    for path in files:
        parsed = _parse_mmcif(path)
        if parsed is None or not _as_list(parsed.get("_atom_site.label_atom_id")):
            unreadable = True
            continue
        entries = [str(value).strip() for value in _as_list(parsed.get("_entry.id"))]
        if len(entries) != 1:
            unreadable = True
            continue
        model_id = entries[0]
        matches = (
            model_id == expected_model_id
            if expected_model_id is not None
            else model_id.startswith(f"AF-{accession}-")
        )
        parsed_records.append(
            {"path": path, "model_id": model_id, "matches": matches, "sha256": sha256_file(path)}
        )
    matching = [record for record in parsed_records if record["matches"]]
    distinct = {record["sha256"] for record in matching}
    ambiguous = len(distinct) > 1
    selected = None
    if matching and not ambiguous:
        selected = min(
            matching,
            key=lambda record: (
                0 if record["path"].name == "model.cif" else 1,
                len(record["path"].parts),
                _logical(paths, record["path"]),
            ),
        )
    return {
        "present": bool(files),
        "path": selected["path"] if selected else (files[0] if files else None),
        "readable": bool(selected),
        "accession_identifiable": bool(selected),
        "model_id": selected["model_id"] if selected else expected_model_id,
        "ambiguous": ambiguous,
        "unreadable": unreadable,
        "other_identity": bool(parsed_records and not matching),
        "input_files": files,
    }


def _mapping_evidence(
    paths: ProjectPaths,
    *,
    pair_id: str,
    pdb_id: str,
    stage0_mapping_ids: set[str],
    stage0_manifest_path: Path,
) -> dict[str, Any]:
    pair_mapping = paths.processed_root / "pairs" / pair_id / "residue_mapping.parquet"
    sifts = paths.raw_root / "mappings" / f"{pdb_id}.xml.gz"
    candidates = [path for path in (pair_mapping, sifts) if path.is_file()]
    valid: list[Path] = []
    unreadable = False
    for path in candidates:
        try:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                if frame.empty or not {
                    "uniprot_residue_number",
                    "uniprot_position",
                }.intersection(frame.columns):
                    raise ValueError("mapping table lacks UniProt positions")
            else:
                with gzip.open(path, "rb") as handle:
                    ET.parse(handle)
        except (OSError, ValueError, ET.ParseError):
            unreadable = True
            continue
        valid.append(path)
    if not valid and pair_id in stage0_mapping_ids:
        valid.append(stage0_manifest_path)
    selected = min(
        valid,
        key=lambda item: (
            0 if item.suffix == ".parquet" else 1,
            _logical(paths, item),
        ),
    ) if valid else None
    return {
        "present": bool(selected),
        "path": selected,
        "unreadable": unreadable,
        "input_files": candidates,
    }


def _stage0_traceability(
    paths: ProjectPaths,
    config: SamplingFrameAuditConfig,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str], tuple[tuple[str, str], ...]]:
    panel_path = _resolve(paths, config.stage0_panel_ref)
    admission_path = _resolve(paths, config.stage0_admission_ref)
    manifest_path = _resolve(paths, config.stage0_protein_manifest_ref)
    panel = _read_jsonl(panel_path, "Stage-0 panel")
    admission = _read_jsonl(admission_path, "Stage-0 admission")
    manifest = _read_json(manifest_path, "Stage-0 protein manifest")
    if len(panel) != config.expected_stage0_declared:
        raise SamplingFrameAuditError(
            "stage0_traceability_count_mismatch", "Stage-0 declared count differs"
        )
    admitted = [
        row for row in admission if row.get("intervention_admission_status") == "ADMITTED"
    ]
    if len(admitted) != config.expected_stage0_admitted:
        raise SamplingFrameAuditError(
            "stage0_traceability_count_mismatch", "Stage-0 admitted count differs"
        )
    by_index = {int(row["candidate_index"]): str(row["pair_id"]) for row in records}
    exceptions: list[dict[str, Any]] = []
    for cohort, values in (("declared", panel), ("admitted", admitted)):
        for row in values:
            index = int(row["candidate_index"])
            protein_id = str(row["protein_id"])
            if by_index.get(index) != protein_id:
                exceptions.append(
                    {
                        "cohort": cohort,
                        "candidate_index": index,
                        "protein_id": protein_id,
                        "sampling_frame_pair_id": by_index.get(index),
                    }
                )
    manifest_proteins = manifest.get("proteins", []) if isinstance(manifest, dict) else []
    mapping_ids = {
        str(row.get("protein_id"))
        for row in manifest_proteins
        if isinstance(row, dict) and isinstance(row.get("mapping_provenance"), dict)
    }
    trace = {
        "declared_count": len(panel),
        "declared_traceable_count": len(panel)
        - sum(item["cohort"] == "declared" for item in exceptions),
        "admitted_count": len(admitted),
        "admitted_traceable_count": len(admitted)
        - sum(item["cohort"] == "admitted" for item in exceptions),
        "declared_candidate_indices": sorted(int(row["candidate_index"]) for row in panel),
        "admitted_candidate_indices": sorted(int(row["candidate_index"]) for row in admitted),
        "exceptions": exceptions,
        "historical_artifacts_modified": False,
    }
    inputs = tuple(
        (ref, sha256_file(_resolve(paths, ref)))
        for ref in (
            config.stage0_panel_ref,
            config.stage0_admission_ref,
            config.stage0_protein_manifest_ref,
        )
    )
    return trace, mapping_ids, inputs


def _candidate_record(
    paths: ProjectPaths,
    *,
    source: Mapping[str, Any],
    sampling_frame_source: str,
    resolution_sources: Mapping[int, list[Path]],
    stage0_mapping_ids: set[str],
    stage0_manifest_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    index, pair_id, pdb_id, chain, accession = _candidate_identity(source)
    polymer_entity_id = str(source["polymer_entity_id"])
    metadata = _metadata_evidence(
        paths,
        candidate_index=index,
        accession=accession,
        resolution_sources=resolution_sources,
    )
    pdb = _pdb_evidence(
        paths,
        pdb_id=pdb_id,
        chain=chain,
        polymer_entity_id=polymer_entity_id,
    )
    afdb = _afdb_evidence(
        paths, accession=accession, expected_model_id=metadata["model_id"]
    )
    mapping = _mapping_evidence(
        paths,
        pair_id=pair_id,
        pdb_id=pdb_id,
        stage0_mapping_ids=stage0_mapping_ids,
        stage0_manifest_path=stage0_manifest_path,
    )
    identity_metadata_present = True  # the authoritative inventory carries the exact tuple
    provenance_metadata_present = True  # source row + source hash are bound by the inventory
    missing_pdb = not pdb["present"]
    missing_afdb = not afdb["present"]
    flags = {
        "ambiguous_local_source": bool(metadata["ambiguous"] or afdb["ambiguous"]),
        "unreadable_local_file": bool(
            metadata["unreadable"]
            or pdb["present"] and not pdb["readable"]
            or afdb["unreadable"]
            or mapping["unreadable"]
        ),
        "missing_pdb": missing_pdb,
        "missing_afdb": missing_afdb,
        "missing_both_structures": missing_pdb and missing_afdb,
        "missing_canonical_sequence": not metadata["canonical_present"],
        "missing_identity_metadata": not identity_metadata_present,
        "missing_mapping_metadata": not mapping["present"],
        "missing_provenance_metadata": not provenance_metadata_present,
        "other_structured_local_blocker": bool(
            pdb["present"]
            and pdb["readable"]
            and not pdb["identity_resolved"]
            or afdb["present"]
            and not afdb["accession_identifiable"]
            and not afdb["ambiguous"]
            and not afdb["unreadable"]
        ),
    }
    status, reason = classify_local_availability(flags)
    local_pair_complete = bool(
        pdb["present"]
        and pdb["readable"]
        and pdb["identity_resolved"]
        and afdb["present"]
        and afdb["readable"]
        and afdb["accession_identifiable"]
        and not flags["ambiguous_local_source"]
    )
    local_metadata_complete = bool(
        metadata["canonical_present"]
        and identity_metadata_present
        and mapping["present"]
        and provenance_metadata_present
        and not flags["unreadable_local_file"]
        and not flags["ambiguous_local_source"]
    )
    record = {
        "candidate_id": pair_id,
        "sampling_frame_index": index,
        "canonical_source_row": int(source["canonical_source_row"]),
        "polymer_entity_id": polymer_entity_id,
        "pair_id": pair_id,
        "canonical_accession": accession,
        "pdb_id": pdb_id,
        "pdb_chain": chain,
        "afdb_id": afdb["model_id"],
        "sampling_frame_source": sampling_frame_source,
        "pdb_file_present": bool(pdb["present"]),
        "pdb_file_path_relative": _logical(paths, pdb["path"]) if pdb["path"] else None,
        "pdb_file_readable": bool(pdb["readable"]),
        "pdb_chain_identifiable": bool(pdb["chain_identifiable"]),
        "pdb_entity_identifiable": bool(pdb["entity_identifiable"]),
        "afdb_file_present": bool(afdb["present"]),
        "afdb_file_path_relative": _logical(paths, afdb["path"]) if afdb["path"] else None,
        "afdb_file_readable": bool(afdb["readable"]),
        "afdb_accession_identifiable": bool(afdb["accession_identifiable"]),
        "canonical_accession_present": True,
        "canonical_sequence_present": bool(metadata["canonical_present"]),
        "canonical_sequence_source_relative": (
            _logical(paths, metadata["selected_path"])
            if metadata["canonical_present"] and metadata["selected_path"]
            else None
        ),
        "canonical_sequence_sha256": metadata["canonical_sequence_sha256"],
        "identity_metadata_present": identity_metadata_present,
        "identity_metadata_source_relative": sampling_frame_source,
        "mapping_metadata_present": bool(mapping["present"]),
        "mapping_metadata_source_relative": (
            _logical(paths, mapping["path"]) if mapping["path"] else None
        ),
        "provenance_metadata_present": provenance_metadata_present,
        "provenance_metadata_source_relative": sampling_frame_source,
        "local_pair_complete": local_pair_complete,
        "local_metadata_complete": local_metadata_complete,
        "missing_pdb": flags["missing_pdb"],
        "missing_afdb": flags["missing_afdb"],
        "missing_canonical_sequence": flags["missing_canonical_sequence"],
        "missing_identity_metadata": flags["missing_identity_metadata"],
        "missing_mapping_metadata": flags["missing_mapping_metadata"],
        "missing_provenance_metadata": flags["missing_provenance_metadata"],
        "ambiguous_local_source": flags["ambiguous_local_source"],
        "unreadable_local_file": flags["unreadable_local_file"],
        "other_structured_local_blocker": flags["other_structured_local_blocker"],
        "local_availability_status": status,
        "terminal_local_missing_reason": reason,
        "stage0_declared_member": False,
        "stage0_admitted_member": False,
    }
    input_files = [
        *metadata["input_files"],
        *pdb["input_files"],
        *afdb["input_files"],
        *mapping["input_files"],
    ]
    return record, input_files


def _summary(candidates: pd.DataFrame, *, scale1a0_status: str) -> dict[str, Any]:
    statuses = candidates["local_availability_status"].value_counts().sort_index().to_dict()
    missing_columns = [
        "missing_pdb",
        "missing_afdb",
        "missing_canonical_sequence",
        "missing_identity_metadata",
        "missing_mapping_metadata",
        "missing_provenance_metadata",
    ]
    combinations: Counter[str] = Counter()
    for row in candidates[missing_columns].itertuples(index=False, name=None):
        names = [name for name, value in zip(missing_columns, row, strict=True) if value]
        combinations["+".join(names) if names else "none"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "scale1a0_status": scale1a0_status,
        "n_sampling_frame": len(candidates),
        "n_unique_identity": int(candidates["candidate_id"].nunique()),
        "n_local_ready": int(
            candidates["local_availability_status"].eq("LOCAL_READY_FOR_SCALE1A").sum()
        ),
        "n_local_pair_complete": int(candidates["local_pair_complete"].sum()),
        "n_local_metadata_complete": int(candidates["local_metadata_complete"].sum()),
        "n_missing_pdb": int(candidates["missing_pdb"].sum()),
        "n_missing_afdb": int(candidates["missing_afdb"].sum()),
        "n_missing_both_structures": int(
            (candidates["missing_pdb"] & candidates["missing_afdb"]).sum()
        ),
        "n_missing_canonical_sequence": int(
            candidates["missing_canonical_sequence"].sum()
        ),
        "n_missing_identity_metadata": int(
            candidates["missing_identity_metadata"].sum()
        ),
        "n_missing_mapping_metadata": int(
            candidates["missing_mapping_metadata"].sum()
        ),
        "n_missing_provenance_metadata": int(
            candidates["missing_provenance_metadata"].sum()
        ),
        "n_ambiguous_local_source": int(candidates["ambiguous_local_source"].sum()),
        "n_unreadable_file": int(candidates["unreadable_local_file"].sum()),
        "n_canonical_sequence_available": int(candidates["canonical_sequence_present"].sum()),
        "n_identity_metadata_available": int(candidates["identity_metadata_present"].sum()),
        "n_mapping_metadata_available": int(candidates["mapping_metadata_present"].sum()),
        "n_provenance_metadata_available": int(
            candidates["provenance_metadata_present"].sum()
        ),
        "status_counts": {str(key): int(value) for key, value in statuses.items()},
        "missing_field_combinations": dict(sorted(combinations.items())),
        "interpretation": (
            "Local readiness is descriptive input availability for future Scale-1A formal "
            "admission; it is not candidate admission or an estimated admission rate."
        ),
    }


def audit_sampling_frame(
    paths: ProjectPaths,
    *,
    config: SamplingFrameAuditConfig | None = None,
) -> SamplingFrameAuditResult:
    """Build the canonical offline Scale-1A0 audit in memory."""
    config = config or SamplingFrameAuditConfig()
    source_records, inventory, source_inputs = _load_sampling_frame(paths, config)
    neutrality, neutrality_inputs = _audit_neutrality(paths, config)
    trace, stage0_mapping_ids, stage0_inputs = _stage0_traceability(
        paths, config, source_records
    )
    resolution_sources = _forensic_metadata_sources(paths)
    stage0_manifest_path = _resolve(paths, config.stage0_protein_manifest_ref)
    records: list[dict[str, Any]] = []
    local_files: set[Path] = set()
    for source in source_records:
        record, candidate_files = _candidate_record(
            paths,
            source=source,
            sampling_frame_source=config.inventory_ref,
            resolution_sources=resolution_sources,
            stage0_mapping_ids=stage0_mapping_ids,
            stage0_manifest_path=stage0_manifest_path,
        )
        records.append(record)
        local_files.update(candidate_files)
    frame = pd.DataFrame(records)
    declared = set(trace["declared_candidate_indices"])
    admitted = set(trace["admitted_candidate_indices"])
    frame["stage0_declared_member"] = frame["sampling_frame_index"].isin(declared)
    frame["stage0_admitted_member"] = frame["sampling_frame_index"].isin(admitted)
    frame = frame.sort_values("sampling_frame_index", kind="mergesort").reset_index(drop=True)
    for column in frame.columns:
        if frame[column].isna().any():
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    scale1a0_status = (
        PASS_STATUS if neutrality["status"] == SAMPLING_FRAME_VALID else BLOCKED_FRAME_STATUS
    )
    summary = _summary(frame, scale1a0_status=scale1a0_status)
    all_inputs = {
        *source_inputs,
        *neutrality_inputs,
        *stage0_inputs,
        *((_logical(paths, path), sha256_file(path)) for path in local_files if path.is_file()),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scale1a0_status": scale1a0_status,
        "source_candidate_artifact": {
            "path": config.inventory_ref,
            "sha256": sha256_file(_resolve(paths, config.inventory_ref)),
            "schema_version": inventory.get("schema_version"),
            "inventory_status": inventory.get("inventory_status"),
        },
        "source_candidate_artifacts": [
            {
                "path": config.inventory_ref,
                "sha256": sha256_file(_resolve(paths, config.inventory_ref)),
                "format": "JSON",
                "schema_version": inventory.get("schema_version"),
            },
            {
                "path": config.inventory_table_ref,
                "sha256": sha256_file(_resolve(paths, config.inventory_table_ref)),
                "format": "TSV",
                "columns": list(
                    pd.read_csv(
                        _resolve(paths, config.inventory_table_ref), sep="\t"
                    ).columns
                ),
            },
        ],
        "canonical_discovery_source": {
            "path": config.discovery_ref,
            "sha256": sha256_file(_resolve(paths, config.discovery_ref)),
            "total_row_count": config.expected_total_discovered,
            "eligible_row_count": config.expected_candidates,
        },
        "candidate_count": len(frame),
        "unique_identity_count": int(frame["candidate_id"].nunique()),
        "candidate_ordering_semantics": (
            "Preserve canonical discovery source row order after filtering "
            "discovery_status == eligible; candidate_index is 1-based source-order convenience ID."
        ),
        "deduplication_semantics": (
            "No rows are silently removed by this audit. The historical source checkpoint "
            "deduplicated polymer_entity_id with keep=last; the frozen eligible artifact must be "
            "unique by polymer_entity_id, pair_id, and (PDB, chain, UniProt)."
        ),
        "candidate_index_semantics": (
            "Inventory convenience ID only; not historical screening_index, admission, or "
            "permanent protein identity."
        ),
        "sampling_frame_neutrality": neutrality,
        "stage0_traceability": trace,
        "local_path_resolution": {
            "implementation": "dual_uq.core.paths.ProjectPaths",
            "serialized_paths": "project-relative logical references only",
            "current_working_directory_dependency": False,
        },
        "availability_contract": {
            "local_pair_complete": (
                "Both exact PDB and AFDB mmCIF files are present, readable, identity-bound, and "
                "not locally ambiguous."
            ),
            "local_metadata_complete": (
                "Canonical full-span sequence, inventory identity, pair/mapping metadata, and "
                "source provenance are locally present and readable."
            ),
            "identity_metadata_present": (
                "The authoritative inventory row supplies the exact polymer entity, PDB, chain, "
                "and UniProt tuple; AFDB exact-record identity is separately reported."
            ),
            "provenance_metadata_present": (
                "The authoritative inventory binds each source row to the discovery artifact and "
                "its SHA; this audit additionally hashes every consumed local asset."
            ),
            "status_precedence": [status for _, status in STATUS_PRECEDENCE]
            + ["LOCAL_READY_FOR_SCALE1A"],
            "availability_is_not_admission": True,
        },
        "input_artifacts": [
            {"path": logical_ref, "sha256": digest}
            for logical_ref, digest in sorted(all_inputs)
        ],
        "output_artifacts": {},
        "scope_confirmations": {
            "network_access_used": False,
            "external_download_performed": False,
            "candidate_admission_assigned": False,
            "proteinmpnn_executed": False,
            "fixed_probes_generated": False,
            "stage0_analysis_stop": True,
            "stage0_artifacts_modified": False,
            "scale1a_started": False,
            "scale1b_started": False,
        },
        "next_task": (
            "SCALE1A_FORMAL_ADMISSION_CENSUS"
            if scale1a0_status == PASS_STATUS
            else "RECONSTRUCT_SOURCE_NEUTRAL_SAMPLING_FRAME"
        ),
    }
    return SamplingFrameAuditResult(
        candidates=frame,
        summary=summary,
        manifest=manifest,
        input_artifacts=tuple(sorted(all_inputs)),
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
        raise SamplingFrameAuditError(
            "invalid_audit_output", "Audit JSON cannot be serialized"
        ) from exc


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise SamplingFrameAuditError(
            "immutable_audit_artifact_conflict", f"immutable output differs: {path.name}"
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
            if path.stat().st_size == temporary.stat().st_size and sha256_file(path) == sha256_file(
                temporary
            ):
                return "reused_identical"
            raise SamplingFrameAuditError(
                "immutable_audit_artifact_conflict", f"immutable output differs: {path.name}"
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SamplingFrameAuditError(
                "immutable_audit_artifact_conflict", f"Output appeared concurrently: {path.name}"
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_sampling_frame_audit(
    result: SamplingFrameAuditResult,
    paths: ProjectPaths,
    *,
    config: SamplingFrameAuditConfig | None = None,
) -> dict[str, Any]:
    """Materialize table and summary first, then the manifest, without overwrite."""
    config = config or SamplingFrameAuditConfig()
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "scale1_sampling_frame_audit.parquet"
    summary_path = output_root / "scale1_local_availability_summary.json"
    manifest_path = output_root / "scale1_sampling_frame_manifest.json"
    write_status = {
        "candidate_audit": _write_immutable_parquet(table_path, result.candidates),
        "availability_summary": _write_immutable_bytes(
            summary_path, _render_json(result.summary)
        ),
    }
    manifest = {
        **result.manifest,
        "output_artifacts": {
            "candidate_audit": {
                "path": _logical(paths, table_path),
                "rows": len(result.candidates),
                "sha256": sha256_file(table_path),
            },
            "availability_summary": {
                "path": _logical(paths, summary_path),
                "sha256": sha256_file(summary_path),
            },
            "manifest": {
                "path": _logical(paths, manifest_path),
                "self_hash_policy": "reported_externally_to_avoid_recursive_self-hash",
            },
        },
    }
    write_status["manifest"] = _write_immutable_bytes(manifest_path, _render_json(manifest))
    return {
        "scale1a0_status": manifest["scale1a0_status"],
        "write_status": write_status,
        "outputs": {
            "candidate_audit": {
                "path": _logical(paths, table_path),
                "rows": len(result.candidates),
                "sha256": sha256_file(table_path),
            },
            "availability_summary": {
                "path": _logical(paths, summary_path),
                "sha256": sha256_file(summary_path),
            },
            "manifest": {
                "path": _logical(paths, manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
    }


def run_sampling_frame_audit(
    paths: ProjectPaths,
    *,
    config: SamplingFrameAuditConfig | None = None,
) -> dict[str, Any]:
    """Build and immutably materialize the complete offline audit."""
    checked = config or SamplingFrameAuditConfig()
    result = audit_sampling_frame(paths, config=checked)
    return materialize_sampling_frame_audit(result, paths, config=checked)
