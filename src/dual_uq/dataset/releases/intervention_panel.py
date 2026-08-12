"""Materialize immutable, human-reviewed intervention-panel declarations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from dual_uq.core.atomic_io import (
    atomic_write_json,
    atomic_write_new_bytes,
)
from dual_uq.core.hashing import sha256_bytes, sha256_file
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.policies.identity import (
    extract_canonical_sequence,
    extract_prediction_record_sequence,
)

PANEL_VERSION = "stage0_intervention_panel_v1"
SOURCE_LOGICAL_PATH = "docs/design/DATASET_SCALE_AND_VALIDATION_PLAN_V1.md"
PANEL_STATEMENT = (
    "This file is a frozen Stage-0 experiment panel declaration transcribed from "
    "prior human review. It is not a canonical reconstruction of the full Batch-1 "
    "census."
)
EXPECTED_STAGE0_PANEL: tuple[tuple[int, str], ...] = (
    (21, "5gv8_A__P83686"),
    (22, "6jgj_A__P42212"),
    (26, "5mn1_A__P00760"),
    (49, "1fn8_A__P35049"),
    (77, "2ykz_A__P00138"),
    (88, "1pjx_A__Q7SIG4"),
    (96, "3pyp_A__P16113"),
    (125, "6s2s_A__P02689"),
    (179, "1ix9_A__P00448"),
    (196, "4ce8_A__Q9HYN5"),
    (200, "5avh_A__P24300"),
)


class InterventionPanelError(ValueError):
    """A structured blocker while materializing an experiment panel."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class HistoricalPanelEntry:
    candidate_index: int
    protein_id: str
    pdb_id: str
    pdb_chain: str
    uniprot_accession: str
    pair_qc_pass_asserted: bool
    mechanism_observable_asserted: bool


def parse_protein_id(protein_id: str) -> tuple[str, str, str]:
    """Parse only the documented ``PDB_chain__UniProt`` identity syntax."""
    match = re.fullmatch(r"([0-9A-Za-z]{4})_([^_]+)__([0-9A-Za-z]+)", protein_id)
    if match is None:
        raise InterventionPanelError(
            "invalid_documented_protein_id",
            f"Invalid documented protein_id: {protein_id!r}",
        )
    return match.group(1), match.group(2), match.group(3)


def _strict_indices(document: str) -> tuple[int, ...]:
    match = re.search(
        r"strict quality \+ observability \(11\):\s*\n"
        r"(?P<indices>[0-9,\s]+?)\n\s*\nobservable but quality",
        document,
    )
    if match is None:
        raise InterventionPanelError(
            "historical_panel_declaration_missing",
            "Planning document lacks the strict quality + observability (11) declaration",
        )
    return tuple(int(value) for value in re.findall(r"\d+", match.group("indices")))


def parse_historical_intervention_panel(source_path: Path) -> tuple[HistoricalPanelEntry, ...]:
    """Verify and transcribe the exact human-reviewed Stage-0 declaration."""
    document = source_path.read_text(encoding="utf-8")
    expected_indices = tuple(index for index, _ in EXPECTED_STAGE0_PANEL)
    observed_indices = _strict_indices(document)
    if observed_indices != expected_indices:
        raise InterventionPanelError(
            "historical_panel_membership_conflict",
            f"Documented strict panel differs: {observed_indices}",
        )
    latest_section = document.split("## 1. Executive decision", maxsplit=1)[0]
    if not re.search(
        r"\|\s*Pair-QC pass and mechanism observable\s*\|\s*11\s*\|",
        latest_section,
    ):
        raise InterventionPanelError(
            "historical_assertion_unsupported",
            "Latest planning update does not support both historical assertions",
        )

    entries = []
    for index, protein_id in EXPECTED_STAGE0_PANEL:
        documented_row = re.search(
            rf"^\|\s*{index}\s*\|\s*`{re.escape(protein_id)}`\s*\|",
            document,
            flags=re.MULTILINE,
        )
        if documented_row is None:
            raise InterventionPanelError(
                "historical_identity_conflict",
                f"Planning document does not contain {index} -> {protein_id}",
            )
        pdb_id, chain, accession = parse_protein_id(protein_id)
        entries.append(
            HistoricalPanelEntry(
                candidate_index=index,
                protein_id=protein_id,
                pdb_id=pdb_id.lower(),
                pdb_chain=chain,
                uniprot_accession=accession,
                pair_qc_pass_asserted=True,
                mechanism_observable_asserted=True,
            )
        )
    return tuple(entries)


def _portable_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise InterventionPanelError(
            "nonportable_repository_path", f"Path is outside repository: {path}"
        ) from exc


def _status(value: Any, status: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "value": value, **extra}


def _load_inventory_record(
    inventory_path: Path, candidate_index: int
) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InterventionPanelError(
            "repository_inventory_unreadable", f"Cannot read {inventory_path}"
        ) from exc
    rows = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise InterventionPanelError(
            "repository_inventory_invalid", "Candidate inventory lacks records"
        )
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("candidate_index") == candidate_index
    ]
    if len(matches) != 1:
        raise InterventionPanelError(
            "repository_identity_ambiguous",
            f"Expected one inventory row for candidate {candidate_index}; found {len(matches)}",
        )
    return matches[0], sha256_file(inventory_path)


def _unique_identical_file(paths: list[Path], description: str) -> Path | None:
    matches = sorted({path.resolve() for path in paths if path.is_file()})
    if not matches:
        return None
    if len({sha256_file(path) for path in matches}) != 1:
        raise InterventionPanelError(
            "repository_identity_conflict",
            f"Multiple incompatible repository assets for {description}",
        )
    return matches[0]


def verify_repository_identity(
    *,
    candidate_index: int,
    protein_id: str,
    repository_root: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    """Verify one documented identity without using repository state for selection."""
    repository_root = repository_root.resolve()
    pdb_id, chain, accession = parse_protein_id(protein_id)
    pdb_id = pdb_id.lower()
    row, inventory_sha256 = _load_inventory_record(inventory_path, candidate_index)
    repository_identity = {
        "pair_id": row.get("pair_id"),
        "pdb_id": str(row.get("PDB", "")).lower(),
        "pdb_chain": row.get("chain"),
        "uniprot_accession": row.get("UniProt"),
    }
    documented_identity = {
        "pair_id": protein_id,
        "pdb_id": pdb_id,
        "pdb_chain": chain,
        "uniprot_accession": accession,
    }
    if repository_identity != documented_identity:
        raise InterventionPanelError(
            "repository_identity_conflict",
            f"repository identity conflict for {candidate_index}: {repository_identity}",
        )

    result: dict[str, Any] = {
        "candidate_inventory_identity": _status(
            repository_identity,
            "verified",
            source_path=_portable_relative(inventory_path, repository_root),
            source_sha256=inventory_sha256,
        ),
        "pdb_entity_id": _status(row.get("polymer_entity_id"), "verified"),
    }
    pdb_path = repository_root / f"data/raw/pdb/{pdb_id}.cif"
    result["pdb_backbone_path"] = (
        _status(
            _portable_relative(pdb_path, repository_root),
            "verified",
            sha256=sha256_file(pdb_path),
        )
        if pdb_path.is_file()
        else _status(None, "unresolved_not_found")
    )

    afdb_root = repository_root / "data/raw/afdb" / accession
    metadata_path = _unique_identical_file(
        [*afdb_root.glob("metadata.json"), *afdb_root.glob("*/metadata.json")],
        f"{accession} metadata",
    )
    if metadata_path is None:
        result.update(
            {
                "afdb_metadata_path": _status(None, "unresolved_not_found"),
                "afdb_model_id": _status(None, "unresolved_not_found"),
                "canonical_sequence_provenance": _status(
                    None, "unresolved_not_found"
                ),
                "afdb_backbone_path": _status(None, "unresolved_not_found"),
            }
        )
        return result

    metadata_bytes = metadata_path.read_bytes()
    metadata_ref = _portable_relative(metadata_path, repository_root)
    result["afdb_metadata_path"] = _status(
        metadata_ref, "verified", sha256=sha256_bytes(metadata_bytes)
    )
    try:
        prediction = extract_prediction_record_sequence(metadata_bytes, accession)
    except DerivationError as exc:
        raise InterventionPanelError(
            "repository_identity_conflict",
            f"AFDB exact identity conflict for {accession}: {exc}",
        ) from exc
    model_id = str(prediction["model_entity_id"])
    result["afdb_model_id"] = _status(model_id, "verified")
    try:
        canonical = extract_canonical_sequence(metadata_bytes, accession)
    except DerivationError as exc:
        result["canonical_sequence_provenance"] = _status(
            None, "unresolved_not_found", failure_code=exc.code
        )
    else:
        result["canonical_sequence_provenance"] = _status(
            {
                "provenance": canonical["canonical_sequence_provenance"],
                "sequence_length": canonical["sequence_length"],
                "sequence_sha256": canonical["sequence_sha256"],
                "source_field": canonical["sequence_source_field"],
                "source_path": metadata_ref,
            },
            "verified",
        )

    model_path = _unique_identical_file(
        [
            afdb_root / "model.cif",
            afdb_root / f"{model_id}-model_v6.cif",
            afdb_root / model_id / "model.cif",
            afdb_root / model_id / f"{model_id}-model_v6.cif",
        ],
        f"{accession} model {model_id}",
    )
    result["afdb_backbone_path"] = (
        _status(
            _portable_relative(model_path, repository_root),
            "verified",
            sha256=sha256_file(model_path),
        )
        if model_path is not None
        else _status(None, "unresolved_not_found")
    )
    return result


def build_panel_record(
    *,
    candidate_index: int,
    protein_id: str,
    source_sha256: str,
    repository_verification: dict[str, Any],
) -> dict[str, Any]:
    pdb_id, chain, accession = parse_protein_id(protein_id)
    return {
        "candidate_index": candidate_index,
        "mechanism_observable_asserted": True,
        "pair_qc_pass_asserted": True,
        "panel_version": PANEL_VERSION,
        "pdb_chain": chain,
        "pdb_id": pdb_id.lower(),
        "protein_id": protein_id,
        "repository_verification": repository_verification,
        "selection_basis": {
            "source_path": SOURCE_LOGICAL_PATH,
            "source_sha256": source_sha256,
            "type": "historical_human_review",
        },
        "selection_status": "human_transcribed_from_prior_review",
        "uniprot_accession": accession,
    }


def render_panel_jsonl(records: list[dict[str, Any]]) -> bytes:
    """Return deterministic LF-terminated JSONL for the immutable panel."""
    ordered = sorted(records, key=lambda row: int(row["candidate_index"]))
    observed = tuple((int(row["candidate_index"]), row["protein_id"]) for row in ordered)
    if observed != EXPECTED_STAGE0_PANEL:
        raise InterventionPanelError(
            "historical_panel_membership_conflict",
            f"Rendered panel differs from frozen declaration: {observed}",
        )
    lines = [
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        for row in ordered
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_immutable_panel(path: Path, payload: bytes) -> str:
    """Create or byte-verify a versioned panel without overwriting it."""
    if path.exists():
        if path.read_bytes() != payload:
            raise InterventionPanelError(
                "immutable_panel_conflict",
                f"immutable panel conflict at {path}",
            )
        return "reused_identical"
    atomic_write_new_bytes(path, payload)
    return "created"


def validate_experiment_binding(
    binding_path: Path, *, expected_panel_path: str
) -> str:
    """Require the established experiment config to bind only Stage-0 semantics."""
    payload = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), dict):
        raise InterventionPanelError(
            "invalid_experiment_binding", "Experiment binding lacks inputs"
        )
    if "canonical_census" in payload["inputs"]:
        raise InterventionPanelError(
            "canonical_census_binding_forbidden",
            "Stage-0 panel must not be bound as canonical_census",
        )
    binding = payload["inputs"].get("stage0_intervention_panel")
    if not isinstance(binding, dict):
        raise InterventionPanelError(
            "stage0_binding_missing", "stage0_intervention_panel binding is missing"
        )
    observed = binding.get("path")
    path = PurePosixPath(str(observed))
    if path.is_absolute() or ".." in path.parts or str(path) != expected_panel_path:
        raise InterventionPanelError(
            "stage0_binding_path_conflict",
            f"Stage-0 binding differs: {observed!r}",
        )
    if binding.get("panel_version") != PANEL_VERSION:
        raise InterventionPanelError(
            "stage0_binding_version_conflict", "Stage-0 binding version differs"
        )
    return str(path)


def materialize_intervention_panel(
    *,
    repository_root: Path,
    source_path: Path,
    inventory_path: Path,
    binding_path: Path,
    panel_path: Path,
    metadata_path: Path,
    git_commit: str | None,
    git_dirty: bool | None,
    creation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Materialize and validate the frozen declaration without scientific reruns."""
    entries = parse_historical_intervention_panel(source_path)
    source_sha256 = sha256_file(source_path)
    records = [
        build_panel_record(
            candidate_index=entry.candidate_index,
            protein_id=entry.protein_id,
            source_sha256=source_sha256,
            repository_verification=verify_repository_identity(
                candidate_index=entry.candidate_index,
                protein_id=entry.protein_id,
                repository_root=repository_root,
                inventory_path=inventory_path,
            ),
        )
        for entry in entries
    ]
    panel_bytes = render_panel_jsonl(records)
    panel_write_status = write_immutable_panel(panel_path, panel_bytes)
    panel_sha256 = sha256_bytes(panel_bytes)
    expected_panel_path = _portable_relative(panel_path, repository_root)
    bound_path = validate_experiment_binding(
        binding_path, expected_panel_path=expected_panel_path
    )
    timestamp = creation_timestamp or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    metadata = {
        "creation_timestamp": timestamp,
        "git_commit_sha": git_commit,
        "git_dirty": git_dirty,
        "panel_jsonl_sha256": panel_sha256,
        "panel_statement": PANEL_STATEMENT,
        "panel_version": PANEL_VERSION,
        "record_count": len(records),
        "source_planning_document_path": _portable_relative(
            source_path, repository_root
        ),
        "source_planning_document_sha256": source_sha256,
        "stage0_intervention_panel_binding": {
            "config_path": _portable_relative(binding_path, repository_root),
            "path": bound_path,
            "semantic_key": "stage0_intervention_panel",
        },
    }
    atomic_write_json(metadata_path, metadata)
    return {
        "metadata": metadata,
        "panel_path": expected_panel_path,
        "panel_write_status": panel_write_status,
        "records": records,
    }
