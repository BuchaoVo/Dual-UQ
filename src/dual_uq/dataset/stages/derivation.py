"""P1: build and validate deterministic paired canonical backbones."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_canonical, sha256_file

from ..models import (
    BACKBONE_ATOM_NAMES,
    AtomRecord,
    BackboneSelection,
    ResidueKey,
    ResidueProvenance,
    canonical_amino_acid,
    group_residue_records,
    select_backbone_atoms,
)
from ..pipeline.resume import (
    SUCCESSFUL_UPSTREAM_STATUSES,
    evaluate_resume,
    verify_declared_outputs,
)
from ..pipeline.status import (
    STAGE_MANIFEST_SCHEMA_VERSION,
    LifecycleStatus,
    OutputDeclaration,
    StageManifest,
    ValidationRecord,
    read_stage_manifest,
    write_stage_manifest,
)
from .resolution import P0_STAGE_NAME

P1_STAGE_NAME = "P1_build_and_validate_paired_backbones"

_MAPPING_COLUMNS = {
    "pdb_residue_name",
    "uniprot_residue_number",
    "uniprot_residue_name",
    "auth_asym_id",
    "auth_seq_id",
    "insertion_code",
    "label_asym_id",
    "label_seq_id",
}
_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
_PROVENANCE_COLUMNS = (
    "protein_id",
    "output_position",
    "uniprot_position",
    "canonical_aa",
    "source",
    "raw_resname",
    "record_type",
    "auth_chain_id",
    "auth_seq_id",
    "insertion_code",
    "label_chain_id",
    "label_seq_id",
    "model_entity_id",
    "model_residue_position",
    "atom_name",
    "altloc",
    "occupancy",
    "x",
    "y",
    "z",
)


class P1ValidationError(ValueError):
    """One structured P1 validation failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class PairedResidue:
    output_position: int
    uniprot_position: int
    canonical_aa: str
    model_residue_position: int
    pdb_backbone: BackboneSelection
    afdb_backbone: BackboneSelection


@dataclass(frozen=True)
class P1Resolution:
    protein_id: str
    tier: int
    selected_model_entity_id: str
    input_digest: str
    config_digest: str
    residues: tuple[PairedResidue, ...]
    canonical_sequence: str
    pdb_backbone: bytes
    afdb_backbone: bytes
    provenance: pd.DataFrame

    @property
    def residue_count(self) -> int:
        return len(self.residues)


@dataclass(frozen=True)
class P1RunResult:
    status: LifecycleStatus
    validation: ValidationRecord
    failure_code: str | None
    failure_message: str | None
    input_digest: str | None
    config_digest: str | None
    residue_count: int | None
    canonical_sequence: str | None


@dataclass(frozen=True)
class _P1Inputs:
    protein_id: str
    tier: int
    selected_model_entity_id: str
    fragment_start: int
    fragment_end: int
    mapping_path: Path
    pdb_path: Path
    afdb_path: Path
    input_digest: str
    config_digest: str
    frozen_hashes: dict[str, str]
    current_hashes: dict[str, str]


def _canonical_text(value: Any, field: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise P1ValidationError(
            "invalid_p1_identity", f"{field} must be a non-empty canonical string"
        )
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise P1ValidationError("invalid_p1_identity", f"{field} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise P1ValidationError(
            "invalid_p1_identity", f"{field} must be an integer"
        ) from exc
    if not np.isfinite(number) or not number.is_integer() or (positive and number < 1):
        raise P1ValidationError("invalid_p1_identity", f"{field} must be an integer")
    return int(number)


def _config_digest(config: Mapping[str, Any], pipeline_version: str) -> str:
    if not isinstance(config, Mapping):
        raise P1ValidationError("invalid_config", "P1 config must be a mapping")
    try:
        return sha256_canonical(
            {"pipeline_version": pipeline_version, "p1_config": dict(config)}
        )
    except (TypeError, ValueError) as exc:
        raise P1ValidationError("invalid_config", "P1 config is not canonical JSON") from exc


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P1ValidationError(
            "invalid_p0_artifact", f"Unable to read {description}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise P1ValidationError(
            "invalid_p0_artifact", f"{description} must be a JSON object"
        )
    return payload


def _declared_path(stage_dir: Path, manifest: StageManifest, logical_name: str) -> Path:
    matches = [
        output for output in manifest.declared_outputs if output.logical_name == logical_name
    ]
    if len(matches) != 1:
        raise P1ValidationError(
            "invalid_p0_contract",
            f"P0 must declare exactly one {logical_name} output",
            logical_name=logical_name,
        )
    return stage_dir / matches[0].relative_path


def _input_record(lock: Mapping[str, Any], logical_name: str) -> dict[str, Any]:
    records = lock.get("input_files")
    if not isinstance(records, list):
        raise P1ValidationError("invalid_p0_contract", "P0 lock lacks input_files")
    matches = [
        item
        for item in records
        if isinstance(item, dict) and item.get("logical_name") == logical_name
    ]
    if len(matches) != 1:
        raise P1ValidationError(
            "invalid_p0_contract",
            f"P0 lock must contain exactly one {logical_name} input",
        )
    return dict(matches[0])


def _resolve_frozen_path(record: Mapping[str, Any], project_root: Path) -> Path:
    project_relative = record.get("project_relative_path")
    original = record.get("original_path")
    if isinstance(project_relative, str) and project_relative:
        path = (project_root.resolve() / project_relative).resolve()
    elif isinstance(original, str) and original:
        candidate = Path(original)
        path = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    else:
        raise P1ValidationError(
            "invalid_p0_contract", "P0 input record lacks an explicit path"
        )
    if not path.exists():
        raise P1ValidationError(
            "missing_frozen_source", f"Frozen P0 source does not exist: {path}", path=str(path)
        )
    if not path.is_file():
        raise P1ValidationError(
            "invalid_frozen_source", f"Frozen P0 source is not a file: {path}", path=str(path)
        )
    return path


def _prepare_inputs(
    *,
    project_root: Path,
    p0_stage_dir: Path,
    config: Mapping[str, Any],
    pipeline_version: str,
) -> _P1Inputs:
    checked_pipeline = _canonical_text(pipeline_version, "pipeline_version")
    config_digest = _config_digest(config, checked_pipeline)
    manifest_path = p0_stage_dir / "stage_manifest.json"
    try:
        manifest = read_stage_manifest(manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise P1ValidationError(
            "invalid_p0_stage_manifest", "P0 stage manifest is missing or invalid"
        ) from exc
    if manifest.stage != P0_STAGE_NAME or manifest.status not in SUCCESSFUL_UPSTREAM_STATUSES:
        raise P1ValidationError(
            "p0_not_complete",
            "P1 requires a validated complete P0 stage",
            p0_status=manifest.status.value,
        )
    integrity = verify_declared_outputs(p0_stage_dir, manifest.declared_outputs)
    if not integrity.validation_pass:
        raise P1ValidationError(
            "p0_output_integrity_failed",
            "P0 declared output integrity validation failed",
            p0_validation=integrity.as_dict(),
        )
    lock_path = _declared_path(p0_stage_dir, manifest, "input_lock")
    mapping_path = _declared_path(p0_stage_dir, manifest, "residue_mapping")
    lock = _load_json(lock_path, "P0 input lock")
    if lock.get("validation_pass") is not True:
        raise P1ValidationError("invalid_p0_contract", "P0 lock is not validation-pass")
    if lock.get("input_digest") != manifest.input_digest:
        raise P1ValidationError(
            "invalid_p0_contract", "P0 lock and stage manifest input digests disagree"
        )
    protein_id = _canonical_text(lock.get("protein_id"), "protein_id")
    if protein_id != manifest.protein_id:
        raise P1ValidationError(
            "invalid_p0_contract", "P0 protein identity is internally inconsistent"
        )
    tier = _integer(lock.get("tier"), "tier", positive=True)
    selected_model = _canonical_text(
        lock.get("selected_model_identity"), "selected_model_identity"
    )
    fragment = lock.get("fragment_interval")
    if not isinstance(fragment, dict):
        raise P1ValidationError("invalid_p0_contract", "P0 lock lacks fragment_interval")
    fragment_start = _integer(fragment.get("uniprot_start"), "uniprot_start", positive=True)
    fragment_end = _integer(fragment.get("uniprot_end"), "uniprot_end", positive=True)
    model_length = _integer(fragment.get("model_length"), "model_length", positive=True)
    if fragment_end < fragment_start or model_length != fragment_end - fragment_start + 1:
        raise P1ValidationError("invalid_p0_contract", "P0 fragment interval is inconsistent")

    pdb_record = _input_record(lock, "pdb_structure_path")
    afdb_record = _input_record(lock, "afdb_model_path")
    pdb_path = _resolve_frozen_path(pdb_record, project_root)
    afdb_path = _resolve_frozen_path(afdb_record, project_root)
    current_hashes = {
        "pdb_structure_path": sha256_file(pdb_path),
        "afdb_model_path": sha256_file(afdb_path),
    }
    frozen_hashes: dict[str, str] = {}
    for logical_name, record in (
        ("pdb_structure_path", pdb_record),
        ("afdb_model_path", afdb_record),
    ):
        digest = record.get("sha256")
        if not isinstance(digest, str):
            raise P1ValidationError(
                "invalid_p0_contract", f"P0 {logical_name} record lacks raw SHA"
            )
        frozen_hashes[logical_name] = digest

    input_digest = sha256_canonical(
        {
            "p0": {
                "input_digest": manifest.input_digest,
                "config_digest": manifest.config_digest,
                "input_lock_sha256": sha256_file(lock_path),
                "residue_mapping_sha256": sha256_file(mapping_path),
            },
            "sources": [
                {
                    "logical_name": name,
                    "path": str(path),
                    "sha256": current_hashes[name],
                }
                for name, path in sorted(
                    (
                        ("pdb_structure_path", pdb_path),
                        ("afdb_model_path", afdb_path),
                    )
                )
            ],
        }
    )
    return _P1Inputs(
        protein_id=protein_id,
        tier=tier,
        selected_model_entity_id=selected_model,
        fragment_start=fragment_start,
        fragment_end=fragment_end,
        mapping_path=mapping_path,
        pdb_path=pdb_path,
        afdb_path=afdb_path,
        input_digest=input_digest,
        config_digest=config_digest,
        frozen_hashes=frozen_hashes,
        current_hashes=current_hashes,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _column(mmcif: Mapping[str, Any], name: str, size: int) -> list[Any]:
    values = _as_list(mmcif.get(name))
    if len(values) != size:
        raise P1ValidationError(
            "invalid_mmcif_atom_site", f"mmCIF column {name} is missing or mis-sized"
        )
    return values


def _optional_text(value: Any) -> str | None:
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def _optional_float(value: Any) -> float | None:
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise P1ValidationError("invalid_atom_record", "Invalid numeric atom value") from exc


def _load_atom_records(path: Path, source_id: str) -> tuple[AtomRecord, ...]:
    try:
        mmcif = MMCIF2Dict(str(path))
    except Exception as exc:
        raise P1ValidationError(
            "invalid_structure_mmcif", f"Unable to parse structure mmCIF: {path}"
        ) from exc
    atom_names = _as_list(mmcif.get("_atom_site.label_atom_id"))
    if not atom_names:
        raise P1ValidationError("invalid_mmcif_atom_site", "mmCIF has no atom_site rows")
    size = len(atom_names)
    columns = {
        "record": _column(mmcif, "_atom_site.group_PDB", size),
        "element": _column(mmcif, "_atom_site.type_symbol", size),
        "altloc": _column(mmcif, "_atom_site.label_alt_id", size),
        "resname": _column(mmcif, "_atom_site.label_comp_id", size),
        "label_chain": _column(mmcif, "_atom_site.label_asym_id", size),
        "label_seq": _column(mmcif, "_atom_site.label_seq_id", size),
        "auth_chain": _column(mmcif, "_atom_site.auth_asym_id", size),
        "auth_seq": _column(mmcif, "_atom_site.auth_seq_id", size),
        "insertion": _column(mmcif, "_atom_site.pdbx_PDB_ins_code", size),
        "x": _column(mmcif, "_atom_site.Cartn_x", size),
        "y": _column(mmcif, "_atom_site.Cartn_y", size),
        "z": _column(mmcif, "_atom_site.Cartn_z", size),
        "occupancy": _column(mmcif, "_atom_site.occupancy", size),
        "model": _column(mmcif, "_atom_site.pdbx_PDB_model_num", size),
    }
    try:
        models = [_integer(value, "model_number", positive=True) for value in columns["model"]]
    except P1ValidationError as exc:
        raise P1ValidationError("invalid_mmcif_atom_site", str(exc)) from exc
    model_ids = set(models)
    if len(model_ids) != 1:
        raise P1ValidationError(
            "ambiguous_structure_model", "P1 requires exactly one coordinate model"
        )
    records: list[AtomRecord] = []
    for index, atom_name in enumerate(atom_names):
        raw_resname = str(columns["resname"][index]).strip().upper()
        if canonical_amino_acid(raw_resname) is None:
            continue
        auth_chain = _optional_text(columns["auth_chain"][index])
        auth_seq = _optional_float(columns["auth_seq"][index])
        label_chain = _optional_text(columns["label_chain"][index])
        label_seq = _optional_float(columns["label_seq"][index])
        if (
            auth_chain is None
            or auth_seq is None
            or not auth_seq.is_integer()
            or (label_chain is None) != (label_seq is None)
            or (label_seq is not None and not label_seq.is_integer())
        ):
            raise P1ValidationError(
                "invalid_atom_residue_identity",
                "Protein atom lacks explicit, consistent auth/label identity",
                atom_row=index,
            )
        try:
            residue = ResidueProvenance(
                key=ResidueKey(
                    source_id,
                    auth_chain,
                    int(auth_seq),
                    str(columns["insertion"][index]),
                ),
                label_chain_id=label_chain,
                label_seq_id=None if label_seq is None else int(label_seq),
                raw_resname=raw_resname,
                record_type=str(columns["record"][index]).strip(),
            )
            records.append(
                AtomRecord(
                    residue=residue,
                    atom_name=str(atom_name),
                    element=_optional_text(columns["element"][index]),
                    altloc=str(columns["altloc"][index]),
                    occupancy=_optional_float(columns["occupancy"][index]),
                    x=float(columns["x"][index]),
                    y=float(columns["y"][index]),
                    z=float(columns["z"][index]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise P1ValidationError(
                "invalid_atom_record", "Structure atom failed A4 validation", atom_row=index
            ) from exc
    if not records:
        raise P1ValidationError("invalid_structure_mmcif", "No supported protein atoms found")
    return tuple(records)


def _canonical_mapping_aa(value: Any, output_position: int) -> str:
    text = str(value).strip().upper()
    if text in _ONE_TO_THREE:
        return text
    try:
        converted = canonical_amino_acid(text)
    except (TypeError, ValueError):
        converted = None
    if converted is None:
        raise P1ValidationError(
            "unsupported_mapping_amino_acid",
            "Mapping amino acid must be an explicit supported canonical residue",
            output_position=output_position,
        )
    return converted


def _load_mapping(path: Path) -> pd.DataFrame:
    try:
        mapping = pd.read_csv(path, sep="\t", keep_default_na=False)
    except Exception as exc:
        raise P1ValidationError(
            "invalid_frozen_mapping", f"Unable to read frozen P0 mapping: {path}"
        ) from exc
    missing = sorted(_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise P1ValidationError(
            "missing_mapping_column", "Frozen mapping lacks P1 columns", missing_columns=missing
        )
    if mapping.empty:
        raise P1ValidationError("invalid_frozen_mapping", "Frozen mapping is empty")
    table = mapping.copy()
    for column in ("uniprot_residue_number", "auth_seq_id"):
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or values.mod(1).ne(0).any():
            raise P1ValidationError(
                "invalid_frozen_mapping", f"{column} must contain explicit integers"
            )
        table[column] = values.astype(int)
    if "output_position" in table:
        output = pd.to_numeric(table["output_position"], errors="coerce")
        if output.isna().any() or output.mod(1).ne(0).any():
            raise P1ValidationError(
                "invalid_output_position", "output_position must contain integers"
            )
        table["output_position"] = output.astype(int)
        table = table.sort_values("output_position", kind="stable").reset_index(drop=True)
        if table["output_position"].tolist() != list(range(1, len(table) + 1)):
            raise P1ValidationError(
                "invalid_output_position", "output_position must be exactly contiguous 1..L"
            )
    else:
        table = table.sort_values("uniprot_residue_number", kind="stable").reset_index(
            drop=True
        )
        table["output_position"] = np.arange(1, len(table) + 1)
    uniprot = table["uniprot_residue_number"].tolist()
    if any(right <= left for left, right in pairwise(uniprot)):
        raise P1ValidationError(
            "uniprot_position_order_mismatch",
            "UniProt positions must increase strictly with output_position",
        )
    table["canonical_aa"] = [
        _canonical_mapping_aa(value, output_position)
        for value, output_position in zip(
            table["uniprot_residue_name"], table["output_position"], strict=True
        )
    ]
    return table


def _select_backbone(
    records: tuple[AtomRecord, ...], *, source: str, output_position: int
) -> BackboneSelection:
    try:
        selection = select_backbone_atoms(records)
    except ValueError as exc:
        code = (
            "ambiguous_backbone_atom"
            if "ambiguous duplicate atom" in str(exc)
            else "invalid_backbone_selection"
        )
        raise P1ValidationError(
            code,
            f"Unable to select unique {source} backbone atoms",
            source=source,
            output_position=output_position,
        ) from exc
    if selection.missing_atoms:
        raise P1ValidationError(
            "missing_backbone_atom",
            f"{source} residue lacks required backbone atoms",
            source=source,
            output_position=output_position,
            missing_atoms=list(selection.missing_atoms),
        )
    return selection


def _index_records(records: tuple[AtomRecord, ...]) -> dict[ResidueKey, tuple[AtomRecord, ...]]:
    return {group.residue.key: group.atoms for group in group_residue_records(records)}


def _optional_mapping_label(value: Any) -> str | None:
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def _pair_residues(
    *,
    mapping: pd.DataFrame,
    pdb_records: tuple[AtomRecord, ...],
    afdb_records: tuple[AtomRecord, ...],
    pair_id: str,
    model_entity_id: str,
    fragment_start: int,
    fragment_end: int,
) -> tuple[PairedResidue, ...]:
    pdb_index = _index_records(pdb_records)
    afdb_groups = group_residue_records(afdb_records)
    model_length = fragment_end - fragment_start + 1
    local_positions: dict[int, tuple[AtomRecord, ...]] = {}
    for group in afdb_groups:
        key = group.residue.key
        if key.insertion_code or key.auth_seq_id in local_positions:
            raise P1ValidationError(
                "afdb_model_position_mismatch",
                "AFDB model positions must be unique insertion-free local positions",
            )
        local_positions[key.auth_seq_id] = group.atoms
    if set(local_positions) != set(range(1, model_length + 1)):
        raise P1ValidationError(
            "afdb_model_position_mismatch",
            "AFDB author positions must be exactly model-local 1..N",
            expected_start=1,
            expected_end=model_length,
        )

    missing_pdb: list[int] = []
    paired: list[PairedResidue] = []
    for row in mapping.itertuples(index=False):
        output_position = int(row.output_position)
        auth_chain = _optional_mapping_label(row.auth_asym_id)
        if auth_chain is None:
            raise P1ValidationError(
                "invalid_auth_residue_identity",
                "PDB author identity must be explicit; label fallback is forbidden",
                output_position=output_position,
            )
        pdb_key = ResidueKey(
            pair_id,
            auth_chain,
            int(row.auth_seq_id),
            str(row.insertion_code),
        )
        pdb_atoms = pdb_index.get(pdb_key)
        if pdb_atoms is None:
            missing_pdb.append(output_position)
            continue
        pdb_selection = _select_backbone(
            pdb_atoms, source="pdb", output_position=output_position
        )
        expected_label_chain = _optional_mapping_label(row.label_asym_id)
        expected_label_seq = (
            None
            if _optional_mapping_label(row.label_seq_id) is None
            else _integer(row.label_seq_id, "label_seq_id")
        )
        provenance = pdb_selection.residue
        if expected_label_chain is not None and (
            provenance.label_chain_id != expected_label_chain
            or provenance.label_seq_id != expected_label_seq
        ):
            raise P1ValidationError(
                "pdb_label_identity_mismatch",
                "Explicit mapping label identity conflicts with PDB atom provenance",
                output_position=output_position,
            )
        canonical_aa = str(row.canonical_aa)
        if provenance.canonical_aa != canonical_aa:
            raise P1ValidationError(
                "pdb_amino_acid_mismatch",
                "PDB residue does not match mapping canonical amino acid",
                output_position=output_position,
                mapping_aa=canonical_aa,
                pdb_aa=provenance.canonical_aa,
            )
        uniprot_position = int(row.uniprot_residue_number)
        model_position = uniprot_position - fragment_start + 1
        if not 1 <= model_position <= model_length:
            raise P1ValidationError(
                "uniprot_position_mismatch",
                "Mapped UniProt position is outside the frozen AFDB fragment",
                output_position=output_position,
                uniprot_position=uniprot_position,
            )
        afdb_atoms = local_positions.get(model_position)
        if afdb_atoms is None:
            raise P1ValidationError(
                "residue_count_mismatch",
                "AFDB source lacks a mapped residue",
                missing_afdb_output_positions=[output_position],
            )
        afdb_selection = _select_backbone(
            afdb_atoms, source="afdb", output_position=output_position
        )
        if afdb_selection.residue.canonical_aa != canonical_aa:
            raise P1ValidationError(
                "afdb_amino_acid_mismatch",
                "AFDB residue does not match mapping canonical amino acid",
                output_position=output_position,
                mapping_aa=canonical_aa,
                afdb_aa=afdb_selection.residue.canonical_aa,
            )
        paired.append(
            PairedResidue(
                output_position=output_position,
                uniprot_position=uniprot_position,
                canonical_aa=canonical_aa,
                model_residue_position=model_position,
                pdb_backbone=pdb_selection,
                afdb_backbone=afdb_selection,
            )
        )
    if missing_pdb:
        raise P1ValidationError(
            "residue_count_mismatch",
            "PDB source lacks mapped residues; intersection trimming is forbidden",
            missing_pdb_output_positions=missing_pdb,
        )
    if len(paired) != len(mapping):
        raise P1ValidationError(
            "residue_count_mismatch", "Paired source residue counts are not identical"
        )
    if [residue.output_position for residue in paired] != list(range(1, len(paired) + 1)):
        raise P1ValidationError(
            "output_position_mismatch", "Paired output positions are not identical 1..L"
        )
    return tuple(paired)


def _pdb_bytes(residues: tuple[PairedResidue, ...], source: str) -> bytes:
    lines: list[str] = []
    serial = 1
    for residue in residues:
        selection = (
            residue.pdb_backbone if source == "pdb" else residue.afdb_backbone
        )
        atoms = {atom.atom_name: atom for atom in selection.selected_atoms}
        resname = _ONE_TO_THREE[residue.canonical_aa]
        for atom_name in BACKBONE_ATOM_NAMES:
            atom = atoms[atom_name]
            element = (atom.element or atom_name[0]).upper()
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A"
                f"{residue.output_position:4d}    "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.extend(("TER", "END"))
    return ("\n".join(lines) + "\n").encode("ascii")


def _provenance_table(
    residues: tuple[PairedResidue, ...], protein_id: str, model_entity_id: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for residue in residues:
        for source, selection in (
            ("pdb", residue.pdb_backbone),
            ("afdb", residue.afdb_backbone),
        ):
            provenance = selection.residue
            atoms = {atom.atom_name: atom for atom in selection.selected_atoms}
            for atom_name in BACKBONE_ATOM_NAMES:
                atom = atoms[atom_name]
                rows.append(
                    {
                        "protein_id": protein_id,
                        "output_position": residue.output_position,
                        "uniprot_position": residue.uniprot_position,
                        "canonical_aa": residue.canonical_aa,
                        "source": source,
                        "raw_resname": provenance.raw_resname,
                        "record_type": provenance.record_type,
                        "auth_chain_id": provenance.key.auth_chain_id,
                        "auth_seq_id": provenance.key.auth_seq_id,
                        "insertion_code": provenance.key.insertion_code,
                        "label_chain_id": provenance.label_chain_id,
                        "label_seq_id": provenance.label_seq_id,
                        "model_entity_id": model_entity_id if source == "afdb" else None,
                        "model_residue_position": (
                            residue.model_residue_position if source == "afdb" else None
                        ),
                        "atom_name": atom_name,
                        "altloc": atom.altloc,
                        "occupancy": atom.occupancy,
                        "x": atom.x,
                        "y": atom.y,
                        "z": atom.z,
                    }
                )
    return pd.DataFrame(rows, columns=_PROVENANCE_COLUMNS)


def resolve_p1_inputs(
    *,
    project_root: Path,
    p0_stage_dir: Path,
    config: Mapping[str, Any],
    pipeline_version: str = "dataset-a.v1",
) -> P1Resolution:
    """Read and validate P0 inputs, then build paired backbones without writing."""
    inputs = _prepare_inputs(
        project_root=project_root,
        p0_stage_dir=p0_stage_dir,
        config=config,
        pipeline_version=pipeline_version,
    )
    drifted = [
        name
        for name in sorted(inputs.current_hashes)
        if inputs.current_hashes[name] != inputs.frozen_hashes[name]
    ]
    if drifted:
        raise P1ValidationError(
            "p0_source_hash_mismatch",
            "Current source bytes do not match the immutable P0 lock",
            drifted_sources=drifted,
        )
    mapping = _load_mapping(inputs.mapping_path)
    pair_id = _load_json(
        _declared_path(
            p0_stage_dir,
            read_stage_manifest(p0_stage_dir / "stage_manifest.json"),
            "input_lock",
        ),
        "P0 input lock",
    ).get("pair_id")
    checked_pair_id = _canonical_text(pair_id, "pair_id")
    pdb_records = _load_atom_records(inputs.pdb_path, checked_pair_id)
    afdb_records = _load_atom_records(
        inputs.afdb_path, inputs.selected_model_entity_id
    )
    residues = _pair_residues(
        mapping=mapping,
        pdb_records=pdb_records,
        afdb_records=afdb_records,
        pair_id=checked_pair_id,
        model_entity_id=inputs.selected_model_entity_id,
        fragment_start=inputs.fragment_start,
        fragment_end=inputs.fragment_end,
    )
    sequence = "".join(residue.canonical_aa for residue in residues)
    return P1Resolution(
        protein_id=inputs.protein_id,
        tier=inputs.tier,
        selected_model_entity_id=inputs.selected_model_entity_id,
        input_digest=inputs.input_digest,
        config_digest=inputs.config_digest,
        residues=residues,
        canonical_sequence=sequence,
        pdb_backbone=_pdb_bytes(residues, "pdb"),
        afdb_backbone=_pdb_bytes(residues, "afdb"),
        provenance=_provenance_table(
            residues, inputs.protein_id, inputs.selected_model_entity_id
        ),
    )


def _failure_result(
    error: P1ValidationError,
    *,
    status: LifecycleStatus = LifecycleStatus.FAILED_VALIDATION,
    input_digest: str | None = None,
    config_digest: str | None = None,
) -> P1RunResult:
    return P1RunResult(
        status=status,
        validation=ValidationRecord(
            validation_pass=False,
            error_codes=(error.code,),
            details={"message": str(error), **error.details},
        ),
        failure_code=error.code,
        failure_message=str(error),
        input_digest=input_digest,
        config_digest=config_digest,
        residue_count=None,
        canonical_sequence=None,
    )


def _atomic_write_new(path: Path, payload: bytes) -> None:
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError as exc:
        raise P1ValidationError(
            "immutable_output_exists", f"Refusing to overwrite P1 output: {path}"
        ) from exc


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _provenance_bytes(table: pd.DataFrame) -> bytes:
    buffer = StringIO(newline="")
    table.to_csv(
        buffer,
        sep="\t",
        index=False,
        na_rep="",
        lineterminator="\n",
        float_format="%.6f",
    )
    return buffer.getvalue().encode("utf-8")


def run_p1(
    *,
    project_root: Path,
    p0_stage_dir: Path,
    stage_dir: Path,
    config: Mapping[str, Any],
    pipeline_version: str,
    run_id: str,
) -> P1RunResult:
    """Build immutable P1 outputs or reuse a validated existing stage via A3."""
    has_history = (stage_dir / "stage_manifest.json").is_file()
    try:
        prepared = _prepare_inputs(
            project_root=project_root,
            p0_stage_dir=p0_stage_dir,
            config=config,
            pipeline_version=pipeline_version,
        )
    except P1ValidationError as exc:
        if has_history:
            drift = P1ValidationError(
                "input_unresolvable_drift",
                "Current inputs cannot reproduce immutable P1 inputs",
                cause_code=exc.code,
                cause_message=str(exc),
            )
            return _failure_result(
                drift, status=LifecycleStatus.BLOCKED_INPUT_DRIFT
            )
        return _failure_result(exc)
    if has_history:
        preliminary = evaluate_resume(
            stage_dir=stage_dir,
            current_input_digest=prepared.input_digest,
            current_config_digest=prepared.config_digest,
            current_validation=ValidationRecord(validation_pass=True),
        )
        if preliminary.target_status is LifecycleStatus.BLOCKED_INPUT_DRIFT:
            error = P1ValidationError(
                preliminary.reason_code, preliminary.reason_code, **preliminary.details
            )
            return _failure_result(
                error,
                status=LifecycleStatus.BLOCKED_INPUT_DRIFT,
                input_digest=prepared.input_digest,
                config_digest=prepared.config_digest,
            )
    try:
        resolution = resolve_p1_inputs(
            project_root=project_root,
            p0_stage_dir=p0_stage_dir,
            config=config,
            pipeline_version=pipeline_version,
        )
    except P1ValidationError as exc:
        return _failure_result(
            exc,
            input_digest=prepared.input_digest,
            config_digest=prepared.config_digest,
        )
    validation = ValidationRecord(
        validation_pass=True,
        details={
            "residue_count": resolution.residue_count,
            "canonical_sequence": resolution.canonical_sequence,
            "selected_model_entity_id": resolution.selected_model_entity_id,
            "paired_atom_count": resolution.residue_count * 8,
        },
    )
    decision = evaluate_resume(
        stage_dir=stage_dir,
        current_input_digest=resolution.input_digest,
        current_config_digest=resolution.config_digest,
        current_validation=validation,
    )
    if decision.can_skip:
        return P1RunResult(
            status=decision.target_status,
            validation=validation,
            failure_code=None,
            failure_message=None,
            input_digest=resolution.input_digest,
            config_digest=resolution.config_digest,
            residue_count=resolution.residue_count,
            canonical_sequence=resolution.canonical_sequence,
        )
    if decision.reason_code != "manifest_missing":
        error = P1ValidationError(decision.reason_code, decision.reason_code, **decision.details)
        return _failure_result(
            error,
            status=decision.target_status,
            input_digest=resolution.input_digest,
            config_digest=resolution.config_digest,
        )

    validation_path = stage_dir / "validation.json"
    pdb_path = stage_dir / "outputs/pdb_backbone.pdb"
    afdb_path = stage_dir / "outputs/afdb_backbone.pdb"
    provenance_path = stage_dir / "outputs/backbone_residue_provenance.tsv"
    targets = (validation_path, pdb_path, afdb_path, provenance_path)
    if any(path.exists() for path in targets):
        return _failure_result(
            P1ValidationError(
                "immutable_output_exists", "Refusing to overwrite existing P1 outputs"
            ),
            input_digest=resolution.input_digest,
            config_digest=resolution.config_digest,
        )
    try:
        _atomic_write_new(validation_path, _json_bytes(validation.as_dict()))
        _atomic_write_new(pdb_path, resolution.pdb_backbone)
        _atomic_write_new(afdb_path, resolution.afdb_backbone)
        _atomic_write_new(provenance_path, _provenance_bytes(resolution.provenance))
    except P1ValidationError as exc:
        return _failure_result(
            exc,
            input_digest=resolution.input_digest,
            config_digest=resolution.config_digest,
        )
    outputs = tuple(
        OutputDeclaration(name, relative, sha256_file(stage_dir / relative), True)
        for name, relative in (
            ("validation", "validation.json"),
            ("pdb_backbone", "outputs/pdb_backbone.pdb"),
            ("afdb_backbone", "outputs/afdb_backbone.pdb"),
            ("backbone_residue_provenance", "outputs/backbone_residue_provenance.tsv"),
        )
    )
    manifest = StageManifest(
        pipeline_name="dataset_a" + "_scale",
        pipeline_version=_canonical_text(pipeline_version, "pipeline_version"),
        schema_version=STAGE_MANIFEST_SCHEMA_VERSION,
        run_id=_canonical_text(run_id, "run_id"),
        protein_id=resolution.protein_id,
        tier=resolution.tier,
        stage=P1_STAGE_NAME,
        status=LifecycleStatus.COMPLETE,
        input_digest=resolution.input_digest,
        config_digest=resolution.config_digest,
        seed_identity=None,
        declared_outputs=outputs,
        validation_pass=True,
    )
    write_stage_manifest(stage_dir / "stage_manifest.json", manifest)
    return P1RunResult(
        status=LifecycleStatus.COMPLETE,
        validation=validation,
        failure_code=None,
        failure_message=None,
        input_digest=resolution.input_digest,
        config_digest=resolution.config_digest,
        residue_count=resolution.residue_count,
        canonical_sequence=resolution.canonical_sequence,
    )
