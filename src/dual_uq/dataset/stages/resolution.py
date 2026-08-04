"""P0: resolve, validate, and immutably freeze Dataset A inputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.errors import PAEMappingError
from dual_uq.core.hashing import sha256_canonical, sha256_file

from ..models import AFDBFragment, ResidueKey, ResidueProvenance
from ..pipeline.resume import evaluate_resume
from ..pipeline.status import (
    STAGE_MANIFEST_SCHEMA_VERSION,
    LifecycleStatus,
    OutputDeclaration,
    StageManifest,
    ValidationRecord,
    write_stage_manifest,
)
from ..services.afdb import load_pae_json, require_fragment_coverage

P0_STAGE_NAME = "P0_resolve_and_freeze_inputs"
P0_INPUT_PATH_FIELDS = (
    "pair_qc_path",
    "residue_mapping_path",
    "pdb_structure_path",
    "afdb_metadata_path",
    "afdb_model_path",
    "afdb_plddt_path",
    "afdb_pae_path",
)

_CORE_FIELDS = (
    "protein_id",
    "screening_index",
    "pair_id",
    "mechanism_label",
    "tier",
)
_MAPPING_COLUMNS = {
    "pdb_residue_name",
    "uniprot_residue_number",
    "auth_asym_id",
    "auth_seq_id",
    "insertion_code",
    "label_asym_id",
    "label_seq_id",
}
_EXPECTED_SUFFIXES = {
    "pair_qc_path": {".json"},
    "residue_mapping_path": {".tsv", ".parquet"},
    "pdb_structure_path": {".cif", ".mmcif"},
    "afdb_metadata_path": {".json"},
    "afdb_model_path": {".cif", ".mmcif"},
    "afdb_plddt_path": {".json"},
    "afdb_pae_path": {".json"},
}


class P0ValidationError(ValueError):
    """One structured P0 input validation failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class FrozenInputFile:
    logical_name: str
    original_path: str
    project_relative_path: str | None
    resolved_path: Path
    sha256: str

    def lock_record(self) -> dict[str, str | None]:
        return {
            "logical_name": self.logical_name,
            "original_path": self.original_path,
            "project_relative_path": self.project_relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class P0Resolution:
    manifest_identity: dict[str, Any]
    input_files: dict[str, FrozenInputFile]
    input_digest: str
    config_digest: str
    selected_model_entity_id: str
    selected_model_version: int
    fragment: AFDBFragment
    mapping: pd.DataFrame
    mapping_summary: dict[str, Any]
    frozen_inputs: dict[str, Any]
    input_lock: dict[str, Any]


@dataclass(frozen=True)
class P0RunResult:
    status: LifecycleStatus
    validation: ValidationRecord
    failure_code: str | None
    failure_message: str | None
    input_digest: str | None
    config_digest: str | None
    selected_model_entity_id: str | None
    input_files: dict[str, FrozenInputFile]
    input_lock: dict[str, Any]


def _as_row(manifest_row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(manifest_row, pd.Series):
        return manifest_row.to_dict()
    if not isinstance(manifest_row, Mapping):
        raise P0ValidationError(
            "invalid_manifest_row", "Manifest row must be an explicit mapping"
        )
    return dict(manifest_row)


def _canonical_text(value: Any, field: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise P0ValidationError(
            "invalid_manifest_identity",
            f"{field} must be a non-empty canonical string",
            field=field,
        )
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise P0ValidationError(
            "invalid_manifest_identity", f"{field} must be a positive integer", field=field
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise P0ValidationError(
            "invalid_manifest_identity", f"{field} must be a positive integer", field=field
        ) from exc
    if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise P0ValidationError(
            "invalid_manifest_identity", f"{field} must be a positive integer", field=field
        )
    return int(numeric)


def _manifest_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _CORE_FIELDS if field not in row]
    if missing:
        raise P0ValidationError(
            "missing_manifest_field",
            "Manifest row is missing required identity fields",
            missing_fields=missing,
        )
    tier = _positive_integer(row["tier"], "tier")
    if tier not in {1, 2}:
        raise P0ValidationError(
            "invalid_manifest_identity", "tier must be integer 1 or 2", field="tier"
        )
    return {
        "protein_id": _canonical_text(row["protein_id"], "protein_id"),
        "screening_index": _positive_integer(
            row["screening_index"], "screening_index"
        ),
        "pair_id": _canonical_text(row["pair_id"], "pair_id"),
        "mechanism_label": _canonical_text(
            row["mechanism_label"], "mechanism_label"
        ),
        "tier": tier,
    }


def _relative_to_root(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _resolve_input_files(
    row: Mapping[str, Any], project_root: Path
) -> dict[str, FrozenInputFile]:
    root = project_root.resolve()
    records: dict[str, FrozenInputFile] = {}
    for logical_name in P0_INPUT_PATH_FIELDS:
        value = row.get(logical_name)
        if type(value) is not str or not value or value != value.strip() or "\0" in value:
            raise P0ValidationError(
                "invalid_input_path",
                f"{logical_name} must be a non-empty canonical path",
                logical_name=logical_name,
            )
        original = value
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if resolved.suffix.lower() not in _EXPECTED_SUFFIXES[logical_name]:
            raise P0ValidationError(
                "invalid_input_file_type",
                f"{logical_name} has an unsupported file type",
                logical_name=logical_name,
                path=original,
            )
        if not resolved.exists():
            raise P0ValidationError(
                "missing_required_file",
                f"Required P0 input does not exist: {original}",
                logical_name=logical_name,
                path=original,
            )
        if not resolved.is_file():
            raise P0ValidationError(
                "input_not_regular_file",
                f"Required P0 input is not a regular file: {original}",
                logical_name=logical_name,
                path=original,
            )
        records[logical_name] = FrozenInputFile(
            logical_name=logical_name,
            original_path=original,
            project_relative_path=_relative_to_root(resolved, root),
            resolved_path=resolved,
            sha256=sha256_file(resolved),
        )
    return records


def _config_digest(config: Mapping[str, Any], pipeline_version: str) -> str:
    if not isinstance(config, Mapping):
        raise P0ValidationError("invalid_config", "P0 config must be a mapping")
    try:
        return sha256_canonical(
            {"pipeline_version": pipeline_version, "config": dict(config)}
        )
    except (TypeError, ValueError) as exc:
        raise P0ValidationError("invalid_config", "P0 config is not canonical JSON") from exc


def _input_digest(
    identity: Mapping[str, Any], files: Mapping[str, FrozenInputFile]
) -> str:
    return sha256_canonical(
        {
            "manifest_identity": dict(identity),
            "input_files": [
                {
                    "logical_name": name,
                    "original_path": files[name].original_path,
                    "project_relative_path": files[name].project_relative_path,
                    "sha256": files[name].sha256,
                }
                for name in sorted(files)
            ],
        }
    )


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P0ValidationError(
            "invalid_json_input", f"Invalid {description}: {path}", description=description
        ) from exc
    if not isinstance(payload, dict):
        raise P0ValidationError(
            "invalid_json_input", f"{description} must be a JSON object", description=description
        )
    return payload


def _load_mapping(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    except Exception as exc:
        raise P0ValidationError(
            "invalid_residue_mapping", f"Unable to read residue mapping: {path}"
        ) from exc


def _optional_integer(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if type(value) is str and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _mapping_positions(mapping: pd.DataFrame) -> pd.Series:
    if "uniprot_residue_number" not in mapping:
        raise P0ValidationError(
            "missing_mapping_column",
            "Residue mapping lacks uniprot_residue_number",
            missing_columns=["uniprot_residue_number"],
        )
    values = pd.to_numeric(mapping["uniprot_residue_number"], errors="coerce")
    if (
        mapping.empty
        or values.isna().any()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
        or values.mod(1).ne(0).any()
        or values.le(0).any()
    ):
        raise P0ValidationError(
            "invalid_uniprot_mapping",
            "Mapped UniProt positions must be explicit positive integers",
        )
    return values.astype(int)


def _metadata_identity(
    metadata: Mapping[str, Any], pair_qc: Mapping[str, Any]
) -> tuple[str, int, int, int]:
    selected = pair_qc.get("afdb_model_entity_id")
    metadata_model = metadata.get("modelEntityId") or metadata.get("entryId")
    if type(selected) is not str or not selected.strip() or metadata_model != selected:
        raise P0ValidationError(
            "afdb_model_identity_mismatch",
            "AFDB metadata identity does not match the explicitly selected model",
            selected_model_entity_id=selected,
            metadata_model_entity_id=metadata_model,
        )
    version = _optional_integer(pair_qc.get("afdb_version"))
    metadata_version = _optional_integer(metadata.get("latestVersion"))
    if version is None or metadata_version != version:
        raise P0ValidationError(
            "afdb_model_version_mismatch",
            "AFDB metadata version does not match the explicitly selected model",
            selected_model_entity_id=selected,
        )
    start = _optional_integer(metadata.get("uniprotStart", metadata.get("sequenceStart")))
    end = _optional_integer(metadata.get("uniprotEnd", metadata.get("sequenceEnd")))
    sequence_start = _optional_integer(metadata.get("sequenceStart"))
    sequence_end = _optional_integer(metadata.get("sequenceEnd"))
    if start is None or end is None or start < 1 or end < start:
        raise P0ValidationError(
            "invalid_afdb_fragment_interval", "AFDB metadata fragment interval is invalid"
        )
    if sequence_start is not None and sequence_start != start:
        raise P0ValidationError(
            "invalid_afdb_fragment_interval", "AFDB metadata start fields disagree"
        )
    if sequence_end is not None and sequence_end != end:
        raise P0ValidationError(
            "invalid_afdb_fragment_interval", "AFDB metadata end fields disagree"
        )
    for field, expected in (
        ("afdb_fragment_start", start),
        ("afdb_fragment_end", end),
        ("afdb_fragment_length", end - start + 1),
    ):
        observed = _optional_integer(pair_qc.get(field))
        if observed is not None and observed != expected:
            raise P0ValidationError(
                "afdb_fragment_length_mismatch",
                f"Pair QC {field} conflicts with selected AFDB metadata",
                selected_model_entity_id=selected,
            )
    return selected, version, start, end


def _artifact_url_identity(url: Any, kind: str) -> tuple[str, int]:
    if type(url) is not str or not url:
        raise P0ValidationError(
            f"{kind}_model_identity_mismatch", f"AFDB metadata lacks explicit {kind} URL"
        )
    suffix = {
        "model": r"model",
        "plddt": r"confidence",
        "pae": r"predicted_aligned_error",
    }[kind]
    name = Path(urlparse(url).path).name
    match = re.fullmatch(rf"(.+)-{suffix}_v([1-9][0-9]*)\.(?:cif|json)", name)
    if match is None:
        raise P0ValidationError(
            f"{kind}_model_identity_mismatch", f"Invalid AFDB metadata {kind} URL"
        )
    return match.group(1), int(match.group(2))


def _validate_artifact_identities(
    metadata: Mapping[str, Any], selected: str, version: int
) -> None:
    for kind, field in (
        ("model", "cifUrl"),
        ("plddt", "plddtDocUrl"),
        ("pae", "paeDocUrl"),
    ):
        identity, artifact_version = _artifact_url_identity(metadata.get(field), kind)
        if identity != selected or artifact_version != version:
            raise P0ValidationError(
                f"{kind}_model_identity_mismatch",
                f"{kind} identity/version does not match selected AFDB model",
                selected_model_entity_id=selected,
                artifact_model_entity_id=identity,
            )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _validate_model_mmcif(path: Path, selected: str, expected_length: int) -> None:
    try:
        mmcif = MMCIF2Dict(str(path))
    except Exception as exc:
        raise P0ValidationError(
            "invalid_afdb_model", f"Unable to parse selected AFDB model: {path}"
        ) from exc
    entry = _as_list(mmcif.get("_entry.id"))
    if len(entry) != 1 or entry[0] != selected:
        raise P0ValidationError(
            "afdb_model_identity_mismatch",
            "AFDB model mmCIF entry identity does not match selected model",
            selected_model_entity_id=selected,
        )
    atom_names = _as_list(mmcif.get("_atom_site.label_atom_id"))
    auth_chains = _as_list(mmcif.get("_atom_site.auth_asym_id"))
    auth_seq_ids = _as_list(mmcif.get("_atom_site.auth_seq_id"))
    insertion_codes = _as_list(mmcif.get("_atom_site.pdbx_PDB_ins_code"))
    lengths = {len(values) for values in (atom_names, auth_chains, auth_seq_ids, insertion_codes)}
    if len(lengths) != 1 or not atom_names:
        raise P0ValidationError(
            "invalid_afdb_model", "AFDB model mmCIF atom-site columns are incomplete"
        )
    residues: set[ResidueKey] = set()
    for atom_name, chain, sequence, insertion in zip(
        atom_names, auth_chains, auth_seq_ids, insertion_codes, strict=True
    ):
        if str(atom_name).strip() != "CA":
            continue
        sequence_id = _optional_integer(sequence)
        if sequence_id is None:
            raise P0ValidationError(
                "invalid_afdb_model", "AFDB model CA residue numbering is invalid"
            )
        residues.add(
            ResidueKey(
                source_id=selected,
                auth_chain_id=str(chain).strip(),
                auth_seq_id=sequence_id,
                insertion_code=str(insertion),
            )
        )
    if len(residues) != expected_length:
        raise P0ValidationError(
            "afdb_fragment_length_mismatch",
            "AFDB model residue count does not match selected fragment interval",
            model_residue_count=len(residues),
            expected_length=expected_length,
            selected_model_entity_id=selected,
        )


def _validate_pdb_mmcif(path: Path, expected_pdb_id: Any) -> None:
    try:
        mmcif = MMCIF2Dict(str(path))
    except Exception as exc:
        raise P0ValidationError(
            "invalid_pdb_structure", f"Unable to parse PDB mmCIF input: {path}"
        ) from exc
    entry = _as_list(mmcif.get("_entry.id"))
    atom_names = _as_list(mmcif.get("_atom_site.label_atom_id"))
    if (
        len(entry) != 1
        or type(expected_pdb_id) is not str
        or str(entry[0]).lower() != expected_pdb_id.lower()
        or not atom_names
    ):
        raise P0ValidationError(
            "invalid_pdb_structure",
            "PDB mmCIF entry identity or atom-site content is invalid",
        )


def _validate_pair_qc(
    pair_qc: Mapping[str, Any],
    identity: Mapping[str, Any],
    files: Mapping[str, FrozenInputFile],
) -> None:
    if pair_qc.get("quality_flag") != "pass":
        raise P0ValidationError(
            "pair_qc_not_acceptable", "Pair QC quality_flag must be pass"
        )
    expected_pair = (
        f"{str(pair_qc.get('pdb_id', '')).lower()}_"
        f"{pair_qc.get('chain_id')}__{pair_qc.get('uniprot_id')}"
    )
    if expected_pair.lower() != identity["pair_id"].lower():
        raise P0ValidationError(
            "pair_qc_identity_mismatch", "Pair QC identity does not match manifest pair_id"
        )
    for qc_field, logical_name in (
        ("pdb_path", "pdb_structure_path"),
        ("mapping_path", "residue_mapping_path"),
        ("afdb_model_path", "afdb_model_path"),
        ("plddt_path", "afdb_plddt_path"),
        ("pae_path", "afdb_pae_path"),
    ):
        value = pair_qc.get(qc_field)
        if type(value) is not str or Path(value).resolve() != files[logical_name].resolved_path:
            raise P0ValidationError(
                "pair_qc_path_mismatch",
                f"Pair QC {qc_field} does not match explicit manifest input",
                logical_name=logical_name,
            )


def _integer_series(mapping: pd.DataFrame, column: str, code: str) -> pd.Series:
    values = pd.to_numeric(mapping[column], errors="coerce")
    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
        or values.mod(1).ne(0).any()
    ):
        raise P0ValidationError(code, f"Mapping {column} must contain explicit integers")
    return values.astype(int)


def _optional_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?"} else text


def _validate_mapping(
    mapping: pd.DataFrame,
    positions: pd.Series,
    identity: Mapping[str, Any],
    pair_qc: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise P0ValidationError(
            "missing_mapping_column",
            "Residue mapping lacks required provenance columns",
            missing_columns=missing,
        )
    if positions.duplicated().any():
        raise P0ValidationError(
            "ambiguous_uniprot_mapping", "Mapped UniProt positions must be unique"
        )
    table = mapping.copy()
    table["uniprot_residue_number"] = positions
    if "uniprot_id" in table:
        observed = {str(value).strip() for value in table["uniprot_id"]}
        if observed != {str(pair_qc.get("uniprot_id"))}:
            raise P0ValidationError(
                "mapping_uniprot_identity_mismatch",
                "Residue mapping UniProt identity conflicts with pair QC",
            )
    auth_sequence = _integer_series(
        table, "auth_seq_id", "invalid_auth_residue_identity"
    )
    if auth_sequence.isna().any():
        raise P0ValidationError(
            "invalid_auth_residue_identity", "Author residue identity is missing"
        )
    auth_keys: list[ResidueKey] = []
    for row_index, row in table.iterrows():
        auth_chain = str(row["auth_asym_id"]).strip()
        if not auth_chain or auth_chain in {".", "?"}:
            raise P0ValidationError(
                "invalid_auth_residue_identity",
                "Author chain/residue identity must be explicit; label fallback is forbidden",
                row=int(row_index),
            )
        label_chain = _optional_label(row["label_asym_id"])
        label_seq = _optional_integer(row["label_seq_id"])
        if (label_chain is None) != (label_seq is None):
            raise P0ValidationError(
                "invalid_label_residue_identity",
                "Label chain and sequence identity must be jointly present or absent",
                row=int(row_index),
            )
        try:
            key = ResidueKey(
                source_id=str(identity["pair_id"]),
                auth_chain_id=auth_chain,
                auth_seq_id=int(auth_sequence.loc[row_index]),
                insertion_code=str(row["insertion_code"]),
            )
            ResidueProvenance(
                key=key,
                label_chain_id=label_chain,
                label_seq_id=label_seq,
                raw_resname=str(row["pdb_residue_name"]).strip(),
                record_type="ATOM",
            )
        except (TypeError, ValueError) as exc:
            raise P0ValidationError(
                "invalid_auth_residue_identity",
                "Residue mapping provenance is invalid",
                row=int(row_index),
            ) from exc
        auth_keys.append(key)
    if len(set(auth_keys)) != len(auth_keys):
        raise P0ValidationError(
            "ambiguous_auth_residue_mapping", "Author residue keys must be unique"
        )
    if "output_position" in table:
        output_positions = _integer_series(
            table, "output_position", "invalid_output_position"
        )
        if output_positions.le(0).any() or output_positions.duplicated().any():
            raise P0ValidationError(
                "ambiguous_output_position",
                "Output positions must be unique positive integers",
            )
        table["output_position"] = output_positions
    if _optional_integer(pair_qc.get("mapped_residue_count")) != len(table):
        raise P0ValidationError(
            "mapping_pair_qc_mismatch", "Mapping row count conflicts with pair QC"
        )
    for field, observed in (
        ("mapped_uniprot_start", int(positions.min())),
        ("mapped_uniprot_end", int(positions.max())),
    ):
        qc_value = _optional_integer(pair_qc.get(field))
        if qc_value is not None and qc_value != observed:
            raise P0ValidationError(
                "mapping_pair_qc_mismatch", f"Mapping interval conflicts with pair QC {field}"
            )
    return table, {
        "row_count": len(table),
        "mapped_uniprot_start": int(positions.min()),
        "mapped_uniprot_end": int(positions.max()),
        "auth_key_count": len(auth_keys),
        "label_identity_count": int(
            sum(_optional_label(value) is not None for value in table["label_asym_id"])
        ),
        "output_position_present": "output_position" in table,
    }


def _validate_plddt(path: Path, expected_length: int) -> None:
    payload = _load_json_object(path, "AFDB pLDDT")
    residue_numbers = payload.get("residueNumber")
    scores = payload.get("confidenceScore")
    if not isinstance(residue_numbers, list) or not isinstance(scores, list):
        raise P0ValidationError(
            "invalid_plddt_schema", "AFDB pLDDT lacks residueNumber/confidenceScore"
        )
    if len(residue_numbers) != expected_length or len(scores) != expected_length:
        raise P0ValidationError(
            "plddt_length_mismatch", "pLDDT residue count does not match selected model"
        )
    try:
        residue_array = np.asarray(residue_numbers)
        score_array = np.asarray(scores)
    except (TypeError, ValueError) as exc:
        raise P0ValidationError("invalid_plddt_schema", "Invalid pLDDT arrays") from exc
    if (
        np.issubdtype(residue_array.dtype, np.bool_)
        or not np.issubdtype(residue_array.dtype, np.number)
        or not np.array_equal(residue_array, np.arange(1, expected_length + 1))
    ):
        raise P0ValidationError(
            "invalid_plddt_residue_numbering",
            "pLDDT residueNumber must be exactly model-local 1..N",
        )
    if (
        np.issubdtype(score_array.dtype, np.bool_)
        or not np.issubdtype(score_array.dtype, np.number)
        or np.issubdtype(score_array.dtype, np.complexfloating)
    ):
        raise P0ValidationError("invalid_plddt_schema", "pLDDT must be real numeric")
    numeric_scores = score_array.astype(float)
    if (
        not np.isfinite(numeric_scores).all()
        or (numeric_scores < 0.0).any()
        or (numeric_scores > 100.0).any()
    ):
        raise P0ValidationError("invalid_plddt_values", "pLDDT values are invalid")


def _ensure_afdb_artifact_directory(files: Mapping[str, FrozenInputFile]) -> None:
    parents = {
        files[name].resolved_path.parent
        for name in (
            "afdb_metadata_path",
            "afdb_model_path",
            "afdb_plddt_path",
            "afdb_pae_path",
        )
    }
    if len(parents) != 1:
        raise P0ValidationError(
            "afdb_artifact_directory_mismatch",
            "Selected AFDB metadata/model/pLDDT/PAE must share one explicit artifact directory",
        )


def resolve_p0_inputs(
    manifest_row: Mapping[str, Any] | pd.Series,
    *,
    project_root: Path,
    config: Mapping[str, Any],
    pipeline_version: str = "dataset-a.v1",
) -> P0Resolution:
    """Resolve and validate one manifest row without writing P0 outputs."""
    row = _as_row(manifest_row)
    identity = _manifest_identity(row)
    pipeline_version = _canonical_text(pipeline_version, "pipeline_version")
    config_digest = _config_digest(config, pipeline_version)
    files = _resolve_input_files(row, project_root)
    input_digest = _input_digest(identity, files)
    pair_qc = _load_json_object(files["pair_qc_path"].resolved_path, "pair QC")
    _validate_pair_qc(pair_qc, identity, files)
    _validate_pdb_mmcif(
        files["pdb_structure_path"].resolved_path, pair_qc.get("pdb_id")
    )
    mapping = _load_mapping(files["residue_mapping_path"].resolved_path)
    positions = _mapping_positions(mapping)
    metadata = _load_json_object(
        files["afdb_metadata_path"].resolved_path, "AFDB metadata"
    )
    selected, version, fragment_start, fragment_end = _metadata_identity(
        metadata, pair_qc
    )
    fragment = AFDBFragment(
        model_entity_id=selected,
        uniprot_start=fragment_start,
        uniprot_end=fragment_end,
        model_residue_count=fragment_end - fragment_start + 1,
    )
    try:
        require_fragment_coverage(
            fragment, (int(positions.min()), int(positions.max()))
        )
    except PAEMappingError as exc:
        raise P0ValidationError(
            exc.code,
            str(exc),
            selected_model_entity_id=selected,
            mapped_interval=[int(positions.min()), int(positions.max())],
        ) from exc
    _ensure_afdb_artifact_directory(files)
    _validate_artifact_identities(metadata, selected, version)
    _validate_model_mmcif(
        files["afdb_model_path"].resolved_path,
        selected,
        fragment.model_residue_count,
    )
    validated_mapping, mapping_summary = _validate_mapping(
        mapping, positions, identity, pair_qc
    )
    _validate_plddt(
        files["afdb_plddt_path"].resolved_path, fragment.model_residue_count
    )
    try:
        pae = load_pae_json(files["afdb_pae_path"].resolved_path, fragment)
    except PAEMappingError as exc:
        raise P0ValidationError(
            exc.code, str(exc), selected_model_entity_id=selected
        ) from exc
    if pae.model_entity_id != selected:
        raise P0ValidationError(
            "pae_model_identity_mismatch", "PAE identity does not match selected model"
        )
    input_records = [files[name].lock_record() for name in sorted(files)]
    fragment_record = {
        "uniprot_start": fragment_start,
        "uniprot_end": fragment_end,
        "model_length": fragment.model_residue_count,
    }
    frozen_inputs = {
        "pipeline_version": pipeline_version,
        **identity,
        "selected_model_identity": selected,
        "selected_model_version": version,
        "fragment_interval": fragment_record,
        "input_files": input_records,
    }
    input_lock = {
        **frozen_inputs,
        "input_digest": input_digest,
        "config_digest": config_digest,
        "mapping_summary": mapping_summary,
        "validation_pass": True,
    }
    return P0Resolution(
        manifest_identity=identity,
        input_files=files,
        input_digest=input_digest,
        config_digest=config_digest,
        selected_model_entity_id=selected,
        selected_model_version=version,
        fragment=fragment,
        mapping=validated_mapping,
        mapping_summary=mapping_summary,
        frozen_inputs=frozen_inputs,
        input_lock=input_lock,
    )


def _atomic_write_new(path: Path, payload: bytes) -> None:
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError as exc:
        raise P0ValidationError(
            "immutable_output_exists",
            f"Refusing to overwrite immutable P0 output: {path}",
        ) from exc


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


def _mapping_bytes(mapping: pd.DataFrame) -> bytes:
    buffer = StringIO(newline="")
    mapping.to_csv(buffer, sep="\t", index=False, na_rep="", lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _failure_result(
    error: P0ValidationError,
    *,
    status: LifecycleStatus = LifecycleStatus.FAILED_VALIDATION,
    config_digest: str | None = None,
) -> P0RunResult:
    validation = ValidationRecord(
        validation_pass=False,
        error_codes=(error.code,),
        details={"message": str(error), **error.details},
    )
    selected = error.details.get("selected_model_entity_id")
    return P0RunResult(
        status=status,
        validation=validation,
        failure_code=error.code,
        failure_message=str(error),
        input_digest=None,
        config_digest=config_digest,
        selected_model_entity_id=(selected if isinstance(selected, str) else None),
        input_files={},
        input_lock={},
    )


def run_p0(
    *,
    manifest_row: Mapping[str, Any] | pd.Series,
    project_root: Path,
    stage_dir: Path,
    config: Mapping[str, Any],
    pipeline_version: str,
    run_id: str,
) -> P0RunResult:
    """Validate and immutably materialize one P0 stage, or reuse it via A3."""
    has_frozen_stage = (stage_dir / "stage_manifest.json").is_file()
    try:
        row = _as_row(manifest_row)
        identity = _manifest_identity(row)
        checked_pipeline_version = _canonical_text(
            pipeline_version, "pipeline_version"
        )
        current_config_digest = _config_digest(config, checked_pipeline_version)
        current_files = _resolve_input_files(row, project_root)
        current_input_digest = _input_digest(identity, current_files)
    except P0ValidationError as exc:
        if has_frozen_stage:
            drift = P0ValidationError(
                "input_unresolvable_drift",
                "Current inputs cannot reproduce the immutable P0 lock",
                cause_code=exc.code,
                cause_message=str(exc),
            )
            return _failure_result(
                drift,
                status=LifecycleStatus.BLOCKED_INPUT_DRIFT,
            )
        return _failure_result(exc)
    if has_frozen_stage:
        preliminary = evaluate_resume(
            stage_dir=stage_dir,
            current_input_digest=current_input_digest,
            current_config_digest=current_config_digest,
            current_validation=ValidationRecord(validation_pass=True),
        )
        if preliminary.target_status is LifecycleStatus.BLOCKED_INPUT_DRIFT:
            error = P0ValidationError(
                preliminary.reason_code,
                preliminary.reason_code,
                **preliminary.details,
            )
            failed = _failure_result(
                error,
                status=LifecycleStatus.BLOCKED_INPUT_DRIFT,
                config_digest=current_config_digest,
            )
            return P0RunResult(
                **{
                    **failed.__dict__,
                    "input_digest": current_input_digest,
                    "input_files": current_files,
                }
            )
    try:
        resolution = resolve_p0_inputs(
            row,
            project_root=project_root,
            config=config,
            pipeline_version=pipeline_version,
        )
    except P0ValidationError as exc:
        return _failure_result(exc)
    validation = ValidationRecord(
        validation_pass=True,
        details={
            "selected_model_entity_id": resolution.selected_model_entity_id,
            "mapping_summary": resolution.mapping_summary,
            "checked_input_count": len(resolution.input_files),
        },
    )
    decision = evaluate_resume(
        stage_dir=stage_dir,
        current_input_digest=resolution.input_digest,
        current_config_digest=resolution.config_digest,
        current_validation=validation,
    )
    if decision.can_skip:
        return P0RunResult(
            status=decision.target_status,
            validation=validation,
            failure_code=None,
            failure_message=None,
            input_digest=resolution.input_digest,
            config_digest=resolution.config_digest,
            selected_model_entity_id=resolution.selected_model_entity_id,
            input_files=resolution.input_files,
            input_lock=resolution.input_lock,
        )
    if decision.reason_code != "manifest_missing":
        error = P0ValidationError(decision.reason_code, decision.reason_code, **decision.details)
        result = _failure_result(
            error,
            status=decision.target_status,
            config_digest=resolution.config_digest,
        )
        return P0RunResult(
            **{
                **result.__dict__,
                "input_digest": resolution.input_digest,
                "selected_model_entity_id": resolution.selected_model_entity_id,
                "input_files": resolution.input_files,
                "input_lock": resolution.input_lock,
            }
        )

    validation_path = stage_dir / "validation.json"
    frozen_path = stage_dir / "outputs/frozen_inputs.json"
    mapping_path = stage_dir / "outputs/residue_mapping.tsv"
    lock_path = stage_dir / "outputs/input_lock.json"
    targets = (validation_path, frozen_path, mapping_path, lock_path)
    if any(path.exists() for path in targets):
        return _failure_result(
            P0ValidationError(
                "immutable_output_exists", "Refusing to overwrite existing P0 outputs"
            ),
            config_digest=resolution.config_digest,
        )
    try:
        _atomic_write_new(validation_path, _json_bytes(validation.as_dict()))
        _atomic_write_new(frozen_path, _json_bytes(resolution.frozen_inputs))
        _atomic_write_new(mapping_path, _mapping_bytes(resolution.mapping))
        _atomic_write_new(lock_path, _json_bytes(resolution.input_lock))
    except P0ValidationError as exc:
        return _failure_result(exc, config_digest=resolution.config_digest)
    outputs = tuple(
        OutputDeclaration(name, relative, sha256_file(stage_dir / relative), True)
        for name, relative in (
            ("validation", "validation.json"),
            ("frozen_inputs", "outputs/frozen_inputs.json"),
            ("residue_mapping", "outputs/residue_mapping.tsv"),
            ("input_lock", "outputs/input_lock.json"),
        )
    )
    manifest = StageManifest(
        pipeline_name="dataset_a" + "_scale",
        pipeline_version=pipeline_version,
        schema_version=STAGE_MANIFEST_SCHEMA_VERSION,
        run_id=_canonical_text(run_id, "run_id"),
        protein_id=resolution.manifest_identity["protein_id"],
        tier=resolution.manifest_identity["tier"],
        stage=P0_STAGE_NAME,
        status=LifecycleStatus.COMPLETE,
        input_digest=resolution.input_digest,
        config_digest=resolution.config_digest,
        seed_identity=None,
        declared_outputs=outputs,
        validation_pass=True,
    )
    write_stage_manifest(stage_dir / "stage_manifest.json", manifest)
    return P0RunResult(
        status=LifecycleStatus.COMPLETE,
        validation=validation,
        failure_code=None,
        failure_message=None,
        input_digest=resolution.input_digest,
        config_digest=resolution.config_digest,
        selected_model_entity_id=resolution.selected_model_entity_id,
        input_files=resolution.input_files,
        input_lock=resolution.input_lock,
    )
