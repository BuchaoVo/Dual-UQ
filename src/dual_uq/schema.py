from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

RESIDUE_MAPPING_SCHEMA_VERSION = 2
RESIDUE_NUMBERING_COLUMNS = (
    "auth_asym_id",
    "label_asym_id",
    "auth_seq_id",
    "label_seq_id",
    "insertion_code",
)
_LEGACY_RESIDUE_PATTERN = re.compile(r"^([+-]?\d+)([A-Za-z]?)$")


class AmbiguousLegacyResidueIdentifier(ValueError):
    """Raised when a legacy residue alias cannot be migrated without guessing."""


def _clean_identifier(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    cleaned = str(value).strip()
    return pd.NA if cleaned in {"", ".", "?"} else cleaned


def _nullable_integer(series: pd.Series, field: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.map(_clean_identifier).notna() & converted.isna()
    if invalid.any():
        values = series.loc[invalid].astype(str).tolist()
        raise AmbiguousLegacyResidueIdentifier(
            f"{field} contains non-integer identifiers: {values[:5]}"
        )
    return converted.astype("Int64")


def _split_legacy_residue_number(value: Any) -> tuple[int, str]:
    cleaned = _clean_identifier(value)
    if pd.isna(cleaned):
        raise AmbiguousLegacyResidueIdentifier(
            "Legacy pdb_residue_number contains a missing identifier."
        )
    match = _LEGACY_RESIDUE_PATTERN.fullmatch(str(cleaned))
    if match is None:
        raise AmbiguousLegacyResidueIdentifier(
            f"Legacy residue identifier {cleaned!r} is ambiguous."
        )
    return int(match.group(1)), match.group(2).upper()


def normalize_residue_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Normalize explicit or legacy mapping data at its I/O boundary.

    Explicit author and label namespaces remain separate. Legacy aliases are
    accepted solely as migration input and are never used to invent label IDs.
    """
    normalized = mapping.copy()
    if "residue_mapping_schema_version" in normalized:
        version_values = normalized["residue_mapping_schema_version"].map(
            _clean_identifier
        )
        numeric_versions = pd.to_numeric(version_values, errors="coerce")
        invalid_versions = (
            version_values.isna()
            | numeric_versions.isna()
            | numeric_versions.mod(1).ne(0)
        )
        if invalid_versions.any():
            raise AmbiguousLegacyResidueIdentifier(
                "Declared residue mapping schema version is missing or malformed."
            )
        declared_versions = set(numeric_versions.astype(int))
        if not declared_versions.issubset({1, RESIDUE_MAPPING_SCHEMA_VERSION}):
            raise AmbiguousLegacyResidueIdentifier(
                f"Unsupported residue mapping schema versions: {declared_versions}"
            )
    has_explicit_author = {"auth_asym_id", "auth_seq_id"}.issubset(normalized.columns)

    if has_explicit_author:
        if "label_asym_id" not in normalized:
            normalized["label_asym_id"] = pd.NA
        if "label_seq_id" not in normalized:
            normalized["label_seq_id"] = pd.NA
        if "insertion_code" not in normalized:
            normalized["insertion_code"] = ""
        if "residue_mapping_provenance" not in normalized:
            normalized["residue_mapping_provenance"] = "explicit"
    else:
        if "pdb_residue_number" not in normalized:
            raise AmbiguousLegacyResidueIdentifier(
                "Mapping has neither explicit author IDs nor pdb_residue_number."
            )
        split = normalized["pdb_residue_number"].map(_split_legacy_residue_number)
        normalized["auth_seq_id"] = split.map(lambda item: item[0])
        normalized["insertion_code"] = split.map(lambda item: item[1])
        chain_aliases = [
            name for name in ("pdb_chain_id", "chain_id") if name in normalized.columns
        ]
        if len(chain_aliases) == 2:
            left = normalized[chain_aliases[0]].map(_clean_identifier)
            right = normalized[chain_aliases[1]].map(_clean_identifier)
            conflicts = left.notna() & right.notna() & left.ne(right)
            if conflicts.any():
                raise AmbiguousLegacyResidueIdentifier(
                    "Legacy chain aliases pdb_chain_id and chain_id disagree."
                )
        if len(chain_aliases) == 2:
            normalized["auth_asym_id"] = normalized[chain_aliases[0]].combine_first(
                normalized[chain_aliases[1]]
            )
        elif chain_aliases:
            normalized["auth_asym_id"] = normalized[chain_aliases[0]]
        else:
            normalized["auth_asym_id"] = pd.NA
        normalized["label_asym_id"] = pd.NA
        normalized["label_seq_id"] = pd.NA
        normalized["residue_mapping_provenance"] = "legacy_alias"

    normalized["auth_asym_id"] = normalized["auth_asym_id"].map(_clean_identifier)
    normalized["label_asym_id"] = normalized["label_asym_id"].map(_clean_identifier)
    normalized["auth_seq_id"] = _nullable_integer(
        normalized["auth_seq_id"], "auth_seq_id"
    )
    normalized["label_seq_id"] = _nullable_integer(
        normalized["label_seq_id"], "label_seq_id"
    )
    normalized["insertion_code"] = (
        normalized["insertion_code"]
        .map(_clean_identifier)
        .fillna("")
        .astype(str)
        .str.upper()
    )
    normalized["residue_mapping_schema_version"] = RESIDUE_MAPPING_SCHEMA_VERSION

    author_available = normalized["auth_seq_id"].notna()
    label_available = (
        normalized["label_asym_id"].notna() & normalized["label_seq_id"].notna()
    )
    if (~author_available & ~label_available).any():
        raise AmbiguousLegacyResidueIdentifier(
            "A mapping row has neither an author residue ID nor a complete label ID."
        )
    return normalized


class ProteinRecord(BaseModel):
    protein_id: str
    uniprot_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    length: int = Field(gt=0)
    sequence_cluster: str | None = None
    domain_count: int | None = Field(default=None, ge=1)
    secondary_structure_class: str | None = None
    split: str = "unassigned"
    mapping_coverage: float = Field(ge=0.0, le=1.0)
    sequence_identity: float = Field(ge=0.0, le=1.0)
    quality_flag: str = "pending"


class StructureRecord(BaseModel):
    structure_id: str
    protein_id: str
    source: str
    parent_structure_id: str | None = None
    coordinate_path: str
    plddt_path: str | None = None
    pae_path: str | None = None
    perturbation_type: str | None = None
    perturbation_strength: float | None = None
    perturbed_residues: str | None = None
    random_seed: int | None = None
    mapping_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    structure_valid: bool = False


class SequenceRecord(BaseModel):
    sequence_id: str
    protein_id: str
    generation_structure_id: str
    generator: str
    checkpoint: str
    temperature: float
    sampling_seed: int
    sequence: str
    sequence_identity_to_native: float | None = Field(default=None, ge=0.0, le=1.0)
    generation_status: str = "pending"


class ScoreRecord(BaseModel):
    protein_id: str
    structure_id: str
    sequence_id: str
    evaluator_id: str
    property_id: str
    score: float | None = None
    score_mean: float | None = None
    score_variance: float | None = Field(default=None, ge=0.0)
    ood_score: float | None = None
    run_status: str = "pending"
    error_message: str | None = None
    runtime: float | None = Field(default=None, ge=0.0)
