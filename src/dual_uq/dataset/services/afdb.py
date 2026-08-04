"""Strict AFDB fragment-to-PAE coordinate mapping foundations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.errors import PAEMappingError

from ..models import (
    AFDBFragment,
    LongRangePAESummary,
    PAEMatrix,
    PAEResidueMapping,
)


def validate_pae_matrix(values: Any, *, expected_size: int) -> np.ndarray:
    """Return a float64 PAE matrix after strict shape and value validation."""
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise PAEMappingError(
            "invalid_fragment_length", "Expected fragment size must be an integer"
        )
    if expected_size < 1:
        raise PAEMappingError(
            "invalid_fragment_length", "Expected fragment size must be positive"
        )
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise PAEMappingError(
            "invalid_pae_matrix", "PAE must be a non-empty square matrix"
        ) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise PAEMappingError(
            "invalid_pae_matrix", "PAE must be a non-empty square matrix"
        )
    if array.shape != (expected_size, expected_size):
        raise PAEMappingError(
            "pae_fragment_length_mismatch",
            "PAE matrix size does not match the selected AFDB fragment length",
        )
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise PAEMappingError(
            "invalid_pae_matrix", "PAE values must have a real numeric dtype"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise PAEMappingError("invalid_pae_matrix", "PAE values must all be finite")
    if (result < 0.0).any():
        raise PAEMappingError("invalid_pae_matrix", "PAE values must be non-negative")
    return result


def load_pae_json(path: str | Path, fragment: AFDBFragment) -> PAEMatrix:
    """Load the audited AFDB list-of-one-record PAE JSON schema."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PAEMappingError(
            "invalid_pae_json_schema", f"Unable to read AFDB PAE JSON: {source}"
        ) from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
        or "predicted_aligned_error" not in payload[0]
    ):
        raise PAEMappingError(
            "invalid_pae_json_schema",
            "AFDB PAE JSON schema must be one record with predicted_aligned_error",
        )
    values = validate_pae_matrix(
        payload[0]["predicted_aligned_error"],
        expected_size=fragment.model_residue_count,
    )
    return PAEMatrix(model_entity_id=fragment.model_entity_id, values=values)


def require_fragment_coverage(
    fragment: AFDBFragment,
    target_interval: tuple[int, int],
) -> None:
    """Require the selected fragment to fully cover an inclusive UniProt interval."""
    if len(target_interval) != 2:
        raise PAEMappingError(
            "invalid_target_interval", "Target UniProt interval must have two endpoints"
        )
    start, end = target_interval
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end < start
    ):
        raise PAEMappingError(
            "invalid_target_interval", "Target UniProt interval is invalid"
        )
    if fragment.uniprot_start > start or fragment.uniprot_end < end:
        raise PAEMappingError(
            "unsupported_afdb_fragment",
            f"Selected AFDB fragment {fragment.model_entity_id} interval "
            f"{fragment.uniprot_start}-{fragment.uniprot_end} does not fully cover "
            f"target UniProt interval {start}-{end}",
        )


def map_uniprot_to_fragment_index(
    fragment: AFDBFragment, uniprot_position: int
) -> int:
    """Map one UniProt position to a zero-based PAE/model-local index."""
    if (
        isinstance(uniprot_position, bool)
        or not isinstance(uniprot_position, int)
        or uniprot_position < 1
    ):
        raise PAEMappingError(
            "invalid_uniprot_position", "UniProt position must be a positive integer"
        )
    if not fragment.uniprot_start <= uniprot_position <= fragment.uniprot_end:
        raise PAEMappingError(
            "unsupported_afdb_fragment",
            f"UniProt position {uniprot_position} is outside selected AFDB fragment "
            f"{fragment.model_entity_id} interval "
            f"{fragment.uniprot_start}-{fragment.uniprot_end}",
        )
    return uniprot_position - fragment.uniprot_start


def _positive_integer_series(table: pd.DataFrame, column: str) -> pd.Series:
    values = table[column]
    if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
        values.dtype
    ):
        raise PAEMappingError(
            "invalid_residue_mapping",
            f"{column} values must be positive integers",
        )
    numeric = pd.to_numeric(values, errors="coerce")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or numeric.mod(1).ne(0).any()
        or numeric.le(0).any()
    ):
        raise PAEMappingError(
            "invalid_residue_mapping",
            f"{column} values must be positive integers",
        )
    return numeric.astype(int)


def map_output_residues_to_pae(
    residue_mapping: pd.DataFrame, fragment: AFDBFragment
) -> PAEResidueMapping:
    """Map explicit output/UniProt rows to deterministic PAE indices."""
    if not isinstance(residue_mapping, pd.DataFrame):
        raise PAEMappingError(
            "invalid_residue_mapping", "Residue mapping must be a pandas DataFrame"
        )
    required = {"output_position", "uniprot_position"}
    missing = sorted(required - set(residue_mapping.columns))
    if missing:
        raise PAEMappingError(
            "invalid_residue_mapping",
            "Residue mapping is missing columns: " + ",".join(missing),
        )
    if residue_mapping.empty:
        raise PAEMappingError(
            "invalid_residue_mapping", "Residue mapping must not be empty"
        )
    table = pd.DataFrame(
        {
            "output_position": _positive_integer_series(
                residue_mapping, "output_position"
            ),
            "uniprot_position": _positive_integer_series(
                residue_mapping, "uniprot_position"
            ),
        }
    )
    if table["output_position"].duplicated().any():
        raise PAEMappingError(
            "ambiguous_residue_mapping", "Duplicate output position in residue mapping"
        )
    if table["uniprot_position"].duplicated().any():
        raise PAEMappingError(
            "ambiguous_residue_mapping", "Duplicate UniProt position in residue mapping"
        )
    table = table.sort_values(
        ["output_position", "uniprot_position"], kind="mergesort"
    )
    pae_indices = tuple(
        map_uniprot_to_fragment_index(fragment, int(position))
        for position in table["uniprot_position"]
    )
    return PAEResidueMapping(
        model_entity_id=fragment.model_entity_id,
        output_positions=tuple(int(value) for value in table["output_position"]),
        uniprot_positions=tuple(int(value) for value in table["uniprot_position"]),
        model_residue_positions=tuple(index + 1 for index in pae_indices),
        pae_indices=pae_indices,
    )


def extract_mapped_pae(
    pae: PAEMatrix, mapping: PAEResidueMapping
) -> np.ndarray:
    """Extract a mapped PAE submatrix without changing shape or coordinates."""
    if not isinstance(pae, PAEMatrix):
        raise PAEMappingError(
            "missing_pae_model_identity",
            "PAE model identity is required before extracting mapped coordinates",
        )
    if pae.model_entity_id != mapping.model_entity_id:
        raise PAEMappingError(
            "pae_model_identity_mismatch",
            "PAE model identity does not match the mapped AFDB fragment identity",
        )
    values = pae.values
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise PAEMappingError(
            "invalid_pae_matrix", "PAE must be a non-empty square matrix"
        ) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise PAEMappingError(
            "invalid_pae_matrix", "PAE must be a non-empty square matrix"
        )
    validated = validate_pae_matrix(array, expected_size=int(array.shape[0]))
    indices = np.asarray(mapping.pae_indices, dtype=int)
    if len(indices) == 0 or (indices < 0).any() or (indices >= len(validated)).any():
        raise PAEMappingError(
            "invalid_residue_mapping", "Mapped PAE index is outside the matrix"
        )
    if len(np.unique(indices)) != len(indices):
        raise PAEMappingError(
            "ambiguous_residue_mapping", "Mapped PAE indices must be unique"
        )
    return validated[np.ix_(indices, indices)]


def summarize_long_range_pae(
    mapping: PAEResidueMapping,
    mapped_pae: np.ndarray,
    *,
    min_sequence_separation: int,
) -> LongRangePAESummary:
    """Summarize unique symmetric mapped PAE pairs by UniProt separation."""
    if (
        isinstance(min_sequence_separation, bool)
        or not isinstance(min_sequence_separation, int)
        or min_sequence_separation < 0
    ):
        raise PAEMappingError(
            "invalid_sequence_separation",
            "Minimum sequence separation must be a non-negative integer",
        )
    values = validate_pae_matrix(mapped_pae, expected_size=len(mapping))
    symmetric = 0.5 * (values + values.T)
    positions = np.asarray(mapping.uniprot_positions, dtype=int)
    separations = np.abs(positions[:, None] - positions[None, :])
    mask = np.triu(np.ones(values.shape, dtype=bool), k=1)
    mask &= separations >= min_sequence_separation
    selected = symmetric[mask]
    if len(selected) == 0:
        return LongRangePAESummary(count=0, mean=None, median=None, maximum=None)
    return LongRangePAESummary(
        count=len(selected),
        mean=float(np.mean(selected)),
        median=float(np.median(selected)),
        maximum=float(np.max(selected)),
    )
