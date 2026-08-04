"""Strict exact-record fragment and AFDB artifact binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import AFDBFragment, DerivationError
from ..services.afdb import validate_pae_matrix
from ..stages.resolution import (
    P0ValidationError,
    _validate_artifact_identities,
    _validate_model_mmcif,
)
from .identity import exact_accession_records


def fragment_from_record(record: Mapping[str, Any]) -> AFDBFragment:
    try:
        start = int(record["sequenceStart"])
        end = int(record["sequenceEnd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DerivationError(
            "invalid_fragment_interval", "Prediction interval is invalid"
        ) from exc
    model = record.get("modelEntityId") or record.get("entryId")
    if not isinstance(model, str):
        raise DerivationError(
            "missing_model_identity", "Prediction record lacks model identity"
        )
    return AFDBFragment(model.strip(), start, end, end - start + 1)


def resolve_exact_fragment(
    records: Sequence[Mapping[str, Any]],
    accession: str,
    mapped_interval: tuple[int, int],
) -> dict[str, Any]:
    """Require exactly one fully covering exact-accession fragment."""
    exact = exact_accession_records(records, accession)
    fragments = [fragment_from_record(record) for record in exact]
    start, end = mapped_interval
    full = [
        fragment
        for fragment in fragments
        if fragment.uniprot_start <= start and fragment.uniprot_end >= end
    ]
    if len(full) == 1:
        status = "fragment_resolved"
        selected = full[0]
    elif not full:
        status = "no_full_covering_fragment"
        selected = None
    else:
        status = "ambiguous_full_covering_fragments"
        selected = None
    return {
        "fragment_resolution_status": status,
        "exact_accession_prediction_record_count": len(exact),
        "mapped_interval": [int(start), int(end)],
        "fragment_candidates": [
            {
                "model_entity_id": fragment.model_entity_id,
                "uniprot_start": fragment.uniprot_start,
                "uniprot_end": fragment.uniprot_end,
                "model_residue_count": fragment.model_residue_count,
            }
            for fragment in fragments
        ],
        "full_cover_count": len(full),
        "selected_model_entity_id": selected.model_entity_id if selected else None,
        "selected_fragment": selected,
    }


def validate_bound_arrays(
    fragment: AFDBFragment, pae: Any, confidence: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly bind PAE and confidence arrays without numerical rescue."""
    checked_pae = validate_pae_matrix(pae, expected_size=fragment.model_residue_count)
    checked_confidence = np.asarray(confidence)
    if (
        checked_confidence.ndim != 1
        or len(checked_confidence) != fragment.model_residue_count
    ):
        raise DerivationError(
            "confidence_fragment_length_mismatch", "confidence/model length mismatch"
        )
    if not np.issubdtype(checked_confidence.dtype, np.number):
        raise DerivationError("invalid_confidence", "Confidence values must be numeric")
    checked_confidence = checked_confidence.astype(float)
    if not np.isfinite(checked_confidence).all():
        raise DerivationError("invalid_confidence", "Confidence values must be finite")
    return checked_pae, checked_confidence


def validate_asset_model_binding(
    *,
    selected_model_id: str,
    asset_records: Mapping[str, Mapping[str, Any] | None],
) -> str:
    """Require all available acquisition records to bind one exact model."""
    required = ("afdb_structure", "afdb_pae", "afdb_confidence")
    present = [asset_records.get(name) is not None for name in required]
    if any(present) and not all(present):
        raise DerivationError(
            "incomplete_asset_model_binding", "asset/model identity binding is incomplete"
        )
    if all(present):
        identities = {
            str(asset_records[name].get("exact_record_model_identity", ""))
            for name in required
            if asset_records[name] is not None
        }
        if identities != {selected_model_id}:
            raise DerivationError(
                "asset_model_identity_mismatch",
                f"asset/model identity mismatch: expected {selected_model_id}, found {identities}",
            )
        return "exact_acquisition_record_binding"
    return "preexisting_canonical_path_plus_exact_metadata_url_binding"


def validate_frozen_model_artifacts(
    metadata: Mapping[str, Any],
    *,
    model_id: str,
    version: int,
    model_path: Path,
    expected_length: int,
) -> None:
    """Delegate AFDB URL and mmCIF validation to the frozen P0 implementation."""
    try:
        _validate_artifact_identities(metadata, model_id, version)
        _validate_model_mmcif(model_path, model_id, expected_length)
    except P0ValidationError as exc:
        raise DerivationError(exc.code, str(exc)) from exc
