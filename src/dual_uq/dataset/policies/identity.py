"""Exact-accession AFDB metadata identity and canonical sequence extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from dual_uq.core.hashing import sha256_bytes

from ..models import DerivationError


def metadata_records(
    payload: bytes | str | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize one AFDB metadata object or collection without choosing a record."""
    if isinstance(payload, bytes):
        try:
            value = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DerivationError("invalid_metadata", "AFDB metadata is not valid JSON") from exc
    elif isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DerivationError("invalid_metadata", "AFDB metadata is not valid JSON") from exc
    else:
        value = payload
    if isinstance(value, Mapping):
        records = [dict(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = [dict(record) for record in value if isinstance(record, Mapping)]
        if len(records) != len(value):
            raise DerivationError(
                "invalid_metadata", "AFDB metadata records must be objects"
            )
    else:
        raise DerivationError(
            "invalid_metadata", "AFDB metadata must be an object or list"
        )
    if not records:
        raise DerivationError("invalid_metadata", "AFDB metadata contains no records")
    return records


def exact_accession_records(
    records: Sequence[Mapping[str, Any]], accession: str
) -> list[dict[str, Any]]:
    """Return only literal exact-accession records; siblings remain nonselected."""
    return [
        dict(record)
        for record in records
        if str(record.get("uniprotAccession", "")).strip() == accession
    ]


def extract_prediction_record_sequence(
    payload: bytes | str | Sequence[Mapping[str, Any]], accession: str
) -> dict[str, Any]:
    """Extract prediction-record sequence provenance without calling it canonical."""
    records = metadata_records(payload)
    exact = exact_accession_records(records, accession)
    if len(exact) != 1:
        code = "exact_identity_mismatch" if not exact else "ambiguous_exact_record_set"
        raise DerivationError(
            code, f"Expected one exact record for {accession}; found {len(exact)}"
        )
    record = exact[0]
    source_field = "uniprotSequence" if record.get("uniprotSequence") else "sequence"
    sequence = record.get(source_field)
    if not isinstance(sequence, str) or not sequence:
        raise DerivationError(
            "missing_prediction_record_sequence",
            "Exact record lacks a prediction-record sequence",
        )
    if sequence != sequence.strip():
        raise DerivationError(
            "invalid_prediction_record_sequence",
            "Prediction-record sequence must not be trimmed",
        )
    model_id = record.get("modelEntityId") or record.get("entryId")
    if not isinstance(model_id, str) or not model_id.strip():
        raise DerivationError("missing_model_identity", "Exact record lacks model identity")
    start = record.get("sequenceStart")
    end = record.get("sequenceEnd")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end < start
    ):
        raise DerivationError(
            "invalid_prediction_interval",
            "Exact prediction record lacks a valid UniProt interval",
        )
    return {
        "accession": accession,
        "prediction_record_sequence": sequence,
        "prediction_sequence_source_field": source_field,
        "prediction_sequence_length": len(sequence),
        "prediction_sequence_sha256": sha256_bytes(sequence.encode()),
        "prediction_interval": [start, end],
        "model_entity_id": model_id.strip(),
        "prediction_record_count": len(exact),
        "metadata_record_count": len(records),
        "nonselected_sibling_record_count": len(records) - len(exact),
        "exact_record": record,
    }


def extract_canonical_sequence(
    payload: bytes | str | Sequence[Mapping[str, Any]], accession: str
) -> dict[str, Any]:
    """Extract canonical sequence only when the exact record proves full coverage."""
    prediction = extract_prediction_record_sequence(payload, accession)
    start, end = prediction["prediction_interval"]
    length = prediction["prediction_sequence_length"]
    if start != 1 or end != length:
        raise DerivationError(
            "missing_canonical_sequence_provenance",
            "Prediction-record fragment does not establish the canonical UniProt sequence",
        )
    return {
        **prediction,
        "sequence": prediction["prediction_record_sequence"],
        "sequence_source_field": prediction["prediction_sequence_source_field"],
        "sequence_length": length,
        "sequence_sha256": prediction["prediction_sequence_sha256"],
        "canonical_sequence_provenance": "full_span_exact_prediction_record",
    }
