"""Acquire the H6-authorized Dataset-A Batch-1 raw assets without fallback."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.core.atomic_io import atomic_write_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.policies.identity import (
    exact_accession_records,
    extract_canonical_sequence_from_source,
    metadata_records,
)
from dual_uq.dataset.services.afdb import PAEMappingError, validate_pae_matrix

EXPECTED_COMMIT = "bf81c19e085982aa90f28a4de98586586d0b2b4c"
PLAN_PATH = Path(
    "artifacts/dataset/reports/acquisition/batch1_acquisition_plan_v1.json"
)
INVENTORY_PATH = Path(
    "artifacts/dataset/reports/census/candidate_inventory_v1.json"
)
OUTPUT_DIR = Path("artifacts/dataset/reports/acquisition")

EXPECTED_BINDINGS = {
    "discovery_source_sha256": (
        "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436"
    ),
    "candidate_inventory_sha256": (
        "05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf"
    ),
    "batch1_plan_sha256": (
        "013ea4d4dd78a0269cf08afb7fe9294fb3ac1ba27bc131b875ff8ba43fec8f15"
    ),
}
EXPECTED_ASSET_COUNTS = {
    "pdb_mmcif": 10,
    "sifts": 10,
    "afdb_metadata": 38,
    "afdb_structure": 38,
    "afdb_pae": 38,
    "afdb_confidence": 38,
}
ASSET_ORDER = tuple(EXPECTED_ASSET_COUNTS)
PDB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz"
AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


class AcquisitionError(RuntimeError):
    """One structured acquisition or validation failure."""

    def __init__(
        self, code: str, message: str, *, http_status: int | None = None
    ) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(message)


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class AssetSpec:
    candidate_index: int
    polymer_entity_id: str
    pair_id: str
    asset_type: str
    source_url: str
    local_path: Path
    expected_identity: str
    metadata_record: Mapping[str, Any] | None = None
    display_path: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _one(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise AcquisitionError("malformed_payload", "mmCIF identity is ambiguous")
        value = value[0]
    return str(value).strip()


def _parse_mmcif(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        parsed = MMCIF2Dict(StringIO(text))
    except (UnicodeError, ValueError, OSError, KeyError) as exc:
        raise AcquisitionError("malformed_payload", "Payload is not parseable mmCIF") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise AcquisitionError("malformed_payload", "Payload is not parseable mmCIF")
    return parsed


def _metadata_record(data: bytes, accession: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("malformed_payload", "AFDB metadata is not valid JSON") from exc
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        records = payload
    else:
        raise AcquisitionError("malformed_payload", "AFDB metadata schema is invalid")
    if not records:
        raise AcquisitionError("malformed_payload", "AFDB metadata contains no records")
    exact = [
        dict(record)
        for record in records
        if str(record.get("uniprotAccession", "")).strip() == accession
    ]
    if not exact:
        raise AcquisitionError(
            "exact_identity_mismatch",
            "AFDB metadata contains no record for the exact manifest accession",
        )
    if len(exact) > 1:
        raise AcquisitionError(
            "ambiguous_exact_record_set",
            "AFDB metadata contains multiple records for the exact manifest accession",
        )
    record = exact[0]
    sequence = record.get("uniprotSequence") or record.get("sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise AcquisitionError(
            "malformed_payload", "Exact AFDB metadata record lacks a canonical sequence field"
        )
    model_id = record.get("modelEntityId") or record.get("entryId")
    if not isinstance(model_id, str) or not model_id.strip():
        raise AcquisitionError("malformed_payload", "Exact AFDB metadata record lacks model identity")
    if (
        record.get("entryId")
        and record.get("modelEntityId")
        and str(record["entryId"]).strip()
        != str(record["modelEntityId"]).strip()
    ):
        raise AcquisitionError(
            "model_identity_mismatch", "Exact AFDB metadata model identity fields disagree"
        )
    nonexact_accessions = sorted(
        {
            str(item.get("uniprotAccession", "")).strip()
            for item in records
            if str(item.get("uniprotAccession", "")).strip() != accession
        }
    )
    record.update(
        {
            "_metadata_record_count": len(records),
            "_prediction_record_count": len(exact),
            "_exact_accession_record_count": len(exact),
            "_nonexact_record_count": len(records) - len(exact),
            "_isoform_suffix_record_count": sum(
                str(item.get("uniprotAccession", "")).strip().startswith(
                    f"{accession}-"
                )
                for item in records
            ),
            "_nonexact_accessions": nonexact_accessions,
            "_identity_resolution_status": "exact_record_resolved",
        }
    )
    return record


def _json_payload(data: bytes, label: str) -> Any:
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("malformed_payload", f"{label} is not valid JSON") from exc
    if not isinstance(payload, (dict, list)) or not payload:
        raise AcquisitionError("malformed_payload", f"{label} schema is empty or invalid")
    return payload


def _metadata_length(metadata: Mapping[str, Any]) -> int:
    try:
        start = int(metadata["sequenceStart"])
        end = int(metadata["sequenceEnd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcquisitionError(
            "malformed_payload", "AFDB metadata fragment interval is invalid"
        ) from exc
    if start < 1 or end < start:
        raise AcquisitionError("malformed_payload", "AFDB metadata fragment interval is invalid")
    return end - start + 1


def validate_payload(
    asset_type: str,
    data: bytes,
    expected_identity: str,
    metadata_record: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate raw bytes without transforming or deriving scientific artifacts."""
    if not data:
        raise AcquisitionError("malformed_payload", "Downloaded payload is empty")
    if asset_type == "pdb_mmcif":
        parsed = _parse_mmcif(data)
        entry = parsed.get("_entry.id")
        if entry is None:
            raise AcquisitionError("malformed_payload", "PDB mmCIF lacks _entry.id")
        if _one(entry).upper() != expected_identity.upper():
            raise AcquisitionError("identity_mismatch", "PDB mmCIF identity mismatch")
        return None
    if asset_type == "sifts":
        try:
            root = ET.fromstring(gzip.decompress(data))
        except (gzip.BadGzipFile, EOFError, ET.ParseError, OSError) as exc:
            raise AcquisitionError("malformed_payload", "SIFTS payload is not parseable XML.gz") from exc
        observed = str(root.attrib.get("dbAccessionId", "")).strip()
        if observed.lower() != expected_identity.lower():
            raise AcquisitionError("identity_mismatch", "SIFTS PDB identity mismatch")
        return None
    if asset_type == "afdb_metadata":
        return _metadata_record(data, expected_identity)
    if asset_type == "afdb_metadata_collection":
        try:
            records = metadata_records(data)
            exact = exact_accession_records(records, expected_identity)
        except DerivationError as exc:
            raise AcquisitionError(exc.code, str(exc)) from exc
        if not exact:
            raise AcquisitionError(
                "exact_identity_mismatch",
                "AFDB metadata contains no exact frozen accession record",
            )
        model_ids: list[str] = []
        for record in exact:
            model_id = record.get("modelEntityId") or record.get("entryId")
            start = record.get("sequenceStart")
            end = record.get("sequenceEnd")
            if (
                not isinstance(model_id, str)
                or not model_id.strip()
                or isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
            ):
                raise AcquisitionError(
                    "malformed_payload",
                    "Exact AFDB metadata record lacks model identity or interval",
                )
            model_ids.append(model_id.strip())
        return {
            "metadata_record_count": len(records),
            "exact_accession_record_count": len(exact),
            "nonexact_record_count": len(records) - len(exact),
            "exact_model_identifiers": sorted(model_ids),
            "source_metadata_versions": sorted(
                {
                    int(record["latestVersion"])
                    for record in exact
                    if isinstance(record.get("latestVersion"), int)
                    and not isinstance(record.get("latestVersion"), bool)
                }
            ),
            "identity_resolution_status": "exact_accession_collection_resolved",
        }
    if asset_type == "uniprot_canonical":
        try:
            return extract_canonical_sequence_from_source(data, expected_identity)
        except DerivationError as exc:
            raise AcquisitionError(exc.code, str(exc)) from exc
    if asset_type == "afdb_structure":
        parsed = _parse_mmcif(data)
        entry = parsed.get("_entry.id")
        if entry is None:
            raise AcquisitionError("malformed_payload", "AFDB mmCIF lacks _entry.id")
        if _one(entry) != expected_identity:
            raise AcquisitionError(
                "model_identity_mismatch", "AFDB structure model identity mismatch"
            )
        return None
    if asset_type == "afdb_pae":
        payload = _json_payload(data, "AFDB PAE")
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
            or "predicted_aligned_error" not in payload[0]
        ):
            raise AcquisitionError("malformed_payload", "AFDB PAE schema is invalid")
        if metadata_record is None:
            raise AcquisitionError("metadata_asset_missing", "PAE validation requires metadata")
        try:
            validate_pae_matrix(
                payload[0]["predicted_aligned_error"],
                expected_size=_metadata_length(metadata_record),
            )
        except PAEMappingError as exc:
            raise AcquisitionError("malformed_payload", str(exc)) from exc
        return None
    if asset_type == "afdb_confidence":
        payload = _json_payload(data, "AFDB confidence")
        records = payload if isinstance(payload, list) else [payload]
        keys = {"confidenceScore", "plddt", "confidence", "scores"}
        if not any(isinstance(row, dict) and keys.intersection(row) for row in records):
            raise AcquisitionError("malformed_payload", "AFDB confidence schema is invalid")
        return None
    raise AcquisitionError("non_manifest_asset", f"Unsupported asset type: {asset_type}")


def default_transport(url: str, timeout: float) -> TransportResponse:
    request = urllib.request.Request(url, headers={"User-Agent": "Dual-UQ-ACQ-1/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return TransportResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return TransportResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(),
        )


def _ledger_base(spec: AssetSpec, timestamp: str) -> dict[str, Any]:
    return {
        "candidate_index": spec.candidate_index,
        "polymer_entity_id": spec.polymer_entity_id,
        "pair_id": spec.pair_id,
        "asset_type": spec.asset_type,
        "source_url": spec.source_url,
        "HTTP_status": None,
        "byte_count": None,
        "SHA256": None,
        "local_path": spec.display_path or spec.local_path.as_posix(),
        "timestamp": timestamp,
        "status": "failed",
        "attempt_count": 0,
        "failure_code": None,
        "response_headers": None,
        "rejected_payload_path": None,
        "rejected_payload_SHA256": None,
        "rejected_evidence_status": None,
        "parsed_returned_accessions": None,
        "parsed_sequence_field_presence": None,
        "parsed_model_identifiers": None,
        "metadata_record_count": None,
        "exact_accession_record_count": None,
        "nonexact_record_count": None,
        "isoform_suffix_record_count": None,
        "identity_resolution_status": None,
        "source_metadata_version_if_available": None,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    try:
        atomic_write_bytes(path, data)
    except OSError as exc:
        raise AcquisitionError("atomic_write_failure", f"Atomic write failed: {path}") from exc


def preserve_rejected_payload(
    data: bytes,
    *,
    rejected_dir: Path,
    candidate_index: int,
    asset_type: str,
) -> dict[str, str]:
    """Preserve validation-failed bytes at an immutable content-addressed path."""
    digest = sha256_bytes(data)
    suffix = ".json" if asset_type == "afdb_metadata" else ".payload"
    path = rejected_dir / str(candidate_index) / asset_type / f"{digest}{suffix}"
    if path.exists():
        existing = path.read_bytes()
        if sha256_bytes(existing) != digest or existing != data:
            raise AcquisitionError(
                "existing_file_conflict",
                f"Rejected evidence conflicts with content-addressed path: {path}",
            )
        status = "reused_immutable"
    else:
        _atomic_write(path, data)
        status = "preserved_new"
    return {"path": path.as_posix(), "SHA256": digest, "evidence_status": status}


def inspect_afdb_metadata_evidence(
    data: bytes, expected_accession: str | None = None
) -> dict[str, Any]:
    """Extract identity evidence without deciding or repairing accession relations."""
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError):
        return {
            "parsed_returned_accessions": None,
            "parsed_sequence_field_presence": "unparseable",
            "parsed_model_identifiers": None,
            "metadata_record_count": None,
            "exact_accession_record_count": None,
            "nonexact_record_count": None,
            "isoform_suffix_record_count": None,
            "identity_resolution_status": "payload_unparseable",
        }
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        records = payload
    else:
        return {
            "parsed_returned_accessions": None,
            "parsed_sequence_field_presence": "schema_unrecognized",
            "parsed_model_identifiers": None,
            "metadata_record_count": None,
            "exact_accession_record_count": None,
            "nonexact_record_count": None,
            "isoform_suffix_record_count": None,
            "identity_resolution_status": "payload_schema_unrecognized",
        }
    accessions = sorted(
        {
            str(row["uniprotAccession"]).strip()
            for row in records
            if row.get("uniprotAccession") is not None
        }
    )
    sequence_presence = [
        isinstance(row.get("uniprotSequence") or row.get("sequence"), str)
        and bool(str(row.get("uniprotSequence") or row.get("sequence")).strip())
        for row in records
    ]
    models = sorted(
        {
            str(row.get("modelEntityId") or row.get("entryId")).strip()
            for row in records
            if row.get("modelEntityId") or row.get("entryId")
        }
    )
    exact_count = (
        sum(
            str(row.get("uniprotAccession", "")).strip() == expected_accession
            for row in records
        )
        if expected_accession is not None
        else None
    )
    if exact_count is None:
        resolution = None
    elif exact_count == 0:
        resolution = "exact_identity_mismatch"
    elif exact_count == 1:
        resolution = "exact_record_resolved"
    else:
        resolution = "ambiguous_exact_record_set"
    return {
        "parsed_returned_accessions": accessions,
        "parsed_sequence_field_presence": sequence_presence,
        "parsed_model_identifiers": models,
        "metadata_record_count": len(records),
        "exact_accession_record_count": exact_count,
        "nonexact_record_count": (
            len(records) - exact_count if exact_count is not None else None
        ),
        "isoform_suffix_record_count": (
            sum(
                str(row.get("uniprotAccession", "")).strip().startswith(
                    f"{expected_accession}-"
                )
                for row in records
            )
            if expected_accession is not None
            else None
        ),
        "identity_resolution_status": resolution,
    }


def interpret_refetch_outcome(
    *, historical_sha: str, current_sha: str, current_validation: str
) -> dict[str, Any]:
    """Interpret current evidence without converting a historical failure to success."""
    same_sha = historical_sha == current_sha
    if same_sha and current_validation == "exact_match":
        raise AcquisitionError(
            "validation_implementation_inconsistency",
            "The historical rejected payload now passes exact identity validation",
        )
    if not same_sha:
        outcome = (
            "source_payload_drift"
            if current_validation == "exact_match"
            else "source_payload_drift_with_identity_mismatch"
        )
    else:
        outcome = "same_payload_still_mismatch"
    return {
        "sha_relation": "same" if same_sha else "different",
        "outcome": outcome,
        "historical_failure_recovered": False,
    }


def _retry_delay(response: TransportResponse, attempt: int) -> float:
    retry_after = next(
        (value for key, value in response.headers.items() if key.lower() == "retry-after"),
        None,
    )
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), 60.0))
        except ValueError:
            pass
    return min(float(2 ** (attempt - 1)), 8.0)


def acquire_asset(
    spec: AssetSpec,
    *,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    timeout: float = 60.0,
    rejected_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate/reuse or atomically download one explicitly declared raw asset."""
    row = _ledger_base(spec, clock())
    if spec.local_path.exists():
        try:
            data = spec.local_path.read_bytes()
            validation_result = validate_payload(
                spec.asset_type, data, spec.expected_identity, spec.metadata_record
            )
        except (OSError, AcquisitionError):
            try:
                existing = spec.local_path.read_bytes()
            except OSError:
                existing = b""
            row.update(
                byte_count=len(existing),
                SHA256=sha256_bytes(existing) if existing else None,
                failure_code="existing_file_conflict",
            )
            return row
        row.update(
            byte_count=len(data),
            SHA256=sha256_bytes(data),
            status="reused_valid",
        )
        if validation_result and validation_result.get("source_metadata_versions"):
            row["source_metadata_version_if_available"] = json.dumps(
                validation_result["source_metadata_versions"], separators=(",", ":")
            )
        return row

    last_code = "transport_failure"
    last_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        row["attempt_count"] = attempt
        try:
            response = transport(spec.source_url, timeout)
        except (TimeoutError, ConnectionResetError, urllib.error.URLError, OSError):
            if attempt < max_attempts:
                sleeper(min(float(2 ** (attempt - 1)), 8.0))
                continue
            row["failure_code"] = "transport_failure"
            return row
        last_status = response.status
        row["HTTP_status"] = response.status
        if response.status == 404:
            row["failure_code"] = "http_not_found"
            return row
        if response.status == 429 or 500 <= response.status <= 599:
            last_code = (
                "rate_limited_exhausted" if response.status == 429 else "transport_failure"
            )
            if attempt < max_attempts:
                sleeper(_retry_delay(response, attempt))
                continue
            row["failure_code"] = last_code
            return row
        if response.status < 200 or response.status >= 300:
            row["failure_code"] = "transport_failure"
            return row
        try:
            validation_result = validate_payload(
                spec.asset_type,
                response.body,
                spec.expected_identity,
                spec.metadata_record,
            )
        except AcquisitionError as exc:
            evidence: dict[str, str] | None = None
            parsed: dict[str, Any] = {}
            if rejected_evidence_dir is not None:
                evidence = preserve_rejected_payload(
                    response.body,
                    rejected_dir=rejected_evidence_dir,
                    candidate_index=spec.candidate_index,
                    asset_type=spec.asset_type,
                )
                if spec.asset_type == "afdb_metadata":
                    parsed = inspect_afdb_metadata_evidence(
                        response.body, spec.expected_identity
                    )
            row.update(
                HTTP_status=response.status,
                byte_count=len(response.body),
                SHA256=sha256_bytes(response.body),
                failure_code=exc.code,
                response_headers=dict(sorted(response.headers.items())),
                rejected_payload_path=evidence["path"] if evidence else None,
                rejected_payload_SHA256=evidence["SHA256"] if evidence else None,
                rejected_evidence_status=(
                    evidence["evidence_status"] if evidence else None
                ),
                **parsed,
            )
            return row
        try:
            _atomic_write(spec.local_path, response.body)
        except AcquisitionError as exc:
            row.update(
                HTTP_status=response.status,
                byte_count=len(response.body),
                SHA256=sha256_bytes(response.body),
                failure_code=exc.code,
                response_headers=dict(sorted(response.headers.items())),
            )
            return row
        row.update(
            HTTP_status=response.status,
            byte_count=len(response.body),
            SHA256=sha256_bytes(response.body),
            status="downloaded_new",
            failure_code=None,
            response_headers=dict(sorted(response.headers.items())),
        )
        if validation_result and validation_result.get("source_metadata_versions"):
            row["source_metadata_version_if_available"] = json.dumps(
                validation_result["source_metadata_versions"], separators=(",", ":")
            )
        if spec.asset_type == "afdb_metadata" and validation_result is not None:
            row.update(
                metadata_record_count=validation_result.get("_metadata_record_count"),
                exact_accession_record_count=validation_result.get(
                    "_exact_accession_record_count"
                ),
                nonexact_record_count=validation_result.get("_nonexact_record_count"),
                isoform_suffix_record_count=validation_result.get(
                    "_isoform_suffix_record_count"
                ),
                identity_resolution_status=validation_result.get(
                    "_identity_resolution_status"
                ),
            )
        return row
    row.update(HTTP_status=last_status, failure_code=last_code)
    return row


def select_manifest_records(
    plan: Mapping[str, Any], requested_indices: Iterable[int] | None = None
) -> list[dict[str, Any]]:
    records = [dict(row) for row in plan.get("records", [])]
    index_map = {int(row["candidate_index"]): row for row in records}
    if len(index_map) != len(records):
        raise AcquisitionError("manifest_scope_error", "Manifest candidate indices are not unique")
    if requested_indices is None:
        return records
    requested = list(requested_indices)
    missing = [value for value in requested if value not in index_map]
    if missing:
        raise AcquisitionError(
            "non_manifest_candidate", f"Candidate {missing[0]} is not present in manifest"
        )
    return [index_map[value] for value in requested]


def _asset_path(root: Path, row: Mapping[str, Any], asset_type: str) -> tuple[Path, str]:
    pdb_id = str(row["PDB"]).lower()
    accession = str(row["UniProt"])
    if asset_type == "pdb_mmcif":
        relative = Path(f"data/raw/pdb/{pdb_id}.cif")
    elif asset_type == "sifts":
        relative = Path(f"data/raw/mappings/{pdb_id}.xml.gz")
    elif asset_type == "afdb_metadata":
        relative = Path(f"data/raw/afdb/{accession}/metadata.json")
    else:
        raise AcquisitionError("non_manifest_asset", f"Invalid initial asset: {asset_type}")
    return root / relative, relative.as_posix()


def build_initial_asset_specs(
    plan: Mapping[str, Any], project_root: Path
) -> list[AssetSpec]:
    specs: list[AssetSpec] = []
    for row in select_manifest_records(plan):
        required = list(row.get("required_external_assets", []))
        unknown = set(required) - set(ASSET_ORDER)
        if unknown:
            raise AcquisitionError(
                "non_manifest_asset", f"Manifest contains unsupported assets: {sorted(unknown)}"
            )
        for asset_type in ASSET_ORDER[:3]:
            if asset_type not in required:
                continue
            path, display = _asset_path(project_root, row, asset_type)
            if asset_type == "pdb_mmcif":
                url = PDB_MMCIF_URL.format(pdb_id=str(row["PDB"]).lower())
                identity = str(row["PDB"])
            elif asset_type == "sifts":
                url = SIFTS_URL.format(pdb_id=str(row["PDB"]).lower())
                identity = str(row["PDB"])
            else:
                url = AFDB_PREDICTION_API.format(accession=row["UniProt"])
                identity = str(row["UniProt"])
            specs.append(
                AssetSpec(
                    candidate_index=int(row["candidate_index"]),
                    polymer_entity_id=str(row["polymer_entity_id"]),
                    pair_id=str(row["pair_id"]),
                    asset_type=asset_type,
                    source_url=url,
                    local_path=path,
                    expected_identity=identity,
                    display_path=display,
                )
            )
    return specs


def _valid_metadata_url(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionError("metadata_asset_missing", f"AFDB metadata lacks {field}")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AcquisitionError("metadata_asset_missing", f"AFDB metadata {field} is invalid")
    return value


def _dependent_spec(
    root: Path,
    row: Mapping[str, Any],
    asset_type: str,
    metadata: Mapping[str, Any],
) -> AssetSpec:
    accession = str(row["UniProt"])
    if int(metadata.get("_prediction_record_count", 1)) != 1:
        raise AcquisitionError(
            "metadata_asset_missing",
            "Multiple AFDB prediction records require fragment resolution, which is out of scope",
        )
    model_id = str(metadata.get("modelEntityId") or metadata.get("entryId") or "").strip()
    if not model_id:
        raise AcquisitionError("model_identity_mismatch", "AFDB metadata lacks model identity")
    fields = {
        "afdb_structure": ("cifUrl", "model.cif"),
        "afdb_pae": ("paeDocUrl", "pae.json"),
        "afdb_confidence": ("plddtDocUrl", "plddt.json"),
    }
    field, filename = fields[asset_type]
    url = _valid_metadata_url(metadata, field)
    relative = Path(f"data/raw/afdb/{accession}/{model_id}/{filename}")
    return AssetSpec(
        candidate_index=int(row["candidate_index"]),
        polymer_entity_id=str(row["polymer_entity_id"]),
        pair_id=str(row["pair_id"]),
        asset_type=asset_type,
        source_url=url,
        local_path=root / relative,
        expected_identity=model_id,
        metadata_record=metadata,
        display_path=relative.as_posix(),
    )


def _failure_without_request(
    row: Mapping[str, Any], asset_type: str, code: str, timestamp: str
) -> dict[str, Any]:
    accession = str(row["UniProt"])
    return {
        "candidate_index": int(row["candidate_index"]),
        "polymer_entity_id": str(row["polymer_entity_id"]),
        "pair_id": str(row["pair_id"]),
        "asset_type": asset_type,
        "source_url": None,
        "HTTP_status": None,
        "byte_count": None,
        "SHA256": None,
        "local_path": f"data/raw/afdb/{accession}/<metadata-bound>/{asset_type}",
        "timestamp": timestamp,
        "status": "failed",
        "failure_code": code,
    }


def _candidate_completeness(
    plan: Mapping[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_candidate[int(record["candidate_index"])].append(record)
    output = []
    for row in plan["records"]:
        index = int(row["candidate_index"])
        planned = len(row["required_external_assets"])
        rows = by_candidate[index]
        successes = sum(item["status"] in {"reused_valid", "downloaded_new"} for item in rows)
        failures = sum(item["status"] == "failed" for item in rows)
        if planned == 0 or successes == planned:
            completeness = "fully_raw_complete"
        elif successes == 0:
            completeness = "zero_success"
        else:
            completeness = "partially_complete"
        output.append(
            {
                "candidate_index": index,
                "polymer_entity_id": row["polymer_entity_id"],
                "pair_id": row["pair_id"],
                "planned_asset_count": planned,
                "successful_asset_count": successes,
                "failed_asset_count": failures,
                "completeness": completeness,
            }
        )
    return output


def _summary(
    plan: Mapping[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix: dict[str, dict[str, int]] = {}
    for asset_type in ASSET_ORDER:
        subset = [row for row in records if row["asset_type"] == asset_type]
        statuses = Counter(row["status"] for row in subset)
        matrix[asset_type] = {
            "planned": len(subset),
            "reused": statuses["reused_valid"],
            "downloaded": statuses["downloaded_new"],
            "failed": statuses["failed"],
        }
    completeness = _candidate_completeness(plan, records)
    complete_counts = Counter(row["completeness"] for row in completeness)
    return (
        {
            "planned_assets": len(records),
            "reused_assets": sum(row["status"] == "reused_valid" for row in records),
            "new_downloads": sum(row["status"] == "downloaded_new" for row in records),
            "failed_assets": sum(row["status"] == "failed" for row in records),
            "failure_code_counts": dict(
                sorted(Counter(row["failure_code"] for row in records if row["failure_code"]).items())
            ),
            "fully_raw_complete_candidates": complete_counts["fully_raw_complete"],
            "partially_complete_candidates": complete_counts["partially_complete"],
            "zero_success_candidates": complete_counts["zero_success"],
            "asset_matrix": matrix,
        },
        completeness,
    )


def run_plan(
    plan: Mapping[str, Any],
    project_root: Path,
    *,
    transport: Callable[[str, float], TransportResponse] = default_transport,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], str] = utc_now,
    max_attempts: int = 3,
    rejected_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Run sequential acquisition while continuing after independent failures."""
    root = project_root.resolve()
    if rejected_evidence_dir is None:
        rejected_evidence_dir = root / "artifacts/dataset/audits/acquisition_rejected"
    records: list[dict[str, Any]] = []
    initial = build_initial_asset_specs(plan, root)
    specs_by_candidate: dict[int, list[AssetSpec]] = defaultdict(list)
    for spec in initial:
        specs_by_candidate[spec.candidate_index].append(spec)
    for row in select_manifest_records(plan):
        index = int(row["candidate_index"])
        metadata_record: dict[str, Any] | None = None
        for spec in specs_by_candidate[index]:
            result = acquire_asset(
                spec,
                transport=transport,
                sleeper=sleeper,
                clock=clock,
                max_attempts=max_attempts,
                rejected_evidence_dir=rejected_evidence_dir,
            )
            records.append(result)
            if spec.asset_type == "afdb_metadata" and result["status"] != "failed":
                try:
                    validated = validate_payload(
                        "afdb_metadata",
                        spec.local_path.read_bytes(),
                        str(row["UniProt"]),
                        None,
                    )
                    metadata_record = dict(validated or {})
                except (OSError, AcquisitionError):
                    metadata_record = None
        for asset_type in ("afdb_structure", "afdb_pae", "afdb_confidence"):
            if asset_type not in row.get("required_external_assets", []):
                continue
            if metadata_record is None:
                records.append(
                    _failure_without_request(row, asset_type, "metadata_asset_missing", clock())
                )
                continue
            try:
                spec = _dependent_spec(root, row, asset_type, metadata_record)
            except AcquisitionError as exc:
                records.append(_failure_without_request(row, asset_type, exc.code, clock()))
                continue
            records.append(
                acquire_asset(
                    spec,
                    transport=transport,
                    sleeper=sleeper,
                    clock=clock,
                    max_attempts=max_attempts,
                    rejected_evidence_dir=rejected_evidence_dir,
                )
            )
    summary, completeness = _summary(plan, records)
    return {
        "schema_version": "dataset-a.batch1-acquisition-run.v1",
        "run_status": "complete_with_failures" if summary["failed_assets"] else "complete",
        "acquisition_plan_commit": EXPECTED_COMMIT,
        "input_bindings": dict(plan.get("input_bindings", {})),
        "records": records,
        "candidate_completeness": completeness,
        "summary": summary,
    }


def _git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def verify_repository_bindings(project_root: Path) -> dict[str, Any]:
    """Stop before network unless the committed plan and its sources match exactly."""
    root = project_root.resolve()
    plan_file = root / PLAN_PATH
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if _git_output(root, "rev-parse", "HEAD").decode().strip() != EXPECTED_COMMIT:
        raise AcquisitionError("input_drift", "Current HEAD is not the authorized plan commit")
    committed = _git_output(root, "show", f"{EXPECTED_COMMIT}:{PLAN_PATH.as_posix()}")
    if committed != plan_file.read_bytes():
        raise AcquisitionError("input_drift", "Acquisition plan differs from authorized commit")
    bindings = dict(plan.get("input_bindings", {}))
    for key, expected in EXPECTED_BINDINGS.items():
        if bindings.get(key) != expected:
            raise AcquisitionError("input_drift", f"Manifest binding differs: {key}")
    bound_paths = {
        "discovery_source_sha256": bindings["discovery_source_path"],
        "candidate_inventory_sha256": bindings["candidate_inventory_path"],
        "batch1_plan_sha256": bindings["batch1_plan_path"],
    }
    for key, relative in bound_paths.items():
        if sha256_file(root / relative) != EXPECTED_BINDINGS[key]:
            raise AcquisitionError("input_drift", f"Bound file digest differs: {relative}")
    if len(plan.get("records", [])) != 48:
        raise AcquisitionError("input_drift", "Authorized candidate count is not 48")
    if any(bool(row.get("round1_member")) for row in plan["records"]):
        raise AcquisitionError("input_drift", "Authorized panel overlaps Round-1")
    counts = Counter(
        asset
        for row in plan["records"]
        for asset in row.get("required_external_assets", [])
    )
    if dict(counts) != EXPECTED_ASSET_COUNTS:
        raise AcquisitionError("input_drift", "Authorized asset counts differ")
    return plan


def _failure_concentration(
    ledger: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = {int(row["candidate_index"]): row for row in inventory["batch1"]}
    failed = {
        int(row["candidate_index"])
        for row in ledger["records"]
        if row["status"] == "failed"
    }

    def length_bucket(value: Any) -> str:
        if value is None:
            return "unknown"
        length = int(value)
        if length < 150:
            return "lt150"
        if length < 300:
            return "150_299"
        if length < 600:
            return "300_599"
        return "ge600"

    dimensions: dict[str, dict[str, dict[str, int | float]]] = {}
    groupers = {
        "protein_length": lambda row: length_bucket(row.get("canonical_uniprot_length")),
        "afdb_fragment_status": lambda row: (
            "unknown"
            if row.get("fragment_count") is None
            else "multi"
            if int(row["fragment_count"]) > 1
            else "single"
        ),
        "source_side_availability": lambda row: str(row.get("estimated_P0_readiness", "unknown")),
    }
    for dimension, grouper in groupers.items():
        totals = Counter(grouper(row) for row in candidates.values())
        failures = Counter(grouper(candidates[index]) for index in failed)
        dimensions[dimension] = {
            group: {
                "candidate_count": total,
                "candidate_with_failure_count": failures[group],
                "failure_fraction": failures[group] / total if total else 0.0,
            }
            for group, total in sorted(totals.items())
        }
    return {
        "analysis_type": "descriptive_only_no_biological_inference",
        "candidate_with_any_failure_count": len(failed),
        "dimensions": dimensions,
    }


def _atomic_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def write_reports(ledger: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch1_acquisition_run_v1.json"
    tsv_path = output_dir / "batch1_acquisition_run_v1.tsv"
    md_path = output_dir / "batch1_acquisition_run_v1.md"
    _atomic_text(json_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    fields = [
        "candidate_index",
        "polymer_entity_id",
        "pair_id",
        "asset_type",
        "source_url",
        "HTTP_status",
        "byte_count",
        "SHA256",
        "local_path",
        "timestamp",
        "metadata_record_count",
        "exact_accession_record_count",
        "nonexact_record_count",
        "isoform_suffix_record_count",
        "identity_resolution_status",
        "failure_code",
        "status",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(ledger["records"])
    _atomic_text(tsv_path, stream.getvalue())

    summary = ledger["summary"]
    lines = [
        "# Dataset-A Batch-1 Raw Acquisition v1",
        "",
        f"- Run status: `{ledger['run_status']}`",
        f"- Acquisition plan commit: `{ledger['acquisition_plan_commit']}`",
        f"- Planned assets: {summary['planned_assets']}",
        f"- Reused valid: {summary['reused_assets']}",
        f"- Downloaded new: {summary['new_downloads']}",
        f"- Failed: {summary['failed_assets']}",
        f"- Fully raw-complete candidates: {summary['fully_raw_complete_candidates']}",
        f"- Partially complete candidates: {summary['partially_complete_candidates']}",
        f"- Zero-success candidates: {summary['zero_success_candidates']}",
        "",
        "## Asset matrix",
        "",
        "| Asset type | Planned | Reused | Downloaded | Failed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for asset_type in ASSET_ORDER:
        row = summary["asset_matrix"][asset_type]
        lines.append(
            f"| `{asset_type}` | {row['planned']} | {row['reused']} | "
            f"{row['downloaded']} | {row['failed']} |"
        )
    lines.extend(["", "## Failure codes", ""])
    if summary["failure_code_counts"]:
        lines.extend(
            f"- `{code}`: {count}"
            for code, count in summary["failure_code_counts"].items()
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Missingness concentration",
            "",
            "Descriptive only; no biological inference.",
            "",
            "```json",
            json.dumps(ledger.get("failure_concentration", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    _atomic_text(md_path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ProjectPaths.discover(anchor=Path(__file__)).repository_root,
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_repository_bindings(args.project_root)
    ledger = run_plan(plan, args.project_root, max_attempts=args.max_attempts)
    inventory = json.loads((args.project_root / INVENTORY_PATH).read_text(encoding="utf-8"))
    ledger["failure_concentration"] = _failure_concentration(ledger, inventory)
    write_reports(ledger, args.project_root / OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
