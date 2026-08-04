"""Audit AFDB metadata collections using exact-accession record partitions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text

from ..stages import acquisition as _BATCH1

LEDGER_PATH = Path(
    "artifacts/dataset/reports/acquisition/batch1_acquisition_run_v1.json"
)
ACQ3_PATH = Path(
    "artifacts/dataset/audits/acquisition/acq1_rejected_metadata_forensics_v1.json"
)
OUTPUT_DIR = Path("artifacts/dataset/audits/acquisition")


class MetadataResolutionAuditError(RuntimeError):
    """The local metadata evidence does not satisfy the ACQ-4 audit contract."""


def _payload_records(data: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MetadataResolutionAuditError("AFDB metadata payload is not valid JSON") from exc
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        return [dict(row) for row in payload]
    raise MetadataResolutionAuditError("AFDB metadata collection schema is invalid")


def _model_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("modelEntityId") or record.get("entryId")
    return str(value).strip() if value else None


def analyze_metadata_collection(data: bytes, candidate_accession: str) -> dict[str, Any]:
    """Partition prediction records by literal exact candidate accession."""
    records = _payload_records(data)
    exact = [
        row
        for row in records
        if str(row.get("uniprotAccession", "")).strip() == candidate_accession
    ]
    nonexact = [row for row in records if row not in exact]
    all_accessions = sorted(
        {
            str(row.get("uniprotAccession", "")).strip()
            for row in records
            if row.get("uniprotAccession") is not None
        }
    )
    nonexact_accessions = sorted(
        {
            str(row.get("uniprotAccession", "")).strip()
            for row in nonexact
            if row.get("uniprotAccession") is not None
        }
    )
    if not exact:
        resolution = "exact_identity_mismatch"
    elif len(exact) == 1:
        resolution = "exact_record_resolved"
    else:
        resolution = "ambiguous_exact_record_set"
    return {
        "record_count": len(records),
        "exact_accession_record_count": len(exact),
        "nonexact_record_count": len(nonexact),
        "isoform_suffix_record_count": sum(
            accession.startswith(f"{candidate_accession}-")
            for accession in nonexact_accessions
        ),
        "all_returned_accessions": all_accessions,
        "exact_record_ids": sorted(
            model for model in (_model_id(row) for row in exact) if model is not None
        ),
        "nonexact_accessions": nonexact_accessions,
        "resolution_status": resolution,
    }


def build_audit(root: Path) -> dict[str, Any]:
    ledger = json.loads((root / LEDGER_PATH).read_text())
    acq3 = json.loads((root / ACQ3_PATH).read_text())
    records: list[dict[str, Any]] = []

    for row in ledger["records"]:
        if row.get("asset_type") != "afdb_metadata" or row.get("status") == "failed":
            continue
        accession = str(row["pair_id"]).split("__")[-1]
        analysis = analyze_metadata_collection(
            (root / row["local_path"]).read_bytes(), accession
        )
        records.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "polymer_entity_id": row["polymer_entity_id"],
                "pair_id": row["pair_id"],
                "candidate_accession": accession,
                "payload_provenance": "canonical_success",
                "payload_path": row["local_path"],
                "payload_SHA256": row["SHA256"],
                "transport_success": True,
                "payload_parse_success": True,
                "previous_acquisition_status": row["status"],
                "updated_primary_failure_stage": None,
                "updated_primary_failure_code": None,
                "dependent_failure_count": 0,
                "dependent_asset_status": "unchanged_from_acq1",
                "model_fragment_resolution_status": "not_evaluated_acq4",
                **analysis,
            }
        )

    for row in acq3["records"]:
        accession = str(row["expected_accession"])
        analysis = analyze_metadata_collection(
            (root / row["raw_rejected_payload_path"]).read_bytes(), accession
        )
        if analysis["resolution_status"] == "exact_record_resolved":
            primary_stage = None
            primary_code = None
        elif analysis["resolution_status"] == "exact_identity_mismatch":
            primary_stage = "metadata_identity_validation"
            primary_code = "exact_identity_mismatch"
        else:
            primary_stage = "metadata_identity_resolution"
            primary_code = "ambiguous_exact_record_set"
        records.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "polymer_entity_id": row["polymer_entity_id"],
                "pair_id": row["pair_id"],
                "candidate_accession": accession,
                "payload_provenance": "acq3_immutable_rejected_evidence",
                "payload_path": row["raw_rejected_payload_path"],
                "payload_SHA256": row["current_SHA256"],
                "transport_success": row["current_HTTP_status"] == 200,
                "payload_parse_success": True,
                "previous_acquisition_status": "failed_identity_mismatch",
                "updated_primary_failure_stage": primary_stage,
                "updated_primary_failure_code": primary_code,
                "dependent_failure_count": 3,
                "dependent_asset_status": "historically_not_acquired_after_metadata_rejection",
                "model_fragment_resolution_status": "not_evaluated_acq4",
                **analysis,
            }
        )

    records.sort(key=lambda row: int(row["candidate_index"]))
    if len(records) != 38:
        raise MetadataResolutionAuditError(
            f"Expected 38 AFDB metadata payloads, found {len(records)}"
        )
    resolution_counts = Counter(row["resolution_status"] for row in records)
    exact_count_distribution = Counter(
        int(row["exact_accession_record_count"]) for row in records
    )
    previous_success = [
        row for row in records if row["payload_provenance"] == "canonical_success"
    ]
    former_failures = [
        row
        for row in records
        if row["payload_provenance"] == "acq3_immutable_rejected_evidence"
    ]
    hidden_success_ambiguity = [
        row
        for row in previous_success
        if row["resolution_status"] != "exact_record_resolved"
    ]
    if hidden_success_ambiguity:
        raise MetadataResolutionAuditError(
            "Previously successful metadata contains hidden record-level ambiguity"
        )
    return {
        "schema_version": "dataset-a.afdb-metadata-record-resolution.v1",
        "contract": {
            "identity_partition": "R_exact = records where uniprotAccession == candidate accession",
            "zero_exact": "exact_identity_mismatch",
            "one_exact": "exact_record_resolved",
            "multiple_exact": "ambiguous_exact_record_set",
            "nonexact_record_policy": "provenance_only_never_selected",
            "fragment_resolution": "not_performed; deferred to mapped-interval D2 contract",
        },
        "source_bindings": {
            "acq1_ledger": {
                "path": LEDGER_PATH.as_posix(),
                "SHA256": _BATCH1.sha256_file(root / LEDGER_PATH),
            },
            "acq3_forensics": {
                "path": ACQ3_PATH.as_posix(),
                "SHA256": _BATCH1.sha256_file(root / ACQ3_PATH),
            },
            "repository_HEAD": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "summary": {
            "payload_count": len(records),
            "previous_success_payload_count": len(previous_success),
            "former_failure_payload_count": len(former_failures),
            "record_count_total": sum(int(row["record_count"]) for row in records),
            "exact_accession_record_count_distribution": {
                str(key): value for key, value in sorted(exact_count_distribution.items())
            },
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "previous_success_status_changes": 0,
            "former_failures_exact_record_resolved": sum(
                row["resolution_status"] == "exact_record_resolved"
                for row in former_failures
            ),
            "true_identity_mismatch_count": resolution_counts[
                "exact_identity_mismatch"
            ],
            "ambiguous_exact_record_set_count": resolution_counts[
                "ambiguous_exact_record_set"
            ],
            "historical_dependent_asset_missing_count": sum(
                int(row["dependent_failure_count"]) for row in former_failures
            ),
            "model_fragment_resolution_evaluated_count": 0,
        },
        "records": records,
    }


def _value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_outputs(audit: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "afdb_metadata_record_resolution_v1.json"
    tsv_path = output_dir / "afdb_metadata_record_resolution_v1.tsv"
    md_path = output_dir / "afdb_metadata_record_resolution_v1.md"
    atomic_write_text(json_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

    fields = [
        "candidate_index",
        "polymer_entity_id",
        "pair_id",
        "candidate_accession",
        "payload_provenance",
        "payload_path",
        "payload_SHA256",
        "transport_success",
        "payload_parse_success",
        "record_count",
        "exact_accession_record_count",
        "nonexact_record_count",
        "isoform_suffix_record_count",
        "all_returned_accessions",
        "exact_record_ids",
        "nonexact_accessions",
        "previous_acquisition_status",
        "updated_primary_failure_stage",
        "updated_primary_failure_code",
        "dependent_failure_count",
        "dependent_asset_status",
        "model_fragment_resolution_status",
        "resolution_status",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in audit["records"]:
        writer.writerow({field: _value(row.get(field)) for field in fields})
    atomic_write_text(tsv_path, stream.getvalue())

    summary = audit["summary"]
    record_count_total = summary.get(
        "record_count_total",
        sum(int(row["record_count"]) for row in audit["records"]),
    )
    exact_distribution = summary.get(
        "exact_accession_record_count_distribution",
        {
            str(key): value
            for key, value in sorted(
                Counter(
                    int(row["exact_accession_record_count"])
                    for row in audit["records"]
                ).items()
            )
        },
    )
    resolution_counts = summary.get(
        "resolution_status_counts",
        dict(
            sorted(
                Counter(row["resolution_status"] for row in audit["records"]).items()
            )
        ),
    )
    former = [
        row
        for row in audit["records"]
        if row["payload_provenance"] == "acq3_immutable_rejected_evidence"
    ]
    lines = [
        "# AFDB Metadata Record Resolution v1",
        "",
        "Status: local record-level identity audit; no fragment selection or candidate identity change.",
        "",
        "## Contract",
        "",
        "For candidate accession `U`, only records with literal `uniprotAccession == U` belong to `R_exact`. Zero, one and multiple exact records resolve respectively to `exact_identity_mismatch`, `exact_record_resolved` and `ambiguous_exact_record_set`. Nonexact sibling records are provenance only and are never selected.",
        "",
        "## Summary",
        "",
        f"- Payloads: {summary['payload_count']}",
        f"- Total prediction records: {record_count_total}",
        f"- Exact-record-count distribution: `{json.dumps(exact_distribution, sort_keys=True)}`",
        f"- Resolution status counts: `{json.dumps(resolution_counts, sort_keys=True)}`",
        f"- Previous-success status changes: {summary.get('previous_success_status_changes', 0)}",
        f"- Historical dependent missing assets: {summary.get('historical_dependent_asset_missing_count', 0)}",
        "",
        "## Five former failures",
        "",
        "| Index | Candidate | Returned accessions | Exact count | Sibling count | Exact record IDs | Resolution |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in former:
        lines.append(
            f"| {row['candidate_index']} | `{row['candidate_accession']}` | "
            f"`{','.join(row['all_returned_accessions'])}` | "
            f"{row['exact_accession_record_count']} | {row['isoform_suffix_record_count']} | "
            f"`{','.join(row['exact_record_ids'])}` | `{row['resolution_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Fragment boundary",
            "",
            "Identity resolution does not select a model or fragment. Mapped-interval D2 resolution remains responsible for requiring exactly one full-coverage exact-accession fragment. F1 fallback, first-record wins, nearest-fragment selection, stitching and offset inference remain forbidden.",
        ]
    )
    atomic_write_text(md_path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != _BATCH1.EXPECTED_COMMIT:
        raise MetadataResolutionAuditError("Repository HEAD differs from ACQ-1 binding")
    audit = build_audit(root)
    write_outputs(audit, root / OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
