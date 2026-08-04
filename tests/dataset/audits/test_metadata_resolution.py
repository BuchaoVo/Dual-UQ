from __future__ import annotations

import json
from pathlib import Path

from dual_uq.dataset.audits import metadata


def _load_module():
    return metadata


def _record(accession: str, model_id: str) -> dict[str, object]:
    return {
        "uniprotAccession": accession,
        "uniprotSequence": "AAAA",
        "modelEntityId": model_id,
        "entryId": model_id,
    }


def test_collection_audit_partitions_exact_and_sibling_records() -> None:
    module = _load_module()
    payload = json.dumps(
        [
            _record("P12345-3", "AF-P12345-3-F1"),
            _record("P12345", "AF-P12345-F1"),
            _record("P12345-2", "AF-P12345-2-F1"),
        ]
    ).encode()

    result = module.analyze_metadata_collection(payload, "P12345")

    assert result == {
        "record_count": 3,
        "exact_accession_record_count": 1,
        "nonexact_record_count": 2,
        "isoform_suffix_record_count": 2,
        "all_returned_accessions": ["P12345", "P12345-2", "P12345-3"],
        "exact_record_ids": ["AF-P12345-F1"],
        "nonexact_accessions": ["P12345-2", "P12345-3"],
        "resolution_status": "exact_record_resolved",
    }


def test_collection_audit_reports_zero_and_multiple_exact_records() -> None:
    module = _load_module()
    zero = json.dumps([_record("Q99999", "AF-Q99999-F1")]).encode()
    multiple = json.dumps(
        [
            _record("P12345", "AF-P12345-F1"),
            _record("P12345", "AF-P12345-F2"),
        ]
    ).encode()

    assert module.analyze_metadata_collection(zero, "P12345")[
        "resolution_status"
    ] == "exact_identity_mismatch"
    assert module.analyze_metadata_collection(multiple, "P12345")[
        "resolution_status"
    ] == "ambiguous_exact_record_set"


def test_resolution_report_writer_is_deterministic_and_clean(tmp_path: Path) -> None:
    module = _load_module()
    audit = {
        "schema_version": "dataset-a.afdb-metadata-record-resolution.v1",
        "summary": {"payload_count": 1},
        "records": [
            {
                "candidate_index": 1,
                "pair_id": "1abc_A__P12345",
                "candidate_accession": "P12345",
                "payload_provenance": "canonical_success",
                "record_count": 1,
                "exact_accession_record_count": 1,
                "nonexact_record_count": 0,
                "isoform_suffix_record_count": 0,
                "all_returned_accessions": ["P12345"],
                "exact_record_ids": ["AF-P12345-F1"],
                "nonexact_accessions": [],
                "resolution_status": "exact_record_resolved",
                "previous_acquisition_status": "downloaded_new",
                "updated_primary_failure_stage": None,
                "updated_primary_failure_code": None,
                "dependent_failure_count": 0,
            }
        ],
    }

    module.write_outputs(audit, tmp_path)

    json_path = tmp_path / "afdb_metadata_record_resolution_v1.json"
    tsv_path = tmp_path / "afdb_metadata_record_resolution_v1.tsv"
    md_path = tmp_path / "afdb_metadata_record_resolution_v1.md"
    assert json.loads(json_path.read_text()) == audit
    assert md_path.read_text().startswith("# AFDB Metadata Record Resolution v1")
    payload = tsv_path.read_bytes()
    assert b"\r" not in payload
    assert all(not line.endswith(b"\t") for line in payload.splitlines())
