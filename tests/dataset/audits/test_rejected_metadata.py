from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dual_uq.dataset.audits import rejected_metadata


def _load_module():
    return rejected_metadata


def _historical_row(index: int, accession: str) -> dict[str, object]:
    return {
        "candidate_index": index,
        "polymer_entity_id": f"PDB{index}_1",
        "pair_id": f"pdb{index}_A__{accession}",
        "asset_type": "afdb_metadata",
        "source_url": f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
        "HTTP_status": 200,
        "byte_count": 10,
        "SHA256": str(index) * 64,
        "timestamp": "historical",
        "status": "failed",
        "failure_code": "identity_mismatch",
    }


def test_selects_only_exact_five_historical_urls() -> None:
    module = _load_module()
    pairs = [(24, "Q8N8S7"), (7, "Q13427"), (38, "Q9Y237"), (103, "Q8K4V4"), (208, "O14786")]
    ledger = {"records": [_historical_row(index, accession) for index, accession in pairs]}

    rows = module.select_historical_failures(ledger)

    assert [row["candidate_index"] for row in rows] == [24, 7, 38, 103, 208]
    assert all(
        row["source_url"].endswith(str(row["pair_id"]).split("__")[-1])
        for row in rows
    )


def test_rejects_scope_or_url_drift() -> None:
    module = _load_module()
    pairs = [(24, "Q8N8S7"), (7, "Q13427"), (38, "Q9Y237"), (103, "Q8K4V4"), (208, "O14786")]
    records = [_historical_row(index, accession) for index, accession in pairs]
    records[0]["source_url"] = "https://example.test/alternate"

    with pytest.raises(module.ForensicScopeError, match="URL"):
        module.select_historical_failures({"records": records})


@pytest.mark.parametrize(
    ("expected", "observed", "classification"),
    [
        ("P12345", ["P12345-1"], "possible_isoform_relation"),
        ("P12345", ["P12345", "P12345-2"], "possible_isoform_relation"),
        ("P12345", ["P12345", "Q99999"], "metadata_schema_or_record_anomaly"),
        ("P12345", ["Q99999"], "cross_source_accession_conflict"),
        ("P12345", None, "insufficient_evidence"),
    ],
)
def test_identity_classification_uses_only_explicit_returned_accessions(
    expected: str, observed: list[str] | None, classification: str
) -> None:
    module = _load_module()

    assert module.classify_returned_identity(expected, observed) == classification


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.urls.append(url)
        return self.responses.pop(0)


def _metadata(accession: str) -> bytes:
    return json.dumps(
        [
            {
                "uniprotAccession": accession,
                "uniprotSequence": "AAAA",
                "modelEntityId": f"AF-{accession}-F1",
                "entryId": f"AF-{accession}-F1",
            }
        ]
    ).encode()


def test_refetch_requests_only_five_urls_and_preserves_same_sha_mismatches(
    tmp_path: Path,
) -> None:
    module = _load_module()
    pairs = [(24, "Q8N8S7"), (7, "Q13427"), (38, "Q9Y237"), (103, "Q8K4V4"), (208, "O14786")]
    records = []
    responses = []
    for index, accession in pairs:
        body = _metadata(f"{accession}-1")
        row = _historical_row(index, accession)
        row["byte_count"] = len(body)
        row["SHA256"] = hashlib.sha256(body).hexdigest()
        records.append(row)
        responses.append(
            module.TransportResponse(
                200,
                {"ETag": f'"{accession}"'},
                body,
            )
        )
    transport = FakeTransport(responses)

    audit = module.run_forensic_refetch(
        {"records": records},
        project_root=tmp_path,
        run_dir=tmp_path / "runs/acq3",
        rejected_dir=tmp_path / "artifacts/dataset/audits/acquisition_rejected",
        transport=transport,
        clock=lambda: "current",
        sleeper=lambda _: None,
    )

    assert transport.urls == [
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
        for _, accession in pairs
    ]
    assert len(audit["records"]) == 5
    assert all(row["sha_relation"] == "same" for row in audit["records"])
    assert all(
        row["failure_classification"] == "possible_isoform_relation"
        for row in audit["records"]
    )
    assert all(
        not Path(row["raw_rejected_payload_path"]).is_absolute()
        for row in audit["records"]
    )
    assert all(
        (tmp_path / row["raw_rejected_payload_path"]).exists()
        for row in audit["records"]
    )
    assert not (tmp_path / "data/raw").exists()


def test_refetch_exact_changed_payload_is_drift_not_recovery(tmp_path: Path) -> None:
    module = _load_module()
    pairs = [(24, "Q8N8S7"), (7, "Q13427"), (38, "Q9Y237"), (103, "Q8K4V4"), (208, "O14786")]
    records = [_historical_row(index, accession) for index, accession in pairs]
    responses = [
        module.TransportResponse(200, {}, _metadata(accession))
        for _, accession in pairs
    ]

    audit = module.run_forensic_refetch(
        {"records": records},
        project_root=tmp_path,
        run_dir=tmp_path / "runs/acq3",
        rejected_dir=tmp_path / "artifacts/dataset/audits/acquisition_rejected",
        transport=FakeTransport(responses),
        clock=lambda: "current",
        sleeper=lambda _: None,
    )

    assert all(row["outcome"] == "source_payload_drift" for row in audit["records"])
    assert all(row["historical_failure_recovered"] is False for row in audit["records"])
    assert all(row["raw_rejected_payload_path"] is None for row in audit["records"])
    assert all(not Path(row["current_payload_path"]).is_absolute() for row in audit["records"])
    assert all(
        (tmp_path / row["current_payload_path"]).exists() for row in audit["records"]
    )
    assert not (tmp_path / "data/raw").exists()


def test_forensic_reports_are_machine_readable_and_whitespace_clean(
    tmp_path: Path,
) -> None:
    module = _load_module()
    audit = {
        "schema_version": "dataset-a.acq1-rejected-metadata-forensics.v1",
        "scope": {
            "candidate_indices": [24],
            "metadata_request_count": 1,
            "dependent_asset_request_count": 0,
            "alternate_source_or_accession_used": False,
        },
        "records": [
            {
                "candidate_index": 24,
                "polymer_entity_id": "7A5M_1",
                "pair_id": "7a5m_A__Q8N8S7",
                "expected_accession": "Q8N8S7",
                "source_url": "https://alphafold.ebi.ac.uk/api/prediction/Q8N8S7",
                "historical_HTTP_status": 200,
                "historical_byte_count": 10,
                "historical_SHA256": "a" * 64,
                "historical_timestamp": "old",
                "current_HTTP_status": 200,
                "current_byte_count": 11,
                "current_SHA256": "b" * 64,
                "current_timestamp": "new",
                "current_response_headers": {"ETag": '"new"'},
                "current_validation": "identity_mismatch",
                "current_failure_code": "identity_mismatch",
                "returned_accessions": ["Q8N8S7-1"],
                "sequence_field_presence": [True],
                "model_identifiers": ["AF-Q8N8S7-1-F1"],
                "raw_rejected_payload_path": "artifacts/rejected.json",
                "rejected_evidence_status": "preserved_new",
                "current_payload_path": None,
                "failure_classification": "possible_isoform_relation",
                "sha_relation": "different",
                "outcome": "source_payload_drift_with_identity_mismatch",
                "historical_failure_recovered": False,
            }
        ],
    }

    module.write_reports(audit, tmp_path)

    json_path = tmp_path / "acq1_rejected_metadata_forensics_v1.json"
    tsv_path = tmp_path / "acq1_rejected_metadata_forensics_v1.tsv"
    md_path = tmp_path / "acq1_rejected_metadata_forensics_v1.md"
    assert json.loads(json_path.read_text()) == audit
    assert md_path.read_text().startswith("# ACQ-1 Rejected AFDB Metadata Forensics v1")
    payload = tsv_path.read_bytes()
    assert b"\r" not in payload
    assert all(not line.endswith(b"\t") for line in payload.splitlines())
