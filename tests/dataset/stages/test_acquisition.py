from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/dual_uq/dataset/stages/acquisition.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("batch1_acquire", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pdb_cif(entry_id: str = "1ABC") -> bytes:
    return f"data_{entry_id}\n#\n_entry.id {entry_id}\n#\n".encode()


def _sifts_xml(pdb_id: str = "1abc") -> bytes:
    xml = f"<?xml version='1.0'?><entry dbAccessionId='{pdb_id}'/>".encode()
    return gzip.compress(xml)


def _metadata(accession: str = "P00001", **overrides: object) -> bytes:
    record: dict[str, object] = {
        "entryId": f"AF-{accession}-F1",
        "modelEntityId": f"AF-{accession}-F1",
        "uniprotAccession": accession,
        "uniprotSequence": "A",
        "sequenceStart": 1,
        "sequenceEnd": 1,
        "cifUrl": f"https://example.test/AF-{accession}-F1-model_v6.cif",
        "paeDocUrl": f"https://example.test/AF-{accession}-F1-pae_v6.json",
        "plddtDocUrl": f"https://example.test/AF-{accession}-F1-confidence_v6.json",
    }
    record.update(overrides)
    return json.dumps([record]).encode()


def _plan_record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_index": 7,
        "polymer_entity_id": "1ABC_1",
        "pair_id": "1abc_A__P00001",
        "PDB": "1abc",
        "chain": "A",
        "UniProt": "P00001",
        "round1_member": False,
        "required_external_assets": ["pdb_mmcif", "sifts"],
    }
    row.update(overrides)
    return row


def _plan(*records: dict[str, object]) -> dict[str, object]:
    return {
        "records": list(records),
        "input_bindings": {
            "discovery_source_sha256": "discovery",
            "candidate_inventory_sha256": "inventory",
            "batch1_plan_sha256": "batch",
        },
    }


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _spec(module, tmp_path: Path, **overrides: object):
    values: dict[str, object] = {
        "candidate_index": 7,
        "polymer_entity_id": "1ABC_1",
        "pair_id": "1abc_A__P00001",
        "asset_type": "pdb_mmcif",
        "source_url": "https://example.test/1abc.cif",
        "local_path": tmp_path / "data/raw/pdb/1abc.cif",
        "expected_identity": "1abc",
        "metadata_record": None,
    }
    values.update(overrides)
    return module.AssetSpec(**values)


def test_manifest_scope_builds_only_declared_assets_and_rejects_nonmanifest() -> None:
    module = _load_module()
    plan = _plan(_plan_record())

    specs = module.build_initial_asset_specs(plan, Path("/repo"))

    assert [(spec.candidate_index, spec.asset_type) for spec in specs] == [
        (7, "pdb_mmcif"),
        (7, "sifts"),
    ]
    with pytest.raises(module.AcquisitionError, match="not present in manifest") as exc:
        module.select_manifest_records(plan, [999])
    assert exc.value.code == "non_manifest_candidate"


@pytest.mark.parametrize(
    ("asset_type", "body", "identity", "metadata", "code"),
    [
        ("pdb_mmcif", _pdb_cif("9XYZ"), "1abc", None, "identity_mismatch"),
        ("sifts", gzip.compress(b"<broken"), "1abc", None, "malformed_payload"),
        ("afdb_metadata", b"not-json", "P00001", None, "malformed_payload"),
        ("afdb_structure", b"not-cif", "AF-P00001-F1", {}, "malformed_payload"),
        ("afdb_pae", b"{}", "AF-P00001-F1", {"sequenceStart": 1, "sequenceEnd": 1}, "malformed_payload"),
        ("afdb_confidence", b"not-json", "AF-P00001-F1", {}, "malformed_payload"),
    ],
)
def test_payload_validation_rejects_identity_and_malformed_content(
    asset_type: str,
    body: bytes,
    identity: str,
    metadata: dict[str, object] | None,
    code: str,
) -> None:
    module = _load_module()
    with pytest.raises(module.AcquisitionError) as exc:
        module.validate_payload(asset_type, body, identity, metadata)
    assert exc.value.code == code


def test_metadata_requires_exact_accession_and_sequence() -> None:
    module = _load_module()
    record = module.validate_payload(
        "afdb_metadata", _metadata(), "P00001", None
    )
    assert record["modelEntityId"] == "AF-P00001-F1"

    with pytest.raises(module.AcquisitionError) as exc:
        module.validate_payload(
            "afdb_metadata", _metadata("P99999"), "P00001", None
        )
    assert exc.value.code == "exact_identity_mismatch"


def test_metadata_resolves_exact_base_and_preserves_sibling_counts() -> None:
    module = _load_module()
    base = json.loads(_metadata("P00001"))[0]
    sibling_2 = json.loads(_metadata("P00001-2"))[0]
    sibling_3 = json.loads(_metadata("P00001-3"))[0]

    resolved = module.validate_payload(
        "afdb_metadata",
        json.dumps([sibling_3, base, sibling_2]).encode(),
        "P00001",
        None,
    )

    assert resolved["uniprotAccession"] == "P00001"
    assert resolved["modelEntityId"] == "AF-P00001-F1"
    assert resolved["_metadata_record_count"] == 3
    assert resolved["_exact_accession_record_count"] == 1
    assert resolved["_nonexact_record_count"] == 2
    assert resolved["_isoform_suffix_record_count"] == 2
    assert resolved["_nonexact_accessions"] == ["P00001-2", "P00001-3"]
    assert resolved["_identity_resolution_status"] == "exact_record_resolved"


def test_metadata_rejects_multiple_exact_records_as_ambiguous() -> None:
    module = _load_module()
    first = json.loads(_metadata("P00001"))[0]
    second = dict(first, modelEntityId="AF-P00001-F2", entryId="AF-P00001-F2")

    with pytest.raises(module.AcquisitionError) as exc:
        module.validate_payload(
            "afdb_metadata", json.dumps([first, second]).encode(), "P00001", None
        )

    assert exc.value.code == "ambiguous_exact_record_set"


def test_metadata_record_resolution_is_input_order_invariant() -> None:
    module = _load_module()
    base = json.loads(_metadata("P00001"))[0]
    sibling = json.loads(_metadata("P00001-2"))[0]

    first = module.validate_payload(
        "afdb_metadata", json.dumps([base, sibling]).encode(), "P00001", None
    )
    second = module.validate_payload(
        "afdb_metadata", json.dumps([sibling, base]).encode(), "P00001", None
    )

    assert first == second
    assert first["uniprotAccession"] == "P00001"



def test_multiple_exact_metadata_records_are_ambiguous_without_fragment_selection(
    tmp_path: Path,
) -> None:
    module = _load_module()
    records = [json.loads(_metadata())[0], json.loads(_metadata())[0]]
    records[1]["entryId"] = records[1]["modelEntityId"] = "AF-P00001-F2"
    plan = _plan(
        _plan_record(
            required_external_assets=[
                "afdb_metadata",
                "afdb_structure",
                "afdb_pae",
                "afdb_confidence",
            ]
        )
    )
    transport = FakeTransport(
        [module.TransportResponse(200, {}, json.dumps(records).encode())]
    )

    ledger = module.run_plan(
        plan,
        tmp_path,
        transport=transport,
        clock=lambda: "T",
        sleeper=lambda _: None,
    )

    by_type = {row["asset_type"]: row for row in ledger["records"]}
    assert by_type["afdb_metadata"]["status"] == "failed"
    assert by_type["afdb_metadata"]["failure_code"] == "ambiguous_exact_record_set"
    assert by_type["afdb_metadata"]["exact_accession_record_count"] == 2
    assert Path(by_type["afdb_metadata"]["rejected_payload_path"]).exists()
    for asset_type in ("afdb_structure", "afdb_pae", "afdb_confidence"):
        assert by_type[asset_type]["failure_code"] == "metadata_asset_missing"
    assert transport.urls == [module.AFDB_PREDICTION_API.format(accession="P00001")]


def test_existing_valid_file_is_reused_without_transport(tmp_path: Path) -> None:
    module = _load_module()
    spec = _spec(module, tmp_path)
    spec.local_path.parent.mkdir(parents=True)
    spec.local_path.write_bytes(_pdb_cif())
    transport = FakeTransport([])

    row = module.acquire_asset(spec, transport=transport, clock=lambda: "T")

    assert row["status"] == "reused_valid"
    assert row["HTTP_status"] is None
    assert transport.urls == []


def test_existing_conflicting_file_is_not_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    spec = _spec(module, tmp_path)
    spec.local_path.parent.mkdir(parents=True)
    spec.local_path.write_bytes(b"bad-existing")

    row = module.acquire_asset(
        spec,
        transport=FakeTransport([module.TransportResponse(200, {}, _pdb_cif())]),
        clock=lambda: "T",
    )

    assert row["status"] == "failed"
    assert row["failure_code"] == "existing_file_conflict"
    assert spec.local_path.read_bytes() == b"bad-existing"


def test_invalid_download_never_leaves_canonical_or_partial_file(tmp_path: Path) -> None:
    module = _load_module()
    spec = _spec(
        module,
        tmp_path,
        asset_type="afdb_metadata",
        expected_identity="P00001",
        local_path=tmp_path / "data/raw/afdb/P00001/metadata.json",
    )

    row = module.acquire_asset(
        spec,
        transport=FakeTransport([module.TransportResponse(200, {}, b"bad-json")]),
        clock=lambda: "T",
    )

    assert row["failure_code"] == "malformed_payload"
    assert not spec.local_path.exists()
    assert not list(spec.local_path.parent.glob(".*.tmp"))


def test_identity_mismatch_retains_forensic_payload_outside_canonical(
    tmp_path: Path,
) -> None:
    module = _load_module()
    body = _metadata("P99999")
    spec = _spec(
        module,
        tmp_path,
        asset_type="afdb_metadata",
        expected_identity="P00001",
        local_path=tmp_path / "data/raw/afdb/P00001/metadata.json",
    )
    rejected_dir = tmp_path / "artifacts/dataset/audits/acquisition_rejected"

    row = module.acquire_asset(
        spec,
        transport=FakeTransport(
            [
                module.TransportResponse(
                    200,
                    {"ETag": '"metadata-v1"', "Content-Type": "application/json"},
                    body,
                )
            ]
        ),
        clock=lambda: "T",
        rejected_evidence_dir=rejected_dir,
    )

    digest = hashlib.sha256(body).hexdigest()
    rejected = Path(row["rejected_payload_path"])
    assert row["failure_code"] == "exact_identity_mismatch"
    assert row["response_headers"] == {
        "Content-Type": "application/json",
        "ETag": '"metadata-v1"',
    }
    assert row["parsed_returned_accessions"] == ["P99999"]
    assert row["parsed_sequence_field_presence"] == [True]
    assert row["parsed_model_identifiers"] == ["AF-P99999-F1"]
    assert rejected == rejected_dir / "7" / "afdb_metadata" / f"{digest}.json"
    assert rejected.read_bytes() == body
    assert row["rejected_payload_SHA256"] == digest
    assert not spec.local_path.exists()


def test_valid_metadata_stays_canonical_and_out_of_rejected_store(tmp_path: Path) -> None:
    module = _load_module()
    body = _metadata("P00001")
    spec = _spec(
        module,
        tmp_path,
        asset_type="afdb_metadata",
        expected_identity="P00001",
        local_path=tmp_path / "data/raw/afdb/P00001/metadata.json",
    )
    rejected_dir = tmp_path / "artifacts/dataset/audits/acquisition_rejected"

    row = module.acquire_asset(
        spec,
        transport=FakeTransport([module.TransportResponse(200, {}, body)]),
        clock=lambda: "T",
        rejected_evidence_dir=rejected_dir,
    )

    assert row["status"] == "downloaded_new"
    assert spec.local_path.read_bytes() == body
    assert not rejected_dir.exists()
    assert row["rejected_payload_path"] is None


def test_same_sha_now_exact_is_validation_inconsistency() -> None:
    module = _load_module()

    with pytest.raises(module.AcquisitionError) as exc:
        module.interpret_refetch_outcome(
            historical_sha="a" * 64,
            current_sha="a" * 64,
            current_validation="exact_match",
        )

    assert exc.value.code == "validation_implementation_inconsistency"


def test_changed_sha_exact_is_source_payload_drift() -> None:
    module = _load_module()

    outcome = module.interpret_refetch_outcome(
        historical_sha="a" * 64,
        current_sha="b" * 64,
        current_validation="exact_match",
    )

    assert outcome == {
        "sha_relation": "different",
        "outcome": "source_payload_drift",
        "historical_failure_recovered": False,
    }


def test_rejected_evidence_resume_reuses_immutable_content(tmp_path: Path) -> None:
    module = _load_module()
    body = _metadata("P99999")
    rejected_dir = tmp_path / "artifacts/dataset/audits/acquisition_rejected"

    first = module.preserve_rejected_payload(
        body,
        rejected_dir=rejected_dir,
        candidate_index=7,
        asset_type="afdb_metadata",
    )
    second = module.preserve_rejected_payload(
        body,
        rejected_dir=rejected_dir,
        candidate_index=7,
        asset_type="afdb_metadata",
    )

    assert first["evidence_status"] == "preserved_new"
    assert second["evidence_status"] == "reused_immutable"
    assert first["path"] == second["path"]
    assert Path(second["path"]).read_bytes() == body


def test_run_plan_preserves_rejected_payload_by_default(tmp_path: Path) -> None:
    module = _load_module()
    body = _metadata("P99999")
    plan = _plan(
        _plan_record(
            required_external_assets=["afdb_metadata"],
        )
    )

    ledger = module.run_plan(
        plan,
        tmp_path,
        transport=FakeTransport([module.TransportResponse(200, {}, body)]),
        clock=lambda: "T",
        sleeper=lambda _: None,
    )

    row = ledger["records"][0]
    rejected = Path(row["rejected_payload_path"])
    assert rejected.is_relative_to(
        tmp_path / "artifacts/dataset/audits/acquisition_rejected"
    )
    assert rejected.read_bytes() == body
    assert not (tmp_path / "data/raw/afdb/P00001/metadata.json").exists()


def test_retryable_http_is_bounded_but_terminal_http_is_not_retried(
    tmp_path: Path,
) -> None:
    module = _load_module()
    spec = _spec(module, tmp_path)
    transport = FakeTransport(
        [
            module.TransportResponse(500, {}, b""),
            module.TransportResponse(200, {}, _pdb_cif()),
        ]
    )

    row = module.acquire_asset(
        spec,
        transport=transport,
        sleeper=lambda _: None,
        clock=lambda: "T",
        max_attempts=3,
    )
    assert row["status"] == "downloaded_new"
    assert len(transport.urls) == 2

    missing = _spec(module, tmp_path, local_path=tmp_path / "missing.cif")
    terminal = FakeTransport([module.TransportResponse(404, {}, b"")])
    row = module.acquire_asset(
        missing, transport=terminal, sleeper=lambda _: None, clock=lambda: "T"
    )
    assert row["failure_code"] == "http_not_found"
    assert len(terminal.urls) == 1


def test_resume_reuses_completed_assets_and_ledger_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_module()
    plan = _plan(_plan_record(required_external_assets=["pdb_mmcif"]))
    spec = module.build_initial_asset_specs(plan, tmp_path)[0]
    first_transport = FakeTransport(
        [module.TransportResponse(200, {}, _pdb_cif())]
    )

    first = module.run_plan(
        plan,
        tmp_path,
        transport=first_transport,
        clock=lambda: "2026-08-04T00:00:00Z",
        sleeper=lambda _: None,
    )
    second = module.run_plan(
        plan,
        tmp_path,
        transport=FakeTransport([]),
        clock=lambda: "2026-08-04T00:00:00Z",
        sleeper=lambda _: None,
    )

    assert first["records"][0]["status"] == "downloaded_new"
    assert second["records"][0]["status"] == "reused_valid"
    assert second == module.run_plan(
        plan,
        tmp_path,
        transport=FakeTransport([]),
        clock=lambda: "2026-08-04T00:00:00Z",
        sleeper=lambda _: None,
    )
    assert spec.local_path.exists()


def test_metadata_missing_url_fails_without_alternate_source_fallback(
    tmp_path: Path,
) -> None:
    module = _load_module()
    plan = _plan(
        _plan_record(
            required_external_assets=[
                "afdb_metadata",
                "afdb_structure",
                "afdb_pae",
                "afdb_confidence",
            ]
        )
    )
    transport = FakeTransport(
        [
            module.TransportResponse(
                200,
                {},
                _metadata(cifUrl=None, paeDocUrl=None, plddtDocUrl=None),
            ),
        ]
    )

    ledger = module.run_plan(
        plan,
        tmp_path,
        transport=transport,
        clock=lambda: "T",
        sleeper=lambda _: None,
    )

    by_type = {row["asset_type"]: row for row in ledger["records"]}
    assert by_type["afdb_metadata"]["status"] == "downloaded_new"
    assert by_type["afdb_confidence"]["failure_code"] == "metadata_asset_missing"
    assert transport.urls == [module.AFDB_PREDICTION_API.format(accession="P00001")]


def test_tsv_report_uses_lf_without_carriage_returns(tmp_path: Path) -> None:
    module = _load_module()
    plan = _plan(_plan_record(required_external_assets=["pdb_mmcif"]))
    ledger = module.run_plan(
        plan,
        tmp_path,
        transport=FakeTransport(
            [module.TransportResponse(200, {}, _pdb_cif())]
        ),
        clock=lambda: "T",
        sleeper=lambda _: None,
    )

    module.write_reports(ledger, tmp_path / "reports")

    tsv = (tmp_path / "reports/batch1_acquisition_run_v1.tsv").read_bytes()
    assert b"\r" not in tsv
    assert all(not line.endswith(b"\t") for line in tsv.splitlines())
    header = tsv.splitlines()[0].decode().split("\t")
    assert "metadata_record_count" in header
    assert "exact_accession_record_count" in header
    assert "nonexact_record_count" in header
    assert "identity_resolution_status" in header
