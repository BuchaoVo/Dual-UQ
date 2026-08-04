from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/dual_uq/dataset/audits/acquisition_identity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_identity_mismatches", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata(accession: str, *, model_id: str | None = None) -> bytes:
    model_id = model_id or f"AF-{accession}-F1"
    return json.dumps(
        [
            {
                "uniprotAccession": accession,
                "uniprotSequence": "AAAA",
                "modelEntityId": model_id,
                "entryId": model_id,
            }
        ]
    ).encode()


def test_success_control_requires_exact_accession_not_normalization() -> None:
    module = _load_module()

    result = module.audit_success_metadata(_metadata("P12345"), "P12345")

    assert result == {
        "prediction_record_count": 1,
        "exact_accession_match_count": 1,
        "normalized_but_nonexact_match_count": 0,
        "unexpected_alias_count": 0,
        "sequence_field_present_for_all": True,
        "model_identifiers": ["AF-P12345-F1"],
    }
    with pytest.raises(module.IdentityContractError, match="non-exact"):
        module.audit_success_metadata(_metadata("p12345"), "P12345")


def test_unretained_failed_payload_remains_insufficient_local_evidence() -> None:
    module = _load_module()
    ledger_row = {
        "candidate_index": 7,
        "pair_id": "1abc_A__P12345",
        "asset_type": "afdb_metadata",
        "source_url": "https://alphafold.ebi.ac.uk/api/prediction/P12345",
        "HTTP_status": 200,
        "byte_count": 123,
        "SHA256": "a" * 64,
        "status": "failed",
        "failure_code": "identity_mismatch",
    }

    result = module.classify_unretained_identity_failure(ledger_row)

    assert result["failure_taxonomy"] == "insufficient_local_evidence"
    assert result["returned_uniprot_accession"] is None
    assert result["returned_sequence_field_presence"] == "unknown_not_retained"
    assert result["returned_model_identifiers"] is None
    assert result["recovery_requirement"] == "additional_evidence_only"


def test_failure_accounting_separates_primary_and_dependent_failures() -> None:
    module = _load_module()
    records = []
    for candidate_index in range(1, 6):
        records.append(
            {
                "candidate_index": candidate_index,
                "asset_type": "afdb_metadata",
                "status": "failed",
                "failure_code": "identity_mismatch",
            }
        )
        for asset_type in ("afdb_structure", "afdb_pae", "afdb_confidence"):
            records.append(
                {
                    "candidate_index": candidate_index,
                    "asset_type": asset_type,
                    "status": "failed",
                    "failure_code": "metadata_asset_missing",
                }
            )

    result = module.summarize_failure_accounting(records)

    assert result == {
        "candidate_count_with_primary_metadata_failure": 5,
        "primary_metadata_identity_failure_assets": 5,
        "downstream_metadata_asset_missing_assets": 15,
        "total_failed_assets": 20,
    }
