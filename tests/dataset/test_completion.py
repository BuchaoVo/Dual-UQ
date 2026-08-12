from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.admission import (
    FormalAdmissionConfig,
    evaluate_admission_candidates,
    evaluate_formal_admission_frame,
    validate_a0_gate,
    validate_stage0_admission_ledger,
)
from dual_uq.dataset.completion import (
    BLOCKED_UPSTREAM_INTEGRITY,
    AcquisitionRecord,
    AssetRequirement,
    CandidateIdentity,
    FullFrameConfig,
    FullFrameError,
    FullFrameResult,
    acquisition_records_frame,
    apply_canonical_sequence_overrides,
    build_attrition_summary,
    build_full_frame_census,
    execute_initial_acquisition,
    materialize_full_frame_completion,
    recompute_full_frame_readiness,
    resolve_afdb_structure_requirement,
    resolve_afdb_structure_requirements,
    resolve_asset_requirements,
    resolve_uniprot_requirements,
    validate_full_frame_inputs,
    validate_scale1a1_regression,
)
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.policies.identity import (
    extract_canonical_sequence_from_source,
)
from dual_uq.dataset.stages.acquisition import AcquisitionError, TransportResponse, validate_payload

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _uniprot_record(accession: str = "P12345", sequence: str = "MAG") -> bytes:
    return json.dumps(
        {
            "primaryAccession": accession,
            "uniProtkbId": "EXAMPLE_HUMAN",
            "sequence": {"value": sequence, "length": len(sequence)},
            "entryAudit": {"entryVersion": 30, "sequenceVersion": 1},
        }
    ).encode()


def _afdb_fragment(
    accession: str, model_id: str, start: int, end: int
) -> dict[str, object]:
    return {
        "uniprotAccession": accession,
        "uniprotSequence": "A" * (end - start + 1),
        "sequenceStart": start,
        "sequenceEnd": end,
        "entryId": model_id,
        "modelEntityId": model_id,
        "latestVersion": 6,
        "cifUrl": f"https://example.test/{model_id}-model_v6.cif",
    }


def test_exact_uniprotkb_record_supplies_explicit_canonical_provenance() -> None:
    result = extract_canonical_sequence_from_source(
        _uniprot_record(), "P12345"
    )

    assert result["sequence"] == "MAG"
    assert result["sequence_length"] == 3
    assert result["sequence_source_field"] == "sequence.value"
    assert result["canonical_sequence_provenance"] == (
        "exact_uniprotkb_primary_accession_record"
    )
    assert result["source_metadata_versions"] == {
        "entryVersion": 30,
        "sequenceVersion": 1,
    }


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (_uniprot_record("Q99999"), "exact_identity_mismatch"),
        (
            json.dumps({"primaryAccession": "P12345", "sequence": {}}).encode(),
            "missing_canonical_sequence_provenance",
        ),
    ],
)
def test_uniprotkb_canonical_source_rejects_nonexact_or_missing_sequence(
    payload: bytes, expected_code: str
) -> None:
    with pytest.raises(DerivationError) as caught:
        extract_canonical_sequence_from_source(payload, "P12345")

    assert caught.value.code == expected_code


def test_afdb_metadata_collection_accepts_multiple_exact_fragment_records() -> None:
    records = [
        _afdb_fragment("P12345", "AF-P12345-F1", 1, 100),
        _afdb_fragment("P12345", "AF-P12345-F2", 101, 180),
        _afdb_fragment("P12345-2", "AF-P12345-2-F1", 1, 90),
    ]

    result = validate_payload(
        "afdb_metadata_collection",
        json.dumps(records).encode(),
        "P12345",
        None,
    )

    assert result == {
        "metadata_record_count": 3,
        "exact_accession_record_count": 2,
        "nonexact_record_count": 1,
        "exact_model_identifiers": ["AF-P12345-F1", "AF-P12345-F2"],
        "source_metadata_versions": [6],
        "identity_resolution_status": "exact_accession_collection_resolved",
    }


def test_afdb_metadata_collection_rejects_collection_without_exact_accession() -> None:
    records = [_afdb_fragment("P12345-2", "AF-P12345-2-F1", 1, 90)]

    with pytest.raises(AcquisitionError) as caught:
        validate_payload(
            "afdb_metadata_collection",
            json.dumps(records).encode(),
            "P12345",
            None,
        )

    assert caught.value.code == "exact_identity_mismatch"


def test_uniprot_payload_validator_uses_the_same_exact_identity_contract() -> None:
    result = validate_payload(
        "uniprot_canonical", _uniprot_record(), "P12345", None
    )
    assert result is not None
    assert result["canonical_sequence_provenance"] == (
        "exact_uniprotkb_primary_accession_record"
    )

    with pytest.raises(AcquisitionError) as caught:
        validate_payload(
            "uniprot_canonical", _uniprot_record("Q99999"), "P12345", None
        )
    assert caught.value.code == "exact_identity_mismatch"


def test_frozen_scale1a2_inputs_reconcile_exact_frame_and_initial_state() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    inputs = validate_full_frame_inputs(paths, FullFrameConfig())

    assert len(inputs.source_frame) == 213
    assert inputs.source_frame["sampling_frame_index"].tolist() == list(range(1, 214))
    assert inputs.source_frame["candidate_id"].is_unique
    assert len(inputs.preexisting_local_ready) == 72
    assert len(inputs.acquisition_targets) == 141
    assert inputs.initial_status_counts == {
        "LOCAL_READY_FOR_SCALE1A": 72,
        "MISSING_PDB": 68,
        "MISSING_PDB_AND_AFDB": 68,
        "MISSING_AFDB": 3,
        "MISSING_CANONICAL_SEQUENCE": 2,
    }
    assert len(inputs.scale1b_v1_artifacts) == 5


def test_frozen_input_sha_drift_blocks_before_requirement_resolution() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = replace(FullFrameConfig(), expected_inventory_sha256="0" * 64)

    with pytest.raises(FullFrameError) as caught:
        validate_full_frame_inputs(paths, config)

    assert caught.value.status == BLOCKED_UPSTREAM_INTEGRITY
    assert caught.value.code == "upstream_sha256_mismatch"


def test_candidate_evaluator_supports_an_empty_prospective_subset() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = FormalAdmissionConfig()
    stage0_gate = validate_stage0_admission_ledger(paths, config)

    result = evaluate_admission_candidates(
        pd.DataFrame(), paths=paths, config=config, stage0_gate=stage0_gate
    )

    assert result.census.empty
    assert "admission_status" in result.census.columns
    assert result.common_masks.empty
    assert "common_mask" in result.common_masks.columns


def test_asset_requirements_are_evidence_derived_and_exclude_original_72() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    inputs = validate_full_frame_inputs(paths, FullFrameConfig())

    requirements = resolve_asset_requirements(inputs, paths)
    counts: dict[str, int] = {}
    for requirement in requirements:
        counts[requirement.asset_type] = counts.get(requirement.asset_type, 0) + 1

    assert counts == {
        "pdb_mmcif": 136,
        "sifts": 136,
        "afdb_metadata_collection": 79,
    }
    target_indices = set(inputs.acquisition_targets["sampling_frame_index"].astype(int))
    ready_indices = set(inputs.preexisting_local_ready["sampling_frame_index"].astype(int))
    assert {requirement.sampling_frame_index for requirement in requirements} <= target_indices
    assert not ({requirement.sampling_frame_index for requirement in requirements} & ready_indices)
    sifts = [item for item in requirements if item.asset_type == "sifts"]
    assert len(sifts) == 136
    assert all(item.source_authority == "PDBe_SIFTS" for item in sifts)
    assert all(item.local_path_relative.startswith("data/raw/mappings/") for item in sifts)
    assert len({item.canonical_identifier for item in sifts}) == 136


class _FakeTransport:
    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, url: str, _timeout: float) -> TransportResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def _pdb_payload(pdb_id: str = "1ABC") -> bytes:
    return f"data_{pdb_id}\n#\n_entry.id {pdb_id}\n#\n".encode()


def _sifts_payload(pdb_id: str = "1abc") -> bytes:
    return gzip.compress(
        f"<?xml version='1.0'?><entry dbAccessionId='{pdb_id}'/>".encode()
    )


def _requirement(
    *, asset_type: str = "pdb_mmcif", local_path: str = "data/raw/pdb/1abc.cif"
) -> AssetRequirement:
    return AssetRequirement(
        sampling_frame_index=7,
        candidate_id="1abc_A__P12345",
        polymer_entity_id="1ABC_1",
        asset_type=asset_type,
        canonical_identifier="1abc" if asset_type != "uniprot_canonical" else "P12345",
        source_authority="RCSB_PDB" if asset_type == "pdb_mmcif" else "UniProtKB",
        source_record_identifier="1abc" if asset_type != "uniprot_canonical" else "P12345",
        source_url="https://example.test/asset",
        local_path_relative=local_path,
    )


def test_initial_acquisition_reuses_valid_asset_without_transport(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    path = paths.resolve_logical("data/raw/pdb/1abc.cif")
    path.parent.mkdir(parents=True)
    payload = _pdb_payload()
    path.write_bytes(payload)
    transport = _FakeTransport([])

    records = execute_initial_acquisition(
        (_requirement(),),
        paths,
        transport=transport,
        clock=lambda: "2026-08-10T00:00:00Z",
        sleeper=lambda _delay: None,
    )

    assert len(records) == 1
    assert records[0].retrieval_status == "ALREADY_PRESENT_FROZEN"
    assert records[0].attempt_count == 0
    assert records[0].file_sha256 == hashlib.sha256(payload).hexdigest()
    assert records[0].local_path_relative == "data/raw/pdb/1abc.cif"
    assert transport.urls == []


def test_initial_acquisition_normalizes_success_and_structured_failure(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    acquired = _requirement(
        asset_type="uniprot_canonical",
        local_path="data/raw/uniprot/P12345.json",
    )
    missing = replace(
        acquired,
        sampling_frame_index=8,
        candidate_id="1abd_A__P12345",
        local_path_relative="data/raw/uniprot/P12345-missing.json",
    )
    transport = _FakeTransport(
        [
            TransportResponse(200, {}, _uniprot_record()),
            TransportResponse(404, {}, b"not found"),
        ]
    )

    records = execute_initial_acquisition(
        (acquired, missing),
        paths,
        transport=transport,
        clock=lambda: "2026-08-10T00:00:00Z",
        sleeper=lambda _delay: None,
    )

    assert [record.retrieval_status for record in records] == [
        "ACQUIRED",
        "SOURCE_NOT_FOUND",
    ]
    assert records[0].attempt_count == 1
    assert records[0].validation_status == "VALID"
    assert records[1].attempt_count == 1
    assert records[1].validation_status == "NOT_VALIDATED"
    assert records[1].failure_code == "http_not_found"


def test_rejected_payload_path_is_portable(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    requirement = _requirement(
        asset_type="uniprot_canonical",
        local_path="data/raw/uniprot/P12345.json",
    )
    transport = _FakeTransport(
        [TransportResponse(200, {}, _uniprot_record("Q99999"))]
    )

    record = execute_initial_acquisition(
        (requirement,),
        paths,
        transport=transport,
        clock=lambda: "2026-08-10T00:00:00Z",
        sleeper=lambda _delay: None,
    )[0]

    assert record.retrieval_status == "IDENTIFIER_BINDING_FAIL"
    assert record.rejected_payload_path is not None
    assert record.rejected_payload_path.startswith(
        "artifacts/dataset/audits/acquisition_rejected/scale1a2/"
    )
    assert not Path(record.rejected_payload_path).is_absolute()


def test_afdb_structure_requirement_uses_only_unique_mapped_interval_cover() -> None:
    identity = CandidateIdentity(7, "1abc_A__P12345", "1ABC_1", "1abc", "A", "P12345")
    metadata = json.dumps(
        [
            _afdb_fragment("P12345", "AF-P12345-F1", 1, 50),
            _afdb_fragment("P12345", "AF-P12345-F2", 40, 120),
        ]
    ).encode()

    resolution = resolve_afdb_structure_requirement(
        identity, metadata, mapped_interval=(60, 100)
    )

    assert resolution.failure_code is None
    assert resolution.requirement is not None
    assert resolution.requirement.canonical_identifier == "AF-P12345-F2"
    assert resolution.requirement.source_metadata_version_if_available == "6"
    assert resolution.requirement.source_url.endswith("AF-P12345-F2-model_v6.cif")
    assert resolution.requirement.local_path_relative == (
        "data/raw/afdb/P12345/AF-P12345-F2/model.cif"
    )


@pytest.mark.parametrize(
    ("records", "expected_failure"),
    [
        (
            [_afdb_fragment("P12345", "AF-P12345-F1", 1, 50)],
            "AFDB_ASSET_UNAVAILABLE",
        ),
        (
            [
                _afdb_fragment("P12345", "AF-P12345-F1", 1, 120),
                _afdb_fragment("P12345", "AF-P12345-F2", 40, 140),
            ],
            "AMBIGUOUS_REMOTE_RECORD",
        ),
    ],
)
def test_afdb_structure_requirement_never_falls_back(
    records: list[dict[str, object]], expected_failure: str
) -> None:
    identity = CandidateIdentity(7, "1abc_A__P12345", "1ABC_1", "1abc", "A", "P12345")

    resolution = resolve_afdb_structure_requirement(
        identity, json.dumps(records).encode(), mapped_interval=(60, 100)
    )

    assert resolution.requirement is None
    assert resolution.failure_code == expected_failure


def test_dependency_failure_rerun_preserves_prior_record_and_timestamp(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    frame = pd.DataFrame(
        [
            {
                "sampling_frame_index": 7,
                "pair_id": "1abc_A__P12345",
                "polymer_entity_id": "1ABC_1",
                "pdb_id": "1abc",
                "pdb_chain": "A",
                "canonical_accession": "P12345",
                "missing_afdb": True,
                "afdb_metadata_source_relative": None,
            }
        ]
    )
    prior = AcquisitionRecord(
        sampling_frame_index=7,
        candidate_id="1abc_A__P12345",
        asset_type="afdb_structure",
        canonical_identifier="P12345",
        source_authority="AlphaFold_DB",
        source_record_identifier="P12345",
        source_url="https://example.test/P12345",
        retrieval_status="AFDB_ASSET_UNAVAILABLE",
        retrieval_timestamp="2026-08-10T00:00:00Z",
        retrieval_method="not_attempted_dependency_blocked",
        attempt_count=0,
        local_path_relative="",
        file_sha256=None,
        file_size=None,
        source_metadata_version_if_available=None,
        validation_status="NOT_VALIDATED",
        failure_code="metadata_asset_missing",
        http_status=None,
        dependency_asset_type="afdb_metadata_collection",
    )

    requirements, failures = resolve_afdb_structure_requirements(
        frame,
        {7},
        paths,
        clock=lambda: "2026-08-11T00:00:00Z",
        prior_records=(prior,),
    )

    assert requirements == ()
    assert failures == (prior,)


def test_canonical_override_completes_only_valid_exact_uniprot_source(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    source = paths.resolve_logical("data/raw/uniprot/P12345.json")
    source.parent.mkdir(parents=True)
    source.write_bytes(_uniprot_record())
    frame = pd.DataFrame(
        [
            {
                "sampling_frame_index": 7,
                "candidate_id": "1abc_A__P12345",
                "canonical_accession": "P12345",
                "canonical_sequence_present": False,
                "canonical_sequence_source_relative": None,
                "canonical_sequence_sha256": None,
                "missing_pdb": False,
                "missing_afdb": False,
                "missing_canonical_sequence": True,
                "missing_identity_metadata": False,
                "missing_mapping_metadata": False,
                "missing_provenance_metadata": False,
                "ambiguous_local_source": False,
                "unreadable_local_file": False,
                "other_structured_local_blocker": False,
                "local_availability_status": "MISSING_CANONICAL_SEQUENCE",
                "terminal_local_missing_reason": "canonical_sequence_not_locally_proven",
            }
        ]
    )

    updated = apply_canonical_sequence_overrides(
        frame,
        paths,
        {7: "data/raw/uniprot/P12345.json"},
    )

    assert updated.loc[0, "canonical_sequence_present"]
    assert not updated.loc[0, "missing_canonical_sequence"]
    assert updated.loc[0, "canonical_sequence_source_relative"] == (
        "data/raw/uniprot/P12345.json"
    )
    assert updated.loc[0, "local_availability_status"] == "LOCAL_READY_FOR_SCALE1A"
    assert pd.isna(updated.loc[0, "terminal_local_missing_reason"])

    source.write_bytes(_uniprot_record("Q99999"))
    with pytest.raises(FullFrameError) as caught:
        apply_canonical_sequence_overrides(
            frame,
            paths,
            {7: "data/raw/uniprot/P12345.json"},
        )
    assert caught.value.code == "invalid_canonical_sequence_override"


def test_public_formal_admission_frame_adapter_reproduces_frozen_72() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = FormalAdmissionConfig()
    gate = validate_a0_gate(paths, config)
    stage0_gate = validate_stage0_admission_ledger(paths, config)

    frame_result = evaluate_formal_admission_frame(
        gate.evaluated_frame,
        paths=paths,
        config=config,
        stage0_gate=stage0_gate,
    )
    regression = validate_scale1a1_regression(
        frame_result.census,
        frame_result.common_masks,
        paths,
        FullFrameConfig(),
    )

    assert regression["status"] == "PASS"
    assert regression["candidate_count"] == 72
    assert regression["common_mask_row_count"] == 22_200
    assert frame_result.stage0_regression["status"] == "PASS"


def test_full_frame_census_keeps_acquisition_failure_out_of_scientific_status() -> None:
    readiness = pd.DataFrame(
        [
            {
                "sampling_frame_index": 1,
                "candidate_id": "1abc_A__P12345",
                "canonical_accession": "P12345",
                "pdb_id": "1abc",
                "pdb_chain": "A",
                "afdb_id": "AF-P12345-F1",
                "local_availability_status": "LOCAL_READY_FOR_SCALE1A",
                "terminal_local_missing_reason": None,
            },
            {
                "sampling_frame_index": 2,
                "candidate_id": "2abc_A__Q12345",
                "canonical_accession": "Q12345",
                "pdb_id": "2abc",
                "pdb_chain": "A",
                "afdb_id": None,
                "local_availability_status": "MISSING_AFDB",
                "terminal_local_missing_reason": "afdb_structure_absent",
            },
        ]
    )
    admission = pd.DataFrame(
        [
            {
                "sampling_frame_index": 1,
                "admission_status": "FORMALLY_ADMITTED",
                "terminal_reason_code": "exact_canonical_identity",
                "paired_sequence_identity": 1.0,
                "mismatch_count": 0,
                "mapped_residue_count": 10,
                "common_mask_count": 8,
                "common_mask_fraction_of_mapped": 0.8,
            }
        ]
    )
    acquisition = (
        AcquisitionRecord(
            sampling_frame_index=2,
            candidate_id="2abc_A__Q12345",
            asset_type="afdb_metadata_collection",
            canonical_identifier="Q12345",
            source_authority="AlphaFold_DB",
            source_record_identifier="Q12345",
            source_url="https://example.test/Q12345",
            retrieval_status="SOURCE_NOT_FOUND",
            retrieval_timestamp="2026-08-10T00:00:00Z",
            retrieval_method="https_get",
            attempt_count=1,
            local_path_relative="data/raw/afdb/Q12345/metadata.json",
            file_sha256=None,
            file_size=None,
            source_metadata_version_if_available=None,
            validation_status="NOT_VALIDATED",
            failure_code="http_not_found",
            http_status=404,
            dependency_asset_type=None,
        ),
    )

    census = build_full_frame_census(
        readiness,
        admission,
        acquisition,
        preexisting_indices={1},
        scale1b_v1_ids={"1abc_A__P12345"},
    )

    assert len(census) == 2
    assert bool(census.loc[0, "formal_admission_evaluated"])
    assert census.loc[0, "formal_admission_status"] == "FORMALLY_ADMITTED"
    assert not bool(census.loc[1, "formal_admission_evaluated"])
    assert pd.isna(census.loc[1, "formal_admission_status"])
    assert pd.isna(census.loc[1, "terminal_scientific_reason"])
    assert census.loc[1, "afdb_acquisition_status"] == "SOURCE_NOT_FOUND"

    summary = build_attrition_summary(census, acquisition, n_preexisting_ready=1)
    assert summary["n_source_frame"] == 2
    assert summary["n_remaining_local_incomplete"] == 1
    assert summary["n_total_scientifically_evaluated"] == 1
    assert summary["n_formally_admitted_total"] == 1
    assert summary["acquisition_failure_reason_counts"] == {"http_not_found": 1}


def test_uniprot_requirement_is_only_added_when_metadata_lacks_canonical_span(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    frame = pd.DataFrame(
        [
            {
                "sampling_frame_index": 1,
                "pair_id": "1abc_A__P12345",
                "polymer_entity_id": "1ABC_1",
                "canonical_accession": "P12345",
                "missing_canonical_sequence": True,
            },
            {
                "sampling_frame_index": 2,
                "pair_id": "2abc_A__Q12345",
                "polymer_entity_id": "2ABC_1",
                "canonical_accession": "Q12345",
                "missing_canonical_sequence": True,
            },
        ]
    )
    metadata = paths.resolve_logical("data/raw/afdb/P12345/metadata.json")
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(
        json.dumps([_afdb_fragment("P12345", "AF-P12345-F1", 1, 100)]).encode()
    )

    requirements = resolve_uniprot_requirements(frame, {1, 2}, paths)

    assert [item.sampling_frame_index for item in requirements] == [2]
    assert requirements[0].source_authority == "UniProtKB"
    assert requirements[0].canonical_identifier == "Q12345"


def test_recompute_readiness_binds_acquired_assets_without_changing_identity(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, data_root=tmp_path / "data"
    )
    canonical = paths.resolve_logical("data/raw/uniprot/P12345.json")
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(_uniprot_record())
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "1abc_A__P12345",
                "pair_id": "1abc_A__P12345",
                "sampling_frame_index": 7,
                "canonical_accession": "P12345",
                "pdb_id": "1abc",
                "pdb_chain": "A",
                "afdb_id": "AF-P12345-F1",
                "pdb_file_present": False,
                "pdb_file_readable": False,
                "pdb_chain_identifiable": False,
                "pdb_entity_identifiable": False,
                "pdb_file_path_relative": None,
                "afdb_file_present": True,
                "afdb_file_readable": True,
                "afdb_accession_identifiable": True,
                "afdb_file_path_relative": "data/raw/afdb/P12345/AF-P12345-F1/model.cif",
                "canonical_sequence_present": False,
                "canonical_sequence_source_relative": None,
                "canonical_sequence_sha256": None,
                "identity_metadata_present": True,
                "mapping_metadata_present": False,
                "mapping_metadata_source_relative": None,
                "provenance_metadata_present": True,
                "missing_pdb": True,
                "missing_afdb": False,
                "missing_canonical_sequence": True,
                "missing_identity_metadata": False,
                "missing_mapping_metadata": True,
                "missing_provenance_metadata": False,
                "ambiguous_local_source": False,
                "unreadable_local_file": False,
                "other_structured_local_blocker": False,
                "local_pair_complete": False,
                "local_metadata_complete": False,
                "local_availability_status": "MISSING_PDB",
                "terminal_local_missing_reason": "pdb_structure_absent",
            }
        ]
    )
    base = {
        "sampling_frame_index": 7,
        "candidate_id": "1abc_A__P12345",
        "canonical_identifier": "1abc",
        "source_authority": "test",
        "source_record_identifier": "1abc",
        "source_url": "https://example.test",
        "retrieval_status": "ACQUIRED",
        "retrieval_timestamp": "2026-08-10T00:00:00Z",
        "retrieval_method": "https_get",
        "attempt_count": 1,
        "file_sha256": "a" * 64,
        "file_size": 1,
        "source_metadata_version_if_available": None,
        "validation_status": "VALID",
        "failure_code": None,
        "http_status": 200,
        "dependency_asset_type": None,
    }
    records = (
        AcquisitionRecord(
            **base,
            asset_type="pdb_mmcif",
            local_path_relative="data/raw/pdb/1abc.cif",
        ),
        AcquisitionRecord(
            **base,
            asset_type="sifts",
            local_path_relative="data/raw/mappings/1abc.xml.gz",
        ),
        AcquisitionRecord(
            **{**base, "canonical_identifier": "P12345"},
            asset_type="uniprot_canonical",
            local_path_relative="data/raw/uniprot/P12345.json",
        ),
    )

    updated = recompute_full_frame_readiness(frame, records, paths)

    assert updated.loc[0, "candidate_id"] == "1abc_A__P12345"
    assert updated.loc[0, "canonical_sequence_source_relative"] == (
        "data/raw/uniprot/P12345.json"
    )
    assert updated.loc[0, "local_availability_status"] == "LOCAL_READY_FOR_SCALE1A"


def test_scale1a2_materialization_is_immutable_and_summary_is_last(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT, artifacts_root=tmp_path / "artifacts"
    )
    config = replace(
        FullFrameConfig(), output_root_ref="artifacts/test-scale1a2"
    )
    ledger = acquisition_records_frame(())
    census = pd.DataFrame([{"sampling_frame_index": 1, "candidate_id": "x"}])
    masks = pd.DataFrame(
        [{"sampling_frame_index": 1, "canonical_position": 1}]
    )
    result = FullFrameResult(
        status="SCALE1A2_FULL_FRAME_CENSUS_COMPLETE",
        acquisition_ledger=ledger,
        census=census,
        common_masks=masks,
        attrition_summary={"n_source_frame": 1},
    )

    first = materialize_full_frame_completion(
        result, paths, config=config
    )
    second = materialize_full_frame_completion(
        result, paths, config=config
    )

    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    summary_path = paths.resolve_logical(
        "artifacts/test-scale1a2/scale1_full_frame_attrition_summary.json"
    )
    payload = json.loads(summary_path.read_text())
    assert payload["output_artifacts"]["full_frame_census"]["rows"] == 1
    assert not any("/home/" in str(value) for value in payload.values())

    conflicting = replace(result, census=pd.DataFrame([{"sampling_frame_index": 2}]))
    with pytest.raises(FullFrameError) as caught:
        materialize_full_frame_completion(
            conflicting, paths, config=config
        )
    assert caught.value.code == "immutable_scale1a2_output_conflict"
