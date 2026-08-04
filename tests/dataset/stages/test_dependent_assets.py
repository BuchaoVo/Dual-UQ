from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dual_uq.dataset.stages import dependent_assets

TARGETS = {
    7: ("2wfi_A__Q13427", "Q13427"),
    24: ("7a5m_A__Q8N8S7", "Q8N8S7"),
    38: ("3ui4_A__Q9Y237", "Q9Y237"),
    103: ("5emb_A__Q8K4V4", "Q8K4V4"),
    208: ("6fmc_A__O14786", "O14786"),
}


def _load_module():
    return dependent_assets


def _plan() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index, (pair_id, accession) in TARGETS.items():
        records.append(
            {
                "candidate_index": index,
                "polymer_entity_id": f"ENTITY_{index}",
                "pair_id": pair_id,
                "PDB": pair_id[:4],
                "chain": pair_id.split("__")[0].split("_")[1],
                "UniProt": accession,
                "required_external_assets": [
                    "afdb_metadata",
                    "afdb_structure",
                    "afdb_pae",
                    "afdb_confidence",
                ],
            }
        )
    records.extend(
        {
            "candidate_index": 1000 + offset,
            "polymer_entity_id": f"OTHER_{offset}",
            "pair_id": f"1abc_A__P{offset:05d}",
            "PDB": "1abc",
            "chain": "A",
            "UniProt": f"P{offset:05d}",
            "required_external_assets": [],
        }
        for offset in range(43)
    )
    return {"records": records, "input_bindings": {"binding": "fixed"}}


def _metadata(accession: str) -> bytes:
    def record(value: str) -> dict[str, object]:
        model = f"AF-{value}-F1"
        return {
            "uniprotAccession": value,
            "uniprotSequence": "AAA",
            "entryId": model,
            "modelEntityId": model,
            "sequenceStart": 1,
            "sequenceEnd": 3,
            "cifUrl": f"https://exact.test/{model}.cif",
            "paeDocUrl": f"https://exact.test/{model}-pae.json",
            "plddtDocUrl": f"https://exact.test/{model}-confidence.json",
        }

    return json.dumps([record(f"{accession}-2"), record(accession)]).encode()


def _resolution(tmp_path: Path) -> dict[str, object]:
    records = []
    for index, (pair_id, accession) in TARGETS.items():
        path = tmp_path / f"{index}.json"
        path.write_bytes(_metadata(accession))
        records.append(
            {
                "candidate_index": index,
                "pair_id": pair_id,
                "candidate_accession": accession,
                "payload_path": path.relative_to(tmp_path).as_posix(),
                "payload_SHA256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "exact_accession_record_count": 1,
                "resolution_status": "exact_record_resolved",
            }
        )
    return {"records": records}


def _acq1() -> dict[str, object]:
    completeness = [
        {
            "candidate_index": int(row["candidate_index"]),
            "completeness": (
                "zero_success"
                if int(row["candidate_index"]) in TARGETS
                else "fully_raw_complete"
            ),
        }
        for row in _plan()["records"]
    ]
    return {
        "candidate_completeness": completeness,
        "records": [
            {
                "candidate_index": index,
                "asset_type": "afdb_metadata",
                "status": "failed",
                "failure_code": "identity_mismatch",
            }
            for index in TARGETS
        ],
    }


def _successful_dependents() -> list[dict[str, object]]:
    return [
        {
            "candidate_index": index,
            "asset_type": asset,
            "status": "downloaded_new",
            "failure_code": None,
        }
        for index in TARGETS
        for asset in ("afdb_structure", "afdb_pae", "afdb_confidence")
    ]


def test_builds_only_fifteen_exact_record_dependent_specs(tmp_path: Path) -> None:
    module = _load_module()

    specs = module.build_acq5_specs(_plan(), _resolution(tmp_path), tmp_path)

    assert len(specs) == 15
    assert {spec.candidate_index for spec in specs} == set(TARGETS)
    assert {spec.asset_type for spec in specs} == {
        "afdb_structure",
        "afdb_pae",
        "afdb_confidence",
    }
    assert all("-2" not in spec.source_url for spec in specs)
    assert all(
        spec.metadata_record["uniprotAccession"]
        == TARGETS[spec.candidate_index][1]
        for spec in specs
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exact_accession_record_count", 2, "exact record"),
        ("resolution_status", "exact_identity_mismatch", "exact record"),
        ("candidate_accession", "P99999", "identity"),
    ],
)
def test_scope_rejects_ambiguity_and_candidate_identity_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    module = _load_module()
    resolution = _resolution(tmp_path)
    resolution["records"][0][field] = value

    with pytest.raises(module.ACQ5Error, match=message):
        module.build_acq5_specs(_plan(), resolution, tmp_path)


def test_acquisition_records_preserve_exact_candidate_and_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    specs = module.build_acq5_specs(_plan(), _resolution(tmp_path), tmp_path)

    def acquire(spec, **_kwargs):
        return {
            "candidate_index": spec.candidate_index,
            "asset_type": spec.asset_type,
            "status": "downloaded_new",
            "failure_code": None,
        }

    monkeypatch.setattr(module._BATCH1, "acquire_asset", acquire)

    records = module.acquire_dependents(specs, clock=lambda: "T")

    assert len(records) == 15
    assert all(
        row["candidate_accession"] == TARGETS[row["candidate_index"]][1]
        for row in records
    )
    assert all(
        row["exact_record_model_identity"]
        == f"AF-{row['candidate_accession']}-F1"
        for row in records
    )
    assert all(row["metadata_identity_resolution"] == "exact_record_resolved" for row in records)


def test_current_state_supersedes_history_without_erasing_it() -> None:
    module = _load_module()
    plan = _plan()
    acq1 = _acq1()
    resolution = {
        "records": [
            {
                "candidate_index": index,
                "resolution_status": "exact_record_resolved",
                "exact_accession_record_count": 1,
            }
            for index in TARGETS
        ]
    }
    originals = copy.deepcopy((acq1, resolution))

    report = module.build_current_state(
        plan,
        acq1,
        resolution,
        _successful_dependents(),
        source_bindings={"acq1_ledger_SHA256": "old", "acq4_report_SHA256": "new"},
    )

    assert report["summary"]["candidate_raw_complete"] == 48
    assert report["summary"]["current_candidate_identity_failures"] == 0
    assert report["summary"]["historical_container_identity_failures"] == 5
    assert report["summary"]["historical_dependent_asset_missing"] == 15
    assert all(row["superseded_by"] == "ACQ-4" for row in report["supersession"])
    assert all(not row["candidate_identity_failure_current"] for row in report["supersession"])
    assert (acq1, resolution) == originals


def test_dependent_failure_is_not_counted_as_identity_failure() -> None:
    module = _load_module()
    dependents = _successful_dependents()
    dependents[0] = dict(dependents[0], status="failed", failure_code="transport_failure")
    resolution = {
        "records": [
            {
                "candidate_index": index,
                "resolution_status": "exact_record_resolved",
                "exact_accession_record_count": 1,
            }
            for index in TARGETS
        ]
    }

    report = module.build_current_state(
        _plan(), _acq1(), resolution, dependents, source_bindings={}
    )

    assert report["summary"]["candidate_raw_complete"] == 47
    assert report["summary"]["current_candidate_identity_failures"] == 0
    assert report["summary"]["failed_dependent_assets"] == 1


def test_current_state_writer_is_deterministic_and_clean(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "schema_version": "dataset-a.batch1-acquisition-current-state.v2",
        "summary": {"candidate_count": 1, "candidate_raw_complete": 1},
        "candidate_states": [
            {
                "candidate_index": 1,
                "pair_id": "1abc_A__P00001",
                "candidate_accession": "P00001",
                "metadata_transport_complete": True,
                "metadata_parse_complete": True,
                "record_level_identity_resolved": True,
                "structure_complete": True,
                "PAE_complete": True,
                "confidence_complete": True,
                "candidate_raw_complete": True,
                "candidate_identity_failure_current": False,
            }
        ],
        "acq5_records": [],
        "supersession": [],
    }

    module.write_outputs(report, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    module.write_outputs(report, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert set(first) == {
        "batch1_acquisition_current_state_v2.json",
        "batch1_acquisition_current_state_v2.tsv",
        "batch1_acquisition_current_state_v2.md",
    }
    assert b"\r" not in first["batch1_acquisition_current_state_v2.tsv"]
    assert all(
        not line.endswith(b"\t")
        for line in first["batch1_acquisition_current_state_v2.tsv"].splitlines()
    )
