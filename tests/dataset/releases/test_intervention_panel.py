from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dual_uq.core.hashing import sha256_bytes
from dual_uq.dataset.releases.intervention_panel import (
    EXPECTED_STAGE0_PANEL,
    InterventionPanelError,
    build_panel_record,
    parse_historical_intervention_panel,
    parse_protein_id,
    render_panel_jsonl,
    validate_experiment_binding,
    verify_repository_identity,
    write_immutable_panel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPOSITORY_ROOT / "docs/design/DATASET_SCALE_AND_VALIDATION_PLAN_V1.md"


def test_current_planning_document_contains_exact_human_reviewed_panel() -> None:
    entries = parse_historical_intervention_panel(SOURCE_PATH)

    assert tuple((entry.candidate_index, entry.protein_id) for entry in entries) == tuple(
        EXPECTED_STAGE0_PANEL
    )
    assert all(entry.pair_qc_pass_asserted for entry in entries)
    assert all(entry.mechanism_observable_asserted for entry in entries)


def test_protein_identity_is_parsed_only_from_documented_identifier() -> None:
    assert parse_protein_id("5gv8_A__P83686") == ("5gv8", "A", "P83686")

    with pytest.raises(InterventionPanelError, match="documented protein_id"):
        parse_protein_id("not-a-pair")


def test_repository_verification_uses_exact_identity_and_local_assets(tmp_path: Path) -> None:
    inventory = tmp_path / "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_index": 21,
                        "pair_id": "5gv8_A__P83686",
                        "polymer_entity_id": "5GV8_1",
                        "PDB": "5gv8",
                        "chain": "A",
                        "UniProt": "P83686",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pdb_path = tmp_path / "data/raw/pdb/5gv8.cif"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(b"pdb-backbone")
    afdb_root = tmp_path / "data/raw/afdb/P83686/AF-P83686-F1"
    afdb_root.mkdir(parents=True)
    metadata = [
        {
            "uniprotAccession": "P83686",
            "modelEntityId": "AF-P83686-F1",
            "uniprotSequence": "ACD",
            "sequenceStart": 1,
            "sequenceEnd": 3,
        }
    ]
    (afdb_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (afdb_root / "model.cif").write_bytes(b"afdb-backbone")

    verification = verify_repository_identity(
        candidate_index=21,
        protein_id="5gv8_A__P83686",
        repository_root=tmp_path,
        inventory_path=inventory,
    )

    assert verification["candidate_inventory_identity"]["status"] == "verified"
    assert verification["pdb_entity_id"] == {"status": "verified", "value": "5GV8_1"}
    assert verification["afdb_model_id"] == {
        "status": "verified",
        "value": "AF-P83686-F1",
    }
    assert verification["canonical_sequence_provenance"]["status"] == "verified"
    assert verification["pdb_backbone_path"]["value"] == "data/raw/pdb/5gv8.cif"
    assert verification["afdb_backbone_path"]["value"] == (
        "data/raw/afdb/P83686/AF-P83686-F1/model.cif"
    )


def test_repository_identity_conflict_is_a_task_blocker(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_index": 21,
                        "pair_id": "5gv8_A__OTHER",
                        "polymer_entity_id": "5GV8_1",
                        "PDB": "5gv8",
                        "chain": "A",
                        "UniProt": "OTHER",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InterventionPanelError, match="repository identity conflict"):
        verify_repository_identity(
            candidate_index=21,
            protein_id="5gv8_A__P83686",
            repository_root=tmp_path,
            inventory_path=inventory,
        )


def test_panel_jsonl_is_stable_portable_and_sorted() -> None:
    records = [
        build_panel_record(
            candidate_index=index,
            protein_id=protein_id,
            source_sha256="a" * 64,
            repository_verification={},
        )
        for index, protein_id in reversed(EXPECTED_STAGE0_PANEL)
    ]

    first = render_panel_jsonl(records)
    second = render_panel_jsonl(list(reversed(records)))
    decoded = [json.loads(line) for line in first.decode("utf-8").splitlines()]

    assert first == second
    assert first.endswith(b"\n")
    assert [row["candidate_index"] for row in decoded] == sorted(
        index for index, _ in EXPECTED_STAGE0_PANEL
    )
    assert b"/home/" not in first
    assert all("timestamp" not in row for row in decoded)
    assert sha256_bytes(first) == sha256_bytes(second)


def test_versioned_panel_is_reused_but_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "stage0_intervention_panel_v1.jsonl"

    assert write_immutable_panel(path, b'{"version":1}\n') == "created"
    assert write_immutable_panel(path, b'{"version":1}\n') == "reused_identical"
    with pytest.raises(InterventionPanelError, match="immutable panel conflict"):
        write_immutable_panel(path, b'{"version":2}\n')
    assert path.read_bytes() == b'{"version":1}\n'


def test_experiment_binding_has_stage0_semantics(tmp_path: Path) -> None:
    binding = tmp_path / "stage0.yaml"
    binding.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "p2_design_baseline_stage0",
                "inputs": {
                    "stage0_intervention_panel": {
                        "panel_version": "stage0_intervention_panel_v1",
                        "path": (
                            "experiments/p2_design_baseline/stage0/"
                            "stage0_intervention_panel_v1.jsonl"
                        ),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    resolved = validate_experiment_binding(
        binding,
        expected_panel_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_panel_v1.jsonl"
        ),
    )

    assert resolved == "experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl"
    assert "canonical_census" not in binding.read_text(encoding="utf-8")


def test_materialized_repository_panel_and_metadata_are_consistent() -> None:
    panel_path = (
        REPOSITORY_ROOT
        / "experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl"
    )
    metadata_path = panel_path.with_name("stage0_intervention_panel_v1.meta.json")
    records = [
        json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert tuple((row["candidate_index"], row["protein_id"]) for row in records) == tuple(
        EXPECTED_STAGE0_PANEL
    )
    assert metadata["record_count"] == 11
    assert metadata["panel_jsonl_sha256"] == sha256_bytes(panel_path.read_bytes())
    assert metadata["source_planning_document_sha256"] == sha256_bytes(
        SOURCE_PATH.read_bytes()
    )
    assert all(
        field["status"] == "verified"
        for row in records
        for field in row["repository_verification"].values()
    )
    assert "/home/" not in panel_path.read_text(encoding="utf-8")
    assert "/home/" not in metadata_path.read_text(encoding="utf-8")
