from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dataset_a"
    / "census"
    / "batch1_acquisition_plan.py"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("batch1_acquisition_plan", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_index": 1,
        "canonical_source_row": 0,
        "polymer_entity_id": "1ABC_1",
        "pair_id": "1abc_A__P00001",
        "PDB": "1abc",
        "chain": "A",
        "UniProt": "P00001",
        "estimated_P0_readiness": "partial_local_inputs",
        "sampling_stratum_prior": "uncertain_or_unclassified",
        "prior_evidence_status": "unobserved",
        "round1_member": False,
        "local_PDB_available": True,
        "local_AFDB_available": True,
        "local_PAE_available": True,
        "local_pLDDT_available": True,
        "fragment_metadata_available": True,
        "mapping_available": False,
        "pair_qc_status": None,
        "missing_local_inputs": ["pair_qc", "residue_mapping"],
        "selection_reason": "acquisition_panel;synthetic",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        (set(), "needs_mapping_derivation_only"),
        ({"pdb_structure"}, "needs_pdb_side_acquisition"),
        ({"afdb_model"}, "needs_afdb_side_acquisition"),
        ({"pdb_structure", "afdb_model"}, "needs_both_structure_sides"),
    ],
)
def test_primary_acquisition_class_is_deterministic(
    tmp_path: Path, missing: set[str], expected: str
) -> None:
    module = _load_module()
    row = _candidate(
        local_PDB_available="pdb_structure" not in missing,
        local_AFDB_available="afdb_model" not in missing,
        missing_local_inputs=sorted({"pair_qc", "residue_mapping", *missing}),
    )

    record = module.build_acquisition_record(row, project_root=tmp_path)

    assert record["primary_acquisition_class"] == expected
    assert set(record["required_external_assets"]).isdisjoint(
        record["required_local_derivations"]
    )


def test_missing_identity_metadata_is_a_structured_blocker(tmp_path: Path) -> None:
    module = _load_module()
    record = module.build_acquisition_record(
        _candidate(UniProt=None), project_root=tmp_path
    )

    assert record["primary_acquisition_class"] == "blocked_by_identity_metadata"
    assert record["asset_statuses"]["canonical_identity_metadata"] == "unresolved"


def test_asset_statuses_and_local_derivations_are_separate(tmp_path: Path) -> None:
    module = _load_module()
    record = module.build_acquisition_record(
        _candidate(
            local_PDB_available=False,
            local_AFDB_available=False,
            local_PAE_available=False,
            local_pLDDT_available=False,
            fragment_metadata_available=False,
            missing_local_inputs=[
                "pdb_structure",
                "sifts_mapping",
                "pair_qc",
                "residue_mapping",
                "afdb_metadata",
                "afdb_model",
                "afdb_pae",
                "afdb_plddt",
            ],
        ),
        project_root=tmp_path,
    )

    assert record["asset_statuses"]["pdb_mmcif"] == "requires_external_acquisition"
    assert record["asset_statuses"]["sifts"] == "requires_external_acquisition"
    assert record["asset_statuses"]["pair_qc_source_inputs"] == (
        "requires_external_acquisition"
    )
    assert set(record["required_external_assets"]) == {
        "pdb_mmcif",
        "sifts",
        "afdb_metadata",
        "afdb_structure",
        "afdb_pae",
        "afdb_confidence",
    }
    assert set(record["required_local_derivations"]) == {
        "canonical_sequence_metadata_extraction",
        "pair_qc",
        "residue_mapping",
        "fragment_resolution",
        "p0_eligibility_inputs",
        "mechanism_observability_features",
    }
    assert record["requires_pdb_mmcif"] is True
    assert record["requires_afdb_pae"] is True
    assert record["asset_statuses"]["canonical_sequence_metadata"] == (
        "derivable_locally_after_acquisition"
    )


def test_repository_plan_binds_inputs_and_preserves_batch_invariants() -> None:
    module = _load_module()
    first = module.build_plan(PROJECT_ROOT)
    second = module.build_plan(PROJECT_ROOT)

    assert first == second
    assert first["input_bindings"] == {
        "discovery_source_path": "data/processed/discovery/discovered_candidates.parquet",
        "discovery_source_sha256": (
            "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436"
        ),
        "candidate_inventory_path": "reports/dataset_a_scale/candidate_inventory_v1.json",
        "candidate_inventory_sha256": module.sha256_file(
            PROJECT_ROOT / "reports/dataset_a_scale/candidate_inventory_v1.json"
        ),
        "batch1_plan_path": "reports/dataset_a_scale/batch1_plan_v1.tsv",
        "batch1_plan_sha256": module.sha256_file(
            PROJECT_ROOT / "reports/dataset_a_scale/batch1_plan_v1.tsv"
        ),
    }
    assert len(first["records"]) == 48
    assert [row["candidate_index"] for row in first["records"]] == [
        row["candidate_index"] for row in first["source_batch_identity"]
    ]
    assert len({row["UniProt"] for row in first["records"]}) == 48
    assert all(row["round1_member"] is False for row in first["records"])
    assert {
        row["sampling_stratum_prior"] for row in first["records"]
    } == {"uncertain_or_unclassified"}
    assert set(first["summary"]["asset_status_counts"]) == set(module.ASSET_FIELDS)
    assert set(first["summary"]["external_acquisition_counts"]) == set(
        module.EXTERNAL_ASSETS
    )
    assert set(first["summary"]["local_derivation_counts"]) == set(
        module.LOCAL_DERIVATIONS
    )


def test_post_acquisition_and_h6_contracts_are_explicit() -> None:
    module = _load_module()
    payload = module.build_plan(PROJECT_ROOT)

    assert payload["panel_semantics"] == {
        "current_role": "acquisition_panel",
        "final_inferential_panel": False,
        "formal_mechanism_balanced_panel": False,
        "post_acquisition_restratification_required": True,
    }
    assert payload["post_acquisition_flow"] == [
        "acquire_assets",
        "verify_hashes_and_identities",
        "construct_pair_qc_and_residue_mapping",
        "evaluate_fragment_and_model_identity",
        "recompute_sampling_priors",
        "select_final_round2_census_panel",
        "run_final_p0_p1_protocol_only_after_selection",
    ]
    assert payload["h6_authorization_package"]["prohibited_actions"] == [
        "model_download",
        "training",
        "evaluator_execution",
        "proteinmpnn_generation",
    ]


def test_sequence_and_confidence_provenance_are_closed_without_double_counting() -> None:
    module = _load_module()
    payload = module.build_plan(PROJECT_ROOT)

    assert payload["asset_provenance"]["canonical_sequence_metadata"] == {
        "resolution": "deterministic_extraction_from_planned_afdb_metadata",
        "identity_field": "uniprotAccession",
        "sequence_fields_in_priority_order": ["uniprotSequence", "sequence"],
        "independent_external_acquisition_required": False,
        "no_sequence_identity_inference": True,
    }
    assert payload["asset_provenance"]["afdb_confidence"] == {
        "classification": "direct_external_afdb_asset",
        "metadata_source_field": "plddtDocUrl",
        "local_normalized_filename": "plddt.json",
        "content_transform": "none_raw_downloaded_bytes",
    }
    assert payload["summary"]["asset_status_counts"]["canonical_sequence_metadata"] == {
        "available_local": 10,
        "derivable_locally_after_acquisition": 38,
    }
    assert "canonical_sequence_metadata" not in module.EXTERNAL_ASSETS
    assert "afdb_confidence" not in module.LOCAL_DERIVATIONS
    h6_assets = {
        entry["asset_type"]: entry
        for entry in payload["h6_authorization_package"]["entries"]
    }
    assert "canonical_sequence_metadata" not in h6_assets
    assert h6_assets["afdb_confidence"]["source_binding"] == (
        "AFDB metadata.plddtDocUrl raw JSON bytes -> local plddt.json; no numeric transform"
    )
    local_derivations = {
        entry["local_derivation"]: entry
        for entry in payload["h6_authorization_package"]["local_derivations"]
    }
    assert local_derivations["canonical_sequence_metadata_extraction"] == {
        "local_derivation": "canonical_sequence_metadata_extraction",
        "candidate_count": 38,
        "input_dependency": (
            "identity-validated AFDB metadata.uniprotAccession plus "
            "uniprotSequence/sequence"
        ),
    }
