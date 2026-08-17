from __future__ import annotations

import pandas as pd

from dual_uq.dataset.apo_holo import (
    ApoHoloConfig,
    admit_pair,
    build_candidate_pairs,
    build_common_residue_mapping,
    select_primary_and_alternatives,
)


def test_pair_builder_preserves_unresolved_state_and_admission_reasons(tmp_path) -> None:
    structures = pd.DataFrame(
        [
            {
                "pdb_id": "1holo",
                "polymer_entity_id": "1holo_1",
                "uniprot_id": "P00001",
                "state_label": "holo",
                "common_mapped_count": 95,
                "canonical_sequence_length": 100,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "assembly_label": "dimer",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution": 1.8,
            },
            {
                "pdb_id": "1apo",
                "polymer_entity_id": "1apo_1",
                "uniprot_id": "P00001",
                "state_label": "unresolved",
                "common_mapped_count": 95,
                "canonical_sequence_length": 100,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "assembly_label": "dimer",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution": 2.0,
            },
        ]
    )
    pairs = build_candidate_pairs(structures, mappings={}, ligand_annotations={})
    assert len(pairs) == 1
    assert pairs.loc[0, "state_pair_status"] == "unresolved"
    assert pairs.loc[0, "admission_status"] == "EXCLUDED"
    assert "unresolved_ligand_state" in pairs.loc[0, "exclusion_reasons"]


def test_primary_selection_keeps_alternative_valid_pairs(tmp_path) -> None:
    pairs = pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "apo_pdb_id": "1apo",
                "holo_pdb_id": "1holo",
                "admission_status": "ADMITTED",
                "common_mapped_count": 100,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "experimental_method_comparable": True,
                "apo_resolution": 2.0,
                "holo_resolution": 2.0,
            },
            {
                "protein_id": "P1",
                "apo_pdb_id": "2apo",
                "holo_pdb_id": "2holo",
                "admission_status": "ADMITTED",
                "common_mapped_count": 95,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "experimental_method_comparable": True,
                "apo_resolution": 1.8,
                "holo_resolution": 1.8,
            },
        ]
    )
    primary, alternatives = select_primary_and_alternatives(pairs)
    assert primary[["apo_pdb_id", "holo_pdb_id"]].values.tolist() == [["1apo", "1holo"]]
    assert alternatives[["apo_pdb_id", "holo_pdb_id"]].values.tolist() == [["2apo", "2holo"]]


def test_common_residue_mapping_joins_only_exact_canonical_positions() -> None:
    apo = pd.DataFrame(
        [{"uniprot_residue_number": 1, "pdb_residue_number": "10", "pdb_residue_name": "ALA"}]
    )
    holo = pd.DataFrame(
        [{"uniprot_residue_number": 1, "pdb_residue_number": "20", "pdb_residue_name": "ALA"}]
    )
    result = build_common_residue_mapping(apo, holo)
    assert result[["canonical_position", "apo_auth_seq_id", "holo_auth_seq_id"]].to_dict("records") == [
        {"canonical_position": 1, "apo_auth_seq_id": 10, "holo_auth_seq_id": 20}
    ]


def test_admit_pair_does_not_promote_missing_reasons_to_literal_nan(tmp_path) -> None:
    result = admit_pair(
        {
            "state_pair_status": "resolved",
            "sequence_identity": 1.0,
            "common_fraction": 1.0,
            "construct_mismatch_count": 0,
            "assembly_comparability": "comparable",
            "exclusion_reasons": float("nan"),
        },
        ApoHoloConfig(tmp_path),
    )
    assert result["admitted"] is True
    assert result["exclusion_reasons"] is None
