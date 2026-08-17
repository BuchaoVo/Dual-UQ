from __future__ import annotations

import pandas as pd

from dual_uq.dataset.apo_holo import (
    LigandClass,
    StateLabel,
    assembly_comparability,
    classify_ligand_component,
    classify_state,
    construct_overlap,
    select_primary_pair,
    sequence_identity,
)


def test_ligand_classification_keeps_biological_and_nonbiological_components_separate() -> None:
    assert classify_ligand_component("ATP", "adenosine triphosphate", 31) is LigandClass.BIOLOGICAL_SMALL_MOLECULE
    assert classify_ligand_component("MG", "magnesium ion", 1) is LigandClass.ION
    assert classify_ligand_component("SO4", "sulfate ion", 5) is LigandClass.ADDITIVE_OR_SOLVENT
    assert classify_ligand_component("UNK", None, None) is LigandClass.AMBIGUOUS


def test_state_classification_does_not_infer_apo_from_missing_ccd() -> None:
    assert classify_state([]) is StateLabel.UNRESOLVED
    assert classify_state([], annotation_complete=True) is StateLabel.APO
    assert classify_state([{"ligand_class": LigandClass.BIOLOGICAL_SMALL_MOLECULE.value}]) is StateLabel.HOLO
    assert classify_state([{"ligand_class": LigandClass.COFACTOR.value}]) is StateLabel.HOLO
    assert classify_state([{"ligand_class": LigandClass.ION.value}]) is StateLabel.UNRESOLVED
    assert classify_state([{"ligand_class": LigandClass.ION.value}], annotation_complete=True) is StateLabel.APO
    assert classify_state([{"ligand_class": LigandClass.AMBIGUOUS.value}], annotation_complete=True) is StateLabel.UNRESOLVED


def test_sequence_identity_is_defined_on_explicit_mapped_overlap() -> None:
    assert sequence_identity("ACD", "ACD") == 1.0
    assert sequence_identity("ACD", "ATD") == 2 / 3
    assert sequence_identity("", "") is None
    assert sequence_identity("AC", "A") is None


def test_construct_and_assembly_comparability_preserve_missingness() -> None:
    assert construct_overlap(1, 10, 1, 10) == 1.0
    assert construct_overlap(None, 10, 1, 10) is None
    assert assembly_comparability("tetramer", "tetramer") == "comparable"
    assert assembly_comparability("tetramer", None) == "unknown"


def test_primary_pair_selection_is_order_invariant_and_outcome_blind() -> None:
    pairs = pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "apo_pdb_id": "2abc",
                "holo_pdb_id": "1abc",
                "common_mapped_count": 90,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "experimental_method_comparable": True,
                "apo_resolution": 2.0,
                "holo_resolution": 1.8,
                "rmsd": 99.0,
            },
            {
                "protein_id": "P1",
                "apo_pdb_id": "4abc",
                "holo_pdb_id": "3abc",
                "common_mapped_count": 90,
                "sequence_identity": 1.0,
                "construct_mismatch_count": 0,
                "experimental_method_comparable": True,
                "apo_resolution": 2.2,
                "holo_resolution": 2.2,
                "rmsd": 0.001,
            },
        ]
    )
    first = select_primary_pair(pairs)
    second = select_primary_pair(pairs.iloc[::-1].reset_index(drop=True))
    assert first.iloc[0]["apo_pdb_id"] == second.iloc[0]["apo_pdb_id"] == "2abc"
