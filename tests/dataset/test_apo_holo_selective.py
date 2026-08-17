from __future__ import annotations

import pandas as pd

from dual_uq.dataset.apo_holo import LigandClass, StateLabel
from dual_uq.dataset.apo_holo_selective import (
    assess_entity_asset_readiness,
    bind_shared_assets_to_entities,
    build_component_reference,
    build_metadata_pairs,
    reduce_non_dominated_pairs,
    resolve_structure_states,
)


def test_structure_state_resolution_requires_complete_inventory() -> None:
    census = pd.DataFrame(
        [
            {
                "polymer_entity_id": "1abc_1",
                "pdb_id": "1abc",
                "uniprot_id": "P1",
                "nonpolymer_entity_count": 0,
                "nonpolymer_entity_ids": "",
            },
            {
                "polymer_entity_id": "2abc_1",
                "pdb_id": "2abc",
                "uniprot_id": "P1",
                "nonpolymer_entity_count": 1,
                "nonpolymer_entity_ids": "2",
            },
            {
                "polymer_entity_id": "3abc_1",
                "pdb_id": "3abc",
                "uniprot_id": "P1",
                "nonpolymer_entity_count": 1,
                "nonpolymer_entity_ids": "3",
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "pdb_id": "2abc",
                "nonpolymer_entity_id": "2",
                "ccd_id": "ATP",
                "chemical_name": "adenosine triphosphate",
                "ligand_class": LigandClass.BIOLOGICAL_SMALL_MOLECULE.value,
                "annotation_status": "complete",
            },
            {
                "pdb_id": "3abc",
                "nonpolymer_entity_id": "3",
                "ccd_id": "UNK",
                "chemical_name": None,
                "ligand_class": LigandClass.AMBIGUOUS.value,
                "annotation_status": "complete",
            },
        ]
    )
    states = resolve_structure_states(census, components)
    by_id = states.set_index("polymer_entity_id")
    assert by_id.loc["1abc_1", "state_label"] == StateLabel.APO.value
    assert by_id.loc["2abc_1", "state_label"] == StateLabel.HOLO.value
    assert by_id.loc["3abc_1", "state_label"] == StateLabel.UNRESOLVED.value


def test_component_reference_is_deduplicated_and_classified() -> None:
    raw = pd.DataFrame(
        [
            {"pdb_id": "1abc", "nonpolymer_entity_id": "2", "ccd_id": "ATP", "chemical_name": "ATP", "heavy_atom_count": 31},
            {"pdb_id": "2abc", "nonpolymer_entity_id": "4", "ccd_id": "ATP", "chemical_name": "ATP", "heavy_atom_count": 31},
        ]
    )
    reference = build_component_reference(raw)
    assert len(reference) == 1
    assert reference.iloc[0]["ccd_id"] == "ATP"
    assert reference.iloc[0]["ligand_class"] == LigandClass.BIOLOGICAL_SMALL_MOLECULE.value


def test_pair_reduction_is_outcome_blind_and_keeps_non_dominated_pairs() -> None:
    structures = pd.DataFrame(
        [
            {"polymer_entity_id": "apo1", "pdb_id": "1apo", "uniprot_id": "P1", "state_label": "apo", "length": 100, "sequence": "A" * 100, "experimental_method": "X-ray", "resolution": 1.5, "assembly_ids": "1"},
            {"polymer_entity_id": "apo2", "pdb_id": "2apo", "uniprot_id": "P1", "state_label": "apo", "length": 100, "sequence": "A" * 100, "experimental_method": "X-ray", "resolution": 2.0, "assembly_ids": "1"},
            {"polymer_entity_id": "holo1", "pdb_id": "1hol", "uniprot_id": "P1", "state_label": "holo", "length": 100, "sequence": "A" * 100, "experimental_method": "X-ray", "resolution": 1.5, "assembly_ids": "1"},
            {"polymer_entity_id": "holo2", "pdb_id": "2hol", "uniprot_id": "P1", "state_label": "holo", "length": 100, "sequence": "A" * 100, "experimental_method": "X-ray", "resolution": 3.0, "assembly_ids": "2"},
        ]
    )
    pairs = build_metadata_pairs(structures)
    reduced = reduce_non_dominated_pairs(pairs)
    assert "1apo_1hol__P1" in set(reduced["pair_id"])
    assert "2apo_2hol__P1" not in set(reduced["pair_id"])
    assert reduced["protein_id"].nunique() == 1


def test_shared_asset_binding_and_readiness_are_entity_explicit() -> None:
    structures = pd.DataFrame(
        [{
            "polymer_entity_id": "1abc_1",
            "pdb_id": "1abc",
            "uniprot_id": "P1",
            "nonpolymer_entity_ids": "2",
        }]
    )
    components = pd.DataFrame(
        [{"pdb_id": "1abc", "nonpolymer_entity_id": "2", "ccd_id": "ATP"}]
    )
    ledger = pd.DataFrame(
        [
            {"asset_type": "pdb_mmcif", "polymer_entity_id": "1abc_1", "retrieval_status": "VALID"},
            {"asset_type": "sifts_xml", "polymer_entity_id": "1abc_1", "retrieval_status": "VALID"},
            {"asset_type": "uniprot_canonical_json", "uniprot_id": "P1", "retrieval_status": "VALID"},
            {"asset_type": "ccd_json", "ccd_id": "ATP", "retrieval_status": "VALID"},
        ]
    )
    bound = bind_shared_assets_to_entities(ledger, structures, components)
    ready = assess_entity_asset_readiness(structures, bound, components)
    assert ready.iloc[0]["raw_asset_status"] == "raw_valid"
