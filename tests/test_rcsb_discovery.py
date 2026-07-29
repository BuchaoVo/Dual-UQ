import pandas as pd

from dual_uq.rcsb_discovery import (
    assign_provisional_strata,
    build_polymer_entity_query,
    select_screening_pool,
)


def test_query_has_expected_return_type() -> None:
    query = build_polymer_entity_query(
        rows=100,
        methods=["X-RAY DIFFRACTION"],
        resolution_max=3.0,
        length_min=100,
        length_max=500,
        sequence_identity_grouping=30,
    )
    assert query["return_type"] == "polymer_entity"
    assert query["request_options"]["paginate"]["rows"] == 100


def test_provisional_strata() -> None:
    table = pd.DataFrame(
        {
            "afdb_global_plddt": [95.0, 70.0, 92.0],
            "length": [200, 220, 400],
        }
    )
    definitions = {
        "high_global_confidence": {
            "global_plddt_min": 90.0,
            "length_max": 320,
        },
        "lower_global_confidence": {"global_plddt_max": 85.0},
        "long_backbone_proxy": {"length_min": 300},
    }
    result = assign_provisional_strata(table, definitions)
    assert result["provisional_stratum"].tolist() == [
        "high_global_confidence",
        "lower_global_confidence",
        "long_backbone_proxy",
    ]


def test_selection_respects_unique_uniprot() -> None:
    table = pd.DataFrame(
        {
            "pdb_id": ["1aaa", "1aab", "1aac", "1aad"],
            "chain_id": ["A", "A", "A", "A"],
            "uniprot_id": ["P1", "P1", "P2", "P3"],
            "length": [150, 160, 350, 250],
            "resolution": [1.0, 1.1, 1.2, 1.3],
            "afdb_global_plddt": [95.0, 94.0, 92.0, 70.0],
            "sequence_cluster": ["30:C1", "30:C1", "30:C2", "30:C3"],
            "organism": ["A", "A", "B", "C"],
            "protein_chain_count": [1, 1, 1, 1],
        }
    )
    definitions = {
        "high_global_confidence": {
            "global_plddt_min": 90.0,
            "length_max": 320,
        },
        "lower_global_confidence": {"global_plddt_max": 85.0},
        "long_backbone_proxy": {"length_min": 300},
    }
    selected = select_screening_pool(
        table,
        quotas={
            "high_global_confidence": 1,
            "lower_global_confidence": 1,
            "long_backbone_proxy": 1,
            "balanced_background": 0,
        },
        definitions=definitions,
        length_bins=[[100, 180], [181, 300], [301, 500]],
        unique_sequence_cluster=True,
        max_per_organism=4,
        seed=1,
        prefer_single_protein_chain=True,
    )
    assert selected["uniprot_id"].is_unique
