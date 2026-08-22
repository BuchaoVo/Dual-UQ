from __future__ import annotations

import pandas as pd
import pytest

from dual_uq.dataset.structcal_release import (
    StructCalReleaseError,
    StructCalSourcePaths,
    _protocol_metadata,
    assign_global_cluster_split,
    audit_public_core_schema,
    build_global_protein_universe,
    normalize_structcal_core_dtypes,
)


def test_split_protocol_generation_names_the_joint_pre_outcome_objective() -> None:
    protocol = _protocol_metadata(
        clustering_protocol={},
        source_paths=StructCalSourcePaths(),
        summary={},
    )["split_protocol"]

    assert protocol["target_fraction_scope"] == "each_normalized_balance_metric"
    assert protocol["optimization_metric_families"] == [
        "protein_count",
        "pair_count",
        "arm:<value>",
        "state_family:<value>",
    ]
    assert protocol["identity_cluster_count_targeted"] is False


def test_global_universe_collapses_cross_arm_protein_identity() -> None:
    arms = {
        "representation_variation": pd.DataFrame(
            [{"protein_id": "P1", "uniprot_id": "P1", "canonical_sequence": "AAAA"}]
        ),
        "ligand_state": pd.DataFrame(
            [
                {"protein_id": "P1", "uniprot_id": "P1", "canonical_sequence": "AAAA"},
                {"protein_id": "P2", "uniprot_id": "P2", "canonical_sequence": "CCCC"},
            ]
        ),
    }

    universe = build_global_protein_universe(arms)

    assert universe["protein_id"].tolist() == ["P1", "P2"]
    assert universe["canonical_length"].tolist() == [4, 4]
    assert universe["canonical_sequence_status"].tolist() == ["STANDARD_20AA", "STANDARD_20AA"]


def test_global_universe_rejects_cross_arm_sequence_conflict() -> None:
    arms = {
        "a": pd.DataFrame(
            [{"protein_id": "P1", "uniprot_id": "P1", "canonical_sequence": "AAAA"}]
        ),
        "b": pd.DataFrame(
            [{"protein_id": "P1", "uniprot_id": "P1", "canonical_sequence": "AAAC"}]
        ),
    }

    with pytest.raises(StructCalReleaseError, match="conflicting canonical sequences"):
        build_global_protein_universe(arms)


def test_global_split_is_deterministic_and_cluster_atomic() -> None:
    proteins = pd.DataFrame(
        {
            "protein_id": [f"P{i}" for i in range(8)],
            "canonical_length": [100, 110, 120, 130, 140, 150, 160, 170],
        }
    )
    clusters = pd.DataFrame(
        {
            "protein_id": [f"P{i}" for i in range(8)],
            "identity_cluster_id": ["C1", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_id": [f"pair-{i}" for i in range(8)],
            "protein_id": [f"P{i}" for i in range(8)],
            "arm": ["representation_variation"] * 2 + ["ligand_state"] * 3 + ["functional_state"] * 3,
            "state_family": [None, None, None, None, None, "OPEN_CLOSED", "PRE_POST", "ACTIVE_INACTIVE"],
        }
    )

    first = assign_global_cluster_split(proteins, clusters, pairs, seed=20260822)
    second = assign_global_cluster_split(proteins, clusters, pairs, seed=20260822)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("identity_cluster_id")["split"].nunique().max() == 1
    assert set(first["split"]) == {"TRAIN", "VALIDATION", "LOCKED_TEST"}
    assert set(first["split_seed"]) == {20260822}


def test_public_core_schema_audit_rejects_hash_model_and_outcome_fields() -> None:
    clean = {"proteins": pd.DataFrame({"protein_id": ["P1"]})}
    audit_public_core_schema(clean)

    for field in ("sequence_sha256", "proteinmpnn_local_eligible", "model_response", "D_excess"):
        dirty = {"proteins": pd.DataFrame({"protein_id": ["P1"], field: [0]})}
        with pytest.raises(StructCalReleaseError, match="forbidden public Core field"):
            audit_public_core_schema(dirty)


def test_core_dtype_normalization_preserves_nullable_scientific_counts() -> None:
    core = {
        "condition_pairs": pd.DataFrame(
            {
                "common_mapped_count": [214.0, None],
                "common_coordinate_visible_count": [207.0, None],
            }
        )
    }

    normalized = normalize_structcal_core_dtypes(core)

    assert str(normalized["condition_pairs"]["common_mapped_count"].dtype) == "Int64"
    assert normalized["condition_pairs"].loc[0, "common_mapped_count"] == 214
    assert pd.isna(normalized["condition_pairs"].loc[1, "common_mapped_count"])
