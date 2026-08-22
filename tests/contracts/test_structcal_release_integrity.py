from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from dual_uq.dataset.structcal_release import validate_materialized_structcal_v1

ROOT = Path(__file__).parents[2]


def test_frozen_structcal_v1_release_round_trips_and_passes_hard_gates() -> None:
    result = validate_materialized_structcal_v1(
        ROOT / "artifacts/releases/structcal_v1",
        schema_root=ROOT / "schemas",
    )

    assert result["release_id"] == "structcal_v1"
    assert result["protein_count"] == 1897
    assert result["pair_count"] == 2545
    assert result["identity_cluster_leakage"] == 0
    assert result["outcome_leakage"] == 0
    assert result["protocol_complete"] is True


def test_frozen_split_contract_names_the_actual_joint_optimization_target() -> None:
    release = ROOT / "artifacts/releases/structcal_v1"
    protocol = json.loads(
        (release / "metadata/split_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["assignment_unit"] == "identity_cluster_30"
    assert protocol["target_fraction_scope"] == "each_normalized_balance_metric"
    assert protocol["optimization_metric_families"] == [
        "protein_count",
        "pair_count",
        "arm:<value>",
        "state_family:<value>",
    ]
    assert protocol["identity_cluster_count_targeted"] is False
    assert protocol["seed_search"] is False
    assert protocol["outcome_attributes_used"] == []

    design = (ROOT / "docs/scientific/BENCHMARK_DESIGN_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "Identity-cluster count is not an optimization target" in design
    assert "127 structural pairs covering 113 canonical proteins" in design
    assert "567 pair-level benchmark observations" in design


def test_frozen_pair_protein_cluster_and_controlled_hierarchies() -> None:
    release = ROOT / "artifacts/releases/structcal_v1"
    pairs = pd.read_parquet(release / "core/condition_pairs.parquet")
    splits = pd.read_parquet(release / "core/splits.parquet")
    track_i = pd.read_parquet(release / "tracks/track_i_invariance.parquet")
    track_ii = pd.read_parquet(release / "tracks/track_ii_sensitivity.parquet")
    joined = pairs.merge(
        splits[["protein_id", "identity_cluster_id", "split"]],
        on="protein_id",
        validate="many_to_one",
    )

    assert len(splits) == splits["protein_id"].nunique() == 1897
    assert splits["identity_cluster_id"].notna().all()
    assert splits.groupby("identity_cluster_id")["split"].nunique().max() == 1
    assert set(pairs["protein_id"]) <= set(splits["protein_id"])
    assert (
        len(joined),
        joined["protein_id"].nunique(),
        joined["identity_cluster_id"].nunique(),
    ) == (2545, 1897, 1350)

    expected_arm_counts = {
        "representation_variation": (127, 113, 107),
        "ligand_state": (1410, 1410, 1014),
        "functional_state": (441, 413, 283),
        "controlled_perturbation": (567, 98, 98),
    }
    for arm, expected in expected_arm_counts.items():
        frame = joined.loc[joined["arm"].eq(arm)]
        assert (
            len(frame),
            frame["protein_id"].nunique(),
            frame["identity_cluster_id"].nunique(),
        ) == expected

    expected_split_counts = {
        "TRAIN": (1164, 1349, 1739),
        "VALIDATION": (92, 276, 400),
        "LOCKED_TEST": (94, 272, 406),
    }
    for split, expected in expected_split_counts.items():
        assert (
            splits.loc[splits["split"].eq(split), "identity_cluster_id"].nunique(),
            splits.loc[splits["split"].eq(split), "protein_id"].nunique(),
            len(joined.loc[joined["split"].eq(split)]),
        ) == expected

    representation = joined.loc[joined["arm"].eq("representation_variation")]
    assert (len(representation), representation["protein_id"].nunique()) == (127, 113)
    multiplicity = representation.groupby("protein_id").size()
    assert multiplicity.value_counts().sort_index().to_dict() == {1: 99, 2: 14}
    assert int(multiplicity.max()) == 2
    assert (
        len(track_i),
        track_i["protein_id"].nunique(),
        track_i["identity_cluster_id"].nunique(),
    ) == (68, 60, 57)
    assert (
        len(track_ii),
        track_ii["protein_id"].nunique(),
        track_ii["identity_cluster_id"].nunique(),
    ) == (1851, 1794, 1257)

    controlled_root = (
        ROOT
        / "experiments/interventions/controlled_perturbations"
        / "relational_geometry_confirmatory/geometry"
    )
    selected = pd.read_parquet(controlled_root / "selected_instances.parquet")
    strata = selected.groupby(["parent_structure_id", "requested_dose"])
    expected_tiers = {"LOW_RELATIONAL", "MEDIUM_RELATIONAL", "HIGH_RELATIONAL"}

    assert len(selected) == 567
    assert selected["parent_structure_id"].nunique() == 98
    assert strata.ngroups == 189
    assert all(len(frame) == 3 for _, frame in strata)
    assert all(set(frame["relational_tier"]) == expected_tiers for _, frame in strata)
    assert len(selected) == strata.ngroups * 3


def test_frozen_public_core_keeps_split_and_model_fields_separate() -> None:
    release = ROOT / "artifacts/releases/structcal_v1"
    core_names = (
        "proteins",
        "structures",
        "condition_pairs",
        "residue_mappings",
        "benchmark_instances",
        "splits",
    )
    forbidden = re.compile(
        r"(^|_)(sha256|hash|checksum)($|_)|^proteinmpnn_|^esm_if1_|"
        r"^(model_score|model_response|R_local|D_excess|J_full)$",
        re.IGNORECASE,
    )

    for name in core_names:
        frame = pd.read_parquet(release / "core" / f"{name}.parquet")
        assert not [column for column in frame.columns if forbidden.search(column)]
        if name != "splits":
            assert "split" not in frame.columns
            assert "identity_cluster_id" not in frame.columns
