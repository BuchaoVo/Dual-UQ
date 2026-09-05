import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.evaluation.cross_model_representation_cases import (
    apply_exact_se3,
    build_controlled_cross_model_cases,
    build_exact_se3_cases,
    build_identical_input_cases,
    load_full_structcal_cohort,
    make_degenerate_dynamic_case,
)

ROOT = Path(__file__).resolve().parents[2]


def test_controlled_projection_accepts_pair_consensus_and_preserves_canonical_mismatch_provenance() -> None:
    cohort = load_full_structcal_cohort(ROOT, cohort="controlled")
    cases, exclusions = build_controlled_cross_model_cases(
        ROOT, cohort, atom_names=("N", "CA", "C")
    )

    assert exclusions.empty
    assert cases["pair_id"].nunique() == len(cohort) == 567
    assert cases["protein_id"].nunique() == cohort["protein_id"].nunique() == 98

    release = ROOT / "artifacts/releases/structcal_v1"
    mappings = pd.read_parquet(release / "core/residue_mappings.parquet")
    mappings = mappings.loc[mappings["pair_id"].isin(cohort["pair_id"])]
    mappings = mappings.loc[mappings["common_coordinate_visible"].astype(bool)]
    assert mappings["condition_1_aa"].eq(mappings["condition_2_aa"]).all()
    expected = mappings.loc[
        mappings["condition_1_aa"].astype(str).str.upper()
        != mappings["canonical_aa"].astype(str).str.upper()
    ]
    expected_by_pair = {
        str(pair_id): tuple(sorted(group["canonical_position"].astype(int)))
        for pair_id, group in expected.groupby("pair_id", sort=True)
    }
    metadata = cases.drop_duplicates("pair_id").set_index("pair_id")
    observed_pairs = set(
        metadata.index[metadata["canonical_identity_mismatch_count"].astype(int).gt(0)]
    )
    assert observed_pairs == set(expected_by_pair)
    for pair_id, positions in expected_by_pair.items():
        assert tuple(json.loads(metadata.loc[pair_id, "canonical_identity_mismatch_positions_json"])) == positions
        details = json.loads(metadata.loc[pair_id, "canonical_identity_mismatch_details_json"])
        assert {row["canonical_position"] for row in details} == set(positions)
        assert all(row["condition_1_aa"] == row["condition_2_aa"] for row in details)
    assert metadata["sequence_context_basis"].eq("CANONICAL_SEQUENCE").all()


def test_exact_se3_transform_preserves_internal_geometry() -> None:
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[2.0, 1.0, 0.5], [3.0, 1.0, 0.5], [3.0, 2.0, 0.5]],
        ],
        dtype=np.float32,
    )

    transformed = apply_exact_se3(coordinates)

    assert transformed.dtype == coordinates.dtype
    assert not np.array_equal(transformed, coordinates)
    left_distances = np.linalg.norm(
        coordinates.reshape(-1, 3)[:, None] - coordinates.reshape(-1, 3)[None, :],
        axis=-1,
    )
    right_distances = np.linalg.norm(
        transformed.reshape(-1, 3)[:, None] - transformed.reshape(-1, 3)[None, :],
        axis=-1,
    )
    assert np.allclose(left_distances, right_distances, rtol=0.0, atol=1e-6)


def test_dynamicmpnn_single_view_projection_duplicates_only_that_view() -> None:
    coordinates = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    case = make_degenerate_dynamic_case(
        protein_id="P1",
        pair_id="pair-1/condition-1",
        canonical_positions=(1, 2, 3),
        sequence="ACD",
        coordinates=coordinates,
        chain_id="A",
    )

    assert case.coordinates.shape == (3, 2, 3, 3)
    assert np.array_equal(case.coordinates[:, 0], coordinates)
    assert np.array_equal(case.coordinates[:, 1], coordinates)
    assert np.array_equal(case.coordinate_present, np.ones((3, 2), dtype=bool))
    assert case.sequences == ("ACD", "ACD")
    assert case.chain_ids == ("A", "A")
    assert case.structure_sha256[0] == case.structure_sha256[1]


def test_full_frozen_cohorts_preserve_all_splits_and_expected_membership() -> None:
    controlled = load_full_structcal_cohort(ROOT, cohort="controlled")
    operational = load_full_structcal_cohort(ROOT, cohort="track_i")

    assert len(controlled) == 567
    assert controlled["protein_id"].nunique() == 98
    assert controlled["split"].value_counts().to_dict() == {
        "TRAIN": 360,
        "VALIDATION": 102,
        "LOCKED_TEST": 105,
    }
    assert len(operational) == 68
    assert operational["protein_id"].nunique() == 60
    assert operational["pair_id"].is_unique


def test_exact_se3_cases_use_one_reference_pair_per_protein() -> None:
    rows = []
    for pair_id in ("pair-b", "pair-a"):
        for condition in ("CONDITION_1", "CONDITION_2"):
            for index, position in enumerate((1, 2)):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "protein_id": "P1",
                        "condition": condition,
                        "condition_label": condition,
                        "canonical_position": position,
                        "coordinates": (
                            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
                            if index == 0
                            else None
                        ),
                    }
                )

    exact = build_exact_se3_cases(__import__("pandas").DataFrame(rows))

    assert exact["pair_id"].unique().tolist() == ["exact_se3::P1"]
    assert exact["condition"].unique().tolist() == ["CONDITION_1", "CONDITION_2"]
    original = exact.loc[exact["condition"].eq("CONDITION_1"), "coordinates"].dropna().iloc[0]
    transformed = exact.loc[exact["condition"].eq("CONDITION_2"), "coordinates"].dropna().iloc[0]
    assert np.array_equal(np.asarray(transformed), apply_exact_se3(np.asarray(original)))


def test_identical_cases_call_the_same_reference_view_twice_per_protein() -> None:
    rows = []
    coordinates = np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
    for condition in ("CONDITION_1", "CONDITION_2"):
        for index, position in enumerate((1, 2)):
            rows.append(
                {
                    "pair_id": "pair-1",
                    "protein_id": "P1",
                    "condition": condition,
                    "condition_label": condition,
                    "canonical_position": position,
                    "coordinates": coordinates if index == 0 else None,
                }
            )

    identical = build_identical_input_cases(pd.DataFrame(rows))

    assert identical["pair_id"].unique().tolist() == ["identical::P1"]
    assert identical["condition_label"].unique().tolist() == [
        "ORIGINAL",
        "IDENTICAL_REPEAT",
    ]
    left = identical.loc[identical["condition"].eq("CONDITION_1"), "coordinates"].dropna().iloc[0]
    right = identical.loc[identical["condition"].eq("CONDITION_2"), "coordinates"].dropna().iloc[0]
    assert np.array_equal(np.asarray(left), np.asarray(right))
