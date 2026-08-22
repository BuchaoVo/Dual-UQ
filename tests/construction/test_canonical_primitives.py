from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_uq.construction.admission import AdmissionPolicy, evaluate_pair
from dual_uq.construction.annotations import annotate_pair_geometry
from dual_uq.construction.assets import AssetStatus, resolve_asset
from dual_uq.construction.comparability import compare_mapped_pair, compare_pair
from dual_uq.construction.functional_state import normalize_functional_state
from dual_uq.construction.mapping import (
    CANONICAL_PAIR_MAPPING_COLUMNS,
    COMPARABILITY_MAPPING_COLUMNS,
    map_condition_pair,
)
from dual_uq.construction.proteins import normalize_protein


def test_asset_resolution_reports_operational_status_without_admission():
    result = resolve_asset("missing.cif")
    assert result.status is AssetStatus.UNAVAILABLE
    assert result.admitted is None


def test_protein_normalization_preserves_nonstandard_sequence():
    result = normalize_protein("P1", "ACX")
    assert result["canonical_sequence"] == "ACX"
    assert result["canonical_sequence_status"] == "NONSTANDARD"


def test_mapping_uses_oriented_condition_labels_and_preserves_visibility():
    rows = map_condition_pair(
        protein_id="protein",
        pair_id="pair",
        condition_1_label="OPEN",
        condition_2_label="CLOSED",
        canonical=pd.DataFrame({"canonical_position": [1], "canonical_aa": ["A"]}),
        condition_1=pd.DataFrame({"canonical_position": [1], "residue_id": ["O:1"], "aa": ["A"], "coordinate_visible": [False]}),
        condition_2=pd.DataFrame({"canonical_position": [1], "residue_id": ["C:1"], "aa": ["A"], "coordinate_visible": [True]}),
    )
    assert rows.iloc[0].condition_1_residue_id == "O:1"
    assert not rows.iloc[0].condition_1_coordinate_visible
    assert rows.iloc[0].condition_2_residue_id == "C:1"


def test_mapping_preserves_explicit_missingness_and_insertion_identity():
    rows = map_condition_pair(
        protein_id="protein",
        pair_id="pair",
        condition_1_label="A",
        condition_2_label="B",
        canonical=pd.DataFrame(
            {"canonical_position": [1, 2, 3, 4], "canonical_aa": list("ACDE")}
        ),
        condition_1=pd.DataFrame(
            {
                "canonical_position": [1, 2, 3, 4],
                "residue_id": ["A:10", "A:11A", None, "A:13"],
                "aa": ["A", "C", None, "E"],
                "coordinate_visible": [True, False, False, True],
                "mapped": [True, True, False, True],
                "missing_reason": [None, None, "TRUE_SEQUENCE_GAP", None],
            }
        ),
        condition_2=pd.DataFrame(
            {
                "canonical_position": [1, 2, 3, 4],
                "residue_id": ["B:20", "B:21", "B:22", "B:23"],
                "aa": list("ACDE"),
                "coordinate_visible": [True, True, True, True],
            }
        ),
    )

    assert rows.loc[1, "condition_1_residue_id"] == "A:11A"
    assert bool(rows.loc[2, "condition_1_mapped"]) is False
    assert rows.loc[2, "condition_1_missing_reason"] == "TRUE_SEQUENCE_GAP"
    assert bool(rows.loc[3, "condition_1_mapped"]) is True
    assert bool(rows.loc[3, "condition_1_coordinate_visible"]) is True


def test_mapping_rejects_duplicate_canonical_positions():
    with pytest.raises(ValueError, match="duplicate"):
        map_condition_pair(
            protein_id="protein",
            pair_id="pair",
            condition_1_label="A",
            condition_2_label="B",
            canonical=pd.DataFrame(
                {"canonical_position": [1, 1], "canonical_aa": ["A", "A"]}
            ),
            condition_1=pd.DataFrame(
                {
                    "canonical_position": [1],
                    "residue_id": ["A:1"],
                    "aa": ["A"],
                    "coordinate_visible": [True],
                }
            ),
            condition_2=pd.DataFrame(
                {
                    "canonical_position": [1],
                    "residue_id": ["B:1"],
                    "aa": ["A"],
                    "coordinate_visible": [True],
                }
            ),
        )


def test_comparability_keeps_dimensions_separate():
    result = compare_pair(sequence_identity=0.99, mapping_fraction=0.95, coordinate_fraction=0.70, construct_overlap=True, assembly_comparable=False)
    assert result.sequence_comparable
    assert result.mapping_comparable
    assert not result.coordinate_comparable
    assert not result.assembly_comparable


def test_comparability_derives_pair_facts_from_canonical_mapping():
    mapping = map_condition_pair(
        protein_id="protein",
        pair_id="pair",
        condition_1_label="A",
        condition_2_label="B",
        canonical=pd.DataFrame(
            {"canonical_position": [1, 2, 3, 4], "canonical_aa": list("ACDE")}
        ),
        condition_1=pd.DataFrame(
            {
                "canonical_position": [1, 2, 3, 4],
                "residue_id": ["A:1", "A:2", None, "A:4"],
                "aa": ["A", "X", None, "E"],
                "coordinate_visible": [True, True, False, False],
                "mapped": [True, True, False, True],
                "missing_reason": [None, None, "TRUE_SEQUENCE_GAP", None],
            }
        ),
        condition_2=pd.DataFrame(
            {
                "canonical_position": [1, 2, 3, 4],
                "residue_id": ["B:1", "B:2", "B:3", "B:4"],
                "aa": list("ACDE"),
                "coordinate_visible": [True, True, True, True],
            }
        ),
    )

    facts = compare_mapped_pair(
        mapping=mapping,
        canonical_length=4,
        sequence_identity=1.0,
        construct_overlap_fraction=1.0,
        assembly_comparable=True,
    )

    assert facts.common_mapped_count == 3
    assert facts.common_coordinate_visible_count == 2
    assert facts.mapping_fraction == 0.75
    assert facts.coordinate_fraction == 0.5
    assert facts.coordinate_bearing_mismatch_count == 1
    assert facts.mapping_comparable is False
    assert facts.coordinate_comparable is False


def test_empty_canonical_mapping_keeps_comparability_contract():
    mapping = map_condition_pair(
        protein_id="P0",
        pair_id="pair",
        condition_1_label="A",
        condition_2_label="B",
        canonical=pd.DataFrame(columns=["canonical_position", "canonical_aa"]),
        condition_1=pd.DataFrame(
            columns=["canonical_position", "residue_id", "aa", "coordinate_visible"]
        ),
        condition_2=pd.DataFrame(
            columns=["canonical_position", "residue_id", "aa", "coordinate_visible"]
        ),
    )

    assert tuple(mapping.columns) == CANONICAL_PAIR_MAPPING_COLUMNS
    assert COMPARABILITY_MAPPING_COLUMNS <= set(mapping.columns)
    facts = compare_mapped_pair(
        mapping=mapping,
        canonical_length=0,
        sequence_identity=1.0,
        construct_overlap_fraction=None,
        assembly_comparable=False,
    )

    assert facts.common_mapped_count == 0
    assert facts.mapping_fraction is None


def test_comparability_contract_failure_is_not_pair_unresolved():
    mapping = pd.DataFrame(columns=["common_mapped"])

    with pytest.raises(ValueError, match="canonical pair mapping lacks comparability fields"):
        compare_mapped_pair(
            mapping=mapping,
            canonical_length=0,
            sequence_identity=1.0,
            construct_overlap_fraction=None,
            assembly_comparable=False,
        )


def test_admission_is_outcome_blind():
    decision = evaluate_pair(compare_pair(sequence_identity=1.0, mapping_fraction=1.0, coordinate_fraction=1.0, construct_overlap=True, assembly_comparable=True), AdmissionPolicy())
    assert decision.admitted
    assert decision.coordinate_comparable
    with pytest.raises(TypeError):
        evaluate_pair({}, AdmissionPolicy(), model_score=1.0)


def test_functional_state_policy_normalizes_only_registered_families():
    result = normalize_functional_state("OPEN_CLOSED", "closed")
    assert result == ("OPEN_CLOSED", "CLOSED")
    with pytest.raises(ValueError):
        normalize_functional_state("UNKNOWN", "x")


def test_annotations_use_condition_2_minus_condition_1_for_directional_values():
    result = annotate_pair_geometry(
        "pair",
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert result["pair_id"] == "pair"
    assert result["median_residue_displacement"] == 0.5
