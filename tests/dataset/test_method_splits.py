from __future__ import annotations

import pandas as pd
import pytest

from dual_uq.dataset.splits import (
    SplitPrerequisiteError,
    build_identity_disjoint_split,
    validate_identity_disjoint_split,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["P1", "P1", "P2", "P3", "P4", "P4"],
            "pair_id": ["pair1", "pair1", "pair2", "pair3", "pair4", "pair4"],
            "uniprot_id": ["U1", "U1", "U2", "U3", "U4", "U4"],
        }
    )


def _clusters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4"],
            "sequence_cluster_id": ["C1", "C2", "C2", "C3"],
            "identity_threshold": [0.30] * 4,
            "cluster_source": ["frozen-test-source"] * 4,
        }
    )


def test_split_keeps_all_states_of_a_pair_together() -> None:
    result = build_identity_disjoint_split(_candidates(), _clusters(), threshold=0.30, seed=20260817)
    validate_identity_disjoint_split(result)
    assert result.status == "READY"
    assert result.assignments.groupby("pair_id")["split"].nunique().max() == 1
    assert result.assignments.groupby("sequence_cluster_id")["split"].nunique().max() == 1


def test_split_is_deterministic_for_same_inputs() -> None:
    first = build_identity_disjoint_split(_candidates(), _clusters(), threshold=0.30, seed=20260817)
    second = build_identity_disjoint_split(_candidates(), _clusters(), threshold=0.30, seed=20260817)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert first.summary == second.summary


def test_split_reports_unresolved_without_authoritative_cluster_source() -> None:
    result = build_identity_disjoint_split(_candidates(), pd.DataFrame(), threshold=0.30, seed=20260817)
    assert result.status == "SPLIT_PREREQUISITE_UNRESOLVED"
    assert result.assignments.empty
    validate_identity_disjoint_split(result)


def test_split_rejects_partial_cluster_coverage() -> None:
    with pytest.raises(SplitPrerequisiteError, match="missing cluster assignments"):
        build_identity_disjoint_split(_candidates(), _clusters().iloc[:2], threshold=0.30, seed=20260817)


def test_split_rejects_conflicting_cluster_assignments() -> None:
    clusters = pd.concat([_clusters(), _clusters().iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitPrerequisiteError, match="duplicate"):
        build_identity_disjoint_split(_candidates(), clusters, threshold=0.30, seed=20260817)
