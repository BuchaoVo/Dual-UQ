from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.structcal_local_response import (
    StructCalLocalResponseError,
    build_local_response_rows,
    js_bits,
    validate_probability_matrix,
)


def test_js_bits_is_symmetric_and_uses_base_two() -> None:
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])

    assert js_bits(left, right) == pytest.approx(1.0)
    assert js_bits(right, left) == pytest.approx(1.0)


def test_probability_matrix_requires_finite_normalized_standard_aa_rows() -> None:
    valid = np.full((2, 20), 1.0 / 20.0)
    result = validate_probability_matrix(valid, label="valid")

    assert result.shape == (2, 20)
    assert np.allclose(result.sum(axis=1), 1.0)

    with pytest.raises(StructCalLocalResponseError, match="normalized"):
        validate_probability_matrix(valid * 2.0, label="not_normalized")


def test_build_local_response_rows_keeps_semantic_metadata_and_pair_mean() -> None:
    positions = (1, 2)
    cases = pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "pair_id": "pair-1",
                "condition": condition,
                "canonical_position": position,
                "canonical_positions": positions,
                "wt_sequence_projection": "AC",
                "identity_cluster_id": "cluster-1",
                "split": "VALIDATION",
                "track_or_diagnostic": "TRACK_I",
                "state_family": None,
            }
            for condition in ("CONDITION_1", "CONDITION_2")
            for position in positions
        ]
    )
    distributions = {
        ("pair-1", "CONDITION_1"): np.vstack(
            [np.eye(20)[0], np.eye(20)[1]]
        ),
        ("pair-1", "CONDITION_2"): np.vstack(
            [np.eye(20)[0], np.eye(20)[2]]
        ),
    }

    residue, pair = build_local_response_rows(
        cases,
        distributions,
        model_id="pifold_official_checkpoint_pth",
        semantic_class="L0",
        probe_semantics="geometry_only",
    )

    assert residue["js_bits"].tolist() == [0.0, 1.0]
    assert pair.loc[0, "n_canonical_positions"] == 2
    assert pair.loc[0, "n_evaluable_positions"] == 2
    assert pair.loc[0, "coverage_fraction"] == pytest.approx(1.0)
    assert pair.loc[0, "R_local_bits"] == pytest.approx(0.5)
    assert set(residue["local_semantics_class"]) == {"L0"}
    assert set(residue["probe_semantics"]) == {"geometry_only"}


def test_build_local_response_rejects_misaligned_probability_axes() -> None:
    cases = pd.DataFrame(
        [
            {"protein_id": "P1", "pair_id": "pair-1", "condition": "A", "canonical_position": 1},
            {"protein_id": "P1", "pair_id": "pair-1", "condition": "B", "canonical_position": 2},
        ]
    )
    distributions = {
        ("pair-1", "A"): np.full((1, 20), 1 / 20),
        ("pair-1", "B"): np.full((1, 20), 1 / 20),
    }

    with pytest.raises(StructCalLocalResponseError, match="condition axes"):
        build_local_response_rows(
            cases,
            distributions,
            model_id="pifold_official_checkpoint_pth",
            semantic_class="L0",
            probe_semantics="geometry_only",
        )


def test_pair_coverage_uses_frozen_canonical_axis_when_some_positions_are_not_evaluable() -> None:
    cases = pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "pair_id": "pair-1",
                "condition": condition,
                "canonical_position": position,
                "n_canonical_positions": 4,
            }
            for condition in ("A", "B")
            for position in (1, 2)
        ]
    )
    uniform = np.full((2, 20), 1 / 20)

    _residue, pair = build_local_response_rows(
        cases,
        {("pair-1", "A"): uniform, ("pair-1", "B"): uniform},
        model_id="pifold_official_checkpoint_pth",
        semantic_class="L0",
        probe_semantics="geometry_only",
    )

    assert pair.loc[0, "n_canonical_positions"] == 4
    assert pair.loc[0, "coverage_fraction"] == pytest.approx(0.5)
