from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dual_uq.models.dynamicmpnn import (
    DynamicMPNNInputCase,
    DynamicMPNNInputError,
    build_natural_decoding_order,
    teacher_forcing_edge_context,
    validate_dynamic_input,
)


def _case(**changes: object) -> DynamicMPNNInputCase:
    coords = np.zeros((6, 2, 3, 3), dtype=np.float32)
    coords[2, 1] = np.nan
    values: dict[str, object] = {
        "protein_id": "fixture_A__P00001",
        "pair_id": "fixture_pair",
        "canonical_positions": tuple(range(1, 7)),
        "sequences": ("ACDEFG", "ACDEFG"),
        "coordinates": coords,
        "coordinate_present": np.isfinite(coords).all(axis=(2, 3)),
        "chain_ids": ("A", "A"),
        "structure_sha256": ("1" * 64, "2" * 64),
    }
    values.update(changes)
    return DynamicMPNNInputCase(**values)  # type: ignore[arg-type]


def test_input_case_is_immutable_and_preserves_two_state_axis() -> None:
    case = _case()

    assert case.residue_count == 6
    assert case.coordinates.shape == (6, 2, 3, 3)
    assert case.coordinate_present.shape == (6, 2)
    assert not case.coordinate_present[2, 1]
    with pytest.raises(FrozenInstanceError):
        case.protein_id = "other"  # type: ignore[misc]


def test_validation_accepts_missing_coordinates_without_imputation() -> None:
    result = validate_dynamic_input(_case())

    assert result.evaluable is True
    assert result.reason is None
    assert result.residue_count == 6
    assert result.missing_coordinate_count == 1


def test_validation_rejects_nonstandard_sequence() -> None:
    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(sequences=("ACXEFG", "ACXEFG")))

    assert exc_info.value.reason == "nonstandard_sequence"


def test_validation_rejects_axis_or_state_mismatch() -> None:
    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(canonical_positions=(1, 2, 4, 5, 6, 7)))

    assert exc_info.value.reason == "canonical_axis_mismatch"

    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(sequences=("ACDEFG", "ACDEFA")))

    assert exc_info.value.reason == "conformation_sequence_mismatch"


def test_validation_rejects_coordinate_presence_claim_that_drops_state() -> None:
    present = np.ones((6, 2), dtype=bool)
    present[:, 1] = False

    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(coordinate_present=present))

    assert exc_info.value.reason == "conformation_coordinate_state_missing"


def test_validation_rejects_both_state_coordinate_gap_without_compression() -> None:
    coords = _case().coordinates.copy()
    coords[1, :, :, :] = np.nan
    present = np.isfinite(coords).all(axis=(2, 3))

    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(coordinates=coords, coordinate_present=present))

    assert exc_info.value.reason == "both_conformations_missing_coordinate"


def test_validation_rejects_nonstandard_coordinate_shape() -> None:
    with pytest.raises(DynamicMPNNInputError) as exc_info:
        validate_dynamic_input(_case(coordinates=np.zeros((6, 2, 4, 3))))

    assert exc_info.value.reason == "invalid_backbone_coordinates"


def test_natural_decoding_order_is_explicit_and_stable() -> None:
    assert build_natural_decoding_order(4) == (0, 1, 2, 3)


def test_teacher_forcing_edge_context_excludes_future_tokens() -> None:
    # Edge convention is source -> destination; src < dst is the causal prefix.
    edge_index = np.asarray([[0, 3, 1, 2], [1, 0, 3, 2]], dtype=np.int64)
    sequence_a = np.asarray([0, 1, 2, 3], dtype=np.int64)
    sequence_b = np.asarray([0, 1, 9, 8], dtype=np.int64)
    context_a = teacher_forcing_edge_context(sequence_a, edge_index)
    context_b = teacher_forcing_edge_context(sequence_b, edge_index)

    # The prefix edge 0 -> 1 retains its token; the future edge 3 -> 0 is masked.
    assert context_a[0] == context_b[0] == 0
    assert context_a[1] == context_b[1] == -1


def test_teacher_forcing_edge_context_responds_to_prefix_tokens() -> None:
    edge_index = np.asarray([[0], [3]], dtype=np.int64)
    context_a = teacher_forcing_edge_context(np.asarray([0, 1, 2, 3]), edge_index)
    context_b = teacher_forcing_edge_context(np.asarray([9, 1, 2, 3]), edge_index)
    assert context_a[0] == 0
    assert context_b[0] == 9
