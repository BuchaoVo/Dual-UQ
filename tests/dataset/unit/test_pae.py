from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.core.errors import PAEMappingError
from dual_uq.dataset.models import AFDBFragment, PAEMatrix
from dual_uq.dataset.services.afdb import (
    extract_mapped_pae,
    load_pae_json,
    map_output_residues_to_pae,
    map_uniprot_to_fragment_index,
    require_fragment_coverage,
    summarize_long_range_pae,
    validate_pae_matrix,
)


def _fragment(
    *,
    model_id: str = "AF-PTEST-F2",
    start: int = 101,
    end: int = 104,
) -> AFDBFragment:
    return AFDBFragment(
        model_entity_id=model_id,
        uniprot_start=start,
        uniprot_end=end,
        model_residue_count=end - start + 1,
    )


def _mapping(
    output_positions: list[int] | None = None,
    uniprot_positions: list[int] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "output_position": output_positions or [1, 2, 3, 4],
            "uniprot_position": uniprot_positions or [101, 102, 103, 104],
        }
    )


def _matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 0.0, 5.0, 6.0],
            [7.0, 8.0, 0.0, 9.0],
            [10.0, 11.0, 12.0, 0.0],
        ]
    )


def test_validates_a_square_non_negative_finite_matrix() -> None:
    result = validate_pae_matrix(_matrix(), expected_size=4)

    assert result.shape == (4, 4)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, _matrix())


@pytest.mark.parametrize(
    "values",
    [
        np.ones(4),
        np.ones((2, 2, 1)),
        np.ones((2, 3)),
        [[0.0, 1.0], [2.0]],
        np.empty((0, 0)),
    ],
)
def test_rejects_wrong_or_ragged_pae_dimensions(values: object) -> None:
    with pytest.raises(PAEMappingError, match="square") as caught:
        validate_pae_matrix(values, expected_size=2)

    assert caught.value.code == "invalid_pae_matrix"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_pae_values(value: float) -> None:
    matrix = _matrix()
    matrix[0, 1] = value

    with pytest.raises(PAEMappingError, match="finite") as caught:
        validate_pae_matrix(matrix, expected_size=4)

    assert caught.value.code == "invalid_pae_matrix"


def test_rejects_negative_pae_values() -> None:
    matrix = _matrix()
    matrix[0, 1] = -0.01

    with pytest.raises(PAEMappingError, match="non-negative") as caught:
        validate_pae_matrix(matrix, expected_size=4)

    assert caught.value.code == "invalid_pae_matrix"


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[False, True], [True, False]]),
        np.asarray([["0", "1"], ["1", "0"]]),
        np.asarray([[0j, 1j], [1j, 0j]]),
    ],
)
def test_rejects_non_real_numeric_pae_dtype(values: np.ndarray) -> None:
    with pytest.raises(PAEMappingError, match="numeric") as caught:
        validate_pae_matrix(values, expected_size=2)

    assert caught.value.code == "invalid_pae_matrix"


def test_rejects_pae_fragment_length_mismatch() -> None:
    with pytest.raises(PAEMappingError, match="fragment") as caught:
        validate_pae_matrix(np.zeros((3, 3)), expected_size=4)

    assert caught.value.code == "pae_fragment_length_mismatch"


def test_loads_audited_afdb_pae_json_schema(tmp_path: Path) -> None:
    path = tmp_path / "pae.json"
    path.write_text(
        json.dumps(
            [
                {
                    "predicted_aligned_error": _matrix().tolist(),
                    "max_predicted_aligned_error": 12.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = load_pae_json(path, _fragment())

    assert result.model_entity_id == "AF-PTEST-F2"
    assert result.matrix_size == 4
    np.testing.assert_array_equal(result.values, _matrix())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        [{"pae": [[0.0]]}],
        [
            {"predicted_aligned_error": [[0.0]]},
            {"predicted_aligned_error": [[0.0]]},
        ],
    ],
)
def test_rejects_non_audited_pae_json_schema(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "pae.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PAEMappingError, match="schema") as caught:
        load_pae_json(path, _fragment(start=1, end=1))

    assert caught.value.code == "invalid_pae_json_schema"


def test_maps_first_and_last_fragment_residue_to_boundary_indices() -> None:
    fragment = _fragment(start=101, end=104)

    assert map_uniprot_to_fragment_index(fragment, 101) == 0
    assert map_uniprot_to_fragment_index(fragment, 104) == 3


@pytest.mark.parametrize("position", [100, 105])
def test_residue_outside_fragment_interval_fails(position: int) -> None:
    with pytest.raises(PAEMappingError, match="outside") as caught:
        map_uniprot_to_fragment_index(_fragment(), position)

    assert caught.value.code == "unsupported_afdb_fragment"


def test_mapping_preserves_output_uniprot_and_pae_coordinates() -> None:
    result = map_output_residues_to_pae(_mapping(), _fragment())

    assert result.output_positions == (1, 2, 3, 4)
    assert result.uniprot_positions == (101, 102, 103, 104)
    assert result.model_residue_positions == (1, 2, 3, 4)
    assert result.pae_indices == (0, 1, 2, 3)


def test_duplicate_uniprot_mapping_fails() -> None:
    table = _mapping(uniprot_positions=[101, 102, 102, 104])

    with pytest.raises(PAEMappingError, match="UniProt") as caught:
        map_output_residues_to_pae(table, _fragment())

    assert caught.value.code == "ambiguous_residue_mapping"


def test_duplicate_output_position_fails() -> None:
    table = _mapping(output_positions=[1, 2, 2, 4])

    with pytest.raises(PAEMappingError, match="output") as caught:
        map_output_residues_to_pae(table, _fragment())

    assert caught.value.code == "ambiguous_residue_mapping"


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("output_position", [1, 2.5, 3, 4]),
        ("uniprot_position", [101, np.nan, 103, 104]),
        ("uniprot_position", [101, 0, 103, 104]),
    ],
)
def test_mapping_positions_must_be_positive_integers(
    column: str, values: list[float]
) -> None:
    table = _mapping()
    table[column] = values

    with pytest.raises(PAEMappingError, match="positive integers") as caught:
        map_output_residues_to_pae(table, _fragment())

    assert caught.value.code == "invalid_residue_mapping"


def test_mapping_order_does_not_change_indices_values_or_summary() -> None:
    fragment = _fragment()
    matrix = PAEMatrix(
        model_entity_id=fragment.model_entity_id,
        values=validate_pae_matrix(_matrix(), expected_size=4),
    )
    forward = map_output_residues_to_pae(_mapping(), fragment)
    shuffled = map_output_residues_to_pae(
        _mapping().sample(frac=1.0, random_state=7), fragment
    )

    assert shuffled == forward
    forward_values = extract_mapped_pae(matrix, forward)
    shuffled_values = extract_mapped_pae(matrix, shuffled)
    np.testing.assert_array_equal(shuffled_values, forward_values)
    assert summarize_long_range_pae(
        shuffled, shuffled_values, min_sequence_separation=2
    ) == summarize_long_range_pae(
        forward, forward_values, min_sequence_separation=2
    )


def test_extract_rejects_pae_from_a_different_model_identity() -> None:
    mapping = map_output_residues_to_pae(_mapping(), _fragment())
    wrong_model = PAEMatrix(model_entity_id="AF-PTEST-F3", values=_matrix())

    with pytest.raises(PAEMappingError, match="identity") as caught:
        extract_mapped_pae(wrong_model, mapping)

    assert caught.value.code == "pae_model_identity_mismatch"


def test_extract_rejects_bare_matrix_without_model_identity() -> None:
    mapping = map_output_residues_to_pae(_mapping(), _fragment())

    with pytest.raises(PAEMappingError, match="identity") as caught:
        extract_mapped_pae(_matrix(), mapping)

    assert caught.value.code == "missing_pae_model_identity"


def test_does_not_guess_offset_from_local_or_observed_numbering() -> None:
    fragment = _fragment(start=101, end=104)
    local_numbered_rows = _mapping(uniprot_positions=[1, 2, 3, 4])

    with pytest.raises(PAEMappingError) as caught:
        map_output_residues_to_pae(local_numbered_rows, fragment)

    assert caught.value.code == "unsupported_afdb_fragment"


def test_fragment_exactly_covers_target_interval() -> None:
    require_fragment_coverage(_fragment(start=101, end=104), (101, 104))


@pytest.mark.parametrize(
    ("fragment", "target"),
    [
        (_fragment(start=101, end=103), (101, 104)),
        (_fragment(start=102, end=104), (101, 104)),
        (_fragment(model_id="AF-PTEST-F1", start=1, end=100), (101, 104)),
    ],
)
def test_partial_or_absent_fragment_coverage_fails_without_fallback(
    fragment: AFDBFragment, target: tuple[int, int]
) -> None:
    with pytest.raises(PAEMappingError) as caught:
        require_fragment_coverage(fragment, target)

    assert caught.value.code == "unsupported_afdb_fragment"


def test_index9_non_overlapping_selected_fragment_is_unsupported_not_numbering() -> None:
    selected_fragment = _fragment(
        model_id="AF-0000000365840311", start=1368, end=1493
    )

    with pytest.raises(PAEMappingError) as caught:
        require_fragment_coverage(selected_fragment, (1024, 1192))

    assert caught.value.code == "unsupported_afdb_fragment"
    assert "number" not in str(caught.value).lower()


def test_index9_partial_overlap_is_still_unsupported() -> None:
    partial_fragment = _fragment(
        model_id="AF-0000000365840308", start=880, end=1050
    )

    with pytest.raises(PAEMappingError) as caught:
        require_fragment_coverage(partial_fragment, (1024, 1192))

    assert caught.value.code == "unsupported_afdb_fragment"


def test_long_range_summary_uses_uniprot_separation_and_includes_boundary() -> None:
    mapping = map_output_residues_to_pae(
        _mapping(
            output_positions=[1, 2, 3, 4],
            uniprot_positions=[101, 124, 125, 150],
        ),
        _fragment(start=101, end=150),
    )
    mapped_pae = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 0.0, 5.0, 6.0],
            [7.0, 8.0, 0.0, 9.0],
            [10.0, 11.0, 12.0, 0.0],
        ]
    )

    result = summarize_long_range_pae(
        mapping, mapped_pae, min_sequence_separation=24
    )

    # Unique upper-triangle pairs at separations 24, 49, 26, and 25.
    assert result.count == 4
    # Existing diagnostics summarize symmetric PAE: (PAE[i,j] + PAE[j,i]) / 2.
    assert result.mean == pytest.approx(7.5)
    assert result.median == pytest.approx(7.5)
    assert result.maximum == pytest.approx(10.5)


def test_long_range_summary_is_deterministic_when_no_pairs_qualify() -> None:
    mapping = map_output_residues_to_pae(_mapping(), _fragment())

    result = summarize_long_range_pae(
        mapping, _matrix(), min_sequence_separation=100
    )

    assert result.count == 0
    assert result.mean is None
    assert result.median is None
    assert result.maximum is None
