from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.esm_if1_generalization import (
    ESMIF1ContractError,
    compare_local_response_with_proteinmpnn,
    find_true_gap_proteins,
    generate_esm_if1_ensemble,
    summarize_generation_propagation,
    teacher_forced_local_response,
    validate_common_backbone,
    validate_contiguous_common_positions,
    validate_gap_preserving_backbone,
)
from dual_uq.models.esm_if1 import ESMIF1Adapter, FakeESMIF1Adapter


def _coords(offset: float = 0.0, length: int = 4) -> np.ndarray:
    return np.asarray(
        [[[float(i) + offset, 0.0, 0.0], [float(i) + offset, 1.0, 0.0], [float(i) + offset, 1.0, 1.0]] for i in range(length)],
        dtype=np.float32,
    )


def fixture_cases() -> pd.DataFrame:
    rows = []
    for protein_id, sequence in (("p1", "ACDE"), ("p2", "FGHI")):
        for condition, offset in (("PDB", 0.0), ("AFDB", 0.2)):
            for position in range(1, 5):
                rows.append(
                    {
                        "protein_id": protein_id,
                        "condition": condition,
                        "position": position,
                        "wt_sequence": sequence,
                        "coordinates": _coords(offset),
                    }
                )
    return pd.DataFrame(rows)


def test_local_response_uses_identical_wt_context_and_common_positions() -> None:
    result = teacher_forced_local_response(FakeESMIF1Adapter(implementation_revision="r", checkpoint_sha256="a" * 64), fixture_cases())
    assert set(result["condition"]) == {"PDB", "AFDB"}
    assert result[["protein_id", "condition", "position"]].duplicated().sum() == 0
    assert set(result["primary_metric"]) == {"js_bits"}
    assert result["js_bits"].notna().all()


def test_backbone_coordinate_rows_must_pair_one_to_one() -> None:
    cases = fixture_cases()
    cases = pd.concat([cases, cases.iloc[[0]]], ignore_index=True)
    with pytest.raises(ESMIF1ContractError, match="one-to-one"):
        validate_common_backbone(cases)


def test_true_uniprot_gaps_are_not_compressed_into_a_chain() -> None:
    with pytest.raises(ESMIF1ContractError, match="cannot be compressed"):
        validate_contiguous_common_positions((1, 2, 4))


def test_true_gap_detection_is_outcome_blind_and_position_based() -> None:
    masks = pd.DataFrame(
        {
            "protein_id": ["gap", "gap", "gap", "clean", "clean", "clean"],
            "canonical_position": [1, 3, 4, 1, 2, 3],
            "common_mask": [True, True, True, True, True, True],
        }
    )
    assert find_true_gap_proteins(masks) == ("gap",)


def _gap_fixture() -> pd.DataFrame:
    coordinates = _coords(length=4)
    coordinates[1] = np.nan
    rows = []
    for condition in ("PDB", "AFDB"):
        for position in range(1, 5):
            rows.append(
                {
                    "protein_id": "gap",
                    "condition": condition,
                    "position": position,
                    "wt_sequence": "ACDE",
                    "coordinates": coordinates,
                    "canonical_positions": (1, 2, 3, 4),
                    "comparable": position != 2,
                    "true_gap": position == 2,
                }
            )
    return pd.DataFrame(rows)


def test_gap_preserving_cases_keep_canonical_axis_and_missing_row() -> None:
    cases = _gap_fixture()
    validate_gap_preserving_backbone(cases)
    adapter = FakeESMIF1Adapter(implementation_revision="r", checkpoint_sha256="a" * 64)
    local = teacher_forced_local_response(
        adapter,
        cases,
        allow_missing_coordinates=True,
        comparable_only=True,
    )
    assert set(local["position"]) == {1, 3, 4}
    generated = generate_esm_if1_ensemble(
        adapter,
        cases,
        n_samples=2,
        temperature=0.1,
        seed_namespace="gap",
        allow_missing_coordinates=True,
    )
    assert set(generated["n_residues"]) == {4}
    assert all(tuple(row) == (1, 3, 4) for row in generated["comparable_positions"])
    protein, position, _ = summarize_generation_propagation(generated)
    assert int(protein.loc[0, "comparable_position_count"]) == 3
    assert set(position["position"]) == {1, 3, 4}


def test_finite_esm_adapter_rejects_missing_coordinate_rows() -> None:
    with pytest.raises(ESMIF1ContractError, match="must be finite"):
        ESMIF1Adapter._validate_coordinates(np.full((2, 3, 3), np.nan))


def test_cross_model_local_association_is_protein_level_only() -> None:
    esm = pd.DataFrame({"protein_id": ["p1", "p2", "p3"], "esm_if1_local_burden": [0.1, 0.2, 0.3]})
    mpnn = pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3"],
            "magnitude_p": [1.0, 2.0, 3.0],
            "rank_displacement": [0.3, 0.2, 0.1],
            "breadth_b": [0.1, 0.2, 0.3],
        }
    )
    result = compare_local_response_with_proteinmpnn(esm, mpnn)
    assert set(result["protein_id"]) == {"p1", "p2", "p3"}
    assert "spearman" not in result.columns
    assert "spearman" in result.attrs


def test_generation_uses_nonoverlapping_condition_seed_domains() -> None:
    table = generate_esm_if1_ensemble(
        FakeESMIF1Adapter(implementation_revision="r", checkpoint_sha256="a" * 64),
        fixture_cases(),
        n_samples=4,
        temperature=0.1,
        seed_namespace="test",
    )
    assert len(table) == 2 * 2 * 4
    pdb_seeds = set(table.loc[table.condition == "PDB", "seed"])
    afdb_seeds = set(table.loc[table.condition == "AFDB", "seed"])
    assert pdb_seeds.isdisjoint(afdb_seeds)


def _generated_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"protein_id": "p1", "condition": condition, "sample_index": i, "seed": i, "sequence": seq}
            for condition, sequences in (("PDB", ("AAAA", "AAAC", "AACC", "ACCC")), ("AFDB", ("CCCC", "CCCG", "CCGG", "CGGG")))
            for i, seq in enumerate(sequences)
        ]
    )


def test_propagation_reports_d_excess_and_position_js() -> None:
    protein, position, summary = summarize_generation_propagation(_generated_fixture())
    assert {"d_pp", "d_aa", "d_pa", "d_excess"}.issubset(protein.columns)
    assert "js_burden" in protein.columns
    assert len(position) == 4
    assert summary["protein_unit"] == "protein"
