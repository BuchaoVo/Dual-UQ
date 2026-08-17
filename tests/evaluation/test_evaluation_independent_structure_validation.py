from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.independent_structure_validation import (
    IndependentStructureValidationError,
    build_independent_structure_analysis,
)


def _predictions(wrong_length: bool = False) -> pd.DataFrame:
    target_pdb = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    target_afdb = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.2, 0.0, 0.0], [0.0, 1.8, 0.0]]
    )
    predicted = target_pdb.copy()
    if wrong_length:
        predicted = predicted[:2]
    rows = []
    for condition in ("PDB", "AFDB"):
        rows.append(
            {
                "protein_id": "protein-1",
                "source_condition": condition,
                "sample_index": 0,
                "sample_class": "paired",
                "seed": 0,
                "sequence_hash": f"{condition.lower()}-sequence",
                "sequence": "ACD",
                "predicted_ca_coordinates": predicted.tolist(),
                "target_pdb_ca_coordinates": target_pdb.tolist(),
                "target_afdb_ca_coordinates": target_afdb.tolist(),
                "plddt": 0.8,
                "pae": 2.0,
                "ptm": 0.7,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _summary_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generation = pd.DataFrame(
        {"protein_id": ["protein-1"], "js_burden_mean": [0.2], "d_pa_excess_independent": [0.3]}
    )
    compatibility = pd.DataFrame({"protein_id": ["protein-1"], "delta_cross": [0.4]})
    remodeling = pd.DataFrame(
        {"protein_id": ["protein-1"], "position_magnitude_upper_tail_excess": [0.5]}
    )
    return generation, compatibility, remodeling


def test_preference_uses_the_same_common_mask_for_both_targets() -> None:
    generation, compatibility, remodeling = _summary_inputs()
    result = build_independent_structure_analysis(
        _predictions(), generation, compatibility, remodeling, expected_protein_count=1
    )
    row = result.sequence_comparison.iloc[0]
    assert row["source_preference"] == pytest.approx(
        row["rmsd_to_alternative"] - row["rmsd_to_conditioning"]
    )
    assert row["source_preference"] > 0


def test_wrong_common_mask_length_is_rejected() -> None:
    generation, compatibility, remodeling = _summary_inputs()
    with pytest.raises(IndependentStructureValidationError, match="common mask"):
        build_independent_structure_analysis(
            _predictions(wrong_length=True), generation, compatibility, remodeling, expected_protein_count=1
        )
