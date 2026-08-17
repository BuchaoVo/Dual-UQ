import pandas as pd
import pytest

from dual_uq.evaluation.structural_validation_asymmetry import (
    build_structural_validation_asymmetry,
)
from dual_uq.inference.independent_structure_validation import (
    StructuralValidationAsymmetryError,
)


def _fixture():
    generated = pd.DataFrame(
        [
            {
                "protein_id": "protein-000",
                "source_condition": "PDB",
                "sample_index": 0,
                "sample_class": "paired",
                "sequence_hash": "pdb-seq",
                "rmsd_to_conditioning": 0.8,
                "rmsd_to_alternative": 1.0,
            },
            {
                "protein_id": "protein-000",
                "source_condition": "AFDB",
                "sample_index": 0,
                "sample_class": "paired",
                "sequence_hash": "afdb-seq",
                "rmsd_to_conditioning": 0.7,
                "rmsd_to_alternative": 1.1,
            },
        ]
    )
    wt = pd.DataFrame(
        [{
            "protein_id": "protein-000",
            "wt_rmsd_to_pdb": 0.9,
            "wt_rmsd_to_afdb": 0.8,
        }]
    )
    generation = pd.DataFrame([{
        "protein_id": "protein-000",
        "d_pa_excess_independent": 0.2,
        "js_burden_mean": 0.3,
    }])
    remodeling = pd.DataFrame([{
        "protein_id": "protein-000",
        "position_magnitude_upper_tail_excess": 0.4,
    }])
    return generated, wt, generation, remodeling


def test_absolute_groups_and_wt_baseline_use_afdb_preference_sign():
    generated, wt, generation, remodeling = _fixture()
    result = build_structural_validation_asymmetry(
        generated,
        wt,
        generation,
        remodeling,
        expected_protein_count=1,
    )
    row = result.protein_effects.iloc[0]
    assert row["pdb_generated_afdb_preference"] == pytest.approx(-0.2)
    assert row["afdb_generated_afdb_preference"] == pytest.approx(0.4)
    assert row["wt_afdb_preference"] == pytest.approx(0.1)
    assert row["pdb_generated_baseline_adjusted"] == pytest.approx(-0.3)
    assert row["afdb_generated_baseline_adjusted"] == pytest.approx(0.3)


def test_missing_wt_baseline_is_structured_not_zero_filled():
    generated, wt, generation, remodeling = _fixture()
    with pytest.raises(StructuralValidationAsymmetryError, match="WT baseline"):
        build_structural_validation_asymmetry(
            generated,
            wt.iloc[0:0],
            generation,
            remodeling,
            expected_protein_count=1,
        )
