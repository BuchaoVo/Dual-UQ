from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from dual_uq.inference.independent_structure_validation import (
    ESMFoldPrediction,
    StructuralValidationAsymmetryError,
    load_wt_predictions,
    predict_wt_sequences,
    select_wt_sequences,
)


def _projections(count: int = 2):
    rows = {}
    coordinates = np.zeros((3, 4, 3), dtype=float)
    for index in range(count):
        protein_id = f"protein-{index:03d}"
        for condition in ("PDB", "AFDB"):
            rows[(protein_id, condition)] = SimpleNamespace(
                protein_id=protein_id,
                backbone_condition=condition,
                uniprot_positions=(1, 2, 3),
                wt_sequence_projection="ACD",
                structure_sha256=f"{condition.lower()}-{index}",
                coordinates=coordinates,
            )
    return rows


def test_wt_selection_is_one_reference_sequence_per_clean_protein():
    selected = select_wt_sequences(_projections(), expected_protein_count=2)
    assert len(selected) == 2
    assert [row.protein_id for row in selected] == ["protein-000", "protein-001"]
    assert all(row.sequence == "ACD" for row in selected)
    assert all(row.canonical_positions == (1, 2, 3) for row in selected)


def test_wt_selection_rejects_mismatched_paired_projection():
    projections = _projections()
    projections.pop(("protein-000", "AFDB"))
    with pytest.raises(StructuralValidationAsymmetryError, match="paired projection"):
        select_wt_sequences(projections, expected_protein_count=2)


class _FakeAdapter:
    model_identity: ClassVar[dict[str, str]] = {"family": "fake-esmfold"}

    def predict(self, sequence: str) -> ESMFoldPrediction:
        atoms = []
        for index, residue in enumerate(sequence, start=1):
            x = float(index)
            atoms.append(
                f"ATOM  {index:5d}  CA  ALA A{index:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  "
            )
        return ESMFoldPrediction("\n".join(atoms) + "\nEND\n", 90.0, 2.0, 0.8)


def test_wt_prediction_is_immutable_and_resumable(tmp_path):
    projections = _projections()
    requests = select_wt_sequences(projections, expected_protein_count=2)
    first = predict_wt_sequences(
        requests,
        projections,
        _FakeAdapter(),
        output_root=tmp_path,
        resume=False,
    )
    second = predict_wt_sequences(
        requests,
        projections,
        _FakeAdapter(),
        output_root=tmp_path,
        resume=True,
    )
    loaded = load_wt_predictions(tmp_path, expected_protein_count=2)
    assert first == second == loaded
