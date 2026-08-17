from __future__ import annotations

from pathlib import Path

import pytest

from dual_uq.inference.independent_structure_validation import (
    VALIDATION_SAMPLE_INDICES,
    IndependentStructureValidationError,
    select_validation_records,
)
from dual_uq.models.proteinmpnn import sequence_sha256
from dual_uq.models.proteinmpnn_generation import (
    GeneratedSequenceRecord,
    GenerationRequest,
    generation_seed_plan,
)
from dual_uq.models.scoring import ScorerBinding


def _records() -> tuple[GeneratedSequenceRecord, ...]:
    binding = ScorerBinding("ProteinMPNN", "fixture", "1" * 64, "generation")
    rows: list[GeneratedSequenceRecord] = []
    for protein_index in range(68):
        protein_id = f"protein-{protein_index:03d}"
        for condition in ("PDB", "AFDB"):
            for spec in generation_seed_plan():
                sequence = "ACD"
                request = GenerationRequest(
                    protein_id=protein_id,
                    backbone_condition=condition,
                    structure_sha256=("1" if condition == "PDB" else "2") * 64,
                    canonical_positions=(1, 2, 3),
                    wt_sequence_projection="AAA",
                    temperature=0.1,
                    sample_index=spec.sample_index,
                    seed=spec.pdb_seed if condition == "PDB" else spec.afdb_seed,
                    sample_class=spec.sample_class,
                    decoding_realization=(spec.sample_index + 1).to_bytes(32, "big").hex(),
                )
                rows.append(GeneratedSequenceRecord(request, sequence, sequence_sha256(sequence), binding))
    return tuple(rows)


def test_selection_is_exactly_eight_per_condition_and_outcome_blind() -> None:
    selected = select_validation_records(_records())
    assert len(selected) == 68 * 2 * 8
    assert {row.request.sample_index for row in selected} == set(VALIDATION_SAMPLE_INDICES)
    assert {(row.request.protein_id, row.request.backbone_condition) for row in selected} == {
        (f"protein-{index:03d}", condition)
        for index in range(68)
        for condition in ("PDB", "AFDB")
    }


def test_selection_rejects_incomplete_generation_grid() -> None:
    with pytest.raises(IndependentStructureValidationError, match="generation grid"):
        select_validation_records(_records()[:-1])


def test_missing_local_model_is_structured_without_loading_weights(tmp_path: Path) -> None:
    from dual_uq.inference.independent_structure_validation import LocalESMFoldAdapter

    with pytest.raises(IndependentStructureValidationError, match="local ESMFold"):
        LocalESMFoldAdapter(tmp_path / "missing-model", device="cpu", chunk_size=64)
