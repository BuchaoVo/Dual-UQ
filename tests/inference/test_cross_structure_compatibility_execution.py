from __future__ import annotations

from pathlib import Path

import numpy as np

from dual_uq.inference.cross_structure_compatibility import (
    CrossStructureInputs,
    score_cross_structure,
)
from dual_uq.models.proteinmpnn import (
    ProteinMPNNScore,
    ProteinMPNNStructureInput,
    sequence_sha256,
)
from dual_uq.models.proteinmpnn_generation import (
    GeneratedSequenceRecord,
    GenerationRequest,
    generation_seed_plan,
)
from dual_uq.models.scoring import ScorerBinding


class FakeScoreAdapter:
    binding = ScorerBinding("ProteinMPNN", "fake-implementation", "f" * 64, "cross-v1")

    def score_sequences(self, structure, sequences, realization, *, batch_size):
        del realization, batch_size
        base = 1.0 if structure.backbone_condition == "PDB" else 2.0
        return tuple(
            ProteinMPNNScore(base + 0.01 * index, (base + 0.01 * index) / len(sequence))
            for index, sequence in enumerate(sequences)
        )


def _records() -> tuple[GeneratedSequenceRecord, ...]:
    binding = ScorerBinding("ProteinMPNN", "generation", "a" * 64, "generation")
    rows = []
    for condition in ("PDB", "AFDB"):
        for spec in generation_seed_plan():
            request = GenerationRequest(
                protein_id="protein-1",
                backbone_condition=condition,
                structure_sha256=("1" if condition == "PDB" else "2") * 64,
                canonical_positions=(1, 2),
                wt_sequence_projection="AA",
                temperature=0.1,
                sample_index=spec.sample_index,
                seed=spec.pdb_seed if condition == "PDB" else spec.afdb_seed,
                sample_class=spec.sample_class,
                decoding_realization="0" * 64,
            )
            rows.append(GeneratedSequenceRecord(request, "AC", sequence_sha256("AC"), binding))
    return tuple(rows)


def _inputs() -> CrossStructureInputs:
    projections = {}
    for condition, digest in (("PDB", "1" * 64), ("AFDB", "2" * 64)):
        projections[("protein-1", condition)] = ProteinMPNNStructureInput(
            protein_id="protein-1",
            backbone_condition=condition,
            uniprot_positions=(1, 2),
            wt_sequence_projection="AA",
            coordinates=np.zeros((2, 4, 3), dtype=np.float32),
            structure_sha256=digest,
        )
    return CrossStructureInputs(
        records=_records(),
        projections=projections,
        expected_protein_count=1,
    )


def test_cross_score_covers_both_evaluated_structures_and_is_immutable(tmp_path: Path) -> None:
    result = score_cross_structure(_inputs(), FakeScoreAdapter(), output_root=tmp_path, resume=False)
    assert len(result.rows) == 1024
    assert {(row.generated_condition, row.evaluated_condition) for row in result.rows} == {
        ("PDB", "PDB"),
        ("PDB", "AFDB"),
        ("AFDB", "PDB"),
        ("AFDB", "AFDB"),
    }
    assert result.executed_proteins == 1
    reused = score_cross_structure(_inputs(), FakeScoreAdapter(), output_root=tmp_path, resume=True)
    assert reused.reused_proteins == 1
    assert reused.rows == result.rows
