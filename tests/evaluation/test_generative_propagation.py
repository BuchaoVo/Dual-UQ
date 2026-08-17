from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.generative_propagation import (
    GenerativePropagationInputs,
    build_analysis_result,
    jensen_shannon_bits,
    materialize_generative_propagation,
)
from dual_uq.models.proteinmpnn import sequence_sha256
from dual_uq.models.proteinmpnn_generation import GeneratedSequenceRecord, GenerationRequest
from dual_uq.models.scoring import ScorerBinding


def _records() -> tuple[GeneratedSequenceRecord, ...]:
    binding = ScorerBinding("ProteinMPNN", "impl", "1" * 64, "generation")
    records: list[GeneratedSequenceRecord] = []
    for condition in ("PDB", "AFDB"):
        for sample_index in range(256):
            paired = sample_index < 128
            local_index = sample_index if paired else sample_index - 128
            seed = local_index if paired else (
                1_000_000 + local_index if condition == "PDB" else 2_000_000 + local_index
            )
            sequence = "AC" if (condition == "PDB" or local_index % 2 == 0) else "CA"
            request = GenerationRequest(
                protein_id="protein-1",
                backbone_condition=condition,
                structure_sha256=("1" if condition == "PDB" else "2") * 64,
                canonical_positions=(1, 2),
                wt_sequence_projection="AA",
                temperature=0.1,
                sample_index=sample_index,
                seed=seed,
                sample_class="paired" if paired else "independent",
                decoding_realization=f"{sample_index + 1:064x}",
            )
            records.append(
                GeneratedSequenceRecord(request, sequence, sequence_sha256(sequence), binding)
            )
    return tuple(records)


def _inputs() -> GenerativePropagationInputs:
    descriptors = pd.DataFrame(
        {
            "protein_id": ["protein-1", "protein-1"],
            "position": [1, 2],
            "magnitude_p": [0.1, 0.2],
            "breadth_b": [0.2, 0.8],
            "rank_displacement": [0.3, 0.1],
        }
    )
    compatibility = pd.DataFrame(
        {
            "protein_id": ["protein-1"] * 512,
            "backbone_condition": ["PDB"] * 256 + ["AFDB"] * 256,
            "sample_index": list(range(256)) * 2,
            "pdb_score": [1.0] * 512,
            "afdb_score": [0.5] * 512,
        }
    )
    return GenerativePropagationInputs(
        records=_records(),
        position_descriptors=descriptors,
        cross_compatibility=compatibility,
        expected_protein_count=1,
    )


def test_js_is_symmetric_and_zero_safe() -> None:
    left = np.array([0.5, 0.5] + [0.0] * 18)
    right = np.array([1.0] + [0.0] * 19)
    assert jensen_shannon_bits(left, left) == 0.0
    assert jensen_shannon_bits(left, right) == jensen_shannon_bits(right, left)
    assert jensen_shannon_bits(left, right) > 0.0


def test_analysis_separates_independent_primary_from_paired_sensitivity() -> None:
    result = build_analysis_result(_inputs())
    protein = result.protein_summary.iloc[0]

    assert protein["n_independent"] == 128
    assert protein["n_paired"] == 128
    assert protein["d_pa_independent"] >= protein["d_pa_paired"]
    assert result.position_shift["js_bits_independent"].notna().all()


def test_cross_structure_loss_keeps_direction_and_descriptor_join() -> None:
    result = build_analysis_result(_inputs())
    cross = result.cross_structure_compatibility

    assert set(cross["generated_condition"]) == {"PDB", "AFDB"}
    assert (cross.loc[cross["generated_condition"] == "PDB", "directional_loss"] == 0.5).all()
    assert (cross.loc[cross["generated_condition"] == "AFDB", "directional_loss"] == -0.5).all()
    assert {"magnitude_p", "breadth_b", "rank_displacement"}.issubset(
        result.position_shift.columns
    )
    assert len(result.protein_summary) == 1
    assert result.protein_summary["symmetric_cross_loss"].notna().all()


def test_descriptor_frame_may_cover_larger_frozen_cohort() -> None:
    inputs = _inputs()
    descriptors = pd.concat(
        [
            inputs.position_descriptors,
            pd.DataFrame(
                {
                    "protein_id": ["other-protein"],
                    "position": [1],
                    "magnitude_p": [0.4],
                    "breadth_b": [0.5],
                    "rank_displacement": [0.6],
                }
            ),
        ],
        ignore_index=True,
    )
    result = build_analysis_result(
        GenerativePropagationInputs(
            records=inputs.records,
            position_descriptors=descriptors,
            cross_compatibility=inputs.cross_compatibility,
            expected_protein_count=inputs.expected_protein_count,
        )
    )
    assert set(result.position_shift["protein_id"]) == {"protein-1"}


def test_materialization_is_canonical_and_immutable(tmp_path) -> None:
    result = build_analysis_result(_inputs())
    first = materialize_generative_propagation(result, tmp_path)
    expected = {
        "generated_sequences.parquet",
        "position_generation_shift.parquet",
        "cross_structure_compatibility.parquet",
        "protein_generation_summary.parquet",
        "convergence.parquet",
        "summary.json",
        "report.md",
        "manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert first["manifest_status"] == "created"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for question in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert question in report
    second = materialize_generative_propagation(result, tmp_path)
    assert second["manifest_status"] == "reused_identical"

    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable output conflict"):
        materialize_generative_propagation(result, tmp_path)
