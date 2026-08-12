"""TDD contract tests for the Stage0-2A boundary representation audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset.fixed_probe_boundary_audit import (
    PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE,
    RepresentationAuditError,
    assess_representation_identifiability,
    build_pairwise_boundary_records,
    pairwise_gap_sd,
)


def _synthetic_ranked_scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, values in (
        ("PDB", {"A": 2.0, "C": 1.0, "D": 0.0}),
        ("AFDB", {"A": 1.1, "C": 1.4, "D": 0.0}),
    ):
        values = {aa: values.get(aa, -10.0) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        for repeat in range(2):
            ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            for rank, (aa, score) in enumerate(ordered, 1):
                rows.append(
                    {
                        "protein_id": "protein",
                        "position": 1,
                        "backbone_condition": condition,
                        "repeat_index": repeat,
                        "seed": repeat,
                        "decoding_realization_sha256": f"realization-{repeat}",
                        "aa": aa,
                        "rank": rank,
                        "score_mean_logp_mask": score
                        + (0.1 * repeat if aa == "A" else 0.05 * repeat if aa == "C" else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_pairwise_gap_sd_uses_symmetric_rms_of_same_state_scales() -> None:
    assert pairwise_gap_sd(3.0, 4.0) == pytest.approx(np.sqrt(12.5))


def test_boundary_records_preserve_source_defined_candidate_pairs() -> None:
    records = build_pairwise_boundary_records(
        _synthetic_ranked_scores(), repeat_count=2
    )

    assert set(records["boundary_source"]) == {"PDB", "AFDB"}
    assert set(records["pair_id"]) == {"A>C", "C>A"}
    assert (records["pairwise_gap_sd"] > 0).all()
    assert records["structural_gap_shift"].tolist() == pytest.approx(
        [-1.3, -1.3, 1.3, 1.3]
    )


def test_zero_same_state_scale_is_structured_not_infinite() -> None:
    ranked = _synthetic_ranked_scores()
    ranked["score_mean_logp_mask"] = ranked["score_mean_logp_mask"].where(
        ranked["repeat_index"] == 0,
        ranked.groupby(["backbone_condition", "aa"])["score_mean_logp_mask"].transform("first"),
    )
    records = build_pairwise_boundary_records(ranked, repeat_count=2)
    zero = records["pairwise_gap_sd"] == 0
    assert zero.any()
    assert records.loc[zero, "pairwise_fragility"].isna().all()
    assert set(records.loc[zero, "pairwise_fragility_status"]) == {"zero_scale"}


def test_frozen_score_schema_reports_unavailable_residue_decomposition() -> None:
    result = assess_representation_identifiability(
        {"score_sum_logp_mask", "score_mean_logp_mask"}
    )
    assert result["status"] == PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE
    assert "not identifiable" in result["statement"]


def test_nonfinite_pairwise_scores_fail_structurally() -> None:
    ranked = _synthetic_ranked_scores()
    ranked.loc[0, "score_mean_logp_mask"] = np.nan
    with pytest.raises(RepresentationAuditError, match="nonfinite"):
        build_pairwise_boundary_records(ranked, repeat_count=2)


def test_frozen_boundary_release_has_expected_shape_and_scope() -> None:
    from dual_uq.dataset.fixed_probe_boundary_audit import (
        run_fixed_probe_boundary_audit,
    )

    result = run_fixed_probe_boundary_audit(Path(__file__).parents[3])
    assert len(result.position_boundary) == 1790
    assert len(result.protein_associations) == 8
    assert not {
        "pairwise_gap_sd_pdb",
        "pairwise_gap_sd_afdb",
        "pdb_pairwise_gap_sign_change_fraction_mean",
        "afdb_pairwise_gap_sign_change_fraction_mean",
        "pair_id",
    }.intersection(result.position_boundary.columns)
    assert result.manifest_fields["stage0_2b_status"] == "HOLD_STAGE0_2B"
    assert result.manifest_fields["scope_declarations"]["proteinmpnn_executed"] is False
