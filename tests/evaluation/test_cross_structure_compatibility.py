from __future__ import annotations

import pandas as pd
import pytest

from dual_uq.evaluation.cross_structure_compatibility import (
    build_cross_structure_analysis,
    materialize_cross_structure_analysis,
)


def _scores() -> pd.DataFrame:
    rows = []
    for condition, matching, alternative in (
        ("PDB", -1.0, -2.0),
        ("AFDB", -1.0, -1.5),
    ):
        for index in range(256):
            rows.extend(
                [
                    {
                        "protein_id": "protein-1",
                        "generated_condition": condition,
                        "evaluated_condition": condition,
                        "sample_index": index,
                        "sequence_hash": f"{condition}{index}",
                        "score_mean_logp_mask": matching,
                    },
                    {
                        "protein_id": "protein-1",
                        "generated_condition": condition,
                        "evaluated_condition": "AFDB" if condition == "PDB" else "PDB",
                        "sample_index": index,
                        "sequence_hash": f"{condition}{index}",
                        "score_mean_logp_mask": alternative,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _generation_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["protein-1"],
            "d_pa_excess_independent": [0.2],
            "js_burden_mean": [0.3],
        }
    )


def _remodeling_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["protein-1"],
            "position_magnitude_upper_tail_excess": [0.4],
        }
    )


def test_directional_nll_loss_and_symmetric_summary() -> None:
    result = build_cross_structure_analysis(
        _scores(), _generation_summary(), _remodeling_summary(), expected_protein_count=1
    )
    row = result.directional_loss.query("generated_condition == 'PDB'").iloc[0]
    assert row["delta_p_to_a"] == pytest.approx(1.0)
    reverse_row = result.directional_loss.query("generated_condition == 'AFDB'").iloc[0]
    assert reverse_row["delta_a_to_p"] == pytest.approx(0.5)
    summary = result.protein_summary.iloc[0]
    assert summary["delta_p_to_a_mean"] == pytest.approx(1.0)
    assert summary["delta_a_to_p_mean"] == pytest.approx(0.5)
    assert summary["delta_cross"] == pytest.approx(0.75)
    assert summary["d_pa_excess_independent"] == pytest.approx(0.2)


def test_materialization_is_immutable(tmp_path) -> None:
    result = build_cross_structure_analysis(
        _scores(), _generation_summary(), _remodeling_summary(), expected_protein_count=1
    )
    first = materialize_cross_structure_analysis(result, tmp_path)
    assert first["manifest_status"] == "created"
    assert "positive in 1/1 proteins" in (tmp_path / "report.md").read_text(encoding="utf-8")
    second = materialize_cross_structure_analysis(result, tmp_path)
    assert second["manifest_status"] == "reused_identical"
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable output conflict"):
        materialize_cross_structure_analysis(result, tmp_path)
