from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.inverse_folding_remodeling import (
    RemodelingAnalysisError,
    RemodelingAnalysisInputs,
    _aggregate_paired_file,
    build_remodeling_result,
    materialize_remodeling_result,
)

AA = "ACDEFGHIKLMNPQRSTVWY"
SUBSTITUTIONS = AA.replace("A", "")


def _paired_fixture(repeats: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protein_id, reverse in (("p1", False), ("p2", True)):
        for position in (1, 2):
            for repeat in range(repeats):
                for index, mut_aa in enumerate(SUBSTITUTIONS):
                    base = float(index + 1)
                    if position == 2:
                        base = 0.0
                    pdb_delta = base
                    afdb_delta = base if not reverse else float(20 - index)
                    d = afdb_delta - pdb_delta
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "position": position,
                            "wt_aa": "A",
                            "mut_aa": mut_aa,
                            "repeat_index": repeat,
                            "candidate_remodeling_d": d,
                            "pdb_delta_score_vs_wt": pdb_delta,
                            "afdb_delta_score_vs_wt": afdb_delta,
                        }
                    )
    return pd.DataFrame(rows)


def _inputs() -> RemodelingAnalysisInputs:
    pair_validity = pd.DataFrame(
        [
            {"protein_id": "p1", "high_comparability_eligible": True},
            {"protein_id": "p2", "high_comparability_eligible": False},
        ]
    )
    return RemodelingAnalysisInputs(
        paired=_paired_fixture(),
        pair_validity=pair_validity,
        input_provenance={"fixture": {"path": "fixture.parquet", "sha256": "fixture"}},
    )


def test_magnitude_breadth_and_zero_response_are_explicit() -> None:
    result = build_remodeling_result(_inputs())
    position = result.position.set_index(["protein_id", "position"])

    # p1/position 1 has a constant D=0, while p1/position 2 is also zero;
    # both must remain explicitly undefined for breadth.
    assert position.loc[("p1", 1), "magnitude_p"] == pytest.approx(0.0)
    assert position.loc[("p1", 1), "position_mean_abs_interaction"] == pytest.approx(0.0)
    assert position.loc[("p1", 1), "breadth_status"] == "zero_remodeling"
    assert np.isnan(position.loc[("p1", 1), "breadth_b"])
    assert position.loc[("p1", 2), "breadth_status"] == "zero_remodeling"


def test_tiny_nonzero_response_is_defined_without_epsilon_threshold() -> None:
    paired = _paired_fixture()
    paired.loc[paired["position"] == 1, "candidate_remodeling_d"] = 1.0e-15
    result = build_remodeling_result(
        RemodelingAnalysisInputs(
            paired=paired,
            pair_validity=_inputs().pair_validity,
            input_provenance={},
        )
    )
    row = result.position.set_index(["protein_id", "position"]).loc[("p1", 1)]
    assert row["breadth_status"] == "defined"
    assert row["breadth_b"] == pytest.approx(1.0)


def test_reordering_uses_profile_ranks_not_top1_as_primary() -> None:
    result = build_remodeling_result(_inputs())
    position = result.position.set_index(["protein_id", "position"])

    stable = position.loc[("p1", 1)]
    reversed_profile = position.loc[("p2", 1)]
    assert stable["kendall_tau"] == pytest.approx(1.0)
    assert reversed_profile["kendall_tau"] < -0.99
    assert reversed_profile["rank_displacement"] > 0.5
    assert "top1_flip" in result.position.columns


def test_clean_and_full_cohorts_are_reported_at_protein_unit() -> None:
    result = build_remodeling_result(_inputs())
    assert result.cohort_summary["cohort"]["full"]["protein_count"] == 2
    assert result.cohort_summary["cohort"]["clean"]["protein_count"] == 1
    assert set(result.protein["cohort"]) == {"clean", "full"}
    assert set(result.representatives["selection_reason"]) >= {
        "highest_typical_magnitude",
        "highest_typical_breadth",
        "highest_reordering_burden",
    }


def test_zero_candidate_grid_is_structured_failure() -> None:
    bad = _inputs()
    bad.paired = bad.paired[bad.paired["mut_aa"] != "C"]
    with pytest.raises(RemodelingAnalysisError) as caught:
        build_remodeling_result(bad)
    assert caught.value.code == "candidate_grid_mismatch"


def test_candidate_set_must_be_exactly_non_wild_type_amino_acids() -> None:
    bad = _inputs()
    bad.paired.loc[bad.paired["mut_aa"] == "C", "mut_aa"] = "A"
    with pytest.raises(RemodelingAnalysisError) as caught:
        build_remodeling_result(bad)
    assert caught.value.code == "candidate_grid_mismatch"


def test_cohort_label_must_be_boolean() -> None:
    bad = _inputs()
    bad.pair_validity["high_comparability_eligible"] = ["False", "True"]
    with pytest.raises(RemodelingAnalysisError) as caught:
        build_remodeling_result(bad)
    assert caught.value.code == "cohort_label_schema_mismatch"


def test_streaming_repeat_audit_rejects_out_of_range_realization(tmp_path: Path) -> None:
    rows = _paired_fixture(repeats=3).iloc[:19].copy()
    rows["repeat_index"] = 30
    path = tmp_path / "paired.parquet"
    rows.to_parquet(path, index=False)
    with pytest.raises(RemodelingAnalysisError) as caught:
        _aggregate_paired_file(path)
    assert caught.value.code == "repeat_grid_mismatch"


def test_materialization_is_immutable_and_derived_from_one_result(tmp_path: Path) -> None:
    result = build_remodeling_result(_inputs())
    first = materialize_remodeling_result(result, tmp_path)
    second = materialize_remodeling_result(result, tmp_path)
    assert first["status"] == "COMPLETE"
    assert second["status"] == "COMPLETE"
    assert (tmp_path / "position_remodeling.parquet").is_file()
    assert (tmp_path / "protein_remodeling.parquet").is_file()
    assert (tmp_path / "candidate_remodeling.parquet").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_analysis_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analysis/analyze_inverse_folding_remodeling.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
