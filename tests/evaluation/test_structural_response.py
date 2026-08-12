from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation import structural_response as response


def _fixture_inputs() -> response.StructuralResponseInputs:
    wt_rows = []
    raw_rows = []
    for repeat, fingerprint in enumerate(("f0", "f1")):
        for condition, wt_score in (("PDB", -2.0 - repeat * 0.1), ("AFDB", -1.8 - repeat * 0.1)):
            wt_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "backbone_condition": condition,
                    "backbone_sha256": condition.lower().ljust(64, "0"),
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": fingerprint.ljust(64, "0"),
                    "score_mean_logp_mask": wt_score,
                }
            )
            raw_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "sequence_hash": "s" * 64,
                    "position": 2,
                    "wt_aa": "A",
                    "mut_aa": "C",
                    "backbone_condition": condition,
                    "backbone_sha256": condition.lower().ljust(64, "0"),
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": fingerprint.ljust(64, "0"),
                    "score_mean_logp_mask": wt_score + (0.4 if condition == "PDB" else 0.9),
                    "delta_score_vs_wt": 0.4 if condition == "PDB" else 0.9,
                }
            )
    return response.StructuralResponseInputs(
        project_root=Path("."),
        scoring_manifest_path=Path("scoring.json"),
        scoring_manifest_sha256="a" * 64,
        scoring_manifest={},
        wt_scores=pd.DataFrame(wt_rows),
        raw_scores=pd.DataFrame(raw_rows),
        cohort=pd.DataFrame(
            {
                "protein_id": ["fixture_A__P00001"],
                "sequence_cluster": ["30:fixture"],
            }
        ),
        input_provenance={},
    )


def test_build_paired_response_uses_exact_fingerprint_and_g_c_d_algebra() -> None:
    paired = response.build_paired_structural_response(_fixture_inputs())

    assert len(paired) == 2
    assert list(paired["repeat_index"]) == [0, 1]
    assert paired["wt_baseline_shift_g"].tolist() == pytest.approx([0.2, 0.2])
    assert paired["candidate_absolute_shift_c"].tolist() == pytest.approx([0.7, 0.7])
    assert paired["candidate_remodeling_d"].tolist() == pytest.approx([0.5, 0.5])
    assert np.allclose(
        paired["candidate_remodeling_d"],
        paired["candidate_absolute_shift_c"] - paired["wt_baseline_shift_g"],
    )


def test_build_paired_response_rejects_missing_structural_condition() -> None:
    inputs = _fixture_inputs()
    inputs.raw_scores = inputs.raw_scores.loc[
        ~(
            (inputs.raw_scores["backbone_condition"] == "AFDB")
            & (inputs.raw_scores["repeat_index"] == 1)
        )
    ].reset_index(drop=True)

    with pytest.raises(response.StructuralResponseError) as caught:
        response.build_paired_structural_response(inputs)

    assert caught.value.code == "missing_structural_pair"


def test_summary_aggregates_within_protein_before_cohort() -> None:
    paired = response.build_paired_structural_response(_fixture_inputs())
    summary = response.summarize_structural_response(paired)

    assert len(summary.protein_summary) == 1
    assert len(summary.position_summary) == 1
    assert summary.cohort_summary["protein_count"] == 1
    assert summary.protein_summary.loc[0, "median_abs_d"] == pytest.approx(0.5)


def test_materialization_is_immutable_and_writes_three_diagnostic_figures(tmp_path: Path) -> None:
    inputs = _fixture_inputs()
    paired = response.build_paired_structural_response(inputs)
    summary = response.summarize_structural_response(paired)
    result = response.StructuralResponseResult(inputs=inputs, paired=paired, summary=summary)

    first = response.materialize_structural_response(result, tmp_path)
    second = response.materialize_structural_response(result, tmp_path)

    assert first["manifest_path"] == second["manifest_path"]
    assert first["outputs"] == second["outputs"]
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert (tmp_path / "paired_structural_response.parquet").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert sorted(path.name for path in (tmp_path / "figures").glob("*.png")) == [
        "paired_difference_distribution.png",
        "per_protein_response_summary.png",
        "repeat_stability.png",
    ]


def test_analysis_cli_help_is_thin_and_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analysis/analyze_structural_response.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--project-root" in completed.stdout


def test_frozen_probe_release_schema_is_adapted_without_changing_identity() -> None:
    release = pd.DataFrame(
        {
            "protein_id": ["p"],
            "sequence_hash": ["s" * 64],
            "canonical_position": [7],
            "wt_aa": ["A"],
            "candidate_aa": ["C"],
        }
    )

    normalized = response._normalize_frozen_probe_keys(release)

    assert normalized.loc[0, "position"] == 7
    assert normalized.loc[0, "mut_aa"] == "C"
    assert normalized.loc[0, "sequence_hash"] == "s" * 64


def test_probe_identity_check_is_independent_of_input_column_order() -> None:
    inputs = _fixture_inputs()
    inputs.frozen_probe_keys = pd.DataFrame(
        {
            "mut_aa": pd.Series(["C"], dtype="string"),
            "position": pd.Series([2], dtype="int32"),
            "sequence_hash": pd.Series(["s" * 64], dtype="string"),
            "wt_aa": pd.Series(["A"], dtype="string"),
            "protein_id": pd.Series(["fixture_A__P00001"], dtype="string"),
        }
    )

    response._validate_input_tables(inputs)


def test_build_paired_response_allows_many_candidates_per_wt_realization() -> None:
    inputs = _fixture_inputs()
    extra = inputs.raw_scores.copy()
    extra["sequence_hash"] = "d" * 64
    extra["mut_aa"] = "D"
    inputs.raw_scores = pd.concat([inputs.raw_scores, extra], ignore_index=True)

    paired = response.build_paired_structural_response(inputs)

    assert len(paired) == 4
    assert paired["sequence_hash"].nunique() == 2
