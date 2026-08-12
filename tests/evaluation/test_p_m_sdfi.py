from __future__ import annotations

import subprocess
import sys

import pytest

from dual_uq.evaluation.p_m_sdfi import (
    PMSDFIError,
    compute_p_m_sdfi,
)


def _fixture_paired(repeat_count: int = 2):
    rows = []
    for position in (1, 2):
        for repeat in range(repeat_count):
            for aa_index, mut_aa in enumerate(
                "ACDEFGHIKLMNPQRSTVWY".replace("A", ""), start=1
            ):
                pdb = float(aa_index)
                afdb = pdb + 0.5
                rows.append(
                    {
                        "protein_id": "fixture_A__P00001",
                        "sequence_hash": f"{position:02d}{aa_index:02d}".ljust(64, "0"),
                        "position": position,
                        "repeat_index": repeat,
                        "decoding_realization_sha256": f"r{repeat}".ljust(64, "0"),
                        "wt_aa": "A",
                        "mut_aa": mut_aa,
                        "pdb_score_mean_logp_mask": pdb,
                        "afdb_score_mean_logp_mask": afdb,
                        "pdb_wt_score": 20.0,
                        "afdb_wt_score": 20.2,
                        "wt_baseline_shift_g": 0.2,
                        "candidate_absolute_shift_c": 0.5,
                        "candidate_remodeling_d": 0.3,
                    }
                )
    return __import__("pandas").DataFrame(rows)


def test_p_m_sdfi_formula_and_margin_semantics() -> None:
    result = compute_p_m_sdfi(_fixture_paired(), expected_repeat_count=2)

    position = result.position_metrics.iloc[0]
    assert position["p_position_mean_abs_interaction"] == pytest.approx(0.3)
    assert position["m_margin_mean_both"] == pytest.approx(0.85)
    assert position["sdfi"] == pytest.approx(0.3 / 0.85)
    assert position["sdfi_status"] == "defined"


def test_p_m_sdfi_aggregates_protein_after_position() -> None:
    result = compute_p_m_sdfi(_fixture_paired(), expected_repeat_count=2)

    assert len(result.position_metrics) == 2
    assert len(result.protein_metrics) == 1
    assert result.protein_metrics.loc[0, "position_count"] == 2
    assert result.protein_metrics.loc[0, "p_position_median"] == pytest.approx(0.3)


def test_zero_margin_keeps_sdfi_undefined() -> None:
    paired = _fixture_paired()
    paired.loc[paired["position"] == 2, ["pdb_wt_score", "afdb_wt_score"]] = 1.0
    paired.loc[
        paired["position"] == 2,
        ["pdb_score_mean_logp_mask", "afdb_score_mean_logp_mask"],
    ] = 1.0
    result = compute_p_m_sdfi(paired, expected_repeat_count=2)

    position = result.position_metrics.set_index("position").loc[2]
    assert position["m_margin_mean_both"] == pytest.approx(0.0)
    assert position["sdfi_status"] == "zero_margin"
    assert position["sdfi"] != position["sdfi"]


def test_missing_condition_is_structured_failure() -> None:
    paired = _fixture_paired().drop(columns=["afdb_score_mean_logp_mask"])

    with pytest.raises(PMSDFIError) as caught:
        compute_p_m_sdfi(paired, expected_repeat_count=2)

    assert caught.value.code == "a1_schema_mismatch"


def test_analysis_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analysis/analyze_p_m_sdfi.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
