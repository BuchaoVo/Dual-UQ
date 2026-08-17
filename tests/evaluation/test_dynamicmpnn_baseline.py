from __future__ import annotations

import pandas as pd

from dual_uq.evaluation.dynamicmpnn_baseline import (
    mean_pairwise_diversity,
    summarize_dynamic_scores,
)
from scripts.analysis.summarize_dynamicmpnn_baseline import run as summarize_run


def test_dynamic_summary_uses_protein_as_primary_unit() -> None:
    scores = pd.DataFrame(
        [
            {"protein_id": "P1", "c_apo": 1.0, "c_holo": 2.0, "mean_compat": 1.5, "worst_compat": 1.0, "state_gap": 1.0},
            {"protein_id": "P1", "c_apo": 2.0, "c_holo": 3.0, "mean_compat": 2.5, "worst_compat": 2.0, "state_gap": 1.0},
            {"protein_id": "P2", "c_apo": -1.0, "c_holo": 1.0, "mean_compat": 0.0, "worst_compat": -1.0, "state_gap": 2.0},
        ]
    )
    sequences = pd.DataFrame(
        [
            {"protein_id": "P1", "sample_index": 0, "sequence": "AAAA"},
            {"protein_id": "P1", "sample_index": 1, "sequence": "AAAT"},
            {"protein_id": "P2", "sample_index": 0, "sequence": "CCCC"},
        ]
    )

    result = summarize_dynamic_scores(scores, sequences)

    assert result["protein_id"].tolist() == ["P1", "P2"]
    assert result.loc[result.protein_id == "P1", "dynamic_mean_compat_median"].item() == 2.0
    assert result.loc[result.protein_id == "P1", "dynamic_diversity"].item() == 0.25


def test_hamming_diversity_is_normalized() -> None:
    assert mean_pairwise_diversity(("AAAA", "AAAT", "AATT")) == 1 / 3


def test_summary_run_accepts_cli_parameter_names(tmp_path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    pd.DataFrame(
        [
            {"protein_id": "P1", "sample_index": 0, "sequence": "AAAA"},
            {"protein_id": "P1", "sample_index": 1, "sequence": "AAAT"},
        ]
    ).to_parquet(generation / "sequences.parquet", index=False)
    scores = pd.DataFrame(
        [
            {
                "protein_id": "P1", "c_apo": 1.0, "c_holo": 1.0,
                "mean_compat": 1.0, "worst_compat": 1.0, "state_gap": 0.0,
            },
            {
                "protein_id": "P1", "c_apo": 1.1, "c_holo": 1.1,
                "mean_compat": 1.1, "worst_compat": 1.1, "state_gap": 0.0,
            },
        ]
    )
    pnn_path = tmp_path / "pnn.parquet"
    esm_path = tmp_path / "esm.parquet"
    scores.to_parquet(pnn_path, index=False)
    scores.to_parquet(esm_path, index=False)
    baseline = pd.DataFrame(
        [{
            "protein_id": "P1",
            "proteinmpnn_apo_worst_compat": 0.0,
            "proteinmpnn_holo_worst_compat": 0.0,
            "proteinmpnn_apo_mean_compat": 0.0,
            "proteinmpnn_holo_mean_compat": 0.0,
            "proteinmpnn_multi_worst_compat": 0.0,
            "proteinmpnn_multi_mean_compat": 0.0,
            "esm_if1_apo_worst_compat": 0.0,
            "esm_if1_holo_worst_compat": 0.0,
            "esm_if1_apo_mean_compat": 0.0,
            "esm_if1_holo_mean_compat": 0.0,
            "esm_if1_multi_worst_compat": 0.0,
            "esm_if1_multi_mean_compat": 0.0,
        }]
    )
    baseline_path = tmp_path / "baseline.parquet"
    baseline.to_parquet(baseline_path, index=False)

    summary = summarize_run(
        generation_root=generation,
        proteinmpnn_scores=pnn_path,
        esm_if1_scores=esm_path,
        baseline=baseline_path,
        output_root=tmp_path / "out",
    )

    assert summary["shared_dynamic_protein_count"] == 1
    joined = pd.read_parquet(tmp_path / "out" / "protein_summary.parquet")
    assert "proteinmpnn_dynamic_minus_multi_worst" in joined
    assert "esm_if1_dynamic_minus_multi_worst" in joined
    assert (tmp_path / "out" / "method_summary.parquet").exists()
    assert "Q1" in (tmp_path / "out" / "report.md").read_text()
