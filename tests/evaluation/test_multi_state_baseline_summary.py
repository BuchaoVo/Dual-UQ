import numpy as np
import pandas as pd

from dual_uq.evaluation.multi_state_baseline_summary import (
    build_summary,
    marginal_between_divergence,
    marginal_hamming_diversity,
    pivot_score_summary,
)


def _scores(evaluator: str, multi: bool = False) -> pd.DataFrame:
    rows = []
    for protein_index, protein in enumerate(("p1", "p2", "p3")):
        states = ("APO", "HOLO", "MULTI") if multi else ("APO", "HOLO")
        for state in states:
            for sample in range(2):
                base = float(protein_index + sample + (state == "HOLO"))
                rows.append(
                    {
                        "protein_id": protein,
                        "source_state": state,
                        "mean_compat": base,
                        "worst_compat": base - 0.5,
                        "state_gap": abs(base),
                    }
                )
    return pd.DataFrame(rows)


def test_marginal_divergence_definitions() -> None:
    assert marginal_hamming_diversity(("AA", "AB")) == 0.5
    assert marginal_between_divergence(("AA",), ("BB",)) == 1.0


def test_pivot_requires_all_three_states() -> None:
    table = _scores("ProteinMPNN", multi=True)
    result = pivot_score_summary(table, prefix="proteinmpnn")
    assert result.shape[0] == 3
    assert "proteinmpnn_multi_worst_compat" in result
    assert np.isfinite(result["proteinmpnn_apo_worst_compat"]).all()


def test_build_summary_uses_one_protein_level_result() -> None:
    upstream = pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3"],
            "proteinmpnn_d_excess_64": [0.1, 0.2, 0.3],
            "esm_if1_d_excess_64": [0.2, 0.1, 0.4],
        }
    )
    ensembles = {
        protein: {"APO": ("AA", "AB"), "HOLO": ("AA", "BB"), "MULTI": ("AB", "BB")}
        for protein in ("p1", "p2", "p3")
    }
    result = build_summary(
        single_tables={"ProteinMPNN": _scores("ProteinMPNN")},
        multi_tables={"ProteinMPNN": _scores("ProteinMPNN", multi=True)},
        sequence_ensembles=ensembles,
        upstream=upstream,
    )
    assert result.protein_summary.shape[0] == 3
    assert result.summary["primary_endpoint"] == "worst_compat"
    assert list(result.associations.columns) == []
