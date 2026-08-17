"""Protein-level aggregation for the DynamicMPNN baseline comparison."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def normalized_hamming(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Hamming inputs must have equal non-zero length")
    return float(sum(a != b for a, b in zip(left, right, strict=True)) / len(left))


def mean_pairwise_diversity(sequences: tuple[str, ...]) -> float:
    if len(sequences) < 2:
        return 0.0
    values = [normalized_hamming(left, right) for left, right in combinations(sequences, 2)]
    return float(np.mean(values))


def summarize_dynamic_scores(scores: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "c_apo", "c_holo", "mean_compat", "worst_compat", "state_gap"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"DynamicMPNN score table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for protein, group in scores.groupby("protein_id", sort=True):
        seq_group = sequences[sequences["protein_id"] == protein].sort_values("sample_index")
        seqs = tuple(str(value) for value in seq_group["sequence"])
        rows.append(
            {
                "protein_id": str(protein),
                "n_sequences": int(len(group)),
                "dynamic_c_apo_median": float(group["c_apo"].median()),
                "dynamic_c_holo_median": float(group["c_holo"].median()),
                "dynamic_mean_compat_median": float(group["mean_compat"].median()),
                "dynamic_mean_compat_q10": float(group["mean_compat"].quantile(0.10, interpolation="linear")),
                "dynamic_mean_compat_q90": float(group["mean_compat"].quantile(0.90, interpolation="linear")),
                "dynamic_worst_compat_median": float(group["worst_compat"].median()),
                "dynamic_worst_compat_q10": float(group["worst_compat"].quantile(0.10, interpolation="linear")),
                "dynamic_worst_compat_q90": float(group["worst_compat"].quantile(0.90, interpolation="linear")),
                "dynamic_state_gap_median": float(group["state_gap"].median()),
                "dynamic_diversity": mean_pairwise_diversity(seqs),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("DynamicMPNN score table contains no complete proteins")
    return result


def compare_with_frozen_baseline(
    dynamic: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "protein_id",
        "proteinmpnn_apo_worst_compat",
        "proteinmpnn_holo_worst_compat",
        "proteinmpnn_apo_mean_compat",
        "proteinmpnn_holo_mean_compat",
        "proteinmpnn_multi_worst_compat",
        "proteinmpnn_multi_mean_compat",
        "esm_if1_apo_worst_compat",
        "esm_if1_holo_worst_compat",
        "esm_if1_apo_mean_compat",
        "esm_if1_holo_mean_compat",
        "esm_if1_multi_worst_compat",
        "esm_if1_multi_mean_compat",
    }
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(f"frozen baseline is missing columns: {sorted(missing)}")
    joined = dynamic.merge(baseline, on="protein_id", how="inner", validate="one_to_one")
    for evaluator in ("proteinmpnn", "esm_if1"):
        best_worst = joined[[f"{evaluator}_apo_worst_compat", f"{evaluator}_holo_worst_compat"]].max(axis=1)
        best_mean = joined[[f"{evaluator}_apo_mean_compat", f"{evaluator}_holo_mean_compat"]].max(axis=1)
        dynamic_worst = joined[f"{evaluator}_dynamic_worst_compat_median"]
        dynamic_mean = joined[f"{evaluator}_dynamic_mean_compat_median"]
        joined[f"{evaluator}_dynamic_minus_best_single_worst"] = (
            dynamic_worst - best_worst
        )
        joined[f"{evaluator}_dynamic_minus_best_single_mean"] = (
            dynamic_mean - best_mean
        )
        joined[f"{evaluator}_dynamic_minus_multi_worst"] = (
            dynamic_worst - joined[f"{evaluator}_multi_worst_compat"]
        )
        joined[f"{evaluator}_dynamic_minus_multi_mean"] = (
            dynamic_mean - joined[f"{evaluator}_multi_mean_compat"]
        )
    fraction_rows: list[dict[str, object]] = []
    for comparison in ("best_single", "multi"):
        pnn = joined[f"proteinmpnn_dynamic_minus_{comparison}_worst"]
        esm = joined[f"esm_if1_dynamic_minus_{comparison}_worst"]
        fraction_rows.extend(
            [
                {"comparison": comparison, "category": "improved_both", "count": int(((pnn > 0) & (esm > 0)).sum())},
                {"comparison": comparison, "category": "improved_proteinmpnn_only", "count": int(((pnn > 0) & (esm <= 0)).sum())},
                {"comparison": comparison, "category": "improved_esm_if1_only", "count": int(((pnn <= 0) & (esm > 0)).sum())},
                {"comparison": comparison, "category": "worsened_both", "count": int(((pnn < 0) & (esm < 0)).sum())},
            ]
        )
    fractions = pd.DataFrame(fraction_rows)
    descriptors = {
        "proteinmpnn_local_burden_mean": "proteinmpnn_local_burden_mean",
        "esm_if1_local_burden_mean": "esm_if1_local_burden_mean",
        "proteinmpnn_d_excess_64": "proteinmpnn_d_excess_64",
        "esm_if1_d_excess_64": "esm_if1_d_excess_64",
        "esm_if1_js_bits_mean_64": "esm_if1_js_bits_mean_64",
        "proteinmpnn_js_bits_mean_64": "proteinmpnn_js_bits_mean_64",
        "median_local_pairwise_distance_change": "median_local_pairwise_distance_change",
        "median_residue_displacement": "median_residue_displacement",
    }
    associations: list[dict[str, object]] = []
    for descriptor, column in descriptors.items():
        if column not in joined:
            continue
        for evaluator in ("proteinmpnn", "esm_if1"):
            for comparison in ("best_single", "multi"):
                effect = f"{evaluator}_dynamic_minus_{comparison}_worst"
                valid = joined[[column, effect]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < 3:
                    continue
                rho = spearmanr(valid[column], valid[effect]).statistic
                associations.append(
                    {
                        "descriptor": descriptor,
                        "evaluator": evaluator,
                        "comparison": comparison,
                        "estimand": f"dynamic_minus_{comparison}_worst",
                        "n_proteins": int(len(valid)),
                        "spearman_rho": float(rho),
                    }
                )
    return joined, fractions, pd.DataFrame(associations)
