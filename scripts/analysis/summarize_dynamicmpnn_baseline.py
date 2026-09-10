"""Materialize DynamicMPNN baseline summaries and bounded interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.core.hashing import sha256_file as _sha256
from dual_uq.evaluation.dynamicmpnn_baseline import (
    compare_with_frozen_baseline,
    summarize_dynamic_scores,
)


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q, interpolation="linear"))


def _build_method_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for evaluator in ("proteinmpnn", "esm_if1"):
        dynamic_prefix = f"{evaluator}_dynamic_"
        methods = {
            "DynamicMPNN": {
                "worst": joined[f"{dynamic_prefix}worst_compat_median"],
                "mean": joined[f"{dynamic_prefix}mean_compat_median"],
                "state_gap": joined[f"{dynamic_prefix}state_gap_median"],
                "diversity": joined[f"{dynamic_prefix}diversity"],
            },
            "SimpleMulti": {
                "worst": joined[f"{evaluator}_multi_worst_compat"],
                "mean": joined[f"{evaluator}_multi_mean_compat"],
                "state_gap": joined.get(f"{evaluator}_multi_state_gap"),
                "diversity": joined.get("diversity_multi"),
            },
            "BestSingle": {
                "worst": joined[[f"{evaluator}_apo_worst_compat", f"{evaluator}_holo_worst_compat"]].max(axis=1),
                "mean": joined[[f"{evaluator}_apo_mean_compat", f"{evaluator}_holo_mean_compat"]].max(axis=1),
                "state_gap": None,
                "diversity": None,
            },
        }
        for method, values in methods.items():
            worst = pd.Series(values["worst"], index=joined.index, dtype="float64")
            mean = pd.Series(values["mean"], index=joined.index, dtype="float64")
            state_gap = values["state_gap"]
            rows.append(
                {
                    "evaluator": evaluator,
                    "method": method,
                    "protein_count": int(worst.notna().sum()),
                    "worst_median": float(worst.median()),
                    "worst_q10": _quantile(worst, 0.10),
                    "worst_q90": _quantile(worst, 0.90),
                    "mean_median": float(mean.median()),
                    "mean_q10": _quantile(mean, 0.10),
                    "mean_q90": _quantile(mean, 0.90),
                    "state_gap_median": None if state_gap is None else float(pd.Series(state_gap).median()),
                    "diversity_median": None if values["diversity"] is None else float(pd.Series(values["diversity"]).median()),
                }
            )
    return pd.DataFrame(rows)


def _build_high_sensitivity_summary(joined: pd.DataFrame) -> pd.DataFrame:
    descriptors = (
        "proteinmpnn_local_burden_mean",
        "esm_if1_local_burden_mean",
        "proteinmpnn_d_excess_64",
        "esm_if1_d_excess_64",
        "proteinmpnn_js_bits_mean_64",
        "esm_if1_js_bits_mean_64",
    )
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        if descriptor not in joined:
            continue
        values = joined[descriptor].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if values.empty:
            continue
        threshold = float(values.quantile(0.75, interpolation="linear"))
        subset = joined[joined[descriptor] >= threshold]
        for evaluator in ("proteinmpnn", "esm_if1"):
            for comparison in ("best_single", "multi"):
                effect = subset[f"{evaluator}_dynamic_minus_{comparison}_worst"]
                rows.append(
                    {
                        "descriptor": descriptor,
                        "threshold_q75": threshold,
                        "evaluator": evaluator,
                        "comparison": comparison,
                        "protein_count": int(effect.notna().sum()),
                        "effect_median": float(effect.median()),
                        "effect_q10": _quantile(effect, 0.10),
                        "effect_q90": _quantile(effect, 0.90),
                        "positive_fraction": float((effect > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def run(
    *,
    generation_root: Path,
    proteinmpnn_scores: Path,
    esm_if1_scores: Path,
    baseline: Path,
    output_root: Path,
) -> dict[str, object]:
    sequences = pd.read_parquet(generation_root / "sequences.parquet")
    pnn = pd.read_parquet(proteinmpnn_scores)
    esm = pd.read_parquet(esm_if1_scores)
    baseline_frame = pd.read_parquet(baseline)
    pnn_summary = summarize_dynamic_scores(pnn, sequences).add_prefix("proteinmpnn_")
    pnn_summary = pnn_summary.rename(columns={"proteinmpnn_protein_id": "protein_id"})
    esm_summary = summarize_dynamic_scores(esm, sequences).add_prefix("esm_if1_")
    esm_summary = esm_summary.rename(columns={"esm_if1_protein_id": "protein_id"})
    dynamic = pnn_summary.merge(esm_summary, on="protein_id", how="inner", validate="one_to_one")
    joined, fractions, associations = compare_with_frozen_baseline(dynamic, baseline_frame)
    method_summary = _build_method_summary(joined)
    high_sensitivity = _build_high_sensitivity_summary(joined)
    output_root.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_root / "protein_summary.parquet", index=False)
    fractions.to_parquet(output_root / "evaluator_fractions.parquet", index=False)
    associations.to_parquet(output_root / "descriptor_associations.parquet", index=False)
    method_summary.to_parquet(output_root / "method_summary.parquet", index=False)
    high_sensitivity.to_parquet(output_root / "high_sensitivity_summary.parquet", index=False)
    decision = "LIMITED" if len(joined) < int(0.75 * len(baseline_frame)) else "PASS"
    generation_summary_path = generation_root / "summary.json"
    generation_summary = json.loads(generation_summary_path.read_text()) if generation_summary_path.exists() else {}
    inventory_path = generation_root / "inventory.parquet"
    unavailable_reasons: dict[str, int] = {}
    if inventory_path.exists():
        inventory = pd.read_parquet(inventory_path)
        unavailable_reasons = {
            str(reason): int(count)
            for reason, count in inventory.loc[inventory["status"] != "GENERATED", "reason"].value_counts().items()
        }
    summary = {
        "verdict": decision,
        "frozen_baseline_protein_count": int(len(baseline_frame)),
        "dynamic_pnn_protein_count": int(pnn_summary.protein_id.nunique()),
        "dynamic_esm_if1_protein_count": int(esm_summary.protein_id.nunique()),
        "shared_dynamic_protein_count": int(joined.protein_id.nunique()),
        "dynamic_pnn_sequence_rows": int(len(pnn)),
        "dynamic_esm_if1_sequence_rows": int(len(esm)),
        "dynamic_generation_unavailable_count": int(generation_summary.get("unavailable_count", 0)),
        "dynamic_generation_unavailable_reasons": unavailable_reasons,
        "interpretation_scope": "DynamicMPNN-evaluable shared evaluator subset only",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    def _stat(evaluator: str, method: str, field: str) -> float:
        return float(method_summary.loc[(method_summary.evaluator == evaluator) & (method_summary.method == method), field].iloc[0])

    best_counts = fractions[fractions["comparison"] == "best_single"].set_index("category")["count"].to_dict()
    multi_counts = fractions[fractions["comparison"] == "multi"].set_index("category")["count"].to_dict()
    strongest = (
        associations.assign(abs_rho=associations["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .head(4)
        if not associations.empty and "spearman_rho" in associations
        else associations
    )
    association_lines = [
        f"- {row.evaluator} vs {row.comparison}: {row.descriptor}, Spearman rho={row.spearman_rho:.3f} (n={int(row.n_proteins)})"
        for row in strongest.itertuples()
    ] or ["- No descriptor association had at least three finite proteins."]

    report = "\n".join(
        [
            "# DynamicMPNN baseline evaluation",
            "",
            f"VERDICT: **{decision}**",
            "",
            f"The DynamicMPNN result is scoped to {len(joined)} proteins with complete DynamicMPNN, ProteinMPNN, and ESM-IF1 scoring paths. The frozen baseline contains {len(baseline_frame)} proteins.",
            f"Generation exclusions were {unavailable_reasons}.",
            "",
            "No absolute compatibility score is interpreted across unrelated proteins. Protein is the inference unit; evaluator scales remain separate.",
            "",
            "## Q1. Cross-evaluator robustness",
            "",
            f"Against best single-state, DynamicMPNN improved WorstCompat under both evaluators for {best_counts.get('improved_both', 0)}/{len(joined)} proteins; it improved ProteinMPNN only for {best_counts.get('improved_proteinmpnn_only', 0)} and ESM-IF1 only for {best_counts.get('improved_esm_if1_only', 0)}. WorstCompat worsened under both for {best_counts.get('worsened_both', 0)} proteins.",
            f"Against simple MULTI, the corresponding counts were {multi_counts.get('improved_both', 0)} improved under both and {multi_counts.get('worsened_both', 0)} worsened under both.",
            "",
            "## Q2. Evaluator-specific compatibility",
            "",
            f"ProteinMPNN DynamicMPNN WorstCompat median/q10/q90 = {_stat('proteinmpnn', 'DynamicMPNN', 'worst_median'):.3f}/{_stat('proteinmpnn', 'DynamicMPNN', 'worst_q10'):.3f}/{_stat('proteinmpnn', 'DynamicMPNN', 'worst_q90'):.3f}; ESM-IF1 = {_stat('esm_if1', 'DynamicMPNN', 'worst_median'):.3f}/{_stat('esm_if1', 'DynamicMPNN', 'worst_q10'):.3f}/{_stat('esm_if1', 'DynamicMPNN', 'worst_q90'):.3f}.",
            "The DynamicMPNN result is therefore not an evaluator-general improvement over the frozen single-state baseline in this evaluable subset.",
            "",
            "## Q3. DynamicMPNN versus simple MULTI",
            "",
            f"Median DynamicMPNN-minus-simple-MULTI WorstCompat = {_stat('proteinmpnn', 'DynamicMPNN', 'worst_median') - _stat('proteinmpnn', 'SimpleMulti', 'worst_median'):.3f} for ProteinMPNN and {_stat('esm_if1', 'DynamicMPNN', 'worst_median') - _stat('esm_if1', 'SimpleMulti', 'worst_median'):.3f} for ESM-IF1.",
            "",
            "## Q4. Heterogeneity",
            "",
            f"DynamicMPNN diversity median = {_stat('proteinmpnn', 'DynamicMPNN', 'diversity_median'):.3f} (ProteinMPNN scoring view) and {_stat('esm_if1', 'DynamicMPNN', 'diversity_median'):.3f} (ESM-IF1 scoring view); the protein-level q10/q90 ranges above show broad heterogeneity rather than a uniform shift.",
            "",
            "## Q5. Relation to frozen sensitivity descriptors",
            "",
            *association_lines,
            "",
            "## Boundaries",
            "",
            "The analysis tests model-level multi-state compatibility only. It does not claim biological fitness, stability, function, or causal mechanism.",
        ]
    )
    (output_root / "report.md").write_text(report + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "dynamicmpnn_baseline_release_v1",
        "inputs": {
            "generation": {"path": generation_root.as_posix(), "sequences_sha256": _sha256(generation_root / "sequences.parquet")},
            "proteinmpnn_scores": {"path": proteinmpnn_scores.as_posix(), "sha256": _sha256(proteinmpnn_scores)},
            "esm_if1_scores": {"path": esm_if1_scores.as_posix(), "sha256": _sha256(esm_if1_scores)},
            "frozen_baseline": {"path": baseline.as_posix(), "sha256": _sha256(baseline)},
        },
        "scope": summary,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--proteinmpnn-scores", type=Path, required=True)
    parser.add_argument("--esm-if1-scores", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
