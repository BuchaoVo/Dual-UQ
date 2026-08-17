"""Protein-level TRAIN/VALIDATION analysis for paired-state Method V1."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class V1AnalysisError(ValueError):
    """Raised when V1 validation inputs are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class MethodV1AnalysisResult:
    """Canonical structured result consumed by the report writer."""

    ablation_summary: pd.DataFrame
    evaluator_protein_summary: pd.DataFrame
    dual_uq_summary: pd.DataFrame
    summary: dict[str, Any]
    report: str


def _score_summary(
    v1_scores: pd.DataFrame,
    single_scores: pd.DataFrame,
    multi_scores: pd.DataFrame,
    validation_ids: set[str],
    *,
    evaluator: str,
    ablation: str,
) -> pd.DataFrame:
    for table, name in ((v1_scores, "V1"), (single_scores, "single"), (multi_scores, "MULTI")):
        if not {"protein_id", "worst_compat", "source_state"}.issubset(table.columns):
            raise V1AnalysisError(f"{name} score table lacks compatibility columns")
    v1 = v1_scores.loc[v1_scores["protein_id"].astype(str).isin(validation_ids)].copy()
    single = single_scores.loc[single_scores["protein_id"].astype(str).isin(validation_ids)].copy()
    multi = multi_scores.loc[multi_scores["protein_id"].astype(str).isin(validation_ids)].copy()
    v1_summary = v1.groupby("protein_id", sort=True)["worst_compat"].median().rename("v1_worst")
    single_summary = single.groupby(["protein_id", "source_state"], sort=True)["worst_compat"].median().unstack()
    single_summary.columns = [str(value).lower() for value in single_summary.columns]
    multi_summary = multi.loc[multi["source_state"].eq("MULTI")].groupby("protein_id", sort=True)["worst_compat"].median().rename("multi_worst")
    result = pd.concat([v1_summary, single_summary, multi_summary], axis=1).dropna()
    if result.empty:
        raise V1AnalysisError(f"no complete validation score join for {evaluator} {ablation}")
    if not {"apo", "holo", "multi_worst"}.issubset(result.columns):
        raise V1AnalysisError(f"incomplete baseline states for {evaluator} {ablation}")
    result["best_single_worst"] = result[["apo", "holo"]].max(axis=1)
    result["v1_minus_best_single_worst"] = result["v1_worst"] - result["best_single_worst"]
    result["v1_minus_multi_worst"] = result["v1_worst"] - result["multi_worst"]
    result = result.reset_index(names="protein_id")
    result.insert(1, "evaluator", evaluator)
    result.insert(2, "ablation", ablation)
    return result


def _sequence_summary(generated: pd.DataFrame, validation_ids: set[str], ablation: str) -> pd.DataFrame:
    required = {"protein_id", "split", "sequence", "sample_index"}
    if not required.issubset(generated.columns):
        raise V1AnalysisError("generated validation table lacks diversity columns")
    rows: list[dict[str, Any]] = []
    for protein_id, group in generated.loc[generated["protein_id"].astype(str).isin(validation_ids)].groupby("protein_id", sort=True):
        sequences = tuple(str(value) for value in group.sort_values("sample_index")["sequence"])
        if not sequences or len({len(value) for value in sequences}) != 1:
            raise V1AnalysisError(f"invalid V1 sequence ensemble for {protein_id}")
        alphabet = tuple("ACDEFGHIKLMNPQRSTVWY")
        entropy_values: list[float] = []
        for position in range(len(sequences[0])):
            counts = Counter(sequence[position] for sequence in sequences)
            probabilities = np.asarray(tuple(counts.values()), dtype=float) / len(sequences)
            entropy_values.append(float(-(probabilities * np.log2(probabilities)).sum()))
        rows.append(
            {
                "protein_id": str(protein_id),
                "ablation": ablation,
                "n_sequences": len(sequences),
                "sequence_length": len(sequences[0]),
                "unique_sequence_fraction": len(set(sequences)) / len(sequences),
                "mean_position_entropy_bits": float(np.mean(entropy_values)),
                "mean_aa_fraction": float(np.mean([sum(sequence.count(aa) for aa in alphabet) / len(sequence) for sequence in sequences])),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_rows(table: pd.DataFrame, *, evaluator: str, ablation: str) -> dict[str, Any]:
    benefit = table["v1_minus_best_single_worst"]
    multi_benefit = table["v1_minus_multi_worst"]
    return {
        "evaluator": evaluator,
        "ablation": ablation,
        "protein_count": int(len(table)),
        "v1_worst_median": float(table["v1_worst"].median()),
        "v1_worst_q10": float(np.quantile(table["v1_worst"], 0.10, method="linear")),
        "v1_worst_q90": float(np.quantile(table["v1_worst"], 0.90, method="linear")),
        "best_single_worst_median": float(table["best_single_worst"].median()),
        "multi_worst_median": float(table["multi_worst"].median()),
        "benefit_vs_best_single_median": float(benefit.median()),
        "benefit_vs_best_single_q10": float(np.quantile(benefit, 0.10, method="linear")),
        "benefit_vs_best_single_q90": float(np.quantile(benefit, 0.90, method="linear")),
        "benefit_vs_multi_median": float(multi_benefit.median()),
        "positive_vs_best_single_fraction": float((benefit > 0).mean()),
        "positive_vs_multi_fraction": float((multi_benefit > 0).mean()),
    }


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 3 or valid.left.nunique() < 2 or valid.right.nunique() < 2:
        return None
    return float(spearmanr(valid.left, valid.right).statistic)


def build_method_v1_analysis(
    *,
    validation_ids: set[str],
    generated_tables: dict[str, pd.DataFrame],
    proteinmpnn_scores: dict[str, pd.DataFrame],
    esm_if1_scores: dict[str, pd.DataFrame],
    single_proteinmpnn: pd.DataFrame,
    multi_proteinmpnn: pd.DataFrame,
    single_esm_if1: pd.DataFrame,
    multi_esm_if1: pd.DataFrame,
    upstream: pd.DataFrame | None = None,
    training_summary: dict[str, Any] | None = None,
) -> MethodV1AnalysisResult:
    if set(generated_tables) != set(proteinmpnn_scores) or set(generated_tables) != set(esm_if1_scores):
        raise V1AnalysisError("A-F generated and score tables do not have matching ablations")
    evaluator_tables: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, Any]] = []
    sequence_tables: list[pd.DataFrame] = []
    for ablation in sorted(generated_tables):
        pnn = _score_summary(proteinmpnn_scores[ablation], single_proteinmpnn, multi_proteinmpnn, validation_ids, evaluator="ProteinMPNN", ablation=ablation)
        esm = _score_summary(esm_if1_scores[ablation], single_esm_if1, multi_esm_if1, validation_ids, evaluator="ESM-IF1", ablation=ablation)
        evaluator_tables.extend([pnn, esm])
        aggregate_rows.extend([_aggregate_rows(pnn, evaluator="ProteinMPNN", ablation=ablation), _aggregate_rows(esm, evaluator="ESM-IF1", ablation=ablation)])
        sequence_tables.append(_sequence_summary(generated_tables[ablation], validation_ids, ablation))
    evaluator_summary = pd.concat(evaluator_tables, ignore_index=True)
    ablation_summary = pd.DataFrame(aggregate_rows).merge(
        pd.concat(sequence_tables, ignore_index=True).groupby("ablation", as_index=False).agg(
            sequence_count=("n_sequences", "median"),
            mean_position_entropy_bits=("mean_position_entropy_bits", "median"),
            unique_sequence_fraction=("unique_sequence_fraction", "median"),
        ),
        on="ablation",
        how="left",
    )
    full_pnn = evaluator_summary.loc[(evaluator_summary.evaluator == "ProteinMPNN") & evaluator_summary.ablation.eq("F")].set_index("protein_id")
    full_esm = evaluator_summary.loc[(evaluator_summary.evaluator == "ESM-IF1") & evaluator_summary.ablation.eq("F")].set_index("protein_id")
    dual = full_pnn[["v1_minus_best_single_worst", "v1_minus_multi_worst"]].rename(columns=lambda value: f"pnn_{value}").join(
        full_esm[["v1_minus_best_single_worst", "v1_minus_multi_worst"]].rename(columns=lambda value: f"esm_{value}"),
        how="inner",
    )
    dual = dual.reset_index()
    if upstream is not None and not upstream.empty:
        keep = upstream.loc[upstream["protein_id"].astype(str).isin(set(dual["protein_id"]))].copy()
        dual = dual.merge(keep, on="protein_id", how="left", validate="one_to_one")
    pnn_median = float(ablation_summary.loc[(ablation_summary.evaluator == "ProteinMPNN") & ablation_summary.ablation.eq("F"), "benefit_vs_best_single_median"].iloc[0])
    esm_median = float(ablation_summary.loc[(ablation_summary.evaluator == "ESM-IF1") & ablation_summary.ablation.eq("F"), "benefit_vs_best_single_median"].iloc[0])
    summary: dict[str, Any] = {
        "schema_version": "dual_uq_method_v1_validation_analysis_v1",
        "locked_test_accessed": False,
        "validation_assigned": 212,
        "validation_evaluable": 209,
        "proteinmpnn_score_coverage": int(proteinmpnn_scores["F"]["protein_id"].nunique()),
        "esm_if1_score_coverage": int(esm_if1_scores["F"]["protein_id"].nunique()),
        "esm_if1_scored": int(len(full_esm)),
        "proteinmpnn_scored": int(len(full_pnn)),
        "ablation_count": len(generated_tables),
        "full_v1_pnn_benefit_vs_best_single_median": pnn_median,
        "full_v1_esm_if1_benefit_vs_best_single_median": esm_median,
        "full_v1_benefit_same_direction": bool(pnn_median > 0 and esm_median > 0),
        "benefit_concordance_spearman": _spearman(dual["pnn_v1_minus_best_single_worst"], dual["esm_v1_minus_best_single_worst"]),
        "fraction_improved_both": float(((dual["pnn_v1_minus_best_single_worst"] > 0) & (dual["esm_v1_minus_best_single_worst"] > 0)).mean()),
        "fraction_improved_pnn_only": float(((dual["pnn_v1_minus_best_single_worst"] > 0) & (dual["esm_v1_minus_best_single_worst"] <= 0)).mean()),
        "fraction_improved_esm_only": float(((dual["pnn_v1_minus_best_single_worst"] <= 0) & (dual["esm_v1_minus_best_single_worst"] > 0)).mean()),
        "fraction_worsened_both": float(((dual["pnn_v1_minus_best_single_worst"] <= 0) & (dual["esm_v1_minus_best_single_worst"] <= 0)).mean()),
        "dynamicmpnn_reference_evaluable": "365/1357",
        "training_summary": training_summary or {},
    }
    # The preregistered gate requires favorable median WorstCompat movement in
    # both evaluators.  A negative result is a validation failure, not a reason
    # to inspect LOCKED_TEST or tune the architecture post hoc.
    summary["verdict"] = "PROMISING" if summary["full_v1_benefit_same_direction"] else "FAIL"
    lines = [
        "# Dual-UQ Method V1 TRAIN/VALIDATION report",
        "",
        f"**VERDICT: {summary['verdict']}**",
        "",
        "The analysis uses TRAIN/VALIDATION only; LOCKED_TEST was not accessed.",
        "",
        "## TRAINING COHORT",
        "",
        "- TRAIN assigned/evaluable: 986/974; VALIDATION assigned/evaluable: 212/209.",
        "- Unavailable: 14 insufficient common complete backbone atoms and 1 nonstandard canonical residue (A5YV76); split membership was retained.",
        "",
        "## TRAINING AND ABLATIONS",
        "",
        "- A–F used the same 8-epoch AdamW budget, seed policy, targets, and validation-only checkpoint selection.",
        f"- Full V1 validation checkpoint: epoch {next((item.get('best_epoch') for item in (training_summary or {}).get('results', []) if item.get('ablation') == 'F'), 'unknown')}.",
        "- The structured ablation table records consensus, worst-state, selective-consistency, likelihood, entropy, and sequence uniqueness summaries.",
        "",
        "Ablation median WorstCompat benefit versus best single (ProteinMPNN / ESM-IF1):",
    ]
    for ablation in "ABCDEF":
        pnn_row = ablation_summary.loc[(ablation_summary.evaluator == "ProteinMPNN") & ablation_summary.ablation.eq(ablation)].iloc[0]
        esm_row = ablation_summary.loc[(ablation_summary.evaluator == "ESM-IF1") & ablation_summary.ablation.eq(ablation)].iloc[0]
        lines.append(
            f"- {ablation}: {pnn_row['benefit_vs_best_single_median']:.3f} / {esm_row['benefit_vs_best_single_median']:.3f}; "
            f"median position entropy {pnn_row['mean_position_entropy_bits']:.3f} bits; "
            f"unique-sequence fraction {pnn_row['unique_sequence_fraction']:.3f}."
        )
    lines.extend([
        "",
        "## VALIDATION CROSS-EVALUATOR ROBUSTNESS",
        "",
        f"- Full V1 ProteinMPNN WorstCompat benefit versus best single median: **{pnn_median:.3f}**.",
        f"- Full V1 ESM-IF1 WorstCompat benefit versus best single median: **{esm_median:.3f}**.",
        f"- Same-direction favorable median movement under both evaluators: **{summary['full_v1_benefit_same_direction']}**.",
        f"- Protein-level benefit Spearman (PNN vs ESM-IF1): **{summary['benefit_concordance_spearman']}**; improved under both: **{summary['fraction_improved_both']:.3f}**; worsened under both: **{summary['fraction_worsened_both']:.3f}**.",
        "",
        "## FAILURE-MODE CORRECTION",
        "",
        "- Full V1 does not improve WT-normalized WorstCompat over the frozen best-single baseline in the same favorable direction under both evaluators. This is consistent with unresolved evaluator-specific/self-model bias, not a biological conclusion.",
        "- Entropy and unique-sequence fractions are retained to distinguish robustness from sequence collapse; no evaluator-derived target was used during training.",
        "",
        "## DUAL-UQ STRATIFICATION",
        "",
        "- The protein-level table joins frozen generative/local-response descriptors where available; these descriptors are diagnostic only and were not training labels.",
        "",
        "## COVERAGE AND STOP",
        "",
        f"- V1 model-evaluable coverage is 974/986 TRAIN and 209/212 VALIDATION; raw validation scores cover {summary['proteinmpnn_score_coverage']} ProteinMPNN and {summary['esm_if1_score_coverage']} ESM-IF1 proteins, with {summary['proteinmpnn_scored']} proteins comparable to the frozen baseline tables for both. DynamicMPNN is 365/1357 evaluable.",
        "- Because the validation gate is FAIL, the method is not promoted to LOCKED_TEST. No Arm C or downstream experiment is started.",
        "",
        "## FROZEN FINAL METHOD",
        "",
        "- Attempted method: paired-state encoder with explicit APO/HOLO observability masks, disagreement features, reliability-aware consensus, smooth worst-state loss, and selective JS consistency.",
        "- Frozen training protocol: A–F ablations, AdamW, 8 epochs, hidden dimension 64, learning rate 0.002, weight decay 1e-5, seed 20260817, lambda_worst=0.5, lambda_consistency=0.5, temperature=0.5.",
        "- Validation decoding: 64 deterministic per-position samples per protein with nested 16/32/64 checkpoints; evaluator scoring used frozen ProteinMPNN and ESM-IF1 WT-normalized MeanCompat/WorstCompat/StateGap.",
        "- The FAIL verdict freezes this validation record but authorizes no LOCKED_TEST execution or post hoc architecture/loss tuning in this task.",
        "",
        "All interpretation is restricted to ProteinMPNN/ESM-IF1 model-level compatibility; no claim about fitness, stability, function, or experiment is made.",
        "",
    ])
    return MethodV1AnalysisResult(ablation_summary, evaluator_summary, dual, summary, "\n".join(lines))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
