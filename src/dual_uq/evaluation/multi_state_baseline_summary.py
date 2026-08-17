"""Summaries for the frozen multi-state APO/HOLO baseline experiment.

The module consumes model-specific score tables and generated-sequence shards.
It deliberately keeps the protein as the inference unit: sequence rows are
used only to form within-protein marginal diversity summaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ENDPOINTS = ("mean_compat", "worst_compat", "state_gap")
SINGLE_STATES = ("APO", "HOLO")


@dataclass(frozen=True, slots=True)
class MultiStateSummaryResult:
    """Canonical protein-level result and reproducible summary metadata."""

    protein_summary: pd.DataFrame
    associations: pd.DataFrame
    summary: dict[str, Any]


def _finite(values: Iterable[object]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size and not np.isfinite(array).all():
        raise ValueError("summary input contains non-finite values")
    return array


def summarize_score_table(table: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Aggregate one evaluator's raw score table by protein and source state."""
    required = {"protein_id", "source_state", *ENDPOINTS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"score table missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for (protein_id, source_state), group in table.groupby(
        ["protein_id", "source_state"], sort=True
    ):
        if source_state not in (*SINGLE_STATES, "MULTI"):
            raise ValueError(f"unexpected source_state: {source_state!r}")
        row: dict[str, Any] = {
            "protein_id": str(protein_id),
            "source_state": str(source_state),
            "n_sequences": int(len(group)),
        }
        for endpoint in ENDPOINTS:
            values = _finite(group[endpoint])
            row[f"{prefix}_{source_state.lower()}_{endpoint}"] = float(np.median(values))
            row[f"{prefix}_{source_state.lower()}_{endpoint}_q10"] = float(np.quantile(values, 0.1, method="linear"))
            row[f"{prefix}_{source_state.lower()}_{endpoint}_q90"] = float(np.quantile(values, 0.9, method="linear"))
        rows.append(row)
    if not rows:
        raise ValueError("score table is empty")
    result = pd.DataFrame(rows)
    return result


def pivot_score_summary(table: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Return one row per protein with APO/HOLO/MULTI endpoint columns."""
    summary = summarize_score_table(table, prefix=prefix)
    counts = summary.groupby("protein_id")["source_state"].agg(set)
    required = set((*SINGLE_STATES, "MULTI"))
    incomplete = sorted(protein for protein, states in counts.items() if states != required)
    if incomplete:
        raise ValueError(f"score summary lacks required states for {len(incomplete)} proteins")
    rows: list[dict[str, Any]] = []
    value_columns = [column for column in summary.columns if column not in {"protein_id", "source_state", "n_sequences"}]
    for protein_id, group in summary.groupby("protein_id", sort=True):
        row: dict[str, Any] = {"protein_id": str(protein_id)}
        for state in (*SINGLE_STATES, "MULTI"):
            state_rows = group.loc[group["source_state"].eq(state)]
            if len(state_rows) != 1:
                raise ValueError(f"protein {protein_id} has invalid {state} row count")
            source = state_rows.iloc[0]
            for column in value_columns:
                marker = f"_{state.lower()}_"
                if marker in column:
                    row[column] = source[column]
        rows.append(row)
    return pd.DataFrame(rows)


def marginal_hamming_diversity(sequences: Iterable[str]) -> float:
    """Mean pairwise Hamming fraction from empirical marginal amino-acid counts."""
    values = tuple(str(sequence) for sequence in sequences)
    if len(values) < 2:
        return 0.0
    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1 or not lengths:
        raise ValueError("sequences must have a common length")
    n = len(values)
    denominator = n * (n - 1)
    total = 0.0
    for position in range(next(iter(lengths))):
        counts = Counter(sequence[position] for sequence in values)
        matches = sum(count * (count - 1) for count in counts.values())
        total += 1.0 - matches / denominator
    return total / next(iter(lengths))


def marginal_between_divergence(left: Iterable[str], right: Iterable[str]) -> float:
    """Mean cross-condition Hamming fraction from empirical marginals."""
    left_values, right_values = tuple(map(str, left)), tuple(map(str, right))
    if not left_values or not right_values:
        raise ValueError("both sequence ensembles must be non-empty")
    lengths = {len(sequence) for sequence in (*left_values, *right_values)}
    if len(lengths) != 1:
        raise ValueError("sequences must have a common length")
    left_n, right_n = len(left_values), len(right_values)
    total = 0.0
    for position in range(next(iter(lengths))):
        left_counts = Counter(sequence[position] for sequence in left_values)
        right_counts = Counter(sequence[position] for sequence in right_values)
        matches = sum(left_counts[aa] * right_counts[aa] for aa in left_counts.keys() | right_counts.keys())
        total += 1.0 - matches / (left_n * right_n)
    return total / next(iter(lengths))


def summarize_sequence_ensembles(
    ensembles: Mapping[str, Mapping[str, Iterable[str]]],
) -> pd.DataFrame:
    """Summarize PDB/AFDB/MULTI marginal diversity for each protein."""
    rows: list[dict[str, Any]] = []
    for protein_id, states in sorted(ensembles.items()):
        if set(states) != {"APO", "HOLO", "MULTI"}:
            raise ValueError(f"protein {protein_id} lacks one of APO/HOLO/MULTI ensembles")
        apo, holo, multi = (tuple(map(str, states[key])) for key in ("APO", "HOLO", "MULTI"))
        rows.append(
            {
                "protein_id": str(protein_id),
                "diversity_apo": marginal_hamming_diversity(apo),
                "diversity_holo": marginal_hamming_diversity(holo),
                "diversity_multi": marginal_hamming_diversity(multi),
                "divergence_apo_holo": marginal_between_divergence(apo, holo),
                "divergence_apo_multi": marginal_between_divergence(apo, multi),
                "divergence_holo_multi": marginal_between_divergence(holo, multi),
                "n_sequences_apo": len(apo),
                "n_sequences_holo": len(holo),
                "n_sequences_multi": len(multi),
            }
        )
    if not rows:
        raise ValueError("no sequence ensembles supplied")
    return pd.DataFrame(rows)


def _association_rows(table: pd.DataFrame, *, target: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor in (
        "proteinmpnn_d_excess_64",
        "esm_if1_d_excess_64",
        "proteinmpnn_js_bits_mean_64",
        "esm_if1_js_bits_mean_64",
        "proteinmpnn_local_burden_mean",
        "esm_if1_local_burden_mean",
    ):
        if descriptor not in table or target not in table:
            continue
        valid = table[[descriptor, target]].dropna()
        if len(valid) < 3 or valid[descriptor].nunique() < 2 or valid[target].nunique() < 2:
            continue
        rho, _ = spearmanr(valid[descriptor], valid[target])
        rows.append(
            {
                "evaluator": prefix,
                "target": target,
                "descriptor": descriptor,
                "n_proteins": int(len(valid)),
                "spearman_rho": float(rho),
            }
        )
    return rows


def build_summary(
    *,
    single_tables: Mapping[str, pd.DataFrame],
    multi_tables: Mapping[str, pd.DataFrame],
    sequence_ensembles: Mapping[str, Mapping[str, Iterable[str]]],
    upstream: pd.DataFrame,
) -> MultiStateSummaryResult:
    """Build the sole structured Part F/G result consumed by all renderers."""
    if set(single_tables) != set(multi_tables):
        raise ValueError("single and multi evaluator sets differ")
    if not single_tables:
        raise ValueError("at least one evaluator is required")
    merged: pd.DataFrame | None = None
    for evaluator in sorted(single_tables):
        single = single_tables[evaluator].loc[
            single_tables[evaluator]["source_state"].isin(SINGLE_STATES)
        ].copy()
        multi = multi_tables[evaluator].loc[
            multi_tables[evaluator]["source_state"].eq("MULTI")
        ].copy()
        if single.empty or multi.empty:
            raise ValueError(f"{evaluator} lacks generated APO/HOLO or MULTI rows")
        single = pd.concat([single, multi], ignore_index=True)
        score_summary = pivot_score_summary(single, prefix=evaluator.lower().replace("-", "_"))
        merged = score_summary if merged is None else merged.merge(score_summary, on="protein_id", how="inner", validate="one_to_one")
    diversity = summarize_sequence_ensembles(sequence_ensembles)
    result = merged.merge(diversity, on="protein_id", how="inner", validate="one_to_one")
    if "protein_id" not in upstream:
        raise ValueError("upstream summary lacks protein_id")
    result = result.merge(upstream, on="protein_id", how="left", validate="one_to_one")
    if result.empty:
        raise ValueError("no shared proteins after summary join")
    for evaluator in sorted(single_tables):
        prefix = evaluator.lower().replace("-", "_")
        result[f"{prefix}_multi_minus_best_single_worst"] = result[
            f"{prefix}_multi_worst_compat"
        ] - result[[f"{prefix}_apo_worst_compat", f"{prefix}_holo_worst_compat"]].max(axis=1)
    associations: list[dict[str, Any]] = []
    for evaluator in sorted(single_tables):
        prefix = evaluator.lower().replace("-", "_")
        target = f"{prefix}_multi_minus_best_single_worst"
        associations.extend(_association_rows(result, target=target, prefix=evaluator))
    association_table = pd.DataFrame(associations)
    summary: dict[str, Any] = {
        "protein_count": int(len(result)),
        "evaluators": sorted(single_tables),
        "primary_endpoint": "worst_compat",
        "diversity_definition": "mean marginal pairwise Hamming fraction within protein; between-condition divergence is marginal only",
        "multi_minus_best_single_definition": "MULTI median endpoint minus max(APO median, HOLO median), higher score direction retained without biological interpretation",
        "associations": associations,
    }
    return MultiStateSummaryResult(result, association_table, summary)
