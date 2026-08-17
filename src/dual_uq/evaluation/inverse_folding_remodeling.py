"""Local inverse-folding compatibility remodeling analysis.

This module consumes the frozen paired ProteinMPNN measurement layer and the
Pair Validity table.  It characterizes magnitude (the existing ``P``
definition), candidate-space breadth, and preference reordering.  It does not
run a scorer, alter frozen measurements, construct a composite phenotype, or
make biological/functional claims.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file

ANALYSIS_PROTOCOL = "local_inverse_folding_remodeling"
P_METRIC_DEFINITION = "mean_over_19_candidates(abs(mean_over_repeats_D))"
BREADTH_DEFINITION = "N_eff/19 where N_eff=1/sum((abs(Delta)/sum(abs(Delta)))^2)"
REORDERING_DESCRIPTORS = (
    "kendall_tau",
    "top3_overlap",
    "top5_overlap",
    "rank_displacement",
    "profile_pearson",
)
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EPSILON = 1.0e-12
PAIRED_COLUMNS = [
    "protein_id",
    "position",
    "wt_aa",
    "mut_aa",
    "repeat_index",
    "candidate_remodeling_d",
    "pdb_delta_score_vs_wt",
    "afdb_delta_score_vs_wt",
]
CANDIDATE_COLUMNS = [
    "protein_id",
    "position",
    "wt_aa",
    "mut_aa",
    "candidate_remodeling_d",
    "pdb_delta_score_vs_wt",
    "afdb_delta_score_vs_wt",
    "n_repeats",
]


class RemodelingAnalysisError(ValueError):
    """Structured input or materialization failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass
class RemodelingAnalysisInputs:
    """Validated analysis inputs; ``paired`` may be raw or candidate-aggregated."""

    paired: pd.DataFrame
    pair_validity: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]


@dataclass
class RemodelingAnalysisResult:
    """Canonical result from which all tables, figures, and report are derived."""

    candidate: pd.DataFrame
    position: pd.DataFrame
    protein: pd.DataFrame
    representatives: pd.DataFrame
    cohort_summary: dict[str, Any]
    input_provenance: dict[str, dict[str, Any]]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RemodelingAnalysisError("schema_mismatch", f"{label} is missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RemodelingAnalysisError("nonfinite_input", f"{label} contains non-finite values")


def _quantile(values: pd.Series, probability: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numeric):
        return None
    return float(np.quantile(numeric, probability, method="linear"))


def _aggregate_raw_candidates(paired: pd.DataFrame) -> pd.DataFrame:
    _require_columns(paired, PAIRED_COLUMNS, "paired structural response")
    _finite(
        paired,
        ["candidate_remodeling_d", "pdb_delta_score_vs_wt", "afdb_delta_score_vs_wt"],
        "paired structural response",
    )
    key = ["protein_id", "position", "mut_aa"]
    if paired.duplicated(key + ["repeat_index"]).any():
        raise RemodelingAnalysisError("duplicate_key", "paired scientific keys are duplicated")
    result = (
        paired.groupby(key, sort=False)
        .agg(
            wt_aa=("wt_aa", "first"),
            candidate_remodeling_d=("candidate_remodeling_d", "mean"),
            pdb_delta_score_vs_wt=("pdb_delta_score_vs_wt", "mean"),
            afdb_delta_score_vs_wt=("afdb_delta_score_vs_wt", "mean"),
            n_repeats=("repeat_index", "nunique"),
        )
        .reset_index()
    )
    return result


def _validate_candidate_grid(candidate: pd.DataFrame) -> None:
    _require_columns(candidate, CANDIDATE_COLUMNS, "candidate response")
    _finite(
        candidate,
        ["candidate_remodeling_d", "pdb_delta_score_vs_wt", "afdb_delta_score_vs_wt", "n_repeats"],
        "candidate response",
    )
    if candidate.duplicated(["protein_id", "position", "mut_aa"]).any():
        raise RemodelingAnalysisError("duplicate_key", "candidate response keys are duplicated")
    candidate_sizes = candidate.groupby(["protein_id", "position"], sort=False).size()
    if not (candidate_sizes == 19).all():
        bad = candidate_sizes[candidate_sizes != 19].head(5).to_dict()
        raise RemodelingAnalysisError("candidate_grid_mismatch", f"positions do not contain 19 candidates: {bad}")
    repeat_counts = candidate.groupby(["protein_id", "position"], sort=False)["n_repeats"].unique()
    if any(len(values) != 1 or int(values[0]) < 1 for values in repeat_counts):
        raise RemodelingAnalysisError("repeat_grid_mismatch", "candidate repeat counts are inconsistent")
    invalid_aa = sorted(set(candidate["mut_aa"].astype(str)) - AMINO_ACIDS)
    if invalid_aa:
        raise RemodelingAnalysisError("amino_acid_schema_mismatch", f"unknown candidate amino acids: {invalid_aa}")
    for (protein_id, position), group in candidate.groupby(["protein_id", "position"], sort=False):
        wild_types = set(group["wt_aa"].astype(str))
        if len(wild_types) != 1 or next(iter(wild_types), "") not in AMINO_ACIDS:
            raise RemodelingAnalysisError("candidate_grid_mismatch", f"invalid WT amino acid for {(protein_id, position)}")
        wt_aa = next(iter(wild_types))
        expected = AMINO_ACIDS - {wt_aa}
        observed = set(group["mut_aa"].astype(str))
        if observed != expected:
            raise RemodelingAnalysisError(
                "candidate_grid_mismatch",
                f"candidate amino-acid set differs from non-WT set for {(protein_id, position)}",
            )


def _profile_metrics(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("mut_aa", kind="mergesort")
    pdb = ordered["pdb_delta_score_vs_wt"].to_numpy(dtype=float)
    afdb = ordered["afdb_delta_score_vs_wt"].to_numpy(dtype=float)
    pdb_rank = pd.Series(pdb).rank(method="average", ascending=False).to_numpy()
    afdb_rank = pd.Series(afdb).rank(method="average", ascending=False).to_numpy()
    max_rank_distance = max(len(pdb) - 1, 1)
    tau = kendalltau(pdb, afdb, nan_policy="raise").statistic
    if not np.isfinite(tau):
        tau = 1.0 if np.array_equal(pdb, afdb) else 0.0
    if np.std(pdb) <= EPSILON or np.std(afdb) <= EPSILON:
        pearson = 1.0 if np.allclose(pdb, afdb, atol=EPSILON, rtol=EPSILON) else 0.0
    else:
        pearson = float(np.corrcoef(pdb, afdb)[0, 1])
    metrics: dict[str, Any] = {
        "kendall_tau": float(tau),
        "profile_pearson": float(pearson),
        "rank_displacement": float(np.mean(np.abs(pdb_rank - afdb_rank)) / max_rank_distance),
        "top1_flip": bool(np.argmax(pdb) != np.argmax(afdb)),
    }
    for k in (3, 5):
        top_pdb = set(np.argsort(-pdb, kind="stable")[:k])
        top_afdb = set(np.argsort(-afdb, kind="stable")[:k])
        metrics[f"top{k}_overlap"] = float(len(top_pdb & top_afdb) / k)
    return metrics


def _build_position_table(candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protein_id, position), group in candidate.groupby(["protein_id", "position"], sort=False):
        delta = group["candidate_remodeling_d"].to_numpy(dtype=float)
        absolute = np.abs(delta)
        total = float(absolute.sum())
        if total == 0.0:
            breadth = np.nan
            effective = np.nan
            breadth_status = "zero_remodeling"
        else:
            weights = absolute / total
            effective = float(1.0 / np.square(weights).sum())
            breadth = float(effective / 19.0)
            breadth_status = "defined"
        row: dict[str, Any] = {
            "protein_id": protein_id,
            "position": int(position),
            "magnitude_p": float(absolute.mean()),
            "position_mean_abs_interaction": float(absolute.mean()),
            "effective_participation": effective,
            "breadth_b": breadth,
            "breadth_status": breadth_status,
            "candidate_count": len(group),
            "repeat_count": int(group["n_repeats"].iloc[0]),
        }
        row.update(_profile_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _cohort_label(pair_validity: pd.DataFrame) -> pd.DataFrame:
    _require_columns(pair_validity, {"protein_id", "high_comparability_eligible"}, "Pair Validity")
    if pair_validity["protein_id"].duplicated().any():
        raise RemodelingAnalysisError("duplicate_pair_validity", "Pair Validity contains duplicate proteins")
    values = pair_validity["high_comparability_eligible"]
    if not pd.api.types.is_bool_dtype(values):
        raise RemodelingAnalysisError(
            "cohort_label_schema_mismatch",
            "Pair Validity high_comparability_eligible must be boolean",
        )
    if values.isna().any():
        raise RemodelingAnalysisError("missing_cohort_label", "Pair Validity clean/full label is missing")
    result = pair_validity[["protein_id", "high_comparability_eligible"]].copy()
    result["cohort"] = np.where(result["high_comparability_eligible"].astype(bool), "clean", "full")
    return result[["protein_id", "cohort"]]


def _build_protein_table(position: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "cohort" in position.columns:
        work = position.copy()
    else:
        work = position.merge(labels, on="protein_id", how="left", validate="many_to_one")
    if work["cohort"].isna().any():
        raise RemodelingAnalysisError("cohort_identity_mismatch", "response proteins are absent from Pair Validity")
    rows: list[dict[str, Any]] = []
    for protein_id, group in work.groupby("protein_id", sort=False):
        defined = group.loc[group["breadth_status"] == "defined", "breadth_b"]
        rows.append(
            {
                "protein_id": protein_id,
                "cohort": str(group["cohort"].iloc[0]),
                "position_count": len(group),
                "magnitude_median": float(group["magnitude_p"].median()),
                "magnitude_q90": _quantile(group["magnitude_p"], 0.90),
                "magnitude_q95": _quantile(group["magnitude_p"], 0.95),
                "magnitude_iqr": float(group["magnitude_p"].quantile(0.75) - group["magnitude_p"].quantile(0.25)),
                "breadth_median": float(defined.median()) if len(defined) else np.nan,
                "breadth_q90": _quantile(defined, 0.90),
                "breadth_defined_fraction": float(len(defined) / len(group)),
                "reordering_kendall_median": float(group["kendall_tau"].median()),
                "reordering_rank_displacement_median": float(group["rank_displacement"].median()),
                "reordering_rank_displacement_q90": _quantile(group["rank_displacement"], 0.90),
                "reordering_burden_rank_displacement": float(group["rank_displacement"].mean()),
                "negative_kendall_fraction": float((group["kendall_tau"] < 0).mean()),
                "top3_overlap_median": float(group["top3_overlap"].median()),
                "position_magnitude_upper_tail_excess": float(
                    group["magnitude_p"].quantile(0.90) - group["magnitude_p"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_representatives(protein: pd.DataFrame) -> pd.DataFrame:
    clean = protein.loc[protein["cohort"] == "clean"].copy()
    if clean.empty:
        raise RemodelingAnalysisError("empty_clean_cohort", "cannot select representatives without clean proteins")
    criteria = (
        ("magnitude_median", "highest_typical_magnitude"),
        ("breadth_median", "highest_typical_breadth"),
        ("reordering_burden_rank_displacement", "highest_reordering_burden"),
        ("magnitude_median", "lowest_typical_magnitude"),
    )
    rows: list[dict[str, Any]] = []
    for metric, reason in criteria:
        ordered = clean.sort_values([metric, "protein_id"], ascending=[reason == "lowest_typical_magnitude", True], kind="mergesort")
        selected = ordered.iloc[0]
        rows.append(
            {
                "protein_id": selected["protein_id"],
                "selection_reason": reason,
                "selection_metric": metric,
                "selection_value": float(selected[metric]) if pd.notna(selected[metric]) else None,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["protein_id", "selection_reason"], ignore_index=True)


def _summary_for(cohort: str, position: pd.DataFrame, protein: pd.DataFrame) -> dict[str, Any]:
    # ``full`` is the complete frozen 127-protein sensitivity population;
    # ``clean`` is the strict 68-protein primary subset.
    if cohort == "full":
        p = position
        proteins = protein
    else:
        p = position.loc[position["cohort"] == "clean"]
        proteins = protein.loc[protein["cohort"] == "clean"]
    return {
        "protein_count": int(proteins["protein_id"].nunique()),
        "position_count": len(p),
        "magnitude_p_median": _quantile(p["magnitude_p"], 0.50),
        "magnitude_p_q90": _quantile(p["magnitude_p"], 0.90),
        "magnitude_p_q95": _quantile(p["magnitude_p"], 0.95),
        "breadth_b_median": _quantile(p["breadth_b"], 0.50),
        "breadth_b_q90": _quantile(p["breadth_b"], 0.90),
        "breadth_defined_fraction": float((p["breadth_status"] == "defined").mean()) if len(p) else None,
        "kendall_tau_median": _quantile(p["kendall_tau"], 0.50),
        "rank_displacement_median": _quantile(p["rank_displacement"], 0.50),
        "rank_displacement_q90": _quantile(p["rank_displacement"], 0.90),
        "top1_flip_fraction": float(p["top1_flip"].mean()) if len(p) else None,
        "protein_magnitude_median": _quantile(proteins["magnitude_median"], 0.50),
        "protein_magnitude_q90": _quantile(proteins["magnitude_median"], 0.90),
        "protein_breadth_median": _quantile(proteins["breadth_median"], 0.50),
        "protein_reordering_burden_median": _quantile(
            proteins["reordering_burden_rank_displacement"], 0.50
        ),
    }


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def build_remodeling_result(inputs: RemodelingAnalysisInputs) -> RemodelingAnalysisResult:
    """Build all magnitude, breadth, and reordering tables from one input layer."""
    paired = inputs.paired.copy()
    if "repeat_index" in paired.columns:
        candidate = _aggregate_raw_candidates(paired)
    else:
        candidate = paired.copy()
        _require_columns(candidate, CANDIDATE_COLUMNS, "candidate response")
    _validate_candidate_grid(candidate)
    position = _build_position_table(candidate)
    labels = _cohort_label(inputs.pair_validity)
    position = position.merge(labels, on="protein_id", how="left", validate="many_to_one")
    if position["cohort"].isna().any():
        raise RemodelingAnalysisError("cohort_identity_mismatch", "response proteins are absent from Pair Validity")
    protein = _build_protein_table(position, labels)
    representatives = _select_representatives(protein)
    summary = {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "p_metric_definition": P_METRIC_DEFINITION,
        "breadth_definition": BREADTH_DEFINITION,
        "reordering_descriptors": list(REORDERING_DESCRIPTORS),
        "cohort": {
            "clean": _summary_for("clean", position, protein),
            "full": _summary_for("full", position, protein),
        },
        "scientific_unit": "protein for cohort summaries; position for local response",
        "top1_flip_role": "diagnostic_only_not_primary_endpoint",
        "interpretation_boundary": "local inverse-folding compatibility remodeling only; no biological or functional inference",
    }
    clean_summary = summary["cohort"]["clean"]
    full_summary = summary["cohort"]["full"]
    summary["clean_vs_full_difference"] = {
        "magnitude_p_median_difference_clean_minus_full": _difference(clean_summary["magnitude_p_median"], full_summary["magnitude_p_median"]),
        "magnitude_p_median_ratio_clean_over_full": (
            clean_summary["magnitude_p_median"] / full_summary["magnitude_p_median"]
            if clean_summary["magnitude_p_median"] is not None and full_summary["magnitude_p_median"]
            else None
        ),
        "breadth_b_median_difference_clean_minus_full": _difference(clean_summary["breadth_b_median"], full_summary["breadth_b_median"]),
        "kendall_tau_median_difference_clean_minus_full": _difference(clean_summary["kendall_tau_median"], full_summary["kendall_tau_median"]),
        "rank_displacement_median_difference_clean_minus_full": _difference(clean_summary["rank_displacement_median"], full_summary["rank_displacement_median"]),
        "top1_flip_fraction_difference_clean_minus_full": _difference(clean_summary["top1_flip_fraction"], full_summary["top1_flip_fraction"]),
    }
    return RemodelingAnalysisResult(
        candidate=candidate.merge(labels, on="protein_id", how="left", validate="many_to_one"),
        position=position,
        protein=protein,
        representatives=representatives,
        cohort_summary=summary,
        input_provenance=inputs.input_provenance,
    )


def _aggregate_paired_file(path: Path) -> pd.DataFrame:
    """Stream the large frozen paired table into one row per candidate position."""
    from pyarrow import parquet

    partial: list[pd.DataFrame] = []
    reader = parquet.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=250_000, columns=PAIRED_COLUMNS):
        frame = batch.to_pandas()
        partial.append(
            frame.groupby(["protein_id", "position", "wt_aa", "mut_aa"], sort=False).agg(
                candidate_remodeling_d=("candidate_remodeling_d", "sum"),
                pdb_delta_score_vs_wt=("pdb_delta_score_vs_wt", "sum"),
                afdb_delta_score_vs_wt=("afdb_delta_score_vs_wt", "sum"),
                n_repeats=("repeat_index", "count"),
                repeat_mask=("repeat_index", lambda values: int(np.bitwise_or.reduce(1 << values.to_numpy(dtype=np.int64)))),
            ).reset_index()
        )
    if not partial:
        raise RemodelingAnalysisError("empty_input", "paired structural response is empty")
    combined = pd.concat(partial, ignore_index=True)
    result = combined.groupby(["protein_id", "position", "wt_aa", "mut_aa"], sort=False).agg(
        candidate_remodeling_d=("candidate_remodeling_d", "sum"),
        pdb_delta_score_vs_wt=("pdb_delta_score_vs_wt", "sum"),
        afdb_delta_score_vs_wt=("afdb_delta_score_vs_wt", "sum"),
        n_repeats=("n_repeats", "sum"),
        repeat_mask=("repeat_mask", lambda values: int(np.bitwise_or.reduce(values.to_numpy(dtype=np.int64)))),
    ).reset_index()
    result[["candidate_remodeling_d", "pdb_delta_score_vs_wt", "afdb_delta_score_vs_wt"]] = result[
        ["candidate_remodeling_d", "pdb_delta_score_vs_wt", "afdb_delta_score_vs_wt"]
    ].div(result["n_repeats"], axis=0)
    expected_mask = (1 << 30) - 1
    invalid = result.loc[
        (result["n_repeats"] != 30)
        | (result["repeat_mask"] != expected_mask)
        | (result["n_repeats"] != result["repeat_mask"].map(int.bit_count))
    ]
    if not invalid.empty:
        raise RemodelingAnalysisError("repeat_grid_mismatch", "frozen paired input has missing or duplicate repeat indices")
    return result


def load_remodeling_inputs(project_root: Path) -> RemodelingAnalysisInputs:
    """Load the frozen Pair Validity and structural-response inputs read-only."""
    root = project_root.expanduser().resolve()
    pair_path = root / "experiments/p2_design_baseline/scale1b-v2/pair_validity/pair_validity.parquet"
    paired_path = root / "experiments/p2_design_baseline/scale1b-v2/structural_response/paired_structural_response.parquet"
    for path, label in ((pair_path, "Pair Validity"), (paired_path, "structural response")):
        if not path.is_file():
            raise RemodelingAnalysisError("input_missing", f"{label} is missing: {path}")
    pair_validity = pd.read_parquet(pair_path)
    if len(pair_validity) != 127 or pair_validity["high_comparability_eligible"].sum() != 68:
        raise RemodelingAnalysisError("cohort_population_mismatch", "Pair Validity must contain 127 pairs with 68 clean pairs")
    paired = _aggregate_paired_file(paired_path)
    repeat_counts = sorted(set(paired["n_repeats"].astype(int)))
    if repeat_counts != [30]:
        raise RemodelingAnalysisError(
            "repeat_grid_mismatch",
            f"frozen structural-response candidates must have 30 repeats, observed {repeat_counts}",
        )
    return RemodelingAnalysisInputs(
        paired=paired,
        pair_validity=pair_validity,
        input_provenance={
            "pair_validity": {"path": pair_path.relative_to(root).as_posix(), "sha256": sha256_file(pair_path), "rows": len(pair_validity)},
            "paired_structural_response": {"path": paired_path.relative_to(root).as_posix(), "sha256": sha256_file(paired_path), "rows": int(parquet_rows(paired_path))},
        },
    )


def parquet_rows(path: Path) -> int:
    from pyarrow import parquet

    return int(parquet.ParquetFile(path).metadata.num_rows)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise RemodelingAnalysisError("immutable_output_conflict", f"output differs: {path}")
            return "reused_identical"
        os.link(temporary, path)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_bytes(data: bytes, path: Path) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise RemodelingAnalysisError("immutable_output_conflict", f"output differs: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, data)
    return "created"


def _figure_bytes(result: RemodelingAnalysisResult, name: str) -> bytes:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    position = result.position
    if name == "magnitude_distribution":
        for cohort, color in (("clean", "#355c7d"), ("full", "#c06c84")):
            subset = position if cohort == "full" else position.loc[position["cohort"] == "clean"]
            values = subset["magnitude_p"].to_numpy(float)
            values = np.sort(values)
            axis.plot(np.linspace(0, 1, len(values), endpoint=True), values, label=cohort, color=color, linewidth=1.2)
        axis.set_xlabel("Position quantile")
        axis.set_ylabel("P: mean absolute remodeling")
        axis.legend(title="Pair Validity cohort")
    elif name == "magnitude_vs_breadth":
        clean = position.loc[position["cohort"] == "clean"]
        axis.scatter(clean["magnitude_p"], clean["breadth_b"], s=8, alpha=0.35, color="#355c7d")
        axis.set_xlabel("P: mean absolute remodeling")
        axis.set_ylabel("B: effective candidate participation / 19")
    elif name == "magnitude_vs_reordering":
        clean = position.loc[position["cohort"] == "clean"]
        axis.scatter(clean["magnitude_p"], clean["rank_displacement"], s=8, alpha=0.35, color="#6c5b7b")
        axis.set_xlabel("P: mean absolute remodeling")
        axis.set_ylabel("Normalized rank displacement")
    elif name == "protein_phenotype":
        protein = result.protein
        for cohort, color, marker in (("clean", "#355c7d", "o"), ("full", "#c06c84", "x")):
            subset = protein if cohort == "full" else protein.loc[protein["cohort"] == "clean"]
            axis.scatter(subset["magnitude_median"], subset["breadth_median"], s=22, color=color, marker=marker, label=cohort)
        axis.set_xlabel("Protein median P")
        axis.set_ylabel("Protein median B")
        axis.legend(title="Pair Validity cohort")
    elif name == "representative_profiles":
        reps = result.representatives
        positions = result.position
        candidate = result.candidate
        for _, rep in reps.iterrows():
            protein_id = str(rep["protein_id"])
            subset_pos = positions.loc[positions["protein_id"] == protein_id]
            if rep["selection_reason"] == "highest_typical_magnitude":
                selected_position = int(subset_pos.sort_values(["magnitude_p", "position"], ascending=[False, True]).iloc[0]["position"])
            elif rep["selection_reason"] == "lowest_typical_magnitude":
                selected_position = int(subset_pos.sort_values(["magnitude_p", "position"], ascending=[True, True]).iloc[0]["position"])
            elif rep["selection_reason"] == "highest_typical_breadth":
                selected_position = int(subset_pos.sort_values(["breadth_b", "position"], ascending=[False, True]).iloc[0]["position"])
            else:
                selected_position = int(subset_pos.sort_values(["rank_displacement", "position"], ascending=[False, True]).iloc[0]["position"])
            profile = candidate.loc[(candidate["protein_id"] == protein_id) & (candidate["position"] == selected_position)].sort_values("mut_aa")
            axis.plot(profile["mut_aa"], profile["pdb_delta_score_vs_wt"], linewidth=0.8, alpha=0.75, label=f"{protein_id} PDB")
            axis.plot(profile["mut_aa"], profile["afdb_delta_score_vs_wt"], linewidth=0.8, linestyle="--", alpha=0.75, label=f"{protein_id} AFDB")
        axis.set_xlabel("Candidate amino acid")
        axis.set_ylabel("Mean paired Δ score vs WT")
        axis.legend(fontsize=5, ncol=2)
    else:
        plt.close(figure)
        raise ValueError(f"unknown figure: {name}")
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=220, metadata={"Software": "Dual-UQ"})
    plt.close(figure)
    return output.getvalue()


def _markdown_report(result: RemodelingAnalysisResult) -> str:
    clean = result.cohort_summary["cohort"]["clean"]
    full = result.cohort_summary["cohort"]["full"]
    comparison = result.cohort_summary["clean_vs_full_difference"]
    lines = [
        "# Local Inverse-Folding Remodeling Analysis",
        "",
        "Status: Magnitude + Breadth + Reordering characterization only.",
        "No scorer execution, structure-geometry analysis, generation, M/SDFI, or external validation was performed.",
        "",
        "## Cohorts",
        "",
        f"- Clean high-comparability cohort: {clean['protein_count']} proteins, {clean['position_count']} positions.",
        f"- Full frozen-pair sensitivity cohort: {full['protein_count']} proteins, {full['position_count']} positions.",
        f"- P definition: `{P_METRIC_DEFINITION}`.",
        f"- Breadth definition: `{BREADTH_DEFINITION}`; zero-remodeling positions remain undefined.",
        "- Reordering descriptors: Kendall concordance, top-3/top-5 overlap, normalized rank displacement, and profile Pearson correlation.",
        "",
        "## Quantitative answers",
        "",
        f"- Q1 magnitude: clean position P median={clean['magnitude_p_median']!s}, q90={clean['magnitude_p_q90']!s}; full median={full['magnitude_p_median']!s}, q90={full['magnitude_p_q90']!s}.",
        f"- Q2 breadth: clean B median={clean['breadth_b_median']!s}, q90={clean['breadth_b_q90']!s}; undefined fraction={None if clean['breadth_defined_fraction'] is None else 1-clean['breadth_defined_fraction']:.6g}.",
        f"- Q3 reordering: clean Kendall median={clean['kendall_tau_median']!s}, normalized rank-displacement median={clean['rank_displacement_median']!s}; Top-1 flip is diagnostic only ({clean['top1_flip_fraction']!s}).",
        f"- Q4 protein distribution: clean protein median-of-position-P={clean['protein_magnitude_median']!s}, protein median breadth={clean['protein_breadth_median']!s}; protein-level summaries retain position heterogeneity.",
        f"- Q5 sensitivity: clean/full comparison is descriptive; clean-minus-full median P={comparison['magnitude_p_median_difference_clean_minus_full']!s}, median B={comparison['breadth_b_median_difference_clean_minus_full']!s}, median Kendall={comparison['kendall_tau_median_difference_clean_minus_full']!s}; the clean cohort remains primary.",
        "",
        "## Interpretation boundary",
        "",
        "These results describe local inverse-folding compatibility remodeling under paired structural representation variation. They do not establish biological structural uncertainty, functional fragility, design failure, fitness effects, or any composite phenotype class.",
        "",
        "## Next boundary",
        "",
        "This analysis stops before structural organization, 3D geometry, generation, cross-structure robustness, M/SDFI, or external validation.",
    ]
    return "\n".join(lines) + "\n"


def materialize_remodeling_result(result: RemodelingAnalysisResult, output_dir: Path) -> dict[str, Any]:
    """Write all analysis outputs from the canonical structured result."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    for frame, filename in (
        (result.candidate, "candidate_remodeling.parquet"),
        (result.position, "position_remodeling.parquet"),
        (result.protein, "protein_remodeling.parquet"),
        (result.representatives, "representative_selection.parquet"),
    ):
        statuses[filename] = _immutable_parquet(frame, output_dir / filename)
    summary = {
        "status": "COMPLETE",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "input_provenance": _plain(result.input_provenance),
        "cohort_summary": _plain(result.cohort_summary),
        "output_rows": {
            "candidate_remodeling": len(result.candidate),
            "position_remodeling": len(result.position),
            "protein_remodeling": len(result.protein),
            "representative_selection": len(result.representatives),
        },
    }
    statuses["summary.json"] = _immutable_bytes(
        (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        output_dir / "summary.json",
    )
    statuses["report.md"] = _immutable_bytes(_markdown_report(result).encode(), output_dir / "report.md")
    figure_dir = output_dir / "figures"
    for name in (
        "magnitude_distribution",
        "magnitude_vs_breadth",
        "magnitude_vs_reordering",
        "protein_phenotype",
        "representative_profiles",
    ):
        statuses[f"figures/{name}.png"] = _immutable_bytes(_figure_bytes(result, name), figure_dir / f"{name}.png")
    return {"status": "COMPLETE", "output_dir": output_dir, "write_status": statuses}
