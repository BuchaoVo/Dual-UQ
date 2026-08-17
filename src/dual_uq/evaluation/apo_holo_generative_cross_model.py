"""Protein-level cross-model analysis of APO/HOLO generation summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class ApoHoloGenerativeCrossModelError(ValueError):
    """Raised when model-specific generative summaries cannot be joined."""


@dataclass(frozen=True, slots=True)
class ApoHoloGenerativeCrossModelResult:
    """Canonical structured result for cross-model generative propagation."""

    protein_summary: pd.DataFrame
    convergence: pd.DataFrame
    associations: pd.DataFrame
    ligand_summary: pd.DataFrame
    summary: dict[str, Any]


_GENERATION_COLUMNS = (
    "d_apo_64",
    "d_holo_64",
    "d_ah_64",
    "d_excess_64",
    "js_bits_mean_64",
    "js_bits_q90_64",
)
_GEOMETRY_COLUMNS = (
    "aligned_ca_rmsd",
    "median_residue_displacement",
    "p90_residue_displacement",
    "median_local_pairwise_distance_change",
    "contact_turnover_fraction_mean",
)
_PAIR_GEOMETRY_COLUMNS = tuple(column for column in _GEOMETRY_COLUMNS if column != "contact_turnover_fraction_mean")


def _validate_protein_table(table: pd.DataFrame, required: Iterable[str], name: str) -> pd.DataFrame:
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ApoHoloGenerativeCrossModelError(f"{name} is missing columns: {missing}")
    result = table.copy()
    if result["protein_id"].isna().any() or result["protein_id"].duplicated().any():
        raise ApoHoloGenerativeCrossModelError(f"{name} must have unique protein_id values")
    result["protein_id"] = result["protein_id"].astype(str)
    for column in set(required) - {"protein_id"}:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ApoHoloGenerativeCrossModelError(f"{name}.{column} contains non-finite values")
    return result


def _spearman(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    values = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(values) < 3 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return float("nan"), len(values)
    return float(spearmanr(values["left"], values["right"]).statistic), len(values)


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"q10": None, "median": None, "q90": None}
    return {
        "q10": float(values.quantile(0.10, interpolation="linear")),
        "median": float(values.median()),
        "q90": float(values.quantile(0.90, interpolation="linear")),
    }


def _fallback_convergence(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Keep a valid 64-sample row when a legacy caller has no convergence table."""
    return pd.DataFrame(
        {
            "protein_id": table["protein_id"],
            "sample_count": 64,
            "d_apo": table["d_apo_64"],
            "d_holo": table["d_holo_64"],
            "d_ah": table["d_ah_64"],
            "d_excess": table["d_excess_64"],
            "model": model,
        }
    )


def _summarize_convergence(table: pd.DataFrame, model: str) -> pd.DataFrame:
    required = {"protein_id", "sample_count", "d_apo", "d_holo", "d_ah", "d_excess"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ApoHoloGenerativeCrossModelError(f"{model} convergence is missing columns: {missing}")
    source = table.copy()
    source["protein_id"] = source["protein_id"].astype(str)
    source["sample_count"] = pd.to_numeric(source["sample_count"], errors="raise").astype(int)
    key = ["protein_id", "sample_count"]
    if source.duplicated(key).any():
        raise ApoHoloGenerativeCrossModelError(f"{model} convergence keys are duplicated")
    rows: list[dict[str, Any]] = []
    final = source.loc[source["sample_count"].eq(source["sample_count"].max()), ["protein_id", "d_excess"]]
    final = final.rename(columns={"d_excess": "d_excess_final"})
    for sample_count, group in source.groupby("sample_count", sort=True):
        values = pd.to_numeric(group["d_excess"], errors="coerce")
        joined = group[["protein_id", "d_excess"]].merge(final, on="protein_id", how="inner")
        rho, n = _spearman(joined["d_excess"], joined["d_excess_final"])
        rows.append(
            {
                "model": model,
                "sample_count": int(sample_count),
                "protein_count": len(group),
                "d_excess_q10": float(values.quantile(0.10)),
                "d_excess_median": float(values.median()),
                "d_excess_q90": float(values.quantile(0.90)),
                "positive_count": int((values > 0).sum()),
                "positive_fraction": float((values > 0).mean()),
                "rank_spearman_to_final": rho,
                "rank_spearman_n": n,
            }
        )
    return pd.DataFrame(rows)


def _association_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(label: str, model: str, x: str, y: str) -> None:
        rho, n = _spearman(summary[x], summary[y])
        rows.append({"association": label, "model": model, "x": x, "y": y, "n": n, "spearman": rho})

    for model in ("proteinmpnn", "esm_if1"):
        add(f"{model}_local_to_d_excess", model, f"{model}_local_burden_mean", f"{model}_d_excess_64")
        add(f"{model}_local_to_generative_js", model, f"{model}_local_burden_mean", f"{model}_js_bits_mean_64")
        for descriptor in _GEOMETRY_COLUMNS:
            if descriptor in summary:
                add(f"{model}_d_excess_to_{descriptor}", model, f"{model}_d_excess_64", descriptor)
                add(f"{model}_generative_js_to_{descriptor}", model, f"{model}_js_bits_mean_64", descriptor)
    add("generative_d_excess_cross_model", "cross-model", "proteinmpnn_d_excess_64", "esm_if1_d_excess_64")
    add("generative_js_cross_model", "cross-model", "proteinmpnn_js_bits_mean_64", "esm_if1_js_bits_mean_64")
    add("local_response_cross_model_observed", "cross-model", "proteinmpnn_local_burden_mean", "esm_if1_local_burden_mean")
    return pd.DataFrame(rows)


def _ligand_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "ligand_status" not in summary:
        return pd.DataFrame(columns=["model", "ligand_status", "protein_count", "d_excess_median", "js_median"])
    for model in ("proteinmpnn", "esm_if1"):
        for status, group in summary.groupby("ligand_status", dropna=False, sort=True):
            rows.append(
                {
                    "model": model,
                    "ligand_status": str(status),
                    "protein_count": len(group),
                    "d_excess_median": float(group[f"{model}_d_excess_64"].median()),
                    "js_median": float(group[f"{model}_js_bits_mean_64"].median()),
                }
            )
    return pd.DataFrame(rows)


def build_cross_model_result(
    *,
    proteinmpnn_generation: pd.DataFrame,
    esm_if1_generation: pd.DataFrame,
    proteinmpnn_local: pd.DataFrame,
    esm_if1_local: pd.DataFrame,
    pair_geometry: pd.DataFrame | None = None,
    residue_geometry: pd.DataFrame | None = None,
    proteinmpnn_convergence: pd.DataFrame | None = None,
    esm_if1_convergence: pd.DataFrame | None = None,
    primary_cohort_count: int = 1359,
    local_reference_spearman: float | None = 0.783822,
) -> ApoHoloGenerativeCrossModelResult:
    """Join canonical model summaries and derive all cross-model statistics."""
    generation_required = {"protein_id", *_GENERATION_COLUMNS}
    pnn_gen = _validate_protein_table(proteinmpnn_generation, generation_required, "ProteinMPNN generation")
    esm_gen = _validate_protein_table(esm_if1_generation, generation_required, "ESM-IF1 generation")
    pnn_local = _validate_protein_table(proteinmpnn_local, {"protein_id", "local_burden_mean"}, "ProteinMPNN local response")
    esm_local = _validate_protein_table(esm_if1_local, {"protein_id", "local_burden_mean"}, "ESM-IF1 local response")
    pnn_gen = pnn_gen.rename(columns={column: f"proteinmpnn_{column}" for column in _GENERATION_COLUMNS})
    esm_gen = esm_gen.rename(columns={column: f"esm_if1_{column}" for column in _GENERATION_COLUMNS})
    pnn_local = pnn_local.rename(columns={"local_burden_mean": "proteinmpnn_local_burden_mean"})
    esm_local = esm_local.rename(columns={"local_burden_mean": "esm_if1_local_burden_mean"})
    summary = pnn_gen.merge(esm_gen, on="protein_id", how="inner", validate="one_to_one")
    summary = summary.merge(pnn_local, on="protein_id", how="inner", validate="one_to_one")
    summary = summary.merge(esm_local, on="protein_id", how="inner", validate="one_to_one")
    if summary.empty:
        raise ApoHoloGenerativeCrossModelError("model generation/local-response intersection is empty")

    if pair_geometry is not None:
        geometry = pair_geometry.copy()
        if "protein_id" not in geometry or geometry["protein_id"].duplicated().any():
            raise ApoHoloGenerativeCrossModelError("pair geometry must have unique protein_id values")
        geometry["protein_id"] = geometry["protein_id"].astype(str)
        keep = ["protein_id", "geometry_status", *_PAIR_GEOMETRY_COLUMNS, "ligand_proximal_count", "ligand_distal_count"]
        missing = sorted(set(keep) - set(geometry.columns))
        if missing:
            raise ApoHoloGenerativeCrossModelError(f"pair geometry is missing columns: {missing}")
        summary = summary.merge(geometry[keep], on="protein_id", how="left", validate="one_to_one")
        summary["ligand_status"] = np.select(
            [summary["ligand_proximal_count"].gt(0), summary["ligand_proximal_count"].eq(0)],
            ["ligand_proximal", "explicit_zero_proximal"],
            default="unavailable",
        )
    else:
        summary["ligand_status"] = "unavailable"

    if residue_geometry is not None and not residue_geometry.empty:
        residue = residue_geometry.copy()
        residue["protein_id"] = residue["protein_id"].astype(str)
        contact = residue.groupby("protein_id", as_index=False)["contact_turnover_fraction"].mean()
        contact = contact.rename(columns={"contact_turnover_fraction": "contact_turnover_fraction_mean"})
        summary = summary.merge(contact, on="protein_id", how="left", validate="one_to_one")
    summary = summary.sort_values("protein_id", kind="mergesort").reset_index(drop=True)

    pnn_conv = _fallback_convergence(pnn_gen.rename(columns={f"proteinmpnn_{column}": column for column in _GENERATION_COLUMNS}), "ProteinMPNN") if proteinmpnn_convergence is None else proteinmpnn_convergence
    esm_conv = _fallback_convergence(esm_gen.rename(columns={f"esm_if1_{column}": column for column in _GENERATION_COLUMNS}), "ESM-IF1") if esm_if1_convergence is None else esm_if1_convergence
    convergence = pd.concat(
        [_summarize_convergence(pnn_conv, "ProteinMPNN"), _summarize_convergence(esm_conv, "ESM-IF1")],
        ignore_index=True,
    ).sort_values(["model", "sample_count"], kind="mergesort").reset_index(drop=True)
    associations = _association_rows(summary)
    ligand = _ligand_summary(summary)
    local_observed, local_n = _spearman(summary["proteinmpnn_local_burden_mean"], summary["esm_if1_local_burden_mean"])
    d_excess_rho = associations.loc[associations["association"].eq("generative_d_excess_cross_model"), "spearman"].iloc[0]
    js_rho = associations.loc[associations["association"].eq("generative_js_cross_model"), "spearman"].iloc[0]
    model_summary: dict[str, Any] = {}
    for model in ("proteinmpnn", "esm_if1"):
        excess = summary[f"{model}_d_excess_64"]
        js = summary[f"{model}_js_bits_mean_64"]
        model_summary[model] = {
            "protein_count": len(summary),
            "d_excess_64": _quantiles(excess),
            "d_excess_positive_count": int((excess > 0).sum()),
            "d_excess_positive_fraction": float((excess > 0).mean()),
            "generative_js_burden_bits_64": _quantiles(js),
        }
    pnn_ids = set(pnn_gen["protein_id"])
    esm_ids = set(esm_gen["protein_id"])
    result_summary: dict[str, Any] = {
        "schema_version": "apo_holo_generative_cross_model_summary_v1",
        "verdict": "COMPLETE" if len(summary) == primary_cohort_count else "LIMITED_PRIMARY_INTERSECTION",
        "primary_requested_protein_count": int(primary_cohort_count),
        "protein_unit": "protein",
        "proteinmpnn_generation_protein_count": len(pnn_ids),
        "esm_if1_generation_protein_count": len(esm_ids),
        "shared_generation_protein_count": len(summary),
        "proteinmpnn_only_generation_proteins": sorted(pnn_ids - esm_ids),
        "esm_if1_only_generation_proteins": sorted(esm_ids - pnn_ids),
        "model_summaries": model_summary,
        "cross_model_generation_spearman": {
            "d_excess_64": float(d_excess_rho) if np.isfinite(d_excess_rho) else None,
            "generative_js_burden": float(js_rho) if np.isfinite(js_rho) else None,
        },
        "local_response_reference_spearman": local_reference_spearman,
        "local_response_observed_shared_spearman": float(local_observed) if np.isfinite(local_observed) else None,
        "local_response_observed_n": local_n,
        "convergence_sample_counts": sorted(convergence["sample_count"].unique().tolist()),
        "geometry_valid_protein_count": int(summary.get("geometry_status", pd.Series(dtype=object)).eq("available").sum()) if "geometry_status" in summary else 0,
        "interpretation_boundary": "Protein-level descriptive generative propagation only; no biological fitness, function, stability, causality, or ligand-mechanism claim.",
    }
    return ApoHoloGenerativeCrossModelResult(summary, convergence, associations, ligand, result_summary)
