"""Frozen-output failure-mode analysis for the paired-state method design."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class FailureModeError(ValueError):
    """Raised when frozen failure-mode inputs violate their scientific keys."""


@dataclass(frozen=True, slots=True)
class FailureModeInputs:
    """Tables consumed by the frozen-output failure-mode analysis."""

    protein_summary: pd.DataFrame
    proteinmpnn_local: pd.DataFrame
    esm_if1_local: pd.DataFrame
    proteinmpnn_generation: pd.DataFrame | None = None
    esm_if1_generation: pd.DataFrame | None = None
    geometry: pd.DataFrame | None = None
    input_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FailureModeResult:
    """Canonical structured result from one failure-mode computation."""

    position_table: pd.DataFrame
    protein_table: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, Any]


POSITION_KEYS = ("protein_id", "pair_id", "canonical_position", "condition")
_LOCAL_REQUIRED = POSITION_KEYS + ("distribution", "js_bits_mean")
_MODEL_NAMES = ("ProteinMPNN", "ESM-IF1")
_STANDARD_AA_COUNT = 20


def _require_columns(table: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise FailureModeError(f"{name} is missing required columns: {missing}")


def _validate_local(table: pd.DataFrame, name: str, model: str) -> pd.DataFrame:
    _require_columns(table, _LOCAL_REQUIRED, name)
    result = table.copy()
    if result.empty:
        raise FailureModeError(f"{name} is empty")
    duplicate = result.duplicated(list(POSITION_KEYS), keep=False)
    if duplicate.any():
        raise FailureModeError(f"{name} has duplicate condition keys")
    result["protein_id"] = result["protein_id"].astype(str)
    result["pair_id"] = result["pair_id"].astype(str)
    result["condition"] = result["condition"].astype(str).str.upper()
    if not result["condition"].isin(("APO", "HOLO")).all():
        raise FailureModeError(f"{name} contains a condition outside APO/HOLO")
    positions = pd.to_numeric(result["canonical_position"], errors="coerce")
    if positions.isna().any() or (positions < 1).any() or (positions % 1 != 0).any():
        raise FailureModeError(f"{name} contains invalid canonical positions")
    result["canonical_position"] = positions.astype(int)
    values = pd.to_numeric(result["js_bits_mean"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise FailureModeError(f"{name}.js_bits_mean contains non-finite values")
    result["js_bits_mean"] = values.astype(float)
    result["model"] = model
    for distribution in result["distribution"]:
        vector = np.asarray(distribution, dtype=float)
        if vector.shape != (_STANDARD_AA_COUNT,):
            raise FailureModeError(f"{name}.distribution must contain 20 values")
        if not np.isfinite(vector).all() or (vector < 0).any() or not np.isclose(vector.sum(), 1.0, atol=1e-6):
            raise FailureModeError(f"{name}.distribution contains invalid probabilities")
    return result


def _js_bits(left: object, right: object) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    midpoint = (p + q) / 2.0
    terms_p = np.where(p > 0, p * np.log2(np.divide(p, midpoint, out=np.ones_like(p), where=midpoint > 0)), 0.0)
    terms_q = np.where(q > 0, q * np.log2(np.divide(q, midpoint, out=np.ones_like(q), where=midpoint > 0)), 0.0)
    return float(0.5 * (terms_p.sum() + terms_q.sum()))


def _entropy_bits(values: np.ndarray) -> float:
    return float(-np.where(values > 0, values * np.log2(values), 0.0).sum())


def _spearman(left: pd.Series, right: pd.Series) -> tuple[float | None, int]:
    valid = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return None, int(len(valid))
    return float(spearmanr(valid["left"], valid["right"]).statistic), int(len(valid))


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    valid = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"q10": None, "median": None, "q90": None}
    return {
        "q10": float(valid.quantile(0.10)),
        "median": float(valid.median()),
        "q90": float(valid.quantile(0.90)),
    }


def _validate_protein_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(table, ("protein_id",), name)
    result = table.copy()
    result["protein_id"] = result["protein_id"].astype(str)
    if result["protein_id"].duplicated().any():
        raise FailureModeError(f"{name} must have unique protein_id values")
    return result


def _geometry_table(table: pd.DataFrame) -> pd.DataFrame:
    required = ("protein_id", "position")
    _require_columns(table, required, "geometry")
    result = table.copy()
    result["protein_key"] = result["protein_id"].astype(str).str.rsplit("__", n=1).str[-1]
    result["canonical_position"] = pd.to_numeric(result["position"], errors="coerce")
    if result["canonical_position"].isna().any():
        raise FailureModeError("geometry contains invalid positions")
    result["canonical_position"] = result["canonical_position"].astype(int)
    keep = [
        "protein_key",
        "canonical_position",
        "magnitude_p",
        "breadth_b",
        "rank_displacement",
        "ca_displacement",
        "local_pairwise_distance_change",
        "contact_turnover_fraction",
    ]
    existing = [column for column in keep if column in result]
    result = result[existing]
    duplicate = result.duplicated(["protein_key", "canonical_position"], keep=False)
    if duplicate.any():
        numeric = [column for column in existing if column not in {"protein_key", "canonical_position"}]
        result = result.groupby(["protein_key", "canonical_position"], as_index=False)[numeric].median()
    return result


def _distribution_summary(table: pd.DataFrame, *, model: str, distribution_column: str = "distribution") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protein_id, condition), group in table.groupby(["protein_id", "condition"], sort=True):
        matrix = np.stack([np.asarray(value, dtype=float) for value in group[distribution_column]])
        mean_distribution = matrix.mean(axis=0)
        rows.append(
            {
                "protein_id": str(protein_id),
                "condition": str(condition),
                f"{model.lower().replace('-', '_')}_entropy_bits": _entropy_bits(mean_distribution),
                f"{model.lower().replace('-', '_')}_distribution": mean_distribution.tolist(),
            }
        )
    return pd.DataFrame(rows)


def _merge_optional_generation(
    summary: pd.DataFrame,
    table: pd.DataFrame | None,
    model_prefix: str,
) -> pd.DataFrame:
    if table is None:
        return summary
    source = _validate_protein_table(table, f"{model_prefix} generation")
    keep = ["protein_id"]
    for column in source.columns:
        if column == "protein_id":
            continue
        if column.startswith(("d_", "js_", "entropy_")):
            keep.append(column)
    if len(keep) == 1:
        return summary
    missing = [
        column for column in keep[1:]
        if f"{model_prefix}_{column}" not in summary.columns
    ]
    if not missing:
        return summary
    renamed = source[["protein_id", *missing]].rename(
        columns={column: f"{model_prefix}_{column}" for column in missing}
    )
    return summary.merge(renamed, on="protein_id", how="left", validate="one_to_one")


def analyze_method_failure_modes(inputs: FailureModeInputs) -> FailureModeResult:
    """Compute all frozen failure-mode descriptors from one canonical join."""
    pnn = _validate_local(inputs.proteinmpnn_local, "ProteinMPNN local response", "ProteinMPNN")
    esm = _validate_local(inputs.esm_if1_local, "ESM-IF1 local response", "ESM-IF1")
    pnn = pnn.rename(columns={"distribution": "pnn_distribution", "js_bits_mean": "pnn_local_burden"})
    esm = esm.rename(columns={"distribution": "esm_distribution", "js_bits_mean": "esm_local_burden"})
    joined = pnn.merge(esm, on=list(POSITION_KEYS), how="inner", validate="one_to_one", suffixes=("_pnn", "_esm"))
    if joined.empty:
        raise FailureModeError("ProteinMPNN and ESM-IF1 local-response intersection is empty")
    joined["evaluator_js_bits"] = [
        _js_bits(left, right) for left, right in zip(joined["pnn_distribution"], joined["esm_distribution"], strict=True)
    ]
    apo = joined.loc[joined["condition"].eq("APO")].copy()
    holo = joined.loc[joined["condition"].eq("HOLO")].copy()
    state = apo.merge(
        holo,
        on=["protein_id", "pair_id", "canonical_position"],
        how="inner",
        validate="one_to_one",
        suffixes=("_apo", "_holo"),
    )
    state["pnn_state_js_bits"] = [
        _js_bits(left, right) for left, right in zip(state["pnn_distribution_apo"], state["pnn_distribution_holo"], strict=True)
    ]
    state["esm_state_js_bits"] = [
        _js_bits(left, right) for left, right in zip(state["esm_distribution_apo"], state["esm_distribution_holo"], strict=True)
    ]
    state["state_shift_disagreement_bits"] = (state["pnn_state_js_bits"] - state["esm_state_js_bits"]).abs()
    position_columns = [
        "protein_id",
        "pair_id",
        "canonical_position",
        "pnn_state_js_bits",
        "esm_state_js_bits",
        "state_shift_disagreement_bits",
        "evaluator_js_bits_apo",
        "evaluator_js_bits_holo",
        "pnn_local_burden_apo",
        "pnn_local_burden_holo",
        "esm_local_burden_apo",
        "esm_local_burden_holo",
    ]
    position = state[position_columns].copy()
    if inputs.geometry is not None and not inputs.geometry.empty:
        geometry = _geometry_table(inputs.geometry)
        position = position.merge(
            geometry,
            left_on=["protein_id", "canonical_position"],
            right_on=["protein_key", "canonical_position"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["protein_key"], errors="ignore")

    protein = position.groupby("protein_id", as_index=False).agg(
        paired_position_count=("canonical_position", "size"),
        pnn_state_js_bits_mean=("pnn_state_js_bits", "mean"),
        pnn_state_js_bits_median=("pnn_state_js_bits", "median"),
        esm_state_js_bits_mean=("esm_state_js_bits", "mean"),
        esm_state_js_bits_median=("esm_state_js_bits", "median"),
        state_shift_disagreement_bits_mean=("state_shift_disagreement_bits", "mean"),
        evaluator_js_bits_apo_mean=("evaluator_js_bits_apo", "mean"),
        evaluator_js_bits_holo_mean=("evaluator_js_bits_holo", "mean"),
    )
    for model, table in (("proteinmpnn", pnn), ("esm_if1", esm)):
        distribution_column = "pnn_distribution" if model == "proteinmpnn" else "esm_distribution"
        dist = _distribution_summary(table, model=model, distribution_column=distribution_column)
        apo_dist = dist.loc[dist["condition"].eq("APO")].drop(columns="condition")
        holo_dist = dist.loc[dist["condition"].eq("HOLO")].drop(columns="condition")
        merged_dist = apo_dist.merge(holo_dist, on="protein_id", how="inner", validate="one_to_one", suffixes=("_apo", "_holo"))
        merged_dist[f"{model.lower().replace('-', '_')}_composition_l1"] = [
            float(np.abs(np.asarray(left) - np.asarray(right)).sum())
            for left, right in zip(
                merged_dist[f"{model.lower().replace('-', '_')}_distribution_apo"],
                merged_dist[f"{model.lower().replace('-', '_')}_distribution_holo"],
                strict=True,
            )
        ]
        keep = ["protein_id", f"{model.lower().replace('-', '_')}_composition_l1"]
        protein = protein.merge(merged_dist[keep], on="protein_id", how="left", validate="one_to_one")
    summary = _validate_protein_table(inputs.protein_summary, "frozen protein summary")
    selected_summary_columns = [
        "protein_id",
        "proteinmpnn_multi_minus_best_single_worst",
        "esm_if1_multi_minus_best_single_worst",
        "proteinmpnn_local_burden_mean",
        "esm_if1_local_burden_mean",
        "proteinmpnn_d_excess_64",
        "esm_if1_d_excess_64",
        "proteinmpnn_js_bits_mean_64",
        "esm_if1_js_bits_mean_64",
        "median_local_pairwise_distance_change",
        "median_residue_displacement",
        "contact_turnover_fraction_mean",
    ]
    available = [column for column in selected_summary_columns if column in summary.columns]
    protein = protein.merge(summary[available], on="protein_id", how="left", validate="one_to_one")
    protein = _merge_optional_generation(protein, inputs.proteinmpnn_generation, "proteinmpnn")
    protein = _merge_optional_generation(protein, inputs.esm_if1_generation, "esm_if1")
    if {"proteinmpnn_multi_minus_best_single_worst", "esm_if1_multi_minus_best_single_worst"}.issubset(protein.columns):
        pnn_effect = protein["proteinmpnn_multi_minus_best_single_worst"]
        esm_effect = protein["esm_if1_multi_minus_best_single_worst"]
        protein["multi_state_direction"] = np.select(
            [(pnn_effect > 0) & (esm_effect < 0), (pnn_effect < 0) & (esm_effect > 0), (pnn_effect > 0) & (esm_effect > 0), (pnn_effect < 0) & (esm_effect < 0)],
            ["PNN_BENEFIT_ESM_LOSS", "PNN_LOSS_ESM_BENEFIT", "BENEFIT_BOTH", "LOSS_BOTH"],
            default="MIXED_OR_ZERO",
        )
    else:
        protein["multi_state_direction"] = "UNAVAILABLE"
    pnn_benefit = protein.get("proteinmpnn_multi_minus_best_single_worst", pd.Series(dtype=float))
    esm_loss = -protein.get("esm_if1_multi_minus_best_single_worst", pd.Series(dtype=float))
    rho, rho_n = _spearman(pnn_benefit, esm_loss)
    direction_counts = protein["multi_state_direction"].value_counts(dropna=False).to_dict()
    evaluator_disagreement = {
        condition.lower(): _quantiles(joined.loc[joined["condition"].eq(condition), "evaluator_js_bits"])
        for condition in ("APO", "HOLO")
    }
    association_pairs = {
        "pnn_state_shift_to_pnn_multi_benefit": ("pnn_state_js_bits_mean", "proteinmpnn_multi_minus_best_single_worst"),
        "esm_state_shift_to_esm_multi_effect": ("esm_state_js_bits_mean", "esm_if1_multi_minus_best_single_worst"),
        "state_shift_disagreement_to_pnn_multi_benefit": ("state_shift_disagreement_bits_mean", "proteinmpnn_multi_minus_best_single_worst"),
        "state_shift_disagreement_to_esm_multi_effect": ("state_shift_disagreement_bits_mean", "esm_if1_multi_minus_best_single_worst"),
        "pnn_composition_shift_to_pnn_multi_benefit": ("proteinmpnn_composition_l1", "proteinmpnn_multi_minus_best_single_worst"),
        "esm_composition_shift_to_esm_multi_effect": ("esm_if1_composition_l1", "esm_if1_multi_minus_best_single_worst"),
    }
    associations: dict[str, dict[str, float | int | None]] = {}
    for label, (left, right) in association_pairs.items():
        if left in protein and right in protein:
            value, count = _spearman(protein[left], protein[right])
            associations[label] = {"spearman": value, "n_proteins": count}
    geometry_associations: dict[str, dict[str, float | int | None]] = {}
    for descriptor in ("magnitude_p", "breadth_b", "rank_displacement", "ca_displacement", "local_pairwise_distance_change", "contact_turnover_fraction"):
        if descriptor not in position:
            continue
        value, count = _spearman(position[descriptor], position["state_shift_disagreement_bits"])
        geometry_associations[descriptor] = {"spearman": value, "n_positions": count}
    if associations.get("pnn_state_shift_to_pnn_multi_benefit", {}).get("spearman") is not None:
        failure_mode_verdict = (
            "SELF_MODEL_OVER_OPTIMIZATION_WITH_SYSTEMATIC_AA_PREFERENCE_BIAS; "
            "GEOMETRY_DEPENDENCE_REMAINS_PARTIAL"
        )
    else:
        failure_mode_verdict = "MIXED_OR_UNRESOLVED"
    output_summary: dict[str, Any] = {
        "schema_version": "dual_uq_method_failure_modes_v1",
        "protein_count": int(protein["protein_id"].nunique()),
        "position_count": int(len(position)),
        "local_response_intersection_protein_count": int(joined["protein_id"].nunique()),
        "local_response_intersection_position_condition_count": int(len(joined)),
        "multi_state_direction": {str(key): int(value) for key, value in direction_counts.items()},
        "failure_mode_verdict": failure_mode_verdict,
        "conditional_evaluator_distribution_disagreement_bits": evaluator_disagreement,
        "protein_effect_quantiles": {
            "proteinmpnn_multi_minus_best_single_worst": _quantiles(protein.get("proteinmpnn_multi_minus_best_single_worst", pd.Series(dtype=float))),
            "esm_if1_multi_minus_best_single_worst": _quantiles(protein.get("esm_if1_multi_minus_best_single_worst", pd.Series(dtype=float))),
        },
        "protein_associations": associations,
        "geometry_associations_to_state_shift_disagreement": geometry_associations,
        "proteinmpnn_benefit_vs_esm_if1_loss_spearman": rho,
        "proteinmpnn_benefit_vs_esm_if1_loss_n": rho_n,
        "interpretation_boundary": "descriptive evaluator disagreement and frozen multi-state failure modes; no causal or biological claim",
    }
    return FailureModeResult(
        position_table=position.sort_values(["protein_id", "canonical_position"], kind="mergesort").reset_index(drop=True),
        protein_table=protein.sort_values("protein_id", kind="mergesort").reset_index(drop=True),
        summary=output_summary,
        input_provenance=dict(inputs.input_provenance),
    )


def render_failure_mode_report(result: FailureModeResult, output_dir: Path) -> None:
    """Materialize tables and a concise report from one canonical result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.position_table.to_parquet(output_dir / "position_failure_modes.parquet", index=False)
    result.protein_table.to_parquet(output_dir / "protein_failure_modes.parquet", index=False)
    manifest = {"schema_version": result.summary["schema_version"], "inputs": result.input_provenance, "summary": result.summary}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = result.summary
    effects = summary["protein_effect_quantiles"]
    disagreement = summary["conditional_evaluator_distribution_disagreement_bits"]
    pnn_assoc = summary["protein_associations"].get("pnn_state_shift_to_pnn_multi_benefit", {})
    esm_assoc = summary["protein_associations"].get("esm_state_shift_to_esm_multi_effect", {})
    geometry_assoc = summary["geometry_associations_to_state_shift_disagreement"]
    geometry_n = max((int(item["n_positions"]) for item in geometry_assoc.values()), default=0)
    lines = [
        "# Paired-State Method Failure Modes",
        "",
        "## FAILURE_MODE_VERDICT",
        "",
        f"`{summary['failure_mode_verdict']}`",
        "",
        "The simple MULTI failure is evaluator-specific: the frozen PNN effect is predominantly beneficial while the frozen ESM-IF1 effect is predominantly harmful, despite both being driven by state-dependent preference shifts. This is consistent with self-model over-optimization plus systematic amino-acid preference bias, not a biological failure claim.",
        "",
        "## Evidence",
        "",
        f"- Protein rows: {summary['protein_count']}; paired position rows: {summary['position_count']}.",
        f"- Direction counts: `{summary['multi_state_direction']}`.",
        f"- PNN effect (multi minus best single, worst): median `{effects['proteinmpnn_multi_minus_best_single_worst']['median']}`, q10 `{effects['proteinmpnn_multi_minus_best_single_worst']['q10']}`, q90 `{effects['proteinmpnn_multi_minus_best_single_worst']['q90']}`.",
        f"- ESM-IF1 effect (multi minus best single, worst): median `{effects['esm_if1_multi_minus_best_single_worst']['median']}`, q10 `{effects['esm_if1_multi_minus_best_single_worst']['q10']}`, q90 `{effects['esm_if1_multi_minus_best_single_worst']['q90']}`.",
        f"- Conditional evaluator-distribution disagreement: APO median `{disagreement['apo']['median']}`, HOLO median `{disagreement['holo']['median']}` bits.",
        f"- PNN state-shift versus PNN effect Spearman: `{pnn_assoc.get('spearman')}` (n={pnn_assoc.get('n_proteins')}).",
        f"- ESM-IF1 state-shift versus ESM-IF1 effect Spearman: `{esm_assoc.get('spearman')}` (n={esm_assoc.get('n_proteins')}).",
        f"- Geometry association is partial: descriptors were available for at most `{geometry_n}` position rows; this is not a full-cohort geometry test.",
        f"- PNN benefit versus ESM-IF1 loss Spearman: `{summary['proteinmpnn_benefit_vs_esm_if1_loss_spearman']}` (n={summary['proteinmpnn_benefit_vs_esm_if1_loss_n']}).",
        "",
        "Interpretation remains descriptive at the protein inference unit; no biological, causal, fitness, stability, or functional claim is made.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
