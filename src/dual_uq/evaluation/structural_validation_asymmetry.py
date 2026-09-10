"""Baseline-adjusted analysis of independent structural-validation asymmetry.

The module consumes the existing generated-sequence comparison table and one
WT prediction per clean protein.  It does not rerun ESMFold or modify the
upstream structural-validation artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file as _sha
from dual_uq.evaluation.independent_structure_validation import spearman_correlation
from dual_uq.geometry import kabsch_align, rmsd
from dual_uq.inference.independent_structure_validation import (
    StructuralValidationAsymmetryError,
)


@dataclass(frozen=True, slots=True)
class StructuralValidationAsymmetryResult:
    """All tables and conclusions derived from one canonical analysis result."""

    absolute_comparisons: pd.DataFrame
    wt_baseline: pd.DataFrame
    protein_effects: pd.DataFrame
    associations: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, Any]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StructuralValidationAsymmetryError(f"{label} is not numeric") from exc
    if not np.isfinite(number):
        raise StructuralValidationAsymmetryError(f"{label} is not finite")
    return number


def _coords(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3 or not np.isfinite(array).all():
        raise StructuralValidationAsymmetryError(f"{label} is not a finite coordinate array")
    return array


def _aligned_rmsd(mobile: Any, target: Any, label: str) -> float:
    mobile_array = _coords(mobile, label)
    target_array = _coords(target, label)
    if mobile_array.shape != target_array.shape:
        raise StructuralValidationAsymmetryError(f"{label} common-mask lengths differ")
    aligned, _rotation, _translation = kabsch_align(mobile_array, target_array)
    value = rmsd(aligned, target_array)
    if not np.isfinite(value):
        raise StructuralValidationAsymmetryError(f"{label} RMSD is not finite")
    return float(value)


def _absolute_comparisons(generated: pd.DataFrame) -> pd.DataFrame:
    required = {
        "protein_id",
        "source_condition",
        "sample_index",
        "rmsd_to_conditioning",
        "rmsd_to_alternative",
    }
    missing = sorted(required.difference(generated.columns))
    if missing:
        raise StructuralValidationAsymmetryError(
            f"generated comparison schema lacks: {', '.join(missing)}"
        )
    if generated.empty or set(generated["source_condition"]) != {"PDB", "AFDB"}:
        raise StructuralValidationAsymmetryError("generated comparison conditions are incomplete")
    keys = generated[["protein_id", "source_condition", "sample_index"]]
    if keys.duplicated().any():
        raise StructuralValidationAsymmetryError("generated comparison keys are duplicated")
    rows: list[dict[str, Any]] = []
    for record in generated.to_dict("records"):
        condition = str(record["source_condition"])
        for target, value in (
            ("PDB", record["rmsd_to_conditioning"] if condition == "PDB" else record["rmsd_to_alternative"]),
            ("AFDB", record["rmsd_to_alternative"] if condition == "PDB" else record["rmsd_to_conditioning"]),
        ):
            rows.append({
                "protein_id": str(record["protein_id"]),
                "generated_condition": condition,
                "target_condition": target,
                "sample_index": int(record["sample_index"]),
                "sample_class": record.get("sample_class"),
                "sequence_hash": record.get("sequence_hash"),
                "rmsd": _finite(value, "absolute RMSD"),
            })
    return pd.DataFrame(rows)


def _wt_baseline(wt: pd.DataFrame) -> pd.DataFrame:
    if "protein_id" not in wt.columns or wt.empty:
        raise StructuralValidationAsymmetryError("WT baseline is missing")
    rows: list[dict[str, Any]] = []
    if {"wt_rmsd_to_pdb", "wt_rmsd_to_afdb"}.issubset(wt.columns):
        for record in wt.to_dict("records"):
            pdb = _finite(record["wt_rmsd_to_pdb"], "WT RMSD to PDB")
            afdb = _finite(record["wt_rmsd_to_afdb"], "WT RMSD to AFDB")
            rows.append({
                "protein_id": str(record["protein_id"]),
                "wt_rmsd_to_pdb": pdb,
                "wt_rmsd_to_afdb": afdb,
                "wt_afdb_preference": pdb - afdb,
            })
    elif {"predicted_ca_coordinates", "target_pdb_ca_coordinates", "target_afdb_ca_coordinates"}.issubset(wt.columns):
        for record in wt.to_dict("records"):
            predicted = record["predicted_ca_coordinates"]
            pdb = _aligned_rmsd(predicted, record["target_pdb_ca_coordinates"], "WT/PDB")
            afdb = _aligned_rmsd(predicted, record["target_afdb_ca_coordinates"], "WT/AFDB")
            rows.append({
                "protein_id": str(record["protein_id"]),
                "wt_rmsd_to_pdb": pdb,
                "wt_rmsd_to_afdb": afdb,
                "wt_afdb_preference": pdb - afdb,
            })
    else:
        raise StructuralValidationAsymmetryError("WT baseline schema is incomplete")
    result = pd.DataFrame(rows)
    if result["protein_id"].duplicated().any():
        raise StructuralValidationAsymmetryError("WT baseline keys are duplicated")
    return result


def _descriptor_frame(frame: pd.DataFrame, name: str, columns: tuple[str, ...]) -> pd.DataFrame:
    if "protein_id" not in frame.columns:
        raise StructuralValidationAsymmetryError(f"{name} summary lacks protein_id")
    required = ["protein_id", *columns]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise StructuralValidationAsymmetryError(f"{name} summary lacks: {', '.join(missing)}")
    selected = frame[required].copy()
    if selected["protein_id"].duplicated().any():
        raise StructuralValidationAsymmetryError(f"{name} summary has duplicated proteins")
    return selected


def _quantile(values: pd.Series, probability: float) -> float:
    return float(values.quantile(probability, interpolation="linear"))


def build_structural_validation_asymmetry(
    generated_comparison: pd.DataFrame,
    wt_comparison: pd.DataFrame,
    generation_summary: pd.DataFrame,
    remodeling_summary: pd.DataFrame,
    *,
    expected_protein_count: int = 68,
) -> StructuralValidationAsymmetryResult:
    """Build absolute, baseline, protein-effect, and association tables."""
    generated = generated_comparison.copy()
    absolute = _absolute_comparisons(generated)
    proteins = tuple(sorted(absolute["protein_id"].unique()))
    if len(proteins) != expected_protein_count:
        raise StructuralValidationAsymmetryError("generated comparison cohort cardinality differs")
    wt = _wt_baseline(wt_comparison)
    if set(wt["protein_id"]) != set(proteins):
        raise StructuralValidationAsymmetryError("WT baseline does not match generated cohort")

    preference_rows: list[dict[str, Any]] = []
    for (protein_id, condition), group in absolute.groupby(["protein_id", "generated_condition"], sort=True):
        pivot = group.pivot_table(index="sample_index", columns="target_condition", values="rmsd", aggfunc="first")
        if set(pivot.columns) != {"PDB", "AFDB"}:
            raise StructuralValidationAsymmetryError("absolute comparison target grid is incomplete")
        preference = pivot["PDB"] - pivot["AFDB"]
        for sample_index, value in preference.items():
            preference_rows.append({
                "protein_id": protein_id,
                "generated_condition": condition,
                "sample_index": int(sample_index),
                "generated_afdb_preference": float(value),
            })
    group_table = pd.DataFrame(preference_rows)
    rows: list[dict[str, Any]] = []
    for protein_id in proteins:
        wt_row = wt.loc[wt["protein_id"] == protein_id].iloc[0]
        values: dict[str, Any] = {
            "protein_id": protein_id,
            "wt_afdb_preference": float(wt_row["wt_afdb_preference"]),
            "wt_rmsd_to_pdb": float(wt_row["wt_rmsd_to_pdb"]),
            "wt_rmsd_to_afdb": float(wt_row["wt_rmsd_to_afdb"]),
        }
        for condition, label in (("PDB", "pdb_generated"), ("AFDB", "afdb_generated")):
            selected = group_table.loc[
                (group_table["protein_id"] == protein_id)
                & (group_table["generated_condition"] == condition),
                "generated_afdb_preference",
            ]
            if selected.empty:
                raise StructuralValidationAsymmetryError("generated condition is incomplete")
            values[f"{label}_afdb_preference"] = float(selected.median())
            values[f"{label}_afdb_preference_q10"] = _quantile(selected, 0.10)
            values[f"{label}_afdb_preference_q90"] = _quantile(selected, 0.90)
            values[f"{label}_positive_fraction"] = float((selected > 0).mean())
            values[f"{label}_baseline_adjusted"] = values[f"{label}_afdb_preference"] - values["wt_afdb_preference"]
        rows.append(values)
    effects = pd.DataFrame(rows)

    merged = effects.merge(
        _descriptor_frame(generation_summary, "generation", ("d_pa_excess_independent", "js_burden_mean")),
        on="protein_id", how="left", validate="one_to_one",
    ).merge(
        _descriptor_frame(remodeling_summary, "remodeling", ("position_magnitude_upper_tail_excess",)),
        on="protein_id", how="left", validate="one_to_one",
    )
    if merged[["d_pa_excess_independent", "js_burden_mean", "position_magnitude_upper_tail_excess"]].isna().any().any():
        raise StructuralValidationAsymmetryError("baseline-adjusted descriptor join is incomplete")

    associations: list[dict[str, Any]] = []
    outcomes = ("pdb_generated_baseline_adjusted", "afdb_generated_baseline_adjusted")
    descriptors = ("d_pa_excess_independent", "js_burden_mean", "position_magnitude_upper_tail_excess")
    for outcome in outcomes:
        for descriptor in descriptors:
            associations.append({
                "outcome": outcome,
                "descriptor": descriptor,
                "spearman": spearman_correlation(
                    merged[outcome], merged[descriptor]
                ),
                "n_proteins": int(merged[[outcome, descriptor]].dropna().shape[0]),
            })
    association_table = pd.DataFrame(associations)

    pdb_effect = effects["pdb_generated_baseline_adjusted"]
    afdb_effect = effects["afdb_generated_baseline_adjusted"]
    summary: dict[str, Any] = {
        "cohort_proteins": len(proteins),
        "generated_prediction_rows": len(generated),
        "absolute_comparison_rows": len(absolute),
        "wt_baseline_rows": len(wt),
        "direction_definition": "AFDB_preference = RMSD_to_PDB - RMSD_to_AFDB; positive means closer to AFDB",
        "baseline_adjustment_definition": "generated_AFDB_preference - WT_AFDB_preference",
        "q1_absolute_groups": {
            "PDB_generated_vs_PDB_median": float(absolute.loc[(absolute.generated_condition == "PDB") & (absolute.target_condition == "PDB"), "rmsd"].median()),
            "PDB_generated_vs_AFDB_median": float(absolute.loc[(absolute.generated_condition == "PDB") & (absolute.target_condition == "AFDB"), "rmsd"].median()),
            "AFDB_generated_vs_PDB_median": float(absolute.loc[(absolute.generated_condition == "AFDB") & (absolute.target_condition == "PDB"), "rmsd"].median()),
            "AFDB_generated_vs_AFDB_median": float(absolute.loc[(absolute.generated_condition == "AFDB") & (absolute.target_condition == "AFDB"), "rmsd"].median()),
        },
        "q2_wt": {
            "median_afdb_preference": float(wt.wt_afdb_preference.median()),
            "positive_protein_fraction": float((wt.wt_afdb_preference > 0).mean()),
        },
        "q3_baseline_adjusted": {
            "pdb_generated_median": float(pdb_effect.median()),
            "pdb_generated_q10": _quantile(pdb_effect, 0.10),
            "pdb_generated_q90": _quantile(pdb_effect, 0.90),
            "pdb_generated_positive_fraction": float((pdb_effect > 0).mean()),
            "afdb_generated_median": float(afdb_effect.median()),
            "afdb_generated_q10": _quantile(afdb_effect, 0.10),
            "afdb_generated_q90": _quantile(afdb_effect, 0.90),
            "afdb_generated_positive_fraction": float((afdb_effect > 0).mean()),
            "afdb_generated_direction_larger": bool(abs(float(afdb_effect.median())) > abs(float(pdb_effect.median()))),
        },
        "q4_descriptor_associations": association_table.to_dict("records"),
        "q5_heterogeneity": {
            "pdb_generated_q10": _quantile(pdb_effect, 0.10),
            "pdb_generated_q90": _quantile(pdb_effect, 0.90),
            "afdb_generated_q10": _quantile(afdb_effect, 0.10),
            "afdb_generated_q90": _quantile(afdb_effect, 0.90),
            "pdb_afdb_direction_disagreement_fraction": float(((pdb_effect * afdb_effect) < 0).mean()),
        },
        "interpretation_boundary": "Predicted-structure target preference only; no biological, functional, stability, fitness, or causal claim.",
    }
    provenance = {
        "generated_rows": len(generated),
        "absolute_rows": len(absolute),
        "wt_rows": len(wt),
        "protein_rows": len(effects),
        "association_rows": len(association_table),
    }
    return StructuralValidationAsymmetryResult(absolute, wt, merged, association_table, summary, provenance)


def _write_immutable(path: Path, payload: bytes) -> str:
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise StructuralValidationAsymmetryError(f"immutable asymmetry output conflict: {path.name}") from None
    return _sha(path)


def _render_report(result: StructuralValidationAsymmetryResult) -> str:
    summary = result.summary
    q2 = summary["q2_wt"]
    q3 = summary["q3_baseline_adjusted"]
    q5 = summary["q5_heterogeneity"]
    lines = [
        "# Structural Validation Asymmetry",
        "",
        "## Q1. Absolute generated comparisons",
        "",
        "Absolute RMSD groups are retained separately; no source/alternative difference is used as a substitute for them.",
        f"- PDB-generated vs PDB median: {summary['q1_absolute_groups']['PDB_generated_vs_PDB_median']:.6f}",
        f"- PDB-generated vs AFDB median: {summary['q1_absolute_groups']['PDB_generated_vs_AFDB_median']:.6f}",
        f"- AFDB-generated vs PDB median: {summary['q1_absolute_groups']['AFDB_generated_vs_PDB_median']:.6f}",
        f"- AFDB-generated vs AFDB median: {summary['q1_absolute_groups']['AFDB_generated_vs_AFDB_median']:.6f}",
        "",
        "## Q2. WT baseline",
        "",
        f"The WT baseline AFDB preference median is {q2['median_afdb_preference']:.6f}; the positive-protein fraction is {q2['positive_protein_fraction']:.6f}.",
        "",
        "## Q3. Baseline-adjusted generated preference",
        "",
        f"PDB-generated median effect is {q3['pdb_generated_median']:.6f} (positive fraction {q3['pdb_generated_positive_fraction']:.6f}).",
        f"AFDB-generated median effect is {q3['afdb_generated_median']:.6f} (positive fraction {q3['afdb_generated_positive_fraction']:.6f}).",
        "These are generated preference shifts after subtracting the same-protein WT preference.",
        "",
        "## Q4. Upstream descriptor associations",
        "",
        "Protein-level descriptive Spearman associations are reported in the association table for D_PA excess, generative JS burden, and local remodeling magnitude; no p-values are computed.",
        "",
        "## Q5. Heterogeneity",
        "",
        f"PDB-generated baseline-adjusted q10/q90: {q5['pdb_generated_q10']:.6f}/{q5['pdb_generated_q90']:.6f}.",
        f"AFDB-generated baseline-adjusted q10/q90: {q5['afdb_generated_q10']:.6f}/{q5['afdb_generated_q90']:.6f}.",
        f"The two directional effects have opposite signs in {q5['pdb_afdb_direction_disagreement_fraction']:.6f} of proteins, indicating protein-level heterogeneity.",
        "",
        "Interpretation is limited to ESMFold-predicted geometry relative to paired targets. This analysis does not establish biological fitness, stability, function, experimental failure, or causality.",
        "",
    ]
    return "\n".join(lines)


def materialize_structural_validation_asymmetry(
    result: StructuralValidationAsymmetryResult,
    *,
    output_root: Path,
    input_paths: dict[str, Path] | None = None,
    wt_identity: str | None = None,
    model_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write all asymmetry outputs from the one structured result."""
    root = Path(output_root).expanduser().resolve()
    tables = {
        "absolute_structural_comparisons": result.absolute_comparisons,
        "wt_structural_baseline": result.wt_baseline,
        "baseline_adjusted_protein_effects": result.protein_effects,
        "descriptor_associations": result.associations,
    }
    files: dict[str, str] = {}
    for name, frame in tables.items():
        payload = frame.to_parquet(index=False)
        if not isinstance(payload, bytes):
            raise StructuralValidationAsymmetryError("parquet renderer did not return bytes")
        files[name] = _write_immutable(root / f"{name}.parquet", payload)
    files["summary"] = _write_immutable(
        root / "summary.json",
        (json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    files["report"] = _write_immutable(root / "report.md", _render_report(result).encode())
    input_hashes: dict[str, str] = {}
    for label, path in (input_paths or {}).items():
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise StructuralValidationAsymmetryError(f"missing asymmetry input: {label}")
        input_hashes[label] = _sha(target)
    manifest = {
        "schema_version": "structural_validation_asymmetry_manifest_v1",
        "analysis": "structural_validation_asymmetry",
        "files": files,
        "input_sha256": input_hashes,
        "wt_prediction_identity": wt_identity,
        "model_identity": model_identity or {},
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "definitions": {
            "afdb_preference": "RMSD_to_PDB - RMSD_to_AFDB",
            "baseline_adjusted_effect": "generated_AFDB_preference - WT_AFDB_preference",
            "association_unit": "protein",
        },
        "interpretation_boundary": result.summary["interpretation_boundary"],
    }
    files["manifest"] = _write_immutable(
        root / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )
    manifest["files"] = files
    return manifest
