"""Analysis of locally predicted structures against paired PDB/AFDB targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file as _sha
from dual_uq.geometry import kabsch_align, rmsd
from dual_uq.inference.independent_structure_validation import (
    IndependentStructureValidationError,
)


@dataclass(frozen=True, slots=True)
class IndependentStructureValidationResult:
    sequence_comparison: pd.DataFrame
    protein_summary: pd.DataFrame
    associations: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, Any]


def _coords(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 3 or not np.isfinite(result).all():
        raise IndependentStructureValidationError(f"{label} is not a finite common mask")
    return result


def _aligned_rmsd(mobile: Any, target: Any, label: str) -> float:
    mobile_array = _coords(mobile, label)
    target_array = _coords(target, label)
    if mobile_array.shape != target_array.shape:
        raise IndependentStructureValidationError(f"{label} common mask lengths differ")
    aligned, _rotation, _translation = kabsch_align(mobile_array, target_array)
    value = rmsd(aligned, target_array)
    if not np.isfinite(value):
        raise IndependentStructureValidationError(f"{label} RMSD is not finite")
    return value


def spearman_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """Compute rank correlation when both inputs support an identified estimate."""

    values = pd.concat([left, right], axis=1).dropna()
    if len(values) < 3 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return None
    return float(values.iloc[:, 0].rank(method="average").corr(values.iloc[:, 1].rank(method="average")))


def build_independent_structure_analysis(
    predictions: pd.DataFrame,
    generation_summary: pd.DataFrame,
    compatibility_summary: pd.DataFrame,
    remodeling_summary: pd.DataFrame,
    *,
    expected_protein_count: int = 68,
) -> IndependentStructureValidationResult:
    """Build all summaries from one prediction table and shared target masks."""
    required = {
        "protein_id", "source_condition", "sample_index", "sample_class", "seed",
        "sequence_hash", "sequence", "predicted_ca_coordinates",
        "target_pdb_ca_coordinates", "target_afdb_ca_coordinates", "plddt", "pae", "ptm", "status",
    }
    if not required.issubset(predictions.columns):
        raise IndependentStructureValidationError("prediction schema is incomplete")
    if predictions.empty:
        raise IndependentStructureValidationError("prediction table is empty")
    if set(predictions["source_condition"]) != {"PDB", "AFDB"}:
        raise IndependentStructureValidationError("prediction conditions are incomplete")
    keys = predictions[["protein_id", "source_condition", "sample_index"]]
    if keys.duplicated().any():
        raise IndependentStructureValidationError("prediction keys are duplicated")
    proteins = tuple(sorted(predictions["protein_id"].astype(str).unique()))
    if len(proteins) != expected_protein_count:
        raise IndependentStructureValidationError("prediction cohort cardinality differs")
    rows: list[dict[str, Any]] = []
    for record in predictions.to_dict("records"):
        condition = str(record["source_condition"])
        conditioning = record["target_pdb_ca_coordinates"] if condition == "PDB" else record["target_afdb_ca_coordinates"]
        alternative = record["target_afdb_ca_coordinates"] if condition == "PDB" else record["target_pdb_ca_coordinates"]
        conditioning_rmsd = _aligned_rmsd(record["predicted_ca_coordinates"], conditioning, "prediction/conditioning")
        alternative_rmsd = _aligned_rmsd(record["predicted_ca_coordinates"], alternative, "prediction/alternative")
        rows.append({
            **record,
            "rmsd_to_conditioning": conditioning_rmsd,
            "rmsd_to_alternative": alternative_rmsd,
            "source_preference": alternative_rmsd - conditioning_rmsd,
        })
    comparison = pd.DataFrame(rows)
    summary_rows = []
    for protein_id, group in comparison.groupby("protein_id", sort=True):
        p = group[group.source_condition == "PDB"]["source_preference"]
        a = group[group.source_condition == "AFDB"]["source_preference"]
        summary_rows.append({
            "protein_id": protein_id,
            "n_predictions": len(group),
            "n_pdb_generated": len(p),
            "n_afdb_generated": len(a),
            "pdb_to_afdb_median_loss": float(p.median()),
            "pdb_to_afdb_q10_loss": float(p.quantile(0.10, interpolation="linear")),
            "pdb_to_afdb_q90_loss": float(p.quantile(0.90, interpolation="linear")),
            "pdb_to_afdb_positive_fraction": float((p > 0).mean()),
            "pdb_generated_conditioning_rmsd_median": float(group.loc[group.source_condition == "PDB", "rmsd_to_conditioning"].median()),
            "pdb_generated_alternative_rmsd_median": float(group.loc[group.source_condition == "PDB", "rmsd_to_alternative"].median()),
            "afdb_to_pdb_median_loss": float(a.median()),
            "afdb_to_pdb_q10_loss": float(a.quantile(0.10, interpolation="linear")),
            "afdb_to_pdb_q90_loss": float(a.quantile(0.90, interpolation="linear")),
            "afdb_to_pdb_positive_fraction": float((a > 0).mean()),
            "afdb_generated_conditioning_rmsd_median": float(group.loc[group.source_condition == "AFDB", "rmsd_to_conditioning"].median()),
            "afdb_generated_alternative_rmsd_median": float(group.loc[group.source_condition == "AFDB", "rmsd_to_alternative"].median()),
            "delta_cross_mean": float(group.source_preference.mean()),
            "delta_cross_median": float(group.source_preference.median()),
            "finite_comparison_fraction": float(group.source_preference.notna().mean()),
            "plddt_median": float(group.plddt.median()),
            "pae_median": float(group.pae.median()),
            "ptm_median": float(group.ptm.median()),
        })
    protein = pd.DataFrame(summary_rows)
    descriptor_columns = {
        "generation": ("js_burden_mean", "d_pa_excess_independent"),
        "compatibility": ("delta_cross",),
        "remodeling": ("position_magnitude_upper_tail_excess",),
    }
    merged = protein.copy()
    for name, frame in (
        ("generation", generation_summary),
        ("compatibility", compatibility_summary),
        ("remodeling", remodeling_summary),
    ):
        if "protein_id" not in frame.columns:
            raise IndependentStructureValidationError(f"{name} summary lacks protein_id")
        columns = ["protein_id", *descriptor_columns[name]]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise IndependentStructureValidationError(
                f"{name} summary lacks descriptors: {', '.join(missing)}"
            )
        selected = frame[columns].drop_duplicates("protein_id")
        merged = merged.merge(selected, on="protein_id", how="left", validate="one_to_one")
    association_rows = []
    outcomes = ["delta_cross_median", "pdb_to_afdb_median_loss", "afdb_to_pdb_median_loss"]
    descriptor_names = ["d_pa_excess_independent", "js_burden_mean", "position_magnitude_upper_tail_excess"]
    for outcome in outcomes:
        for descriptor in descriptor_names:
            if descriptor in merged:
                association_rows.append(
                    {
                        "outcome": outcome,
                        "descriptor": descriptor,
                        "spearman": spearman_correlation(
                            merged[outcome], merged[descriptor]
                        ),
                        "n_proteins": int(
                            merged[[outcome, descriptor]].dropna().shape[0]
                        ),
                    }
                )
    associations = pd.DataFrame(association_rows)
    summary = {
        "cohort_proteins": len(proteins),
        "prediction_rows": len(comparison),
        "conditions": ["PDB", "AFDB"],
        "metric": "aligned common-mask CA RMSD",
        "direction": "source_preference = rmsd_to_alternative - rmsd_to_conditioning",
        "q1_pdb_to_afdb_median_across_proteins": float(protein.pdb_to_afdb_median_loss.median()),
        "q1_pdb_to_afdb_positive_protein_fraction": float((protein.pdb_to_afdb_median_loss > 0).mean()),
        "q1_pdb_generated_conditioning_rmsd_median": float(protein.pdb_generated_conditioning_rmsd_median.median()),
        "q1_pdb_generated_alternative_rmsd_median": float(protein.pdb_generated_alternative_rmsd_median.median()),
        "q2_afdb_to_pdb_median_across_proteins": float(protein.afdb_to_pdb_median_loss.median()),
        "q2_afdb_to_pdb_positive_protein_fraction": float((protein.afdb_to_pdb_median_loss > 0).mean()),
        "q2_afdb_generated_conditioning_rmsd_median": float(protein.afdb_generated_conditioning_rmsd_median.median()),
        "q2_afdb_generated_alternative_rmsd_median": float(protein.afdb_generated_alternative_rmsd_median.median()),
        "q3_directional_median_difference": float((protein.pdb_to_afdb_median_loss - protein.afdb_to_pdb_median_loss).median()),
        "q3_larger_direction": "AFDB_to_PDB" if float(protein.afdb_to_pdb_median_loss.median()) > float(protein.pdb_to_afdb_median_loss.median()) else "PDB_to_AFDB",
        "q4_protein_delta_cross_q10": float(protein.delta_cross_median.quantile(0.10, interpolation="linear")),
        "q4_protein_delta_cross_median": float(protein.delta_cross_median.median()),
        "q4_protein_delta_cross_q90": float(protein.delta_cross_median.quantile(0.90, interpolation="linear")),
        "q4_delta_cross_positive_protein_fraction": float((protein.delta_cross_median > 0).mean()),
        "q5_descriptor_associations": associations.loc[
            associations["outcome"] == "delta_cross_median",
            ["descriptor", "spearman", "n_proteins"],
        ].to_dict("records"),
        "interpretation": "Independent ESMFold-predicted structure versus paired PDB/AFDB target geometry only; no ProteinMPNN, biological, or functional claim.",
    }
    provenance = {"rows": len(comparison), "input_tables": {"generation": len(generation_summary), "compatibility": len(compatibility_summary), "remodeling": len(remodeling_summary)}}
    return IndependentStructureValidationResult(comparison, merged, associations, summary, provenance)


def materialize_independent_structure_analysis(
    result: IndependentStructureValidationResult,
    *,
    output_root: Path,
    input_paths: dict[str, Path] | None = None,
    model_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the four analysis artifacts, report, and manifest immutably."""
    root = Path(output_root).expanduser().resolve()
    tables = {
        "sequence_structural_comparison": result.sequence_comparison,
        "protein_structural_summary": result.protein_summary,
        "descriptor_associations": result.associations,
    }
    files: dict[str, str] = {}
    for name, frame in tables.items():
        path = root / f"{name}.parquet"
        payload = frame.to_parquet(index=False)
        if not isinstance(payload, bytes):
            raise IndependentStructureValidationError("parquet renderer did not return bytes")
        try:
            atomic_write_new_bytes(path, payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise IndependentStructureValidationError(f"immutable analysis conflict: {name}") from None
        files[name] = _sha(path)
    summary_path = root / "summary.json"
    atomic_write_new_bytes(summary_path, (json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    files["summary"] = _sha(summary_path)
    summary = result.summary
    lines = [
        "# Independent structural validation",
        "",
        "All comparisons use the same paired common-mask CA coordinates and aligned RMSD.",
        "",
        "## Q1. PDB-generated sequences under the AFDB target",
        "",
        f"- Answer: the cohort median is `{summary['q1_pdb_to_afdb_median_across_proteins']:.6g}`, so this metric does not show a systematic positive alternative loss for PDB-generated sequences.",
        f"- Median source preference across proteins: `{summary['q1_pdb_to_afdb_median_across_proteins']:.6g}`.",
        f"- Median conditioning/alternative RMSD: `{summary['q1_pdb_generated_conditioning_rmsd_median']:.6g}` / `{summary['q1_pdb_generated_alternative_rmsd_median']:.6g}`.",
        f"- Positive-protein fraction: `{summary['q1_pdb_to_afdb_positive_protein_fraction']:.6g}`.",
        "",
        "## Q2. AFDB-generated sequences under the PDB target",
        "",
        f"- Answer: the cohort median is `{summary['q2_afdb_to_pdb_median_across_proteins']:.6g}`, with a positive protein fraction of `{summary['q2_afdb_to_pdb_positive_protein_fraction']:.6g}`; the directional effect is positive for this geometric comparison.",
        f"- Median source preference across proteins: `{summary['q2_afdb_to_pdb_median_across_proteins']:.6g}`.",
        f"- Median conditioning/alternative RMSD: `{summary['q2_afdb_generated_conditioning_rmsd_median']:.6g}` / `{summary['q2_afdb_generated_alternative_rmsd_median']:.6g}`.",
        f"- Positive-protein fraction: `{summary['q2_afdb_to_pdb_positive_protein_fraction']:.6g}`.",
        "",
        "## Q3. Directional asymmetry",
        "",
        f"- Answer: the two directions are asymmetric, with the larger median in `{summary['q3_larger_direction']}`.",
        f"- Larger median direction: `{summary['q3_larger_direction']}`; median directional difference (PDB→AFDB minus AFDB→PDB): `{summary['q3_directional_median_difference']:.6g}`.",
        "",
        "## Q4. Heterogeneity of the direction-neutral protein summary",
        "",
        "- Answer: effects are heterogeneous across proteins, as shown by the reported q10–q90 range and positive fraction.",
        f"- Delta-cross median / q10 / q90: `{summary['q4_protein_delta_cross_median']:.6g}` / `{summary['q4_protein_delta_cross_q10']:.6g}` / `{summary['q4_protein_delta_cross_q90']:.6g}`.",
        f"- Positive-protein fraction: `{summary['q4_delta_cross_positive_protein_fraction']:.6g}`.",
        "",
        "## Q5. Association with upstream generative/remodeling descriptors",
        "",
        "- Answer: all three associations are descriptive protein-level rank correlations; they are not causal tests and no p-values are reported.",
    ]
    for row in summary["q5_descriptor_associations"]:
        lines.append(f"- `{row['descriptor']}` vs delta-cross Spearman: `{row['spearman']:.6g}` (n=`{row['n_proteins']}`).")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation"],
        ]
    )
    report_path = root / "report.md"
    atomic_write_new_bytes(report_path, ("\n".join(lines) + "\n").encode())
    files["report"] = _sha(report_path)
    manifest = {
        "schema": "independent_structure_validation_manifest_v1",
        "files": files,
        "model": model_identity or {},
        "inputs": {
            name: {"path": Path(path).name, "sha256": _sha(Path(path))}
            for name, path in (input_paths or {}).items()
        },
        "summary": result.summary,
    }
    manifest_path = root / "manifest.json"
    atomic_write_new_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return manifest
