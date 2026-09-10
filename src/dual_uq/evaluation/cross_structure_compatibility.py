"""Protein-level analysis of cross-structure ProteinMPNN compatibility loss."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.artifacts import parquet_bytes
from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes

ANALYSIS_PROTOCOL = "cross_structure_compatibility_temperature_0.1_v1"


class CrossStructureCompatibilityError(ValueError):
    """Structured cross-structure analysis input or invariant failure."""


@dataclass(frozen=True, slots=True)
class CrossStructureCompatibilityResult:
    """Canonical result from which every output is rendered."""

    score_rows: pd.DataFrame
    directional_loss: pd.DataFrame
    protein_summary: pd.DataFrame
    descriptor_associations: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CrossStructureCompatibilityError(f"{label} missing columns: {missing}")


def _validate_score_rows(scores: pd.DataFrame, *, expected_protein_count: int) -> pd.DataFrame:
    required = {
        "protein_id",
        "generated_condition",
        "evaluated_condition",
        "sample_index",
        "sequence_hash",
        "score_mean_logp_mask",
    }
    _require_columns(scores, required, "cross-score rows")
    result = scores.copy()
    if set(result["generated_condition"].dropna()) != {"PDB", "AFDB"}:
        raise CrossStructureCompatibilityError("generated conditions must be PDB and AFDB")
    if set(result["evaluated_condition"].dropna()) != {"PDB", "AFDB"}:
        raise CrossStructureCompatibilityError("evaluated conditions must be PDB and AFDB")
    key = ["protein_id", "generated_condition", "evaluated_condition", "sample_index", "sequence_hash"]
    if result.duplicated(key).any():
        raise CrossStructureCompatibilityError("cross-score rows contain duplicate keys")
    result["score_mean_logp_mask"] = pd.to_numeric(result["score_mean_logp_mask"], errors="coerce")
    if not np.isfinite(result["score_mean_logp_mask"].to_numpy(dtype=float)).all():
        raise CrossStructureCompatibilityError("cross-score values must be finite")
    proteins = set(result["protein_id"].unique())
    if len(proteins) != expected_protein_count:
        raise CrossStructureCompatibilityError("cross-score cohort cardinality differs")
    expected = {
        (protein, generated, evaluated, index)
        for protein in result["protein_id"].unique()
        for generated in ("PDB", "AFDB")
        for evaluated in ("PDB", "AFDB")
        for index in range(256)
    }
    observed = set(
        result[["protein_id", "generated_condition", "evaluated_condition", "sample_index"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise CrossStructureCompatibilityError("cross-score grid is incomplete")
    if len(result) != len(expected):
        raise CrossStructureCompatibilityError("cross-score row cardinality differs")
    return result.sort_values(
        ["protein_id", "generated_condition", "evaluated_condition", "sample_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _directional_loss(scores: pd.DataFrame) -> pd.DataFrame:
    key = ["protein_id", "generated_condition", "sample_index", "sequence_hash"]
    pivot = scores.pivot(index=key, columns="evaluated_condition", values="score_mean_logp_mask")
    if set(pivot.columns) != {"PDB", "AFDB"}:
        raise CrossStructureCompatibilityError("each generated sequence requires both evaluated structures")
    pivot = pivot.reset_index()
    pivot["pdb_nll"] = -pivot["PDB"]
    pivot["afdb_nll"] = -pivot["AFDB"]
    pivot["matching_nll"] = np.where(
        pivot["generated_condition"].eq("PDB"), pivot["pdb_nll"], pivot["afdb_nll"]
    )
    pivot["alternative_nll"] = np.where(
        pivot["generated_condition"].eq("PDB"), pivot["afdb_nll"], pivot["pdb_nll"]
    )
    pivot["delta_nll"] = pivot["alternative_nll"] - pivot["matching_nll"]
    pivot["delta_p_to_a"] = np.where(pivot["generated_condition"].eq("PDB"), pivot["delta_nll"], np.nan)
    pivot["delta_a_to_p"] = np.where(pivot["generated_condition"].eq("AFDB"), pivot["delta_nll"], np.nan)
    return pivot[
        [
            "protein_id",
            "generated_condition",
            "sample_index",
            "sequence_hash",
            "matching_nll",
            "alternative_nll",
            "delta_nll",
            "delta_p_to_a",
            "delta_a_to_p",
        ]
    ].sort_values(["protein_id", "generated_condition", "sample_index"], kind="mergesort").reset_index(drop=True)


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return None
    value = joined.iloc[:, 0].rank(method="average").corr(joined.iloc[:, 1].rank(method="average"))
    return None if pd.isna(value) else float(value)


def _protein_summary(
    directional: pd.DataFrame,
    generation: pd.DataFrame,
    remodeling: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        generation,
        {"protein_id", "d_pa_excess_independent", "js_burden_mean"},
        "generation summary",
    )
    _require_columns(remodeling, {"protein_id", "position_magnitude_upper_tail_excess"}, "remodeling summary")
    score_proteins = set(directional["protein_id"])
    if not score_proteins.issubset(set(generation["protein_id"])):
        raise CrossStructureCompatibilityError("generation summary cohort differs")
    if not score_proteins.issubset(set(remodeling["protein_id"])):
        raise CrossStructureCompatibilityError("remodeling summary cohort differs")
    rows: list[dict[str, Any]] = []
    for protein_id, group in directional.groupby("protein_id", sort=True):
        p_to_a = group["delta_p_to_a"].dropna()
        a_to_p = group["delta_a_to_p"].dropna()
        if len(p_to_a) != 256 or len(a_to_p) != 256:
            raise CrossStructureCompatibilityError("directional sequence counts differ")
        p_mean = float(p_to_a.mean())
        a_mean = float(a_to_p.mean())
        rows.append(
            {
                "protein_id": protein_id,
                "n_pdb_generated": len(p_to_a),
                "n_afdb_generated": len(a_to_p),
                "delta_p_to_a_mean": p_mean,
                "delta_p_to_a_median": float(p_to_a.median()),
                "delta_p_to_a_q10": float(p_to_a.quantile(0.10, interpolation="linear")),
                "delta_p_to_a_q90": float(p_to_a.quantile(0.90, interpolation="linear")),
                "delta_p_to_a_positive_fraction": float((p_to_a > 0).mean()),
                "delta_a_to_p_mean": a_mean,
                "delta_a_to_p_median": float(a_to_p.median()),
                "delta_a_to_p_q10": float(a_to_p.quantile(0.10, interpolation="linear")),
                "delta_a_to_p_q90": float(a_to_p.quantile(0.90, interpolation="linear")),
                "delta_a_to_p_positive_fraction": float((a_to_p > 0).mean()),
                "delta_cross": 0.5 * (p_mean + a_mean),
                "directional_asymmetry_p_to_a_minus_a_to_p": p_mean - a_mean,
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.merge(
        generation[["protein_id", "d_pa_excess_independent", "js_burden_mean"]],
        on="protein_id",
        how="left",
        validate="one_to_one",
    ).merge(
        remodeling[["protein_id", "position_magnitude_upper_tail_excess"]].rename(
            columns={"position_magnitude_upper_tail_excess": "local_remodeling_magnitude_burden"}
        ),
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
    if summary[["d_pa_excess_independent", "js_burden_mean", "local_remodeling_magnitude_burden"]].isna().any().any():
        raise CrossStructureCompatibilityError("descriptor summaries do not cover the cross-score cohort")
    associations = pd.DataFrame(
        [
            {
                "descriptor": descriptor,
                "estimand": "protein-level Spearman correlation with delta_cross",
                "spearman": _spearman(summary[descriptor], summary["delta_cross"]),
                "n_proteins": len(summary),
            }
            for descriptor in (
                "d_pa_excess_independent",
                "js_burden_mean",
                "local_remodeling_magnitude_burden",
            )
        ]
    )
    return summary, associations


def _summary(protein: pd.DataFrame, associations: pd.DataFrame) -> dict[str, Any]:
    delta_cross = protein["delta_cross"]
    p_to_a = protein["delta_p_to_a_mean"]
    a_to_p = protein["delta_a_to_p_mean"]
    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "scientific_unit": "protein",
        "protein_count": len(protein),
        "directional_loss": {
            "pdb_generated_to_afdb": {
                "median": float(p_to_a.median()),
                "q10": float(p_to_a.quantile(0.10, interpolation="linear")),
                "q90": float(p_to_a.quantile(0.90, interpolation="linear")),
                "positive_protein_fraction": float((p_to_a > 0).mean()),
            },
            "afdb_generated_to_pdb": {
                "median": float(a_to_p.median()),
                "q10": float(a_to_p.quantile(0.10, interpolation="linear")),
                "q90": float(a_to_p.quantile(0.90, interpolation="linear")),
                "positive_protein_fraction": float((a_to_p > 0).mean()),
            },
        },
        "symmetric_delta_cross": {
            "median": float(delta_cross.median()),
            "q10": float(delta_cross.quantile(0.10, interpolation="linear")),
            "q90": float(delta_cross.quantile(0.90, interpolation="linear")),
            "positive_protein_fraction": float((delta_cross > 0).mean()),
            "median_directional_asymmetry": float(
                protein["directional_asymmetry_p_to_a_minus_a_to_p"].median()
            ),
        },
        "descriptor_associations": associations.to_dict(orient="records"),
        "interpretation_boundary": "ProteinMPNN model-level compatibility loss only; no biological fitness, stability, function, or experimental claim",
    }


def build_cross_structure_analysis(
    score_rows: pd.DataFrame,
    generation_summary: pd.DataFrame,
    remodeling_summary: pd.DataFrame,
    *,
    input_provenance: dict[str, Any] | None = None,
    expected_protein_count: int = 68,
) -> CrossStructureCompatibilityResult:
    """Build the sole structured result consumed by all output renderers."""
    if expected_protein_count <= 0:
        raise CrossStructureCompatibilityError("expected protein count must be positive")
    scores = _validate_score_rows(score_rows, expected_protein_count=expected_protein_count)
    directional = _directional_loss(scores)
    protein, associations = _protein_summary(directional, generation_summary, remodeling_summary)
    summary = _summary(protein, associations)
    return CrossStructureCompatibilityResult(
        score_rows=scores,
        directional_loss=directional,
        protein_summary=protein,
        descriptor_associations=associations,
        summary=summary,
        input_provenance=dict(input_provenance or {}),
    )


def _immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise CrossStructureCompatibilityError(f"immutable output conflict: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, payload)
    return "created"


def materialize_cross_structure_analysis(
    result: CrossStructureCompatibilityResult,
    output_root: Path,
) -> dict[str, Any]:
    """Materialize immutable score, directional, protein, and report artifacts."""
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "cross_structure_scores.parquet": result.score_rows,
        "directional_compatibility_loss.parquet": result.directional_loss,
        "protein_cross_structure_summary.parquet": result.protein_summary,
        "descriptor_associations.parquet": result.descriptor_associations,
    }
    outputs: dict[str, Any] = {}
    statuses = []
    for filename, frame in tables.items():
        payload = parquet_bytes(frame)
        status = _immutable_bytes(output_root / filename, payload)
        statuses.append(status)
        outputs[filename] = {"path": filename, "rows": len(frame), "sha256": sha256_bytes(payload)}
    summary_payload = (json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    report_payload = _render_report(result.summary).encode()
    statuses.append(_immutable_bytes(output_root / "summary.json", summary_payload))
    statuses.append(_immutable_bytes(output_root / "report.md", report_payload))
    outputs["summary.json"] = {"path": "summary.json", "rows": None, "sha256": sha256_bytes(summary_payload)}
    outputs["report.md"] = {"path": "report.md", "rows": None, "sha256": sha256_bytes(report_payload)}
    manifest = {
        "manifest_schema": "cross_structure_compatibility_manifest_v1",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "status": "COMPLETE",
        "scientific_scope": "68-protein generated-sequence cross-structure ProteinMPNN compatibility",
        "estimands": {
            "nll": "-score_mean_logp_mask",
            "delta_p_to_a": "NLL(sequence generated under PDB | AFDB) - NLL(sequence generated under PDB | PDB)",
            "delta_a_to_p": "NLL(sequence generated under AFDB | PDB) - NLL(sequence generated under AFDB | AFDB)",
            "delta_cross": "0.5 * (protein mean Delta_P_to_A + protein mean Delta_A_to_P)",
        },
        "input_provenance": result.input_provenance,
        "outputs": outputs,
        "interpretation_boundary": result.summary["interpretation_boundary"],
    }
    manifest_payload = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    manifest_status = _immutable_bytes(output_root / "manifest.json", manifest_payload)
    statuses.append(manifest_status)
    return {
        "manifest_status": manifest_status,
        "output_statuses": statuses,
        "manifest_sha256": sha256_bytes(manifest_payload),
    }


def _render_report(summary: dict[str, Any]) -> str:
    p = summary["directional_loss"]["pdb_generated_to_afdb"]
    a = summary["directional_loss"]["afdb_generated_to_pdb"]
    cross = summary["symmetric_delta_cross"]
    protein_count = summary["protein_count"]
    associations = ", ".join(
        f"{row['descriptor']}={row['spearman']:.6g}" if row["spearman"] is not None
        else f"{row['descriptor']}=NA"
        for row in summary["descriptor_associations"]
    )
    return "\n".join(
        [
            "# Cross-Structure ProteinMPNN Compatibility Analysis",
            "",
            "Status: protein-level model self-consistency analysis only.",
            "",
            "## Q1. Do PDB-generated sequences lose compatibility on AFDB structures?",
            "",
            f"- Protein-level Delta_P_to_A median/q10/q90={p['median']:.6g}/{p['q10']:.6g}/{p['q90']:.6g}; positive in {p['positive_protein_fraction'] * protein_count:.0f}/{protein_count} proteins.",
            "",
            "## Q2. Do AFDB-generated sequences lose compatibility on PDB structures?",
            "",
            f"- Protein-level Delta_A_to_P median/q10/q90={a['median']:.6g}/{a['q10']:.6g}/{a['q90']:.6g}; positive in {a['positive_protein_fraction'] * protein_count:.0f}/{protein_count} proteins.",
            "",
            "## Q3. Is the effect symmetric?",
            "",
            f"- Delta_cross median/q10/q90={cross['median']:.6g}/{cross['q10']:.6g}/{cross['q90']:.6g}; median PDB-to-AFDB minus AFDB-to-PDB asymmetry={cross['median_directional_asymmetry']:.6g}.",
            "",
            "## Q4. How heterogeneous is cross-structure loss?",
            "",
            f"- Delta_cross is positive in {cross['positive_protein_fraction'] * protein_count:.0f}/{protein_count} proteins; q10/q90 are reported above at the protein unit.",
            "",
            "## Q5. Does generative divergence track compatibility loss?",
            "",
            f"- Protein-level Spearman associations with Delta_cross: {associations}.",
            "- These are descriptive associations without p-values or biological interpretation.",
            "",
            "## Interpretation boundary",
            "",
            "Positive deltas mean lower ProteinMPNN model compatibility under the paired alternative structure. This does not establish biological fitness, stability, functional failure, or experimental failure.",
            "",
        ]
    )
