"""Analysis of free ProteinMPNN generation under paired structures.

The module consumes validated generation records and derives every analysis
table from one in-memory result.  It does not run a model and does not redefine
the frozen fixed-probe descriptors.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.proteinmpnn import PROTEINMPNN_ALPHABET
from dual_uq.models.proteinmpnn_generation import GeneratedSequenceRecord

AMINO_ACIDS = tuple(PROTEINMPNN_ALPHABET[:20])
ANALYSIS_PROTOCOL = "generative_propagation_temperature_0.1_v1"
PREFIXES = (32, 64, 128, 256)
DESCRIPTORS = ("magnitude_p", "breadth_b", "rank_displacement")


class GenerativePropagationError(ValueError):
    """Structured generation-analysis input or invariant failure."""


@dataclass(frozen=True, slots=True)
class GenerativePropagationInputs:
    """Validated upstream records and unchanged frozen descriptors."""

    records: tuple[GeneratedSequenceRecord, ...]
    position_descriptors: pd.DataFrame
    cross_compatibility: pd.DataFrame | None = None
    expected_protein_count: int = 68
    input_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise GenerativePropagationError("generation records are required")
        if self.expected_protein_count <= 0:
            raise GenerativePropagationError("expected protein count must be positive")
        if any(not isinstance(record, GeneratedSequenceRecord) for record in records):
            raise GenerativePropagationError("records must be GeneratedSequenceRecord values")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "position_descriptors", self.position_descriptors.copy())
        if self.cross_compatibility is not None:
            object.__setattr__(self, "cross_compatibility", self.cross_compatibility.copy())
        object.__setattr__(self, "input_provenance", dict(self.input_provenance or {}))


@dataclass(frozen=True, slots=True)
class GenerativePropagationResult:
    """Canonical structured result consumed by all materializers and reports."""

    generated_sequences: pd.DataFrame
    position_shift: pd.DataFrame
    cross_structure_compatibility: pd.DataFrame
    protein_summary: pd.DataFrame
    descriptor_associations: pd.DataFrame
    convergence: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, Any]


def jensen_shannon_bits(left: Iterable[float], right: Iterable[float]) -> float:
    """Return zero-safe Jensen-Shannon divergence in bits."""
    p = np.asarray(tuple(left), dtype=float)
    q = np.asarray(tuple(right), dtype=float)
    if p.shape != q.shape or p.ndim != 1 or not len(p):
        raise GenerativePropagationError("JS inputs must be equal non-empty vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise GenerativePropagationError("JS inputs must be finite non-negative vectors")
    p_total = float(p.sum())
    q_total = float(q.sum())
    if p_total <= 0.0 or q_total <= 0.0:
        raise GenerativePropagationError("JS inputs must have positive mass")
    p = p / p_total
    q = q / q_total
    midpoint = 0.5 * (p + q)
    terms = np.zeros_like(p)
    p_nonzero = p > 0
    q_nonzero = q > 0
    terms[p_nonzero] = p[p_nonzero] * np.log2(p[p_nonzero] / midpoint[p_nonzero])
    terms[q_nonzero] += q[q_nonzero] * np.log2(q[q_nonzero] / midpoint[q_nonzero])
    return float(0.5 * terms.sum())


def _records_frame(records: tuple[GeneratedSequenceRecord, ...]) -> pd.DataFrame:
    rows = []
    for record in records:
        request = record.request
        rows.append(
            {
                "protein_id": request.protein_id,
                "backbone_condition": request.backbone_condition,
                "structure_sha256": request.structure_sha256,
                "canonical_positions": tuple(request.canonical_positions),
                "wt_sequence_projection": request.wt_sequence_projection,
                "temperature": float(request.temperature),
                "sample_index": int(request.sample_index),
                "seed": int(request.seed),
                "sample_class": request.sample_class,
                "decoding_realization": request.decoding_realization,
                "sequence": record.sequence,
                "sequence_hash": record.sequence_hash,
                "scorer_id": record.scorer_binding.scorer_id,
                "implementation_id": record.scorer_binding.implementation_id,
                "checkpoint_id": record.scorer_binding.checkpoint_id,
                "score_contract_id": record.scorer_binding.score_contract_id,
            }
        )
    frame = pd.DataFrame(rows)
    key = ["protein_id", "backbone_condition", "sample_index"]
    if frame.duplicated(key).any():
        raise GenerativePropagationError("generation sample keys are duplicated")
    expected = {(protein, condition, index) for protein in frame["protein_id"].unique() for condition in ("PDB", "AFDB") for index in range(256)}
    if set(map(tuple, frame[key].itertuples(index=False, name=None))) != expected:
        raise GenerativePropagationError("generation sample grid is incomplete")
    for protein_id, group in frame.groupby("protein_id", sort=False):
        conditions = set(group["backbone_condition"])
        if conditions != {"PDB", "AFDB"}:
            raise GenerativePropagationError(f"PDB/AFDB condition pair is incomplete: {protein_id}")
        positions = {tuple(value) for value in group["canonical_positions"]}
        sequences = set(group["wt_sequence_projection"])
        if len(positions) != 1 or len(sequences) != 1:
            raise GenerativePropagationError(f"generation projection differs within protein: {protein_id}")
        if group["temperature"].nunique() != 1 or float(group["temperature"].iloc[0]) != 0.1:
            raise GenerativePropagationError("generation temperature must be exactly 0.1")
        for condition, condition_group in group.groupby("backbone_condition", sort=False):
            if condition_group["sample_class"].value_counts().to_dict() != {"paired": 128, "independent": 128}:
                raise GenerativePropagationError(f"seed-class partition differs: {protein_id}/{condition}")
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def _validate_descriptors(descriptors: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "position", *DESCRIPTORS}
    missing = sorted(required - set(descriptors.columns))
    if missing:
        raise GenerativePropagationError(f"position descriptors missing columns: {missing}")
    result = descriptors.copy()
    key = ["protein_id", "position"]
    if result.duplicated(key).any():
        raise GenerativePropagationError("position descriptors contain duplicate keys")
    for column in DESCRIPTORS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].dropna().to_numpy(dtype=float)).all():
            raise GenerativePropagationError(f"position descriptor is non-finite: {column}")
    generated_proteins = set(records["protein_id"])
    result = result.loc[result["protein_id"].isin(generated_proteins)].copy()
    position_keys = set(map(tuple, result[key].itertuples(index=False, name=None)))
    record_keys = {
        (row.protein_id, int(position))
        for row in records[["protein_id", "canonical_positions"]].itertuples(index=False)
        for position in row.canonical_positions
    }
    if position_keys != record_keys:
        raise GenerativePropagationError("position descriptors do not exactly cover generated positions")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def _distribution(sequences: Iterable[str]) -> np.ndarray:
    values = tuple(sequences)
    if not values:
        raise GenerativePropagationError("cannot estimate a distribution from zero sequences")
    length = len(values[0])
    if any(len(sequence) != length for sequence in values):
        raise GenerativePropagationError("sequence lengths differ within a distribution")
    index = {amino_acid: i for i, amino_acid in enumerate(AMINO_ACIDS)}
    counts = np.zeros((length, len(AMINO_ACIDS)), dtype=float)
    for sequence in values:
        for position, amino_acid in enumerate(sequence):
            if amino_acid not in index:
                raise GenerativePropagationError("generated sequence contains non-canonical amino acid")
            counts[position, index[amino_acid]] += 1.0
    return counts / len(values)


def _mean_pairwise_hamming(left: Iterable[str], right: Iterable[str] | None = None) -> float:
    first = tuple(left)
    second = first if right is None else tuple(right)
    if not first or not second:
        return float("nan")
    length = len(first[0])
    if any(len(sequence) != length for sequence in (*first, *second)):
        raise GenerativePropagationError("sequence lengths differ for Hamming divergence")
    distances = [sum(a != b for a, b in zip(x, y, strict=True)) / length for x in first for y in second]
    if right is None:
        distances = [value for index, value in enumerate(distances) if index // len(first) < index % len(first)]
    return float(np.mean(distances)) if distances else 0.0


def _position_shift(records: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in records.groupby("protein_id", sort=False):
        positions = tuple(group.iloc[0]["canonical_positions"])
        for offset, position in enumerate(positions):
            row: dict[str, Any] = {"protein_id": protein_id, "position": int(position)}
            for sample_class, label in (("independent", "independent"), ("paired", "paired")):
                p = group.loc[(group["backbone_condition"] == "PDB") & (group["sample_class"] == sample_class)]
                a = group.loc[(group["backbone_condition"] == "AFDB") & (group["sample_class"] == sample_class)]
                p_distribution = _distribution(p["sequence"].str.slice(offset, offset + 1)) [0]
                a_distribution = _distribution(a["sequence"].str.slice(offset, offset + 1)) [0]
                row[f"pdb_q_{label}"] = json.dumps(p_distribution.tolist(), separators=(",", ":"))
                row[f"afdb_q_{label}"] = json.dumps(a_distribution.tolist(), separators=(",", ":"))
                row[f"js_bits_{label}"] = jensen_shannon_bits(p_distribution, a_distribution)
            rows.append(row)
    result = pd.DataFrame(rows)
    result = result.merge(descriptors, on=["protein_id", "position"], how="left", validate="one_to_one")
    return result


def _descriptor_associations(position: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in position.groupby("protein_id", sort=False):
        for descriptor in DESCRIPTORS:
            valid = group[[descriptor, "js_bits_independent"]].dropna()
            value = float(valid[descriptor].corr(valid["js_bits_independent"], method="spearman")) if len(valid) >= 2 and valid[descriptor].nunique() > 1 and valid["js_bits_independent"].nunique() > 1 else np.nan
            rows.append({"protein_id": protein_id, "descriptor": descriptor, "n_positions": len(valid), "spearman": value})
    return pd.DataFrame(rows)


def _cross_table(records: pd.DataFrame, cross: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["protein_id", "generated_condition", "sample_index", "sequence_hash", "pdb_score", "afdb_score", "matching_score", "cross_score", "directional_loss"]
    if cross is None:
        return pd.DataFrame(columns=columns)
    required = {"protein_id", "backbone_condition", "sample_index", "pdb_score", "afdb_score"}
    missing = sorted(required - set(cross.columns))
    if missing:
        raise GenerativePropagationError(f"cross-compatibility input missing columns: {missing}")
    source = cross.copy().rename(columns={"backbone_condition": "generated_condition"})
    key = ["protein_id", "generated_condition", "sample_index"]
    if source.duplicated(key).any():
        raise GenerativePropagationError("cross-compatibility keys are duplicated")
    left = records.rename(columns={"backbone_condition": "generated_condition"})
    joined = left.merge(source, on=key, how="left", validate="one_to_one", suffixes=("", "_score"))
    if joined[["pdb_score", "afdb_score"]].isna().any().any():
        raise GenerativePropagationError("cross-compatibility input does not cover generated records")
    joined["matching_score"] = np.where(joined["generated_condition"].eq("PDB"), joined["pdb_score"], joined["afdb_score"])
    joined["cross_score"] = np.where(joined["generated_condition"].eq("PDB"), joined["afdb_score"], joined["pdb_score"])
    joined["directional_loss"] = joined["matching_score"] - joined["cross_score"]
    return joined[columns]


def _protein_summary(records: pd.DataFrame, position: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in records.groupby("protein_id", sort=False):
        independent = group.loc[group["sample_class"] == "independent"]
        paired = group.loc[group["sample_class"] == "paired"]
        p_ind = independent.loc[independent["backbone_condition"] == "PDB", "sequence"]
        a_ind = independent.loc[independent["backbone_condition"] == "AFDB", "sequence"]
        p_pair = paired.loc[paired["backbone_condition"] == "PDB", "sequence"]
        a_pair = paired.loc[paired["backbone_condition"] == "AFDB", "sequence"]
        p_pos = position.loc[position["protein_id"] == protein_id]
        row: dict[str, Any] = {
            "protein_id": protein_id,
            "n_independent": len(p_ind),
            "n_paired": len(p_pair),
            "d_pp_independent": _mean_pairwise_hamming(p_ind),
            "d_aa_independent": _mean_pairwise_hamming(a_ind),
            "d_pa_independent": _mean_pairwise_hamming(p_ind, a_ind),
            "d_pp_paired": _mean_pairwise_hamming(p_pair),
            "d_aa_paired": _mean_pairwise_hamming(a_pair),
            "d_pa_paired": _mean_pairwise_hamming(p_pair, a_pair),
            "js_burden_mean": float(p_pos["js_bits_independent"].mean()),
            "js_burden_median": float(p_pos["js_bits_independent"].median()),
            "js_burden_q90": float(p_pos["js_bits_independent"].quantile(0.90)),
            "magnitude_p_median": float(p_pos["magnitude_p"].median()),
            "breadth_b_median": float(p_pos["breadth_b"].median()),
            "rank_displacement_median": float(p_pos["rank_displacement"].median()),
        }
        row["d_pa_excess_independent"] = row["d_pa_independent"] - 0.5 * (row["d_pp_independent"] + row["d_aa_independent"])
        if cross.empty:
            row.update({"symmetric_cross_loss": np.nan, "cross_loss_status": "unavailable"})
        else:
            values = cross.loc[cross["protein_id"] == protein_id, "directional_loss"]
            row.update({"symmetric_cross_loss": float(values.mean()), "cross_loss_status": "defined"})
        rows.append(row)
    return pd.DataFrame(rows)


def _convergence(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in records.groupby("protein_id", sort=False):
        for mode in ("full", "paired", "independent"):
            for prefix in PREFIXES:
                if mode in {"independent", "paired"} and prefix > 128:
                    continue
                if mode == "paired":
                    subset = group.loc[group["sample_class"] == "paired"]
                elif mode == "independent":
                    subset = group.loc[group["sample_class"] == "independent"]
                else:
                    subset = group
                subset = subset.loc[subset["sample_index"] < (prefix if mode != "independent" else 128 + prefix)]
                p = subset.loc[subset["backbone_condition"] == "PDB", "sequence"]
                a = subset.loc[subset["backbone_condition"] == "AFDB", "sequence"]
                if len(p) != prefix or len(a) != prefix:
                    raise GenerativePropagationError("convergence prefix is not nested and complete")
                rows.append({
                    "protein_id": protein_id,
                    "sample_mode": mode,
                    "prefix": prefix,
                    "d_pp": _mean_pairwise_hamming(p),
                    "d_aa": _mean_pairwise_hamming(a),
                    "d_pa": _mean_pairwise_hamming(p, a),
                    "d_pa_excess": _mean_pairwise_hamming(p, a) - 0.5 * (_mean_pairwise_hamming(p) + _mean_pairwise_hamming(a)),
                })
    return pd.DataFrame(rows)


def _summary(protein: pd.DataFrame, associations: pd.DataFrame, cross_available: bool) -> dict[str, Any]:
    descriptor_medians = associations.groupby("descriptor", sort=False)["spearman"].apply(lambda values: float(values.abs().median()) if values.notna().any() else np.nan).to_dict()
    finite = {key: value for key, value in descriptor_medians.items() if np.isfinite(value)}
    best = max(finite, key=lambda key: (finite[key], key)) if finite else None
    excess = protein["d_pa_excess_independent"]
    js = protein["js_burden_mean"]
    cross = protein.loc[protein["symmetric_cross_loss"].notna(), "symmetric_cross_loss"]
    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "scientific_unit": "protein",
        "n_proteins": len(protein),
        "generated_row_count": int(len(protein) * 2 * 256),
        "independent_sample_count_per_condition": 128,
        "paired_sample_count_per_condition": 128,
        "temperature": 0.1,
        "q1_between_vs_within": {
            "proteins_d_pa_greater": int((excess > 0).sum()),
            "protein_fraction_d_pa_greater": float((excess > 0).mean()),
            "median_excess": float(excess.median()),
            "estimand": "independent D_PA minus mean(D_PP,D_AA)",
        },
        "q2_position_distribution_propagation": {
            "proteins_with_positive_js_burden": int((js > 0).sum()),
            "median_protein_js_burden_bits": float(js.median()),
            "js_units": "bits",
        },
        "q3_descriptor_tracking": {
            "median_absolute_within_protein_spearman": descriptor_medians,
            "best_descriptor_by_descriptive_median_absolute_spearman": best,
            "no_p_values": True,
        },
        "q4_cross_structure_compatibility": {
            "status": "defined" if cross_available else "unavailable",
            "proteins_with_positive_symmetric_loss": int((cross > 0).sum()) if len(cross) else 0,
            "median_symmetric_loss": float(cross.median()) if len(cross) else None,
            "interpretation": "ProteinMPNN model-level compatibility only",
        },
        "q5_heterogeneity": {
            "d_pa_excess_q10": float(excess.quantile(0.10)),
            "d_pa_excess_median": float(excess.median()),
            "d_pa_excess_q90": float(excess.quantile(0.90)),
            "js_burden_q10": float(js.quantile(0.10)),
            "js_burden_q90": float(js.quantile(0.90)),
        },
        "interpretation_boundary": "descriptive generative propagation; no biological fitness, function, experimental stability, or causal mechanism claim",
    }


def build_analysis_result(inputs: GenerativePropagationInputs) -> GenerativePropagationResult:
    """Build the sole canonical result from validated generation measurements."""
    if len({record.request.protein_id for record in inputs.records}) != inputs.expected_protein_count:
        raise GenerativePropagationError("generation cohort cardinality differs")
    records = _records_frame(inputs.records)
    descriptors = _validate_descriptors(inputs.position_descriptors, records)
    position = _position_shift(records, descriptors)
    associations = _descriptor_associations(position)
    cross = _cross_table(records, inputs.cross_compatibility)
    protein = _protein_summary(records, position, cross)
    convergence = _convergence(records)
    return GenerativePropagationResult(
        generated_sequences=records,
        position_shift=position,
        cross_structure_compatibility=cross,
        protein_summary=protein,
        descriptor_associations=associations,
        convergence=convergence,
        summary=_summary(protein, associations, not cross.empty),
        input_provenance=dict(inputs.input_provenance or {}),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _immutable_bytes(path: Path, payload: bytes) -> str:
    if not isinstance(path, Path):
        raise GenerativePropagationError("output path must be a pathlib.Path")
    if path.exists():
        if path.read_bytes() != payload:
            raise GenerativePropagationError(f"immutable output conflict: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, payload)
    return "created"


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _immutable_parquet(path: Any, frame: pd.DataFrame) -> tuple[str, str]:
    payload = _parquet_bytes(frame)
    return _immutable_bytes(path, payload), sha256_bytes(payload)


def _render_report(summary: dict[str, Any]) -> str:
    q1 = summary["q1_between_vs_within"]
    q2 = summary["q2_position_distribution_propagation"]
    q3 = summary["q3_descriptor_tracking"]
    q4 = summary["q4_cross_structure_compatibility"]
    q5 = summary["q5_heterogeneity"]
    descriptor_text = ", ".join(
        f"{key}={value!s}" for key, value in q3["median_absolute_within_protein_spearman"].items()
    )
    return "\n".join(
        [
            "# Generative Propagation Analysis",
            "",
            "Status: descriptive generative propagation only; protein is the inference unit.",
            "",
            "## Q1. Is between-structure divergence greater than within-structure diversity?",
            "",
            f"- Independent-sample D_PA exceeded the mean within-condition diversity for {q1['proteins_d_pa_greater']}/{summary['n_proteins']} proteins ({q1['protein_fraction_d_pa_greater']:.6g}); median excess={q1['median_excess']:.6g}.",
            "- This is a descriptive comparison of independent marginal samples, not a significance test.",
            "",
            "## Q2. Does fixed-probe remodeling propagate to generated position distributions?",
            "",
            f"- Positive independent-sample position JS burden was observed for {q2['proteins_with_positive_js_burden']}/{summary['n_proteins']} proteins; median protein burden={q2['median_protein_js_burden_bits']:.6g} bits.",
            "- Position-level JS is joined to the frozen descriptors without recomputing them.",
            "",
            "## Q3. Which upstream descriptor best tracks propagation?",
            "",
            f"- Median absolute within-protein Spearman values: {descriptor_text}.",
            f"- Descriptive best descriptor: {q3['best_descriptor_by_descriptive_median_absolute_spearman']!s}.",
            "- No p-values or causal interpretation are reported.",
            "",
            "## Q4. Do generated sequences show cross-structure compatibility loss?",
            "",
            f"- Cross-structure scoring status: {q4['status']}; positive symmetric-loss proteins={q4['proteins_with_positive_symmetric_loss']}/{summary['n_proteins']}; median={q4['median_symmetric_loss']!s}.",
            "- The quantity is ProteinMPNN model-level compatibility loss only.",
            "",
            "## Q5. How heterogeneous are effects across proteins?",
            "",
            f"- D_PA excess q10/median/q90={q5['d_pa_excess_q10']:.6g}/{q5['d_pa_excess_median']:.6g}/{q5['d_pa_excess_q90']:.6g}.",
            f"- JS burden q10/q90={q5['js_burden_q10']:.6g}/{q5['js_burden_q90']:.6g} bits.",
            "",
            "## Interpretation boundary",
            "",
            "These outputs do not establish biological fitness, function, experimental instability, or a causal structural-uncertainty mechanism. No external folding, Rosetta, DMS, evaluator-UQ, or M/SDFI analysis is included.",
            "",
        ]
    )


def materialize_generative_propagation(
    result: GenerativePropagationResult,
    output_root: Any,
) -> dict[str, Any]:
    """Write the canonical five tables, summary, report, and final manifest."""
    if not isinstance(result, GenerativePropagationResult):
        raise GenerativePropagationError("a canonical analysis result is required")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated = result.generated_sequences.copy()
    if "canonical_positions" in generated:
        generated["canonical_positions"] = generated["canonical_positions"].map(
            lambda values: json.dumps(list(values), separators=(",", ":"))
        )
    tables = {
        "generated_sequences.parquet": generated,
        "position_generation_shift.parquet": result.position_shift,
        "cross_structure_compatibility.parquet": result.cross_structure_compatibility,
        "protein_generation_summary.parquet": result.protein_summary,
        "convergence.parquet": result.convergence,
    }
    outputs: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    for filename, frame in tables.items():
        status, digest = _immutable_parquet(output_root / filename, frame)
        statuses.append(status)
        outputs[filename] = {"path": filename, "sha256": digest, "rows": len(frame)}
    summary_payload = json.dumps(_json_safe(result.summary), sort_keys=True, indent=2, allow_nan=False) + "\n"
    statuses.append(_immutable_bytes(output_root / "summary.json", summary_payload.encode("utf-8")))
    outputs["summary.json"] = {
        "path": "summary.json",
        "sha256": sha256_bytes(summary_payload.encode("utf-8")),
        "rows": None,
    }
    report_payload = _render_report(_json_safe(result.summary)).encode("utf-8")
    statuses.append(_immutable_bytes(output_root / "report.md", report_payload))
    outputs["report.md"] = {"path": "report.md", "sha256": sha256_bytes(report_payload), "rows": None}
    model_ids = sorted(generated["implementation_id"].dropna().astype(str).unique()) if "implementation_id" in generated else []
    checkpoint_ids = sorted(generated["checkpoint_id"].dropna().astype(str).unique()) if "checkpoint_id" in generated else []
    manifest = {
        "manifest_schema": "generative_propagation_manifest_v1",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "status": "COMPLETE",
        "scientific_scope": "68-protein free ProteinMPNN generation propagation",
        "cohort": {"protein_count": len(result.protein_summary), "generated_rows": len(generated)},
        "sampling": {
            "temperature": 0.1,
            "sequences_per_condition": 256,
            "paired_samples_per_condition": 128,
            "independent_samples_per_condition": 128,
            "prefixes": list(PREFIXES),
            "pdb_independent_seed_namespace": "1000000..1000127",
            "afdb_independent_seed_namespace": "2000000..2000127",
        },
        "model_identity": {"implementation_ids": model_ids, "checkpoint_ids": checkpoint_ids},
        "estimands": {
            "d_primary": "independent sequence mean pairwise normalized Hamming distance",
            "js_primary": "independent empirical position distributions, Jensen-Shannon bits",
            "cross_loss": "signed matching-condition score minus alternative-condition score",
        },
        "input_provenance": _json_safe(result.input_provenance),
        "outputs": outputs,
        "interpretation_boundary": result.summary["interpretation_boundary"],
    }
    manifest_payload = (json.dumps(_json_safe(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    manifest_status = _immutable_bytes(output_root / "manifest.json", manifest_payload)
    statuses.append(manifest_status)
    return {
        "manifest_status": manifest_status,
        "output_statuses": statuses,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "output_count": len(outputs),
    }
