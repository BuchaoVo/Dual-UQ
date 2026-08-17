"""Descriptive PDB/AFDB pair-validity analysis.

This module consumes the frozen Scale-1B-v2 cohort, its common masks, existing
admission evidence, and the protein-level structural-response summary.  It
does not remap residues, rescore probes, or collapse provenance dimensions
into a validity score.  Missing evidence stays missing.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.confidence import load_pae, load_plddt
from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file


class PairValidityError(ValueError):
    """Raised when pair-validity inputs violate an existing schema."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class PairValidityInputs:
    """Authoritative tables and locally observed metadata for one analysis."""

    project_root: Path
    cohort: pd.DataFrame
    common_masks: pd.DataFrame
    admissions: pd.DataFrame
    protein_response: pd.DataFrame
    pdb_metadata: pd.DataFrame
    afdb_confidence: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]


@dataclass
class PairValidityResult:
    """Single structured result from which all output files are rendered."""

    pair_table: pd.DataFrame
    response_comparison: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, dict[str, Any]]


COHORT_COLUMNS = {
    "protein_id",
    "candidate_id",
    "pair_id",
    "canonical_accession",
    "pdb_id",
    "pdb_chain",
    "afdb_model_id",
    "canonical_sequence_length",
    "pdb_structure_ref",
    "afdb_structure_ref",
}
MASK_COLUMNS = {
    "protein_id",
    "canonical_position",
    "mapping_present",
    "pdb_mapped",
    "afdb_mapped",
    "pdb_backbone_complete",
    "afdb_backbone_complete",
    "common_mask",
    "observability_evaluation_status",
}
RESPONSE_COLUMNS = {"protein_id", "median_abs_d"}
FORMAL_ADMISSION = "FORMALLY_ADMITTED"
CLEAN_MAPPED_FRACTION_MIN = 0.90
_MISSING_TEXT = {"", ".", "?", "nan", "None"}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PairValidityError("schema_mismatch", f"{label} is missing columns: {missing}")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in _MISSING_TEXT else text


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _quantile(values: pd.Series, probability: float) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return None
    return float(np.quantile(finite, probability, method="linear"))


def _mask_aggregate(common_masks: pd.DataFrame) -> pd.DataFrame:
    _require_columns(common_masks, MASK_COLUMNS, "common masks")
    work = common_masks.copy()
    bool_columns = [
        "mapping_present",
        "pdb_mapped",
        "afdb_mapped",
        "pdb_backbone_complete",
        "afdb_backbone_complete",
        "common_mask",
    ]
    for column in bool_columns:
        work[column] = work[column].map(_optional_bool)
    work["ambiguous_mapping"] = work["observability_evaluation_status"].astype("string").str.contains(
        "ambig", case=False, na=False
    )
    grouped = work.groupby("protein_id", sort=False, dropna=False).agg(
        mask_row_count=("canonical_position", "size"),
        mapping_present_count=("mapping_present", "sum"),
        pdb_mapped_count=("pdb_mapped", "sum"),
        afdb_mapped_count=("afdb_mapped", "sum"),
        pdb_backbone_complete_count=("pdb_backbone_complete", "sum"),
        afdb_backbone_complete_count=("afdb_backbone_complete", "sum"),
        common_mask_count_observed=("common_mask", "sum"),
        mapping_gap_count=("mapping_present", lambda x: int(x.eq(False).sum())),
        pdb_nonobservable_count=(
            "pdb_backbone_complete",
            lambda x: int(x.eq(False).sum()),
        ),
        afdb_nonobservable_count=(
            "afdb_backbone_complete",
            lambda x: int(x.eq(False).sum()),
        ),
        ambiguous_mapping_count=("ambiguous_mapping", "sum"),
        evaluated_mask_row_count=(
            "observability_evaluation_status",
            lambda x: int(x.eq("EVALUATED").sum()),
        ),
    ).reset_index()
    for column in grouped.columns:
        if column != "protein_id":
            grouped[column] = pd.to_numeric(grouped[column], errors="coerce").astype("Int64")
    if not bool(work["ambiguous_mapping"].any()):
        grouped["ambiguous_mapping_count"] = pd.Series(pd.NA, index=grouped.index, dtype="Int64")
    return grouped


def _coalesce_admission_columns(
    cohort: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    if admissions.empty:
        return cohort.copy()
    if "candidate_id" not in admissions.columns:
        raise PairValidityError("schema_mismatch", "admissions is missing candidate_id")
    if admissions["candidate_id"].duplicated().any():
        raise PairValidityError("duplicate_admission", "admission evidence has duplicate candidates")
    keep = [
        column
        for column in (
            "candidate_id",
            "admission_status",
            "mapping_valid",
            "paired_sequence_identity",
            "mismatch_count",
            "mapped_residue_count",
            "coordinate_nonobservable_count",
            "common_mask_count",
            "common_mask_fraction_of_mapped",
            "exact_variant_authorized",
            "observed_variants_json",
            "terminal_reason_code",
            "terminal_reason_details",
            "selected_fragment_start",
            "selected_fragment_end",
            "admission_evidence_conflict",
            "admission_conflict_fields",
        )
        if column in admissions.columns
    ]
    return cohort.merge(admissions[keep], on="candidate_id", how="left", validate="one_to_one")


def _merge_optional(
    frame: pd.DataFrame, extra: pd.DataFrame, *, key: str, label: str
) -> pd.DataFrame:
    if extra.empty:
        return frame
    if key not in extra.columns:
        raise PairValidityError("schema_mismatch", f"{label} is missing {key}")
    if extra[key].duplicated().any():
        raise PairValidityError("duplicate_metadata", f"{label} has duplicate {key}")
    overlapping = [column for column in extra.columns if column != key and column in frame.columns]
    if overlapping:
        extra = extra.drop(columns=overlapping)
    return frame.merge(extra, on=key, how="left", validate="one_to_one")


def _comparability_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if _optional_bool(row.get("admission_evidence_conflict")) is True:
        reasons.append("conflicting_admission_evidence")
    if row.get("identity_status") == "unresolved":
        reasons.append("missing_explicit_admission_evidence")
    elif row.get("identity_status") != "exact":
        reasons.append("nonexact_sequence_or_variant_evidence")
    if row.get("mapping_status") == "unresolved":
        reasons.append("missing_explicit_mapping_evidence")
    elif row.get("mapping_status") != "formally_admitted":
        reasons.append("mapping_or_admission_not_formally_admitted")
    if pd.isna(row.get("common_mask_count_observed")) or int(row["common_mask_count_observed"]) == 0:
        reasons.append("no_evaluated_common_region")
    mapped_fraction = _optional_float(row.get("mapped_fraction_of_canonical_observed"))
    if mapped_fraction is None:
        reasons.append("missing_analyzed_region_coverage")
    elif mapped_fraction < CLEAN_MAPPED_FRACTION_MIN:
        reasons.append("analyzed_region_coverage_below_existing_090_quality_value")
    if _optional_bool(row.get("known_construct_difference")) is True:
        reasons.append("explicit_construct_difference")
    return reasons


def _assign_comparability_group(row: pd.Series) -> str:
    reasons = row["high_comparability_exclusion_reason"]
    if (
        "missing_explicit_admission_evidence" in reasons
        or "missing_explicit_mapping_evidence" in reasons
        or "conflicting_admission_evidence" in reasons
    ):
        return "unresolved"
    if "explicit_construct_difference" in reasons:
        return "sequence_comparable_context_metadata_different"
    if reasons:
        return "potentially_confounded_coverage_or_mapping"
    return "highly_comparable_identity_mapping"


def build_pair_validity_table(inputs: PairValidityInputs) -> pd.DataFrame:
    """Build one row per frozen protein with separate evidence dimensions."""
    _require_columns(inputs.cohort, COHORT_COLUMNS, "cohort")
    _require_columns(inputs.protein_response, RESPONSE_COLUMNS, "protein response")
    if inputs.cohort["protein_id"].duplicated().any():
        raise PairValidityError("duplicate_cohort_identity", "cohort protein_id values are not unique")
    frame = inputs.cohort.copy()
    frame = _coalesce_admission_columns(frame, inputs.admissions)
    frame = frame.merge(_mask_aggregate(inputs.common_masks), on="protein_id", how="left", validate="one_to_one")
    frame = _merge_optional(frame, inputs.pdb_metadata, key="candidate_id", label="PDB metadata")
    frame = _merge_optional(frame, inputs.afdb_confidence, key="candidate_id", label="AFDB confidence")
    frame = frame.merge(
        inputs.protein_response[[column for column in inputs.protein_response.columns if column in {
            "protein_id", "median_abs_d", "mean_abs_d", "q75_abs_d", "q90_abs_d"
        }]],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )

    mismatch = pd.to_numeric(frame.get("mismatch_count"), errors="coerce")
    identity = pd.to_numeric(frame.get("paired_sequence_identity"), errors="coerce")
    frame["identity_status"] = np.select(
        [mismatch.notna() & identity.notna() & (mismatch.eq(0)) & (identity >= 1.0 - 1.0e-12), mismatch.notna() | identity.notna()],
        ["exact", "nonexact_sequence_or_variant"],
        default="unresolved",
    )
    admission = frame.get("admission_status", pd.Series(pd.NA, index=frame.index))
    mapping_valid = frame.get("mapping_valid", pd.Series(pd.NA, index=frame.index)).map(_optional_bool)
    frame["mapping_status"] = np.select(
        [admission.eq(FORMAL_ADMISSION) & (mapping_valid.isna() | mapping_valid.eq(True)), admission.notna()],
        ["formally_admitted", "not_formally_admitted"],
        default="unresolved",
    )
    canonical_length = pd.to_numeric(frame["canonical_sequence_length"], errors="coerce")
    frame["mapped_fraction_of_canonical_observed"] = (
        pd.to_numeric(frame.get("mapped_residue_count"), errors="coerce") / canonical_length
    )
    frame["common_mask_fraction_of_canonical_observed"] = (
        pd.to_numeric(frame["common_mask_count_observed"], errors="coerce") / canonical_length
    )
    frame["known_construct_difference"] = frame.get(
        "known_construct_difference", pd.Series(pd.NA, index=frame.index)
    )
    frame["high_comparability_exclusion_reason"] = frame.apply(_comparability_reasons, axis=1)
    frame["high_comparability_eligible"] = frame["high_comparability_exclusion_reason"].map(lambda value: len(value) == 0)
    frame["comparability_group"] = frame.apply(_assign_comparability_group, axis=1)
    frame["context_comparability_status"] = np.where(
        frame.get("pdb_metadata_status", pd.Series(pd.NA, index=frame.index)).eq("available"),
        "pdb_context_observed_afdb_context_unavailable",
        "unresolved",
    )
    frame["high_comparability_exclusion_reason"] = frame[
        "high_comparability_exclusion_reason"
    ].map(lambda value: ";".join(value) if value else None)
    order = list(inputs.cohort["protein_id"].astype(str))
    frame["_order"] = frame["protein_id"].astype(str).map({value: index for index, value in enumerate(order)})
    return frame.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)


def select_high_comparability(table: pd.DataFrame) -> pd.DataFrame:
    """Return the transparent subset supported by explicit existing evidence."""
    required = {"high_comparability_eligible"}
    _require_columns(table, required, "pair-validity table")
    return table.loc[table["high_comparability_eligible"].eq(True)].copy().reset_index(drop=True)


def build_response_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Return the protein-unit response table with comparability annotations."""
    required = {"protein_id", "comparability_group", "high_comparability_eligible", "median_abs_d"}
    _require_columns(table, required, "pair-validity table")
    columns = [
        "protein_id",
        "candidate_id",
        "comparability_group",
        "high_comparability_eligible",
        "high_comparability_exclusion_reason",
        "median_abs_d",
        "mean_abs_d",
        "q75_abs_d",
        "q90_abs_d",
        "common_mask_count_observed",
        "common_mask_fraction_of_canonical_observed",
        "mapped_fraction_of_canonical_observed",
        "common_mask_fraction_of_mapped",
        "identity_status",
        "mapping_status",
        "context_comparability_status",
    ]
    return table[[column for column in columns if column in table.columns]].copy()


def summarize_pair_validity(table: pd.DataFrame, comparison: pd.DataFrame | None = None) -> dict[str, Any]:
    """Summarize descriptive groups and protein-level remodeling contrasts."""
    if comparison is None:
        comparison = build_response_comparison(table)
    values = pd.to_numeric(comparison["median_abs_d"], errors="coerce")
    clean = values.loc[comparison["high_comparability_eligible"].eq(True)]
    finite_clean = clean.dropna()
    finite_full = values.dropna()
    group_counts = comparison["comparability_group"].value_counts(dropna=False).to_dict()
    identity_counts = table["identity_status"].value_counts(dropna=False).to_dict()
    mapping_counts = table["mapping_status"].value_counts(dropna=False).to_dict()
    summary: dict[str, Any] = {
        "analysis": "PDB/AFDB pair validity and protein-level structural response",
        "protein_count": len(comparison),
        "high_comparability_count": int(comparison["high_comparability_eligible"].eq(True).sum()),
        "high_comparability_criterion": "FORMALLY_ADMITTED + exact identity + mapped_residue_count/canonical_sequence_length >= 0.90; descriptive sensitivity only",
        "group_counts": {str(key): int(value) for key, value in group_counts.items()},
        "identity_status_counts": {str(key): int(value) for key, value in identity_counts.items()},
        "mapping_status_counts": {str(key): int(value) for key, value in mapping_counts.items()},
        "context_metadata_status_counts": {
            str(key): int(value)
            for key, value in table.get(
                "context_comparability_status", pd.Series("unresolved", index=table.index)
            ).value_counts(dropna=False).to_dict().items()
        },
        "afdb_confidence_status_counts": {
            str(key): int(value)
            for key, value in table.get(
                "afdb_confidence_status", pd.Series("unavailable", index=table.index)
            ).value_counts(dropna=False).to_dict().items()
        },
        "admission_evidence_conflict_count": int(
            table.get("admission_evidence_conflict", pd.Series(False, index=table.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "response_metric": "existing structural_response.protein_summary.median_abs_d",
        "response_full_median": float(finite_full.median()) if len(finite_full) else None,
        "response_full_q25": _quantile(finite_full, 0.25),
        "response_full_q75": _quantile(finite_full, 0.75),
        "response_clean_median": float(finite_clean.median()) if len(finite_clean) else None,
        "response_clean_q25": _quantile(finite_clean, 0.25),
        "response_clean_q75": _quantile(finite_clean, 0.75),
        "response_clean_vs_full_median_difference": (
            float(finite_clean.median() - finite_full.median())
            if len(finite_clean) and len(finite_full)
            else None
        ),
        "no_position_or_repeat_as_independent_unit": True,
        "no_weighted_pair_validity_score": True,
    }
    for column in ("common_mask_fraction_of_canonical_observed", "common_mask_fraction_of_mapped"):
        if column in table:
            finite = pd.to_numeric(table[column], errors="coerce").dropna()
            summary[f"{column}_median"] = float(finite.median()) if len(finite) else None
            summary[f"{column}_q25"] = _quantile(finite, 0.25)
            summary[f"{column}_q75"] = _quantile(finite, 0.75)
    for column in (
        "common_mask_fraction_of_canonical_observed",
        "mapped_fraction_of_canonical_observed",
        "afdb_global_plddt_median",
        "afdb_global_pae_q90",
    ):
        if column not in table:
            continue
        pair = table[[column, "median_abs_d"]].apply(pd.to_numeric, errors="coerce").dropna()
        summary[f"response_spearman_vs_{column}"] = (
            float(pair[column].corr(pair["median_abs_d"], method="spearman"))
            if len(pair) >= 3
            else None
        )
    summary["response_median_by_comparability_group"] = {
        str(group): (
            float(values.median())
            if len(values := pd.to_numeric(
                comparison.loc[comparison["comparability_group"].eq(group), "median_abs_d"],
                errors="coerce",
            ).dropna())
            else None
        )
        for group in sorted(comparison["comparability_group"].dropna().unique())
    }
    return summary


def _mmcif_value(values: Any, index: int | None = None) -> str | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if index is None or index >= len(values):
            return None
        return _optional_text(values[index])
    return _optional_text(values)


def _mmcif_join(values: Any) -> str | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        cleaned = [_optional_text(value) for value in values]
        cleaned = [value for value in cleaned if value is not None]
        return " | ".join(cleaned) if cleaned else None
    return _optional_text(values)


def _parse_pdb_metadata(project_root: Path, row: pd.Series) -> dict[str, Any]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    record: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "pdb_metadata_status": "unavailable",
        "context_metadata_status": "unresolved",
        "pdb_structure_ref": row.get("pdb_structure_ref"),
    }
    path = project_root / str(row["pdb_structure_ref"])
    if not path.is_file():
        return record
    try:
        data = MMCIF2Dict(str(path))
    except (OSError, ValueError, KeyError):
        record["pdb_metadata_status"] = "invalid_local_mmcif"
        return record
    record["pdb_metadata_status"] = "available"
    record["context_metadata_status"] = "pdb_context_observed_afdb_context_unavailable"
    chain_ids = data.get("_struct_asym.id", [])
    entity_ids = data.get("_struct_asym.entity_id", [])
    chain = str(row["pdb_chain"])
    chain_index = next((index for index, value in enumerate(chain_ids) if str(value) == chain), None)
    entity_id = _mmcif_value(entity_ids, chain_index)
    record["pdb_entity_id"] = entity_id
    entity_keys = ["_entity_poly.entity_id", "_entity_poly.pdbx_description", "_entity_poly.pdbx_seq_one_letter_code_can"]
    entity_ids_poly = data.get(entity_keys[0], [])
    entity_index = next((index for index, value in enumerate(entity_ids_poly) if str(value) == str(entity_id)), None)
    record["pdb_entity_description"] = _mmcif_value(data.get(entity_keys[1]), entity_index)
    sequence = _mmcif_value(data.get(entity_keys[2]), entity_index)
    record["pdb_entity_polymer_sequence_length"] = len(re.sub(r"\s+", "", sequence)) if sequence else None
    record["pdb_source_description"] = _mmcif_join(data.get("_entity_src_gen.pdbx_description"))
    record["pdb_source_scientific_name"] = _mmcif_join(data.get("_entity_src_gen.pdbx_gene_src_scientific_name"))
    record["pdb_host_scientific_name"] = _mmcif_join(data.get("_entity_src_gen.pdbx_host_org_scientific_name"))
    record["pdb_source_variant"] = _mmcif_join(data.get("_entity_src_gen.pdbx_gene_src_variant"))
    record["pdb_source_fragment"] = _mmcif_join(data.get("_entity_src_gen.pdbx_gene_src_fragment"))
    record["known_construct_difference"] = bool(
        record["pdb_source_variant"] or record["pdb_source_fragment"]
    )
    record["pdb_assembly_details"] = _mmcif_join(data.get("_pdbx_struct_assembly.details"))
    record["pdb_oligomeric_details"] = _mmcif_join(data.get("_pdbx_struct_assembly.oligomeric_details"))
    nonpoly_ids = data.get("_pdbx_entity_nonpoly.entity_id")
    if isinstance(nonpoly_ids, (list, tuple)):
        record["pdb_nonpoly_entity_count"] = len(nonpoly_ids)
        record["pdb_nonpoly_entity_ids"] = " | ".join(str(value) for value in nonpoly_ids)
    else:
        record["pdb_nonpoly_entity_count"] = None
        record["pdb_nonpoly_entity_ids"] = None
    return record


def _find_confidence_file(root: Path, accession: str, model_id: str, filename: str) -> Path | None:
    candidates = [
        root / "data/raw/afdb" / accession / model_id / filename,
        root / "data/raw/afdb" / accession / filename,
    ]
    if filename == "plddt.json":
        candidates.extend(
            [
                root / "data/raw/afdb" / accession / model_id / f"{model_id}-confidence_v6.json",
                root / "data/raw/afdb" / accession / f"{model_id}-confidence_v6.json",
            ]
        )
    return next((path for path in candidates if path.is_file()), None)


def _parse_afdb_confidence(project_root: Path, row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "afdb_confidence_status": "unavailable",
        "afdb_global_plddt_median": None,
        "afdb_global_plddt_q10": None,
        "afdb_global_plddt_q90": None,
        "afdb_global_pae_median": None,
        "afdb_global_pae_q90": None,
        "afdb_analyzed_region_confidence_status": "unavailable_without_existing_afdb_position_projection",
    }
    accession = str(row["canonical_accession"])
    model_id = str(row["afdb_model_id"])
    plddt_path = _find_confidence_file(project_root, accession, model_id, "plddt.json")
    pae_path = _find_confidence_file(project_root, accession, model_id, "pae.json")
    if plddt_path is not None:
        try:
            values = load_plddt(plddt_path)
            result["afdb_global_plddt_median"] = float(np.median(values))
            result["afdb_global_plddt_q10"] = float(np.quantile(values, 0.10, method="linear"))
            result["afdb_global_plddt_q90"] = float(np.quantile(values, 0.90, method="linear"))
            result["afdb_plddt_source_relative"] = plddt_path.relative_to(project_root).as_posix()
            result["afdb_confidence_status"] = "global_descriptors_available"
        except (OSError, ValueError, json.JSONDecodeError):
            result["afdb_confidence_status"] = "invalid_local_plddt"
    if pae_path is not None:
        try:
            matrix = load_pae(pae_path)
            values = matrix[np.isfinite(matrix)]
            if len(values):
                result["afdb_global_pae_median"] = float(np.median(values))
                result["afdb_global_pae_q90"] = float(np.quantile(values, 0.90, method="linear"))
                result["afdb_pae_source_relative"] = pae_path.relative_to(project_root).as_posix()
                if result["afdb_confidence_status"] == "unavailable":
                    result["afdb_confidence_status"] = "global_descriptors_available"
        except (OSError, ValueError, json.JSONDecodeError):
            result["afdb_pae_status"] = "invalid_local_pae"
    return result


def load_pair_validity_inputs(project_root: Path) -> PairValidityInputs:
    """Load the frozen cohort and existing pair/response evidence."""
    root = project_root.expanduser().resolve()
    base = root / "experiments/p2_design_baseline"
    paths = {
        "cohort": base / "scale1/scale1b_v2/scale1b_v2_primary_cohort.parquet",
        "protein_response": base / "scale1b-v2/structural_response/protein_summary.parquet",
    }
    _require_files(paths)
    cohort = pd.read_parquet(paths["cohort"])
    mask_sources = {
        "ORIGINAL_213_FRAME": base / "scale1/scale1a2/scale1_full_frame_common_masks.parquet",
        "EXPANSION_WAVE1": base / "scale1/expansion_wave1/scale1_expansion_wave1_common_masks.parquet",
        "EXPANSION_WAVE2": base / "scale1/expansion_wave2/scale1_expansion_wave2_common_masks.parquet",
    }
    common_masks = _load_stratum_masks(cohort, mask_sources)
    response = pd.read_parquet(paths["protein_response"])
    admission_paths = [
        base / "scale1/scale1a1/scale1_formal_admission_census.parquet",
        base / "scale1/expansion_wave1/scale1_expansion_wave1_admission.parquet",
        base / "scale1/expansion_wave2/scale1_expansion_wave2_admission.parquet",
    ]
    frames: list[pd.DataFrame] = []
    for priority, path in enumerate(admission_paths):
        if path.is_file():
            frame = pd.read_parquet(path).copy()
            frame["_source_priority"] = priority
            frames.append(frame)
    admissions = _merge_admission_sources(frames, set(cohort["candidate_id"]))
    pdb_metadata = pd.DataFrame([_parse_pdb_metadata(root, row) for _, row in cohort.iterrows()])
    afdb_confidence = pd.DataFrame([_parse_afdb_confidence(root, row) for _, row in cohort.iterrows()])
    provenance = {
        key: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "rows": len(pd.read_parquet(path))}
        for key, path in paths.items()
    }
    provenance["common_masks"] = {
        "sources": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "rows": len(pd.read_parquet(path)),
            }
            for path in mask_sources.values()
            if path.is_file()
        ],
        "selection": "source_stratum-specific existing complete-mask artifact",
        "rows": len(common_masks),
    }
    provenance["admission_sources"] = {
        "sources": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "rows": len(pd.read_parquet(path)),
            }
            for path in admission_paths
            if path.is_file()
        ],
        "rows": len(admissions),
    }
    return PairValidityInputs(
        project_root=root,
        cohort=cohort,
        common_masks=common_masks,
        admissions=admissions,
        protein_response=response,
        pdb_metadata=pdb_metadata,
        afdb_confidence=afdb_confidence,
        input_provenance=provenance,
    )


def _require_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PairValidityError("input_missing", f"required pair-validity inputs are missing: {missing}")


def _load_stratum_masks(
    cohort: pd.DataFrame, sources: dict[str, Path]
) -> pd.DataFrame:
    """Select the existing complete-mask artifact matching each frozen stratum."""
    loaded: dict[str, pd.DataFrame] = {}
    for stratum, path in sources.items():
        if path.is_file():
            frame = pd.read_parquet(path).copy()
            frame["_mask_source_relative"] = path.as_posix()
            loaded[stratum] = frame
    if not loaded:
        raise PairValidityError("input_missing", "no complete common-mask source is available")
    selected: list[pd.DataFrame] = []
    for stratum, group in cohort.groupby("source_stratum", sort=False):
        source = loaded.get(stratum)
        if source is None:
            raise PairValidityError("mask_source_missing", f"no mask source for cohort stratum: {stratum}")
        ids = set(group["candidate_id"].astype(str))
        subset = source.loc[source["candidate_id"].astype(str).isin(ids)].copy()
        missing = ids - set(subset["candidate_id"].astype(str))
        if missing:
            raise PairValidityError(
                "mask_source_incomplete",
                f"mask source for {stratum} is missing candidates: {sorted(missing)}",
            )
        selected.append(subset)
    result = pd.concat(selected, ignore_index=True, sort=False)
    expected = set(cohort["candidate_id"].astype(str))
    observed = set(result["candidate_id"].astype(str))
    if expected != observed:
        raise PairValidityError("mask_cohort_mismatch", "selected masks do not cover the frozen cohort")
    identity = cohort[["candidate_id", "protein_id"]].copy()
    result = result.merge(identity, on="candidate_id", how="left", validate="many_to_one")
    return result


def _merge_admission_sources(frames: list[pd.DataFrame], candidate_ids: set[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["candidate_id"])
    prepared: list[pd.DataFrame] = []
    for priority, frame in enumerate(frames):
        current = frame.copy()
        if "_source_priority" not in current.columns:
            current["_source_priority"] = priority
        prepared.append(current)
    combined = pd.concat(prepared, ignore_index=True, sort=False)
    combined = combined.loc[combined["candidate_id"].isin(candidate_ids)].copy()
    compare_columns = [
        "admission_status",
        "paired_sequence_identity",
        "mismatch_count",
        "mapped_residue_count",
        "coordinate_nonobservable_count",
        "common_mask_count",
        "common_mask_fraction_of_mapped",
    ]
    conflict_fields: dict[str, list[str]] = {}
    for candidate_id, group in combined.groupby("candidate_id", sort=False):
        for column in compare_columns:
            if column not in group:
                continue
            values = group[column].dropna().astype(str).unique()
            if len(values) > 1:
                conflict_fields.setdefault(str(candidate_id), []).append(column)
    combined = combined.sort_values(["candidate_id", "_source_priority"], kind="mergesort")
    selected = combined.drop_duplicates("candidate_id", keep="first").copy()
    selected["admission_evidence_conflict"] = selected["candidate_id"].astype(str).map(
        lambda value: value in conflict_fields
    )
    selected["admission_conflict_fields"] = selected["candidate_id"].astype(str).map(
        lambda value: ";".join(conflict_fields.get(value, [])) or None
    )
    for candidate_id, fields in conflict_fields.items():
        selected.loc[selected["candidate_id"].astype(str).eq(candidate_id), fields] = pd.NA
    return selected.drop(columns="_source_priority")


def _figure_bytes(comparison: pd.DataFrame, name: str) -> bytes:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    if name == "comparability_overview":
        counts = comparison["comparability_group"].value_counts().sort_values()
        axis.barh(counts.index, counts.values, color="#355c7d")
        axis.set_xlabel("Proteins")
        axis.set_title("Descriptive pair-comparability groups")
    elif name == "remodeling_clean_subset":
        groups = []
        labels = []
        full = pd.to_numeric(comparison["median_abs_d"], errors="coerce").dropna()
        clean = pd.to_numeric(
            comparison.loc[comparison["high_comparability_eligible"].eq(True), "median_abs_d"],
            errors="coerce",
        ).dropna()
        if len(full):
            groups.append(full.to_numpy())
            labels.append("All")
        if len(clean):
            groups.append(clean.to_numpy())
            labels.append("Explicit clean subset")
        if groups:
            axis.boxplot(groups, tick_labels=labels, showfliers=False)
        axis.set_ylabel("Existing protein median |D|")
        axis.set_title("Protein-level remodeling characterization")
    elif name == "remodeling_vs_coverage":
        work = comparison.dropna(subset=["common_mask_fraction_of_canonical_observed", "median_abs_d"])
        axis.scatter(
            work["common_mask_fraction_of_canonical_observed"],
            work["median_abs_d"],
            s=24,
            alpha=0.8,
            color="#c06c84",
        )
        axis.set_xlabel("Observed common-mask fraction of canonical length")
        axis.set_ylabel("Existing protein median |D|")
        axis.set_title("Remodeling versus analyzed-region coverage")
    else:
        plt.close(figure)
        raise ValueError(f"unknown pair-validity figure: {name}")
    output = io.BytesIO()
    figure.savefig(output, format="png", metadata={"Software": "Dual-UQ"}, dpi=140)
    plt.close(figure)
    return output.getvalue()


def _report_markdown(summary: dict[str, Any]) -> str:
    groups = summary.get("group_counts", {})
    lines = [
        "# PDB/AFDB Pair Validity Analysis",
        "",
        "Descriptive pair comparability only; no new uncertainty metric, P/M/SDFI, ranking, regret, mechanism, or H1 analysis.",
        "",
        "## Pair validity",
        "",
        f"- Proteins: {summary['protein_count']}",
        f"- Explicit clean identity/mapping subset: {summary['high_comparability_count']}",
        f"- Groups: {json.dumps(groups, sort_keys=True)}",
        "- No weighted pair-validity score was constructed.",
        "- The explicit clean subset uses formal exact admission plus mapped_residue_count/canonical_sequence_length >= 0.90, reusing the existing quality value as a descriptive sensitivity criterion rather than changing admission.",
        "- Missing metadata remains unresolved; AFDB confidence values are global descriptors when present, not analyzed-region ground truth.",
        "",
        "## Relation to remodeling",
        "",
        f"- Existing protein-level median |D|, full cohort: {summary['response_full_median']}",
        f"- Existing protein-level median |D|, explicit clean subset: {summary['response_clean_median']}",
        f"- Clean-minus-full median difference: {summary['response_clean_vs_full_median_difference']}",
        "",
        "## Interpretation",
        "",
        "The comparison uses the frozen cohort, existing common masks, admission evidence, and the existing structural-response protein summary. It does not establish biological structural uncertainty; it describes paired structural representation variation and experimental–predicted structural discrepancy.",
        "",
        "## Next scientific gate",
        "",
        "If pair validity is judged adequate, the next separately authorized analysis is LOCAL_INVERSE_FOLDING_REMODELING_ANALYSIS.",
    ]
    return "\n".join(lines) + "\n"


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    try:
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise PairValidityError("immutable_output_conflict", f"output differs: {path}")
            return "reused_identical"
        temporary.replace(path)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_bytes(data: bytes, path: Path) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise PairValidityError("immutable_output_conflict", f"output differs: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, data)
    return "created"


def materialize_pair_validity(result: PairValidityResult, output_dir: Path) -> dict[str, Any]:
    """Write pair table, response comparison, summary, report, and diagnostics."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for label, frame, filename in (
        ("pair_validity", result.pair_table, "pair_validity.parquet"),
        ("response_comparison", result.response_comparison, "protein_response_comparison.parquet"),
    ):
        path = output_dir / filename
        statuses[label] = _immutable_parquet(frame, path)
        outputs[label] = {"path": filename, "rows": len(frame), "sha256": sha256_file(path)}
    summary_bytes = (
        json.dumps(
            {
                "status": "COMPLETE",
                "analysis": "pdb_afdb_pair_validity_v1",
                "summary": _jsonable(result.summary),
                "input_provenance": _jsonable(result.input_provenance),
                "outputs": outputs,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    statuses["summary"] = _immutable_bytes(summary_bytes, output_dir / "summary.json")
    statuses["report"] = _immutable_bytes(_report_markdown(result.summary).encode(), output_dir / "report.md")
    figure_dir = output_dir / "figures"
    for name in ("comparability_overview", "remodeling_clean_subset", "remodeling_vs_coverage"):
        path = figure_dir / f"{name}.png"
        statuses[name] = _immutable_bytes(_figure_bytes(result.response_comparison, name), path)
        outputs[name] = {"path": f"figures/{path.name}", "sha256": sha256_file(path)}
    return {"outputs": outputs, "write_status": statuses}


def analyze_pair_validity(project_root: Path) -> PairValidityResult:
    """Load frozen inputs and compute the single canonical structured result."""
    inputs = load_pair_validity_inputs(project_root)
    pair_table = build_pair_validity_table(inputs)
    comparison = build_response_comparison(pair_table)
    summary = summarize_pair_validity(pair_table, comparison)
    return PairValidityResult(
        pair_table=pair_table,
        response_comparison=comparison,
        summary=summary,
        input_provenance=inputs.input_provenance,
    )
