"""Pair-QC adapters, residue provenance, true gaps and sequence discrepancies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.dataset_a_scale.pae import AFDBFragment
from dual_uq.preflight import classify_preflight, compute_preflight_metrics
from dual_uq.structure_io import (
    join_residue_mapping_to_ca,
    load_chain_ca_table,
    residue_name_to_one_letter,
)

from .models import DerivationError


def prepare_confidence_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Remove only downstream-derived fields before the frozen confidence join."""
    return mapping.drop(
        columns=["observed_ca", "mapped", "fragment_covered", "plddt"],
        errors="ignore",
    ).copy()


def annotate_gap_semantics(mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach D3 segment/gap semantics using true UniProt coordinates."""
    source_column = (
        "uniprot_position"
        if "uniprot_position" in mapping.columns
        else "uniprot_residue_number"
    )
    if source_column not in mapping.columns or mapping.empty:
        raise DerivationError("invalid_mapping", "Mapping requires UniProt positions")
    table = mapping.copy()
    positions = pd.to_numeric(table[source_column], errors="coerce")
    if positions.isna().any() or positions.mod(1).ne(0).any() or positions.le(0).any():
        raise DerivationError(
            "invalid_mapping", "UniProt positions must be positive integers"
        )
    table["uniprot_position"] = positions.astype(int)
    if table["uniprot_position"].duplicated().any():
        raise DerivationError("ambiguous_mapping", "Duplicate UniProt positions")
    table = table.sort_values("uniprot_position", kind="mergesort").reset_index(drop=True)
    table["output_position"] = np.arange(1, len(table) + 1)
    differences = table["uniprot_position"].diff()
    starts = differences.isna() | differences.gt(1)
    table["segment_id"] = starts.cumsum().astype(int)
    gap_before = differences.sub(1).where(differences.gt(1), 0).astype("Int64")
    gap_before.iloc[0] = pd.NA
    next_diff = table["uniprot_position"].shift(-1).sub(table["uniprot_position"])
    gap_after = next_diff.sub(1).where(next_diff.gt(1), 0).astype("Int64")
    gap_after.iloc[-1] = pd.NA
    table["gap_before"] = gap_before
    table["gap_after"] = gap_after
    boundaries: list[dict[str, int]] = []
    for row_index in range(len(table) - 1):
        left = int(table.loc[row_index, "uniprot_position"])
        right = int(table.loc[row_index + 1, "uniprot_position"])
        if right - left > 1:
            boundaries.append(
                {
                    "left_uniprot_position": left,
                    "right_uniprot_position": right,
                    "left_output_position": int(table.loc[row_index, "output_position"]),
                    "right_output_position": int(
                        table.loc[row_index + 1, "output_position"]
                    ),
                    "missing_uniprot_start": left + 1,
                    "missing_uniprot_end": right - 1,
                    "gap_length": right - left - 1,
                }
            )
    table["nearest_gap_distance"] = pd.Series([None] * len(table), dtype="Int64")
    if boundaries:
        flanks_by_segment: dict[int, list[int]] = {}
        for boundary in boundaries:
            left_output = boundary["left_output_position"]
            right_output = boundary["right_output_position"]
            left_segment = int(table.loc[left_output - 1, "segment_id"])
            right_segment = int(table.loc[right_output - 1, "segment_id"])
            flanks_by_segment.setdefault(left_segment, []).append(
                boundary["left_uniprot_position"]
            )
            flanks_by_segment.setdefault(right_segment, []).append(
                boundary["right_uniprot_position"]
            )
        table["nearest_gap_distance"] = pd.Series(
            [
                min(
                    abs(int(row.uniprot_position) - flank)
                    for flank in flanks_by_segment[int(row.segment_id)]
                )
                if int(row.segment_id) in flanks_by_segment
                else pd.NA
                for row in table.itertuples(index=False)
            ],
            dtype="Int64",
        )
    table["segment_count"] = int(table["segment_id"].max())
    summary = {
        "gap_count": len(boundaries),
        "segment_count": int(table["segment_id"].max()),
        "largest_uniprot_gap": max(
            (boundary["gap_length"] for boundary in boundaries), default=0
        ),
        "longest_gap": max(
            (boundary["gap_length"] for boundary in boundaries), default=0
        ),
        "gap_boundaries": boundaries,
        "nearest_gap_metadata_available": True,
    }
    return table, summary


def are_peptide_adjacent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return true only for true UniProt and segment adjacency."""
    return bool(
        int(right["uniprot_position"]) - int(left["uniprot_position"]) == 1
        and int(right["segment_id"]) == int(left["segment_id"])
    )


def audit_sequence_discrepancies(
    mapping: pd.DataFrame,
    pdb_sequence: Mapping[int, str],
    afdb_sequence: Mapping[int, str],
) -> dict[str, list[dict[str, Any]]]:
    """Keep mapping/PDB/AFDB mismatch sets explicit and independent."""
    result = {"mapping_vs_pdb": [], "mapping_vs_afdb": [], "pdb_vs_afdb": []}
    for row in mapping.sort_values("uniprot_position", kind="mergesort").to_dict(
        "records"
    ):
        position = int(row["uniprot_position"])
        mapping_aa = str(row["mapping_aa"])
        pdb_aa = pdb_sequence.get(position)
        afdb_aa = afdb_sequence.get(position)
        common = {
            "uniprot_position": position,
            "mapping_aa": mapping_aa,
            "pdb_aa": pdb_aa,
            "afdb_aa": afdb_aa,
        }
        if pdb_aa is not None and mapping_aa != pdb_aa:
            result["mapping_vs_pdb"].append(dict(common))
        if afdb_aa is not None and mapping_aa != afdb_aa:
            result["mapping_vs_afdb"].append(dict(common))
        if pdb_aa is not None and afdb_aa is not None and pdb_aa != afdb_aa:
            result["pdb_vs_afdb"].append(dict(common))
    return result


def compute_pair_quality(
    mapping: pd.DataFrame,
    pdb_ca: pd.DataFrame,
    *,
    canonical_length: int,
    pdb_entity_length: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Delegate all pair-QC metrics and classification to the frozen preflight API."""
    metrics = compute_preflight_metrics(
        mapping,
        pdb_ca,
        uniprot_length=canonical_length,
        pdb_entity_length=pdb_entity_length,
    )
    preflight_status, preflight_reason = classify_preflight(metrics, dict(thresholds))
    return {
        **metrics,
        "pair_qc_status": (
            "pair_qc_pass" if preflight_status == "pass_full_length" else "pair_qc_fail"
        ),
        "preflight_status": preflight_status,
        "preflight_reason": preflight_reason,
        "mapping_coverage": metrics["full_length_mapping_coverage"],
        "mapped_interval": [
            metrics["first_mapped_uniprot_position"],
            metrics["last_mapped_uniprot_position"],
        ],
    }


def pair_qc_attrition(status: str) -> dict[str, Any]:
    """Separate a calculated QC predicate from causal stage availability."""
    if status == "pair_qc_pass":
        return {
            "pair_qc_eligible": True,
            "scientific_attrition_flags": [],
            "scientific_attrition_classes": [],
        }
    if status == "pair_qc_fail":
        return {
            "pair_qc_eligible": False,
            "scientific_attrition_flags": ["pair_qc_threshold_not_met"],
            "scientific_attrition_classes": ["protocol_eligibility_issue"],
        }
    return {
        "pair_qc_eligible": None,
        "scientific_attrition_flags": ["pair_qc_unobservable"],
        "scientific_attrition_classes": ["mapping_issue"],
    }


def mapping_with_provenance(
    mapping: pd.DataFrame, pdb_ca: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    auth_keys = ["auth_asym_id", "auth_seq_id", "insertion_code"]
    namespaces = pdb_ca[
        auth_keys + ["label_asym_id", "label_seq_id"]
    ].drop_duplicates(auth_keys)
    table = mapping.drop(
        columns=["label_asym_id", "label_seq_id"], errors="ignore"
    ).merge(namespaces, on=auth_keys, how="left", validate="one_to_one")
    table, gaps = annotate_gap_semantics(table)
    observed, join_diagnostics = join_residue_mapping_to_ca(table, pdb_ca)
    observed_keys = {
        tuple(row) for row in observed[auth_keys].itertuples(index=False, name=None)
    }
    table["observed_ca"] = [
        tuple(row) in observed_keys
        for row in table[auth_keys].itertuples(index=False, name=None)
    ]
    return table, gaps, join_diagnostics


def _nullable_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def residue_mapping_audit_records(mapping: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize explicit D3 and auth/label provenance for audit."""
    records = []
    for row in mapping.itertuples(index=False):
        records.append(
            {
                "uniprot_position": int(row.uniprot_position),
                "output_position": int(row.output_position),
                "segment_id": int(row.segment_id),
                "segment_count": int(row.segment_count),
                "gap_before": _nullable_int(row.gap_before),
                "gap_after": _nullable_int(row.gap_after),
                "nearest_gap_distance": _nullable_int(row.nearest_gap_distance),
                "auth_asym_id": (
                    None if pd.isna(row.auth_asym_id) else str(row.auth_asym_id)
                ),
                "auth_seq_id": _nullable_int(row.auth_seq_id),
                "insertion_code": str(row.insertion_code),
                "label_asym_id": (
                    None if pd.isna(row.label_asym_id) else str(row.label_asym_id)
                ),
                "label_seq_id": _nullable_int(row.label_seq_id),
                "mapping_aa": residue_name_to_one_letter(row.uniprot_residue_name),
                "pdb_aa": residue_name_to_one_letter(row.pdb_residue_name),
                "observed_ca": bool(row.observed_ca),
            }
        )
    return records


def afdb_ca_by_uniprot(
    model_path: Path, fragment: AFDBFragment
) -> tuple[dict[int, str], pd.DataFrame]:
    ca = load_chain_ca_table(model_path)
    local = pd.to_numeric(ca["auth_seq_id"], errors="coerce")
    if local.isna().any() or local.mod(1).ne(0).any():
        raise DerivationError(
            "invalid_afdb_numbering", "AFDB model positions are invalid"
        )
    ca = ca.copy()
    ca["model_residue_position"] = local.astype(int)
    expected = set(range(1, fragment.model_residue_count + 1))
    if set(ca["model_residue_position"]) != expected:
        raise DerivationError(
            "afdb_model_position_mismatch",
            "AFDB model CA positions are not exactly 1..N",
        )
    ca["uniprot_position"] = (
        ca["model_residue_position"] + fragment.uniprot_start - 1
    )
    sequence = dict(
        zip(
            ca["uniprot_position"].astype(int),
            ca["residue_one_letter"].astype(str),
            strict=True,
        )
    )
    return sequence, ca
