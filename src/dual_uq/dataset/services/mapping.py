"""Pair-QC adapters, residue provenance, true gaps and sequence discrepancies."""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.pairing import normalise_residue_name
from dual_uq.schema import (
    RESIDUE_MAPPING_SCHEMA_VERSION,
    AmbiguousLegacyResidueIdentifier,
)
from dual_uq.sifts import parse_sifts_residue_mapping
from dual_uq.structure_io import (
    join_residue_mapping_to_ca,
    load_chain_ca_table,
    residue_name_to_one_letter,
)

from ..models import AFDBFragment, DerivationError

_MISSING_SOURCE_IDENTIFIERS = {None, "", ".", "?", "null"}


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _matching_sifts_records(
    xml_gz_path: str | Path,
    *,
    chain_id: str,
    uniprot_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(xml_gz_path, "rb") as handle:
        for _, residue in ET.iterparse(handle, events=("end",)):
            if _local_xml_name(residue.tag) != "residue":
                continue
            pdb_ref = None
            uniprot_ref = None
            annotations: list[str] = []
            for child in residue:
                child_name = _local_xml_name(child.tag)
                if child_name == "crossRefDb":
                    if child.attrib.get("dbSource") == "PDB" and pdb_ref is None:
                        pdb_ref = dict(child.attrib)
                    elif (
                        child.attrib.get("dbSource") == "UniProt"
                        and uniprot_ref is None
                    ):
                        uniprot_ref = dict(child.attrib)
                elif child_name == "residueDetail":
                    text = (child.text or "").strip()
                    if text:
                        annotations.append(text)
            if pdb_ref is None or uniprot_ref is None:
                residue.clear()
                continue
            if (
                pdb_ref.get("dbChainId") == chain_id
                and str(uniprot_ref.get("dbAccessionId", "")).upper()
                == uniprot_id
            ):
                records.append(
                    {
                        "pdb_chain_id": pdb_ref.get("dbChainId"),
                        "pdb_residue_number": pdb_ref.get("dbResNum"),
                        "pdb_residue_name": pdb_ref.get("dbResName"),
                        "uniprot_id": str(
                            uniprot_ref.get("dbAccessionId", "")
                        ).upper(),
                        "uniprot_residue_number": uniprot_ref.get("dbResNum"),
                        "uniprot_residue_name": uniprot_ref.get("dbResName"),
                        "sifts_pdbe_residue_number": residue.attrib.get("dbResNum"),
                        "sifts_pdbe_residue_name": residue.attrib.get("dbResName"),
                        "sifts_pdbe_source": residue.attrib.get("dbSource"),
                        "sifts_pdbe_coordinate_system": residue.attrib.get(
                            "dbCoordSys"
                        ),
                        "sifts_annotations": tuple(annotations),
                    }
                )
            residue.clear()
    return records


def _as_cif_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _clean_source_identifier(value: Any) -> str | None:
    cleaned = None if value is None else str(value).strip()
    return None if cleaned in _MISSING_SOURCE_IDENTIFIERS else cleaned


def _poly_seq_scheme(cif_path: str | Path) -> list[dict[str, str | None]]:
    data = MMCIF2Dict(str(cif_path))
    fields = (
        "asym_id",
        "entity_id",
        "seq_id",
        "mon_id",
        "auth_mon_id",
        "auth_seq_num",
        "pdb_strand_id",
        "pdb_ins_code",
    )
    columns = {
        field: _as_cif_list(data.get(f"_pdbx_poly_seq_scheme.{field}", []))
        for field in fields
    }
    sizes = {len(values) for values in columns.values()}
    if sizes != {next(iter(sizes), 0)} or sizes == {0}:
        raise AmbiguousLegacyResidueIdentifier(
            "mmCIF lacks a complete explicit mmCIF label relation."
        )
    return [
        {field: _clean_source_identifier(columns[field][index]) for field in fields}
        for index in range(next(iter(sizes)))
    ]


def _parse_author_identifier(value: Any) -> tuple[int, str] | None:
    cleaned = _clean_source_identifier(value)
    if cleaned is None:
        return None
    match = re.fullmatch(r"([+-]?\d+)([A-Za-z]?)", cleaned)
    if match is None:
        raise AmbiguousLegacyResidueIdentifier(
            f"SIFTS PDB residue identifier {cleaned!r} is ambiguous."
        )
    return int(match.group(1)), match.group(2).upper()


def parse_sifts_mapping_with_explicit_labels(
    xml_gz_path: str | Path,
    cif_path: str | Path,
    *,
    chain_id: str,
    uniprot_id: str,
) -> pd.DataFrame:
    """Preserve an explicit PDBe/mmCIF label identity when author ID is absent.

    The established author parser remains the normal path. The richer path is
    entered only when that parser cannot represent a missing author residue ID;
    it requires one exact `_pdbx_poly_seq_scheme` row per SIFTS PDBe residue and
    never converts a label position into an author position.
    """
    chain_id = chain_id.strip()
    uniprot_id = uniprot_id.strip().upper()
    try:
        return parse_sifts_residue_mapping(
            xml_gz_path, chain_id=chain_id, uniprot_id=uniprot_id
        )
    except AmbiguousLegacyResidueIdentifier:
        records = _matching_sifts_records(
            xml_gz_path, chain_id=chain_id, uniprot_id=uniprot_id
        )
        if not records or all(
            _clean_source_identifier(record["pdb_residue_number"]) is not None
            for record in records
        ):
            raise

    scheme = _poly_seq_scheme(cif_path)
    resolved: list[dict[str, Any]] = []
    for record in records:
        if (
            record["sifts_pdbe_source"] != "PDBe"
            or record["sifts_pdbe_coordinate_system"] != "PDBe"
        ):
            raise AmbiguousLegacyResidueIdentifier(
                "SIFTS residue lacks an explicit PDBe coordinate identity."
            )
        label_position = _clean_source_identifier(
            record["sifts_pdbe_residue_number"]
        )
        label_residue = _clean_source_identifier(record["sifts_pdbe_residue_name"])
        candidates = [
            row
            for row in scheme
            if row["pdb_strand_id"] == chain_id
            and row["seq_id"] == label_position
            and row["mon_id"] == label_residue
        ]
        if len(candidates) != 1:
            raise AmbiguousLegacyResidueIdentifier(
                "SIFTS PDBe residue does not have exactly one explicit mmCIF "
                f"label relation: chain={chain_id!r}, label_seq_id={label_position!r}."
            )
        relation = candidates[0]
        author = _parse_author_identifier(record["pdb_residue_number"])
        relation_author = _clean_source_identifier(relation["auth_seq_num"])
        relation_insertion = _clean_source_identifier(relation["pdb_ins_code"])
        if author is None:
            if relation_author is not None:
                raise AmbiguousLegacyResidueIdentifier(
                    "SIFTS missing author ID conflicts with explicit mmCIF author ID."
                )
            auth_seq_id: int | pd.NA = pd.NA
            insertion_code = "" if relation_insertion is None else relation_insertion
        else:
            auth_seq_id, insertion_code = author
            if relation_author != str(auth_seq_id) or (
                relation_insertion or ""
            ).upper() != insertion_code:
                raise AmbiguousLegacyResidueIdentifier(
                    "SIFTS author ID conflicts with explicit mmCIF residue relation."
                )
        resolved.append(
            {
                **record,
                "entity_id": relation["entity_id"],
                "auth_asym_id": chain_id,
                "label_asym_id": relation["asym_id"],
                "auth_seq_id": auth_seq_id,
                "label_seq_id": int(str(relation["seq_id"])),
                "insertion_code": insertion_code,
                "residue_mapping_schema_version": RESIDUE_MAPPING_SCHEMA_VERSION,
                "residue_mapping_provenance": (
                    "sifts_pdbe_mmcif_explicit_label"
                    if author is None
                    else "sifts_auth_with_explicit_label"
                ),
            }
        )
    mapping = pd.DataFrame(resolved)
    mapping["uniprot_residue_number"] = pd.to_numeric(
        mapping["uniprot_residue_number"], errors="coerce"
    ).astype("Int64")
    mapping["auth_seq_id"] = pd.to_numeric(
        mapping["auth_seq_id"], errors="coerce"
    ).astype("Int64")
    mapping["label_seq_id"] = pd.to_numeric(
        mapping["label_seq_id"], errors="coerce"
    ).astype("Int64")
    if (
        mapping["uniprot_residue_number"].isna().any()
        or mapping["uniprot_residue_number"].duplicated().any()
        or mapping[["label_asym_id", "label_seq_id"]].duplicated().any()
    ):
        raise AmbiguousLegacyResidueIdentifier(
            "Explicit SIFTS/mmCIF mapping contains duplicate or invalid residue keys."
        )
    return mapping.sort_values("uniprot_residue_number", kind="mergesort").reset_index(
        drop=True
    )


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


def compute_mapping_quality_metrics(
    mapping: pd.DataFrame,
    pdb_ca_table: pd.DataFrame,
    *,
    uniprot_length: int,
    pdb_entity_length: int,
) -> dict[str, Any]:
    """Measure sequence, mapping, and observed-coordinate support."""

    mapped_positions = np.sort(
        mapping["uniprot_residue_number"].dropna().astype(int).unique()
    )
    mapped_count = len(mapped_positions)
    if mapped_count == 0:
        raise ValueError("No mapped UniProt residue positions.")

    pdb_letters = mapping["pdb_residue_name"].map(normalise_residue_name)
    uniprot_letters = mapping["uniprot_residue_name"].map(normalise_residue_name)
    comparable = pdb_letters.notna() & uniprot_letters.notna()
    sequence_identity = (
        float((pdb_letters[comparable].values == uniprot_letters[comparable].values).mean())
        if comparable.any()
        else float("nan")
    )
    observed, join_diagnostics = join_residue_mapping_to_ca(mapping, pdb_ca_table)
    observed_positions = observed["uniprot_residue_number"].dropna().astype(int).unique()
    first_position = int(mapped_positions.min())
    last_position = int(mapped_positions.max())
    span_length = last_position - first_position + 1
    internal_unmapped_count = span_length - mapped_count
    return {
        "mapped_residue_count": mapped_count,
        "uniprot_length": int(uniprot_length),
        "pdb_entity_length": int(pdb_entity_length),
        "full_length_mapping_coverage": float(mapped_count / int(uniprot_length)),
        "entity_mapping_coverage": float(mapped_count / int(pdb_entity_length)),
        "sequence_identity": sequence_identity,
        "observed_ca_count": len(observed_positions),
        "observed_ca_fraction_of_mapped": float(len(observed_positions) / mapped_count),
        "first_mapped_uniprot_position": first_position,
        "last_mapped_uniprot_position": last_position,
        "n_terminal_unmapped_count": int(first_position - 1),
        "c_terminal_unmapped_count": int(uniprot_length - last_position),
        "internal_unmapped_count": int(internal_unmapped_count),
        "internal_unmapped_fraction": float(internal_unmapped_count / max(span_length, 1)),
        "pdb_to_uniprot_length_ratio": float(pdb_entity_length / uniprot_length),
        **join_diagnostics,
    }


def classify_mapping_quality(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> tuple[str, str]:
    """Classify a mapping from its explicit quality metrics."""

    identity_ok = metrics["sequence_identity"] >= thresholds["min_sequence_identity"]
    entity_ok = (
        metrics["entity_mapping_coverage"]
        >= thresholds["min_entity_mapping_coverage"]
    )
    observed_ok = (
        metrics["observed_ca_fraction_of_mapped"]
        >= thresholds["min_observed_ca_fraction"]
    )
    internal_ok = (
        metrics["internal_unmapped_fraction"]
        <= thresholds["max_internal_unmapped_fraction"]
    )
    full_coverage = metrics["full_length_mapping_coverage"]
    if (
        full_coverage >= thresholds["min_full_length_mapping_coverage"]
        and identity_ok
        and entity_ok
        and observed_ok
        and internal_ok
    ):
        return "pass_full_length", "Suitable for the main A0 screening pipeline."
    if (
        full_coverage >= thresholds["warn_full_length_mapping_coverage"]
        and identity_ok
        and entity_ok
        and observed_ok
    ):
        return (
            "warn_construct_difference",
            "Construct or precursor difference; retain only as an edge-case control.",
        )

    reasons = []
    if full_coverage < thresholds["warn_full_length_mapping_coverage"]:
        reasons.append("low_full_length_mapping_coverage")
    if not identity_ok:
        reasons.append("low_sequence_identity")
    if not entity_ok:
        reasons.append("incomplete_entity_mapping")
    if not observed_ok:
        reasons.append("missing_observed_ca")
    if not internal_ok:
        reasons.append("internal_mapping_gaps")
    return "fail_preflight", ";".join(reasons) or "quality_threshold_failure"


def compute_pair_quality(
    mapping: pd.DataFrame,
    pdb_ca: pd.DataFrame,
    *,
    canonical_length: int,
    pdb_entity_length: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Compute pair-QC metrics and classify the mapping in one owner."""
    metrics = compute_mapping_quality_metrics(
        mapping,
        pdb_ca,
        uniprot_length=canonical_length,
        pdb_entity_length=pdb_entity_length,
    )
    mapping_status, mapping_reason = classify_mapping_quality(metrics, dict(thresholds))
    return {
        **metrics,
        "pair_qc_status": (
            "pair_qc_pass" if mapping_status == "pass_full_length" else "pair_qc_fail"
        ),
        "preflight_status": mapping_status,
        "preflight_reason": mapping_reason,
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
    table = mapping.merge(
        namespaces,
        on=auth_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_ca"),
    )
    for column in ("label_asym_id", "label_seq_id"):
        ca_column = f"{column}_ca"
        if ca_column in table:
            table[column] = table[column].combine_first(table[ca_column])
            table = table.drop(columns=ca_column)
    table, gaps = annotate_gap_semantics(table)
    observed, join_diagnostics = join_residue_mapping_to_ca(table, pdb_ca)
    observed_positions = set(observed["uniprot_residue_number"].astype(int))
    table["observed_ca"] = table["uniprot_residue_number"].astype(int).isin(
        observed_positions
    )
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
