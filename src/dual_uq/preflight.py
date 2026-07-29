from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended

from .afdb import (
    UnsupportedAFDBFragment,
    canonical_uniprot_length,
    prediction_fragment_length,
    prediction_interval,
    select_prediction_for_interval,
)
from .structure_io import join_residue_mapping_to_ca


def normalise_residue_name(name: Any) -> str | None:
    if name is None:
        return None
    value = str(name).strip().upper()
    if len(value) == 1 and value.isalpha():
        return value
    return protein_letters_3to1_extended.get(value)


def evaluate_afdb_fragment_support(
    records: list[dict[str, Any]],
    mapped_interval: tuple[int, int],
) -> dict[str, Any]:
    canonical_length = canonical_uniprot_length(records)
    try:
        prediction = select_prediction_for_interval(records, mapped_interval)
    except UnsupportedAFDBFragment as exc:
        return {
            "afdb_fragment_status": "unsupported_afdb_fragment",
            "preflight_status": "unsupported_afdb_fragment",
            "preflight_reason": "no_afdb_fragment_covers_mapped_interval",
            "canonical_uniprot_length": canonical_length,
            "afdb_fragment_intervals": [list(interval) for interval in exc.fragment_intervals],
            "mapped_uniprot_start": mapped_interval[0],
            "mapped_uniprot_end": mapped_interval[1],
        }

    interval = prediction_interval(prediction)
    if interval is None:
        raise ValueError("Selected AlphaFold DB prediction has no valid interval.")
    return {
        "afdb_fragment_status": "supported",
        "canonical_uniprot_length": canonical_length,
        "afdb_fragment_length": prediction_fragment_length(prediction),
        "afdb_fragment_start": interval[0],
        "afdb_fragment_end": interval[1],
        "afdb_model_entity_id": prediction.get("modelEntityId"),
        "afdb_version": prediction.get("latestVersion"),
        "prediction": prediction,
        "mapped_uniprot_start": mapped_interval[0],
        "mapped_uniprot_end": mapped_interval[1],
    }


def compute_preflight_metrics(
    mapping: pd.DataFrame,
    pdb_ca_table: pd.DataFrame,
    *,
    uniprot_length: int,
    pdb_entity_length: int,
) -> dict[str, Any]:
    mapped_positions = np.sort(
        mapping["uniprot_residue_number"].dropna().astype(int).unique()
    )
    mapped_count = len(mapped_positions)
    if mapped_count == 0:
        raise ValueError("No mapped UniProt residue positions.")

    full_length_coverage = mapped_count / int(uniprot_length)
    entity_mapping_coverage = mapped_count / int(pdb_entity_length)

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
    observed_ca_fraction = len(observed_positions) / mapped_count

    first_position = int(mapped_positions.min())
    last_position = int(mapped_positions.max())
    span_length = last_position - first_position + 1
    internal_unmapped_count = span_length - mapped_count

    return {
        "mapped_residue_count": mapped_count,
        "uniprot_length": int(uniprot_length),
        "pdb_entity_length": int(pdb_entity_length),
        "full_length_mapping_coverage": float(full_length_coverage),
        "entity_mapping_coverage": float(entity_mapping_coverage),
        "sequence_identity": sequence_identity,
        "observed_ca_count": len(observed_positions),
        "observed_ca_fraction_of_mapped": float(observed_ca_fraction),
        "first_mapped_uniprot_position": first_position,
        "last_mapped_uniprot_position": last_position,
        "n_terminal_unmapped_count": int(first_position - 1),
        "c_terminal_unmapped_count": int(uniprot_length - last_position),
        "internal_unmapped_count": int(internal_unmapped_count),
        "internal_unmapped_fraction": float(internal_unmapped_count / max(span_length, 1)),
        "pdb_to_uniprot_length_ratio": float(pdb_entity_length / uniprot_length),
        **join_diagnostics,
    }


def classify_preflight(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[str, str]:
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


def finalize_preflight_status(
    fragment_support: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    mapping_status, mapping_reason = classify_preflight(metrics, thresholds)
    afdb_status = str(fragment_support["afdb_fragment_status"])
    if afdb_status == "unsupported_afdb_fragment":
        preflight_status = "unsupported_afdb_fragment"
        preflight_reason = str(fragment_support["preflight_reason"])
    else:
        preflight_status = mapping_status
        preflight_reason = mapping_reason
    return {
        "preflight_status": preflight_status,
        "preflight_reason": preflight_reason,
        "afdb_coverage_status": afdb_status,
        "mapping_quality_status": mapping_status,
        "mapping_quality_reason": mapping_reason,
        "geometry_launched": False,
    }
