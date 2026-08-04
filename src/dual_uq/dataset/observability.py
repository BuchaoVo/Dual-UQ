"""Mechanism-observability features and sampling-only prior derivation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.a0_classification import compute_pae_strata
from dual_uq.dataset_a_scale.pae import AFDBFragment
from dual_uq.geometry import kabsch_align
from dual_uq.mapped_confidence import (
    build_mapped_confidence_residue_table,
    summarize_mapped_confidence,
)
from dual_uq.robust_stats import contiguous_segments
from dual_uq.structure_io import join_residue_mapping_to_ca

from .mapping import prepare_confidence_mapping
from .models import DerivationError


def recompute_sampling_prior(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a sampling-only prior from positive observable quality-pass evidence."""
    available = dict(evidence.get("available", {}))
    names = [
        ("easy_control", "easy_control_prior"),
        ("low_confidence_local", "low_confidence_local_prior"),
        ("high_pae_long_range", "high_pae_long_range_prior"),
        ("state_disagreement", "state_disagreement_prior"),
    ]
    positives = [
        prior
        for label, prior in names
        if available.get(label) and evidence.get(label)
    ]
    quality_pass = evidence.get("quality_pass") is True
    supported = positives if quality_pass else []
    missing = [label for label, _ in names if not available.get(label)]
    if not any(available.values()):
        status = "still_unobservable"
        primary = "uncertain_or_unclassified"
    elif supported:
        status = "positive_evidence_observed"
        precedence = [
            "state_disagreement_prior",
            "high_pae_long_range_prior",
            "low_confidence_local_prior",
            "easy_control_prior",
        ]
        primary = next(label for label in precedence if label in supported)
    elif positives:
        status = "positive_partial_quality_ineligible"
        primary = "uncertain_or_unclassified"
    else:
        status = "observed_no_positive_prior"
        primary = "observed_no_positive_prior"
    return {
        "mechanism_observability_status": status,
        "positive_evidence_available": positives,
        "supported_sampling_priors": supported,
        "missing_evidence": missing,
        "sampling_stratum_prior_recomputed": primary,
    }

def state_segments(
    table: pd.DataFrame, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    thresholds = config["thresholds"]["high_confidence_state_disagreement"]
    minimum_disagreement = float(thresholds["minimum_segment_disagreement"])
    ordered = table.sort_values("uniprot_position", kind="mergesort")
    segments = contiguous_segments(
        ordered["uniprot_position"].to_numpy(int),
        ordered["ca_disagreement"].to_numpy(float),
        ordered["plddt"].to_numpy(float),
        threshold=minimum_disagreement,
    )
    return [
        {
            "start_uniprot": int(segment["start_position"]),
            "end_uniprot": int(segment["end_position"]),
            "residue_count": int(segment["residue_count"]),
            "median_plddt": float(segment["median_plddt"]),
            "median_disagreement": float(segment["median_disagreement"]),
        }
        for segment in segments
    ]


def mechanism_evidence(
    mapping: pd.DataFrame,
    pdb_observed: pd.DataFrame,
    afdb_ca: pd.DataFrame,
    pae: np.ndarray,
    confidence: np.ndarray,
    fragment: AFDBFragment,
    config: Mapping[str, Any],
    *,
    quality_pass: bool,
) -> dict[str, Any]:
    """Derive the four observability feature families without assigning P6 labels."""
    confidence_mapping = prepare_confidence_mapping(mapping)
    confidence_table = build_mapped_confidence_residue_table(
        confidence_mapping,
        pdb_observed,
        confidence,
        fragment_start=fragment.uniprot_start,
        fragment_end=fragment.uniprot_end,
    )
    confidence_summary = summarize_mapped_confidence(confidence_table)
    observed, _ = join_residue_mapping_to_ca(confidence_mapping, pdb_observed)
    coordinates = observed[
        ["uniprot_residue_number", "x", "y", "z"]
    ].rename(columns={"uniprot_residue_number": "uniprot_position"})
    afdb_coordinates = afdb_ca[
        ["uniprot_position", "x", "y", "z"]
    ].rename(columns={"x": "afdb_x", "y": "afdb_y", "z": "afdb_z"})
    geometry = coordinates.merge(
        afdb_coordinates, on="uniprot_position", how="inner", validate="one_to_one"
    )
    if len(geometry) < 3:
        raise DerivationError(
            "insufficient_geometry_pairs", "Fewer than 3 mapped CA pairs"
        )
    aligned, _, _ = kabsch_align(
        geometry[["afdb_x", "afdb_y", "afdb_z"]].to_numpy(float),
        geometry[["x", "y", "z"]].to_numpy(float),
    )
    target = geometry[["x", "y", "z"]].to_numpy(float)
    geometry["ca_disagreement"] = np.linalg.norm(aligned - target, axis=1)
    geometry["model_index"] = (
        geometry["uniprot_position"].astype(int) - fragment.uniprot_start
    )
    geometry["plddt"] = confidence[geometry["model_index"].to_numpy(int)]
    segment_lookup = mapping[["uniprot_position", "segment_id"]].drop_duplicates(
        "uniprot_position"
    )
    geometry = geometry.merge(
        segment_lookup, on="uniprot_position", validate="one_to_one"
    )
    pae_indices = (
        mapping["uniprot_position"].astype(int).to_numpy() - fragment.uniprot_start
    )
    mapped_pae = pae[np.ix_(pae_indices, pae_indices)]
    symmetric_pae = (mapped_pae + mapped_pae.T) / 2.0
    strata = compute_pae_strata(
        mapping["uniprot_position"].to_numpy(int), symmetric_pae
    )
    strata_dict = {name: asdict(stats) for name, stats in strata.items()}
    segments = state_segments(geometry, config)
    high_pae_thresholds = config["thresholds"]["high_pae_long_range"]
    long_range = [strata[name] for name in ("48_95", "96_plus") if name in strata]
    high_pae = any(
        stats.pair_count > 0
        and (
            stats.pae_q90 >= float(high_pae_thresholds["q90_threshold"])
            or stats.above_10_fraction
            >= float(high_pae_thresholds["above_10_fraction_threshold"])
            or stats.above_15_fraction
            >= float(high_pae_thresholds["above_15_fraction_threshold"])
        )
        for stats in long_range
    )
    state_thresholds = config["thresholds"]["high_confidence_state_disagreement"]
    state_threshold = int(state_thresholds["minimum_segment_length"])
    state_disagreement = any(
        segment["residue_count"] >= state_threshold
        and segment["median_plddt"]
        >= float(state_thresholds["minimum_segment_median_plddt"])
        and segment["median_disagreement"]
        >= float(state_thresholds["minimum_segment_disagreement"])
        for segment in segments
    )
    easy_thresholds = config["thresholds"]["easy_control"]
    disagreement_p90 = float(np.quantile(geometry["ca_disagreement"], 0.90))
    largest_segment = max((segment["residue_count"] for segment in segments), default=0)
    easy = bool(
        confidence_summary.mapped_plddt_median
        >= float(easy_thresholds["mapped_plddt_median_min"])
        and disagreement_p90 <= float(easy_thresholds["ca_disagreement_p90_max"])
        and largest_segment <= int(easy_thresholds["high_conf_segment_max_length"])
        and not confidence_summary.is_low_conf_local
        and not high_pae
        and not state_disagreement
    )
    labels = {
        "easy_control": easy,
        "low_confidence_local": bool(confidence_summary.is_low_conf_local),
        "high_pae_long_range": high_pae,
        "state_disagreement": state_disagreement,
        "quality_pass": quality_pass,
        "available": {
            "easy_control": True,
            "low_confidence_local": True,
            "high_pae_long_range": bool(sum(stats.pair_count for stats in long_range)),
            "state_disagreement": True,
        },
    }
    prior = recompute_sampling_prior(labels)
    return {
        **prior,
        "mechanism_multilabel_evidence": labels,
        "mapped_plddt_median": confidence_summary.mapped_plddt_median,
        "mapped_plddt_q10": confidence_summary.mapped_plddt_q10,
        "longest_internal_below_80_length": (
            confidence_summary.longest_internal_below_80_length
        ),
        "ca_disagreement_p90": disagreement_p90,
        "ca_disagreement_median": float(np.median(geometry["ca_disagreement"])),
        "pae_strata": strata_dict,
        "state_disagreement_segments": segments,
        "geometry_pair_count": len(geometry),
    }
