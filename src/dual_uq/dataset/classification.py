from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PAE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("6_11", 6, 11),
    ("12_23", 12, 23),
    ("24_47", 24, 47),
    ("48_95", 48, 95),
    ("96_plus", 96, None),
)
LONG_RANGE_PAE_BINS = ("48_95", "96_plus")


@dataclass(frozen=True)
class PaeStratumStats:
    pair_count: int
    pae_q50: float
    pae_q90: float
    pae_mean: float
    above_10_fraction: float
    above_15_fraction: float


@dataclass(frozen=True)
class DisagreementSegment:
    start_position: int
    end_position: int
    residue_count: int
    median_plddt: float
    median_disagreement: float


@dataclass(frozen=True)
class A0Evidence:
    preflight_status: str | None = None
    full_length_mapping_coverage: float | None = None
    entity_mapping_coverage: float | None = None
    sequence_identity: float | None = None
    observed_ca_fraction: float | None = None
    pair_status: str | None = None
    geometry_status: str | None = None
    robust_status: str | None = None
    segment_context_status: str | None = None
    confidence_model_match: bool | None = None
    mapped_plddt_median: float | None = None
    longest_internal_below_80_length: int | None = None
    terminal_only_below_80: bool | None = None
    low_conf_evidence_available: bool = False
    pae_strata: Mapping[str, PaeStratumStats] | None = None
    pae_evidence_available: bool = False
    ca_disagreement_p90: float | None = None
    disagreement_segments: tuple[DisagreementSegment, ...] | None = None
    state_disagreement_evidence_available: bool = False
    pilot_role: str | None = None
    secondary_missing_coordinate_stress: bool | None = None
    unsupported_afdb_fragment: bool = False
    protein_length: int | None = None
    provisional_stratum: str | None = None


@dataclass(frozen=True)
class A0Classification:
    is_easy_control: bool
    is_low_conf_local: bool | None
    is_high_pae_long_range: bool | None
    is_high_conf_state_disagreement: bool | None
    is_construct_difference: bool | None
    is_missing_coordinate_stress: bool | None
    easy_control_evidence_available: bool
    low_conf_evidence_available: bool
    high_pae_evidence_available: bool
    state_disagreement_evidence_available: bool
    construct_evidence_available: bool
    missing_coordinate_evidence_available: bool
    easy_control_evidence_summary: str
    low_conf_evidence_summary: str
    high_pae_evidence_summary: str
    state_disagreement_evidence_summary: str
    construct_evidence_summary: str
    missing_coordinate_evidence_summary: str
    primary_category: str
    classification_status: str
    full_diagnostic_complete: bool
    main_quality_pass: bool
    selection_eligible: bool
    classification_explanation: str


def _empty_pae_stats() -> PaeStratumStats:
    return PaeStratumStats(
        pair_count=0,
        pae_q50=float("nan"),
        pae_q90=float("nan"),
        pae_mean=float("nan"),
        above_10_fraction=float("nan"),
        above_15_fraction=float("nan"),
    )


def compute_pae_strata(
    uniprot_positions: np.ndarray,
    symmetric_pae: np.ndarray,
) -> dict[str, PaeStratumStats]:
    positions = np.asarray(uniprot_positions)
    pae = np.asarray(symmetric_pae, dtype=float)
    if positions.ndim != 1:
        raise ValueError("uniprot_positions must be one-dimensional")
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("symmetric_pae must be a square matrix")
    if pae.shape[0] != len(positions):
        raise ValueError("PAE matrix size must match uniprot_positions")
    numeric_positions = positions.astype(float)
    if (
        not np.isfinite(numeric_positions).all()
        or not np.equal(numeric_positions, np.floor(numeric_positions)).all()
        or (numeric_positions <= 0).any()
        or len(np.unique(numeric_positions)) != len(numeric_positions)
    ):
        raise ValueError("uniprot_positions must be unique positive integers")

    separations = np.abs(
        numeric_positions[:, None] - numeric_positions[None, :]
    )
    upper_triangle = np.triu(
        np.ones(separations.shape, dtype=bool),
        k=1,
    )
    strata: dict[str, PaeStratumStats] = {}
    for name, lower, upper in PAE_BINS:
        mask = upper_triangle & (separations >= lower)
        if upper is not None:
            mask &= separations <= upper
        values = pae[mask]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            strata[name] = _empty_pae_stats()
            continue
        strata[name] = PaeStratumStats(
            pair_count=len(values),
            pae_q50=float(np.quantile(values, 0.50)),
            pae_q90=float(np.quantile(values, 0.90)),
            pae_mean=float(np.mean(values)),
            above_10_fraction=float(np.mean(values >= 10.0)),
            above_15_fraction=float(np.mean(values >= 15.0)),
        )
    return strata


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _meets_minimum(value: Any, threshold: float) -> bool:
    number = _finite_number(value)
    return number is not None and number >= threshold


def _quality_pass(evidence: A0Evidence, config: Mapping[str, Any]) -> bool:
    thresholds = config["quality"]
    accepted_preflight = {
        "pass_full_length",
        "reference_quality_pass",
    }
    return (
        evidence.preflight_status in accepted_preflight
        and _meets_minimum(
            evidence.full_length_mapping_coverage,
            float(thresholds["minimum_full_length_mapping_coverage"]),
        )
        and _meets_minimum(
            evidence.entity_mapping_coverage,
            float(thresholds["minimum_entity_mapping_coverage"]),
        )
        and _meets_minimum(
            evidence.sequence_identity,
            float(thresholds["minimum_sequence_identity"]),
        )
        and _meets_minimum(
            evidence.observed_ca_fraction,
            float(thresholds["minimum_observed_ca_fraction"]),
        )
        and evidence.confidence_model_match is True
        and not evidence.unsupported_afdb_fragment
    )


def _pae_is_high(
    strata: Mapping[str, PaeStratumStats],
    config: Mapping[str, Any],
) -> bool:
    thresholds = config["thresholds"]["high_pae_long_range"]
    if int(thresholds["minimum_sequence_separation"]) != 48:
        raise ValueError(
            "high_pae_long_range minimum_sequence_separation must be 48"
        )
    tests: list[bool] = []
    for name in LONG_RANGE_PAE_BINS:
        stats = strata.get(name)
        if stats is None or stats.pair_count < 1:
            continue
        tests.extend(
            [
                stats.pae_q90 >= float(thresholds["q90_threshold"]),
                stats.above_10_fraction
                >= float(thresholds["above_10_fraction_threshold"]),
                stats.above_15_fraction
                >= float(thresholds["above_15_fraction_threshold"]),
            ]
        )
    rule = str(thresholds["rule"])
    if rule == "any":
        return any(tests)
    if rule == "all":
        return bool(tests) and all(tests)
    raise ValueError(f"unsupported high-PAE rule: {rule}")


def _qualifying_state_segments(
    segments: tuple[DisagreementSegment, ...],
    config: Mapping[str, Any],
) -> list[DisagreementSegment]:
    thresholds = config["thresholds"][
        "high_confidence_state_disagreement"
    ]
    minimum_length = int(thresholds["minimum_segment_length"])
    minimum_plddt = float(thresholds["minimum_segment_median_plddt"])
    minimum_disagreement = float(
        thresholds["minimum_segment_disagreement"]
    )
    return [
        segment
        for segment in segments
        if segment.residue_count >= minimum_length
        and segment.median_plddt >= minimum_plddt
        and segment.median_disagreement >= minimum_disagreement
    ]


def _largest_high_conf_disagreement_run(
    segments: tuple[DisagreementSegment, ...],
    config: Mapping[str, Any],
) -> int:
    thresholds = config["thresholds"][
        "high_confidence_state_disagreement"
    ]
    minimum_plddt = float(thresholds["minimum_segment_median_plddt"])
    minimum_disagreement = float(
        thresholds["minimum_segment_disagreement"]
    )
    lengths = [
        segment.residue_count
        for segment in segments
        if segment.median_plddt >= minimum_plddt
        and segment.median_disagreement >= minimum_disagreement
    ]
    return max(lengths, default=0)


def _primary_category(
    labels: Mapping[str, bool | None],
    *,
    unsupported: bool,
    full_diagnostic_complete: bool,
    config: Mapping[str, Any],
) -> str:
    active = {
        category
        for category, value in labels.items()
        if value is True
    }
    if unsupported:
        active.add("unsupported_afdb_fragment")
    if not active:
        active.add(
            "ordinary_or_unclassified"
            if full_diagnostic_complete
            else "insufficient_evidence"
        )
    priority = config["classification"]["primary_category_priority"]
    for category in priority:
        if category in active:
            return str(category)
    raise ValueError(
        "primary_category_priority does not cover active categories: "
        + ",".join(sorted(active))
    )


def classify_candidate(
    evidence: A0Evidence,
    config: Mapping[str, Any],
) -> A0Classification:
    unsupported = bool(
        evidence.unsupported_afdb_fragment
        or evidence.preflight_status == "unsupported_afdb_fragment"
    )
    construct_available = bool(
        evidence.preflight_status or evidence.pilot_role
    )
    is_construct = (
        evidence.preflight_status == "warn_construct_difference"
        or evidence.pilot_role == "construct_difference_negative_control"
    )
    missing_available = (
        _finite_number(evidence.observed_ca_fraction) is not None
        or evidence.secondary_missing_coordinate_stress is not None
    )
    observed_ca = _finite_number(evidence.observed_ca_fraction)
    observed_threshold = float(
        config["quality"]["minimum_observed_ca_fraction"]
    )
    is_missing = (
        (observed_ca is not None and observed_ca < observed_threshold)
        or evidence.secondary_missing_coordinate_stress is True
    )

    low_available = bool(
        not unsupported
        and evidence.low_conf_evidence_available
        and evidence.confidence_model_match is True
        and evidence.longest_internal_below_80_length is not None
    )
    if low_available:
        low_threshold = int(
            config["thresholds"]["low_confidence_local"][
                "internal_below_80_min_length"
            ]
        )
        is_low_conf = bool(
            evidence.preflight_status == "pass_full_length"
            and int(evidence.longest_internal_below_80_length)
            >= low_threshold
        )
    else:
        is_low_conf = None

    long_range_pair_count = 0
    if evidence.pae_strata is not None:
        long_range_pair_count = sum(
            evidence.pae_strata[name].pair_count
            for name in LONG_RANGE_PAE_BINS
            if name in evidence.pae_strata
        )
    high_pae_available = bool(
        not unsupported
        and evidence.pae_evidence_available
        and evidence.pae_strata is not None
        and long_range_pair_count > 0
    )
    is_high_pae = (
        _pae_is_high(evidence.pae_strata, config)
        if high_pae_available and evidence.pae_strata is not None
        else None
    )

    state_available = bool(
        not unsupported
        and evidence.state_disagreement_evidence_available
        and evidence.disagreement_segments is not None
        and evidence.segment_context_status
        in {"complete", "successful_no_segments"}
    )
    segments = evidence.disagreement_segments or ()
    qualifying_segments = (
        _qualifying_state_segments(segments, config)
        if state_available
        else []
    )
    is_state_disagreement = (
        bool(qualifying_segments) if state_available else None
    )

    full_diagnostic_complete = bool(
        not unsupported
        and evidence.pair_status == "complete"
        and evidence.geometry_status == "complete"
        and evidence.robust_status == "complete"
        and evidence.segment_context_status
        in {"complete", "successful_no_segments"}
        and evidence.confidence_model_match is True
        and _finite_number(evidence.mapped_plddt_median) is not None
        and _finite_number(evidence.ca_disagreement_p90) is not None
        and low_available
        and high_pae_available
        and state_available
    )
    main_quality_pass = _quality_pass(evidence, config)

    easy_thresholds = config["thresholds"]["easy_control"]
    median_plddt = _finite_number(evidence.mapped_plddt_median)
    disagreement_p90 = _finite_number(evidence.ca_disagreement_p90)
    largest_high_conf_run = (
        _largest_high_conf_disagreement_run(segments, config)
        if state_available
        else 0
    )
    easy_available = bool(
        full_diagnostic_complete
        and median_plddt is not None
        and disagreement_p90 is not None
    )
    is_easy = bool(
        easy_available
        and main_quality_pass
        and median_plddt
        >= float(easy_thresholds["mapped_plddt_median_min"])
        and disagreement_p90
        <= float(easy_thresholds["ca_disagreement_p90_max"])
        and largest_high_conf_run
        <= int(easy_thresholds["high_conf_segment_max_length"])
        and is_low_conf is False
        and is_high_pae is False
        and is_state_disagreement is False
        and not is_construct
        and not is_missing
    )

    labels = {
        "construct_difference_control": is_construct,
        "missing_coordinate_stress": is_missing,
        "high_confidence_state_disagreement": is_state_disagreement,
        "low_confidence_local": is_low_conf,
        "high_pae_long_range": is_high_pae,
        "easy_control": is_easy,
    }
    primary = _primary_category(
        labels,
        unsupported=unsupported,
        full_diagnostic_complete=full_diagnostic_complete,
        config=config,
    )
    any_known_label = any(value is True for value in labels.values())
    if unsupported:
        classification_status = "insufficient_evidence"
    elif full_diagnostic_complete:
        classification_status = "complete_classification"
    elif any_known_label:
        classification_status = "partial_classification"
    else:
        classification_status = "insufficient_evidence"

    selectable_categories = set(
        config["selection"]["target_per_primary_category"]
    )
    selection_eligible = bool(
        full_diagnostic_complete
        and main_quality_pass
        and not is_construct
        and not is_missing
        and not unsupported
        and primary in selectable_categories
    )

    low_summary = (
        "unknown: mapped-confidence evidence unavailable or mismatched"
        if not low_available
        else (
            "internal below-80 run length="
            f"{evidence.longest_internal_below_80_length}"
        )
    )
    if not high_pae_available:
        high_pae_summary = "unknown: long-range PAE evidence unavailable"
    else:
        high_pae_summary = "; ".join(
            f"{name}:q90={evidence.pae_strata[name].pae_q90:.3g},"
            f">10={evidence.pae_strata[name].above_10_fraction:.3g},"
            f">15={evidence.pae_strata[name].above_15_fraction:.3g},"
            f"n={evidence.pae_strata[name].pair_count}"
            for name in LONG_RANGE_PAE_BINS
            if evidence.pae_strata is not None
            and name in evidence.pae_strata
        )
    state_summary = (
        "unknown: structural-disagreement segment evidence unavailable"
        if not state_available
        else (
            f"qualifying segments={len(qualifying_segments)}, "
            f"largest high-confidence run={largest_high_conf_run}"
        )
    )
    explanation = (
        f"status={classification_status}; primary={primary}; "
        f"active_labels="
        + (
            ",".join(
                category
                for category, value in labels.items()
                if value is True
            )
            or "none"
        )
        + "; missing evidence remains unknown rather than negative."
    )
    return A0Classification(
        is_easy_control=is_easy,
        is_low_conf_local=is_low_conf,
        is_high_pae_long_range=is_high_pae,
        is_high_conf_state_disagreement=is_state_disagreement,
        is_construct_difference=is_construct if construct_available else None,
        is_missing_coordinate_stress=(
            is_missing if missing_available else None
        ),
        easy_control_evidence_available=easy_available,
        low_conf_evidence_available=low_available,
        high_pae_evidence_available=high_pae_available,
        state_disagreement_evidence_available=state_available,
        construct_evidence_available=construct_available,
        missing_coordinate_evidence_available=missing_available,
        easy_control_evidence_summary=(
            f"available={easy_available}; mapped pLDDT median="
            f"{median_plddt}; CA disagreement p90={disagreement_p90}; "
            f"largest high-confidence run={largest_high_conf_run}"
        ),
        low_conf_evidence_summary=low_summary,
        high_pae_evidence_summary=high_pae_summary,
        state_disagreement_evidence_summary=state_summary,
        construct_evidence_summary=(
            f"preflight={evidence.preflight_status}; "
            f"pilot_role={evidence.pilot_role}"
        ),
        missing_coordinate_evidence_summary=(
            f"observed CA fraction={observed_ca}; "
            "secondary stress="
            f"{evidence.secondary_missing_coordinate_stress}"
        ),
        primary_category=primary,
        classification_status=classification_status,
        full_diagnostic_complete=full_diagnostic_complete,
        main_quality_pass=main_quality_pass,
        selection_eligible=selection_eligible,
        classification_explanation=explanation,
    )
