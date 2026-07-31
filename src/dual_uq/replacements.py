from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MIN_TARGET_COUNT = 15
MAX_TARGET_COUNT = 20

HARD_QUALITY_THRESHOLDS = {
    "min_pdb_to_uniprot_length_ratio": 0.90,
    "max_pdb_to_uniprot_length_ratio": 1.10,
    "min_full_length_mapping_coverage": 0.90,
    "min_entity_mapping_coverage": 0.90,
    "min_sequence_identity": 0.95,
    "min_observed_ca_fraction": 0.90,
}

COMPOSITE_KEY = ["screening_index", "pdb_id", "chain_id", "uniprot_id"]
PREFLIGHT_PROVENANCE_FIELDS = {
    "afdb_model_entity_id",
    "afdb_version",
    "afdb_fragment_start",
    "afdb_fragment_end",
    "afdb_fragment_status",
}


@dataclass(frozen=True)
class ReplacementEligibility:
    eligible: bool
    pdb_to_uniprot_length_ratio: float | None
    exclusion_reasons: tuple[str, ...]
    quality_metrics: dict[str, float | bool | None]


def _value(row: Mapping[str, Any] | pd.Series, field: str) -> Any:
    return row.get(field) if hasattr(row, "get") else None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def evaluate_full_length_proxy(
    row: Mapping[str, Any] | pd.Series,
    thresholds: Mapping[str, float],
) -> ReplacementEligibility:
    canonical_length = _finite_float(_value(row, "canonical_uniprot_length"))
    pdb_length = _finite_float(_value(row, "pdb_entity_length"))
    full_coverage = _finite_float(_value(row, "full_length_mapping_coverage"))
    entity_coverage = _finite_float(_value(row, "entity_mapping_coverage"))
    sequence_identity = _finite_float(_value(row, "sequence_identity"))
    observed_fraction = _finite_float(
        _value(row, "observed_ca_fraction_of_mapped")
    )
    global_plddt = _finite_float(_value(row, "afdb_global_plddt"))

    reasons: list[str] = []
    if canonical_length is None or canonical_length <= 0:
        reasons.append("missing_canonical_uniprot_length")
    if pdb_length is None or pdb_length <= 0:
        reasons.append("missing_pdb_entity_length")

    ratio = (
        pdb_length / canonical_length
        if pdb_length is not None
        and pdb_length > 0
        and canonical_length is not None
        and canonical_length > 0
        else None
    )
    if ratio is not None:
        if ratio < float(thresholds["min_pdb_to_uniprot_length_ratio"]):
            reasons.append("low_length_ratio")
        elif ratio > float(thresholds["max_pdb_to_uniprot_length_ratio"]):
            reasons.append("high_length_ratio")

    metric_checks = [
        (
            full_coverage,
            "missing_full_length_mapping_coverage",
            "low_full_length_mapping_coverage",
            "min_full_length_mapping_coverage",
        ),
        (
            entity_coverage,
            "missing_entity_mapping_coverage",
            "low_entity_mapping_coverage",
            "min_entity_mapping_coverage",
        ),
        (
            sequence_identity,
            "missing_sequence_identity",
            "low_sequence_identity",
            "min_sequence_identity",
        ),
        (
            observed_fraction,
            "missing_observed_ca",
            "low_observed_ca_fraction",
            "min_observed_ca_fraction",
        ),
    ]
    for metric, missing_reason, low_reason, threshold_name in metric_checks:
        if metric is None:
            reasons.append(missing_reason)
        elif metric < float(thresholds[threshold_name]):
            reasons.append(low_reason)

    if global_plddt is None:
        reasons.append("missing_afdb_global_plddt")

    preflight_status = str(_value(row, "preflight_status") or "")
    if preflight_status == "unsupported_afdb_fragment":
        reasons.append("unsupported_afdb_fragment")
    elif preflight_status == "failed_runtime":
        reasons.append("failed_runtime")

    metrics: dict[str, float | bool | None] = {
        "canonical_uniprot_length": canonical_length,
        "pdb_entity_length": pdb_length,
        "pdb_to_uniprot_length_ratio": ratio,
        "full_length_mapping_coverage": full_coverage,
        "entity_mapping_coverage": entity_coverage,
        "sequence_identity": sequence_identity,
        "observed_ca_fraction_of_mapped": observed_fraction,
        "afdb_global_plddt": global_plddt,
        "global_plddt_available": global_plddt is not None,
    }
    return ReplacementEligibility(
        eligible=not reasons,
        pdb_to_uniprot_length_ratio=ratio,
        exclusion_reasons=tuple(reasons),
        quality_metrics=metrics,
    )


def _normalise_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pdb_id"] = result["pdb_id"].astype(str).str.strip().str.lower()
    result["chain_id"] = result["chain_id"].astype(str).str.strip()
    result["uniprot_id"] = result["uniprot_id"].astype(str).str.strip().str.upper()
    return result


def _seeded_rank(seed: int, row: pd.Series) -> str:
    identity = "|".join(
        [
            str(seed),
            str(row["pdb_id"]),
            str(row["chain_id"]),
            str(row["uniprot_id"]),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _sort_candidates(
    candidates: pd.DataFrame,
    *,
    confidence_ranking: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    ranked = candidates.copy()
    if not bool(confidence_ranking.get("enabled", False)):
        raise ValueError("confidence_ranking.enabled must be true")
    direction = str(confidence_ranking.get("direction", "")).lower()
    if direction != "ascending":
        raise ValueError("confidence_ranking.direction must be ascending")
    if confidence_ranking.get("hard_max") is not None:
        raise ValueError(
            "confidence_ranking.hard_max must be null; global pLDDT is "
            "a ranking prior only"
        )

    def numeric_series(column: str, *, default: float) -> pd.Series:
        values = (
            ranked[column]
            if column in ranked
            else pd.Series(default, index=ranked.index, dtype=float)
        )
        return pd.to_numeric(values, errors="coerce").fillna(default)

    ranked["_ratio_distance"] = (
        pd.to_numeric(ranked["pdb_to_uniprot_length_ratio"], errors="coerce") - 1.0
    ).abs()
    ranked["_full_coverage"] = numeric_series(
        "full_length_mapping_coverage",
        default=-np.inf,
    )
    ranked["_observed_fraction"] = numeric_series(
        "observed_ca_fraction_of_mapped",
        default=-np.inf,
    )
    ranked["_resolution"] = numeric_series("resolution", default=np.inf)
    ranked["_global_plddt"] = numeric_series(
        "afdb_global_plddt",
        default=np.inf,
    )
    ranked["_seeded_rank"] = ranked.apply(
        lambda row: _seeded_rank(seed, row),
        axis=1,
    )
    return ranked.sort_values(
        [
            "_global_plddt",
            "_ratio_distance",
            "_full_coverage",
            "_observed_fraction",
            "_resolution",
            "_seeded_rank",
            "pdb_id",
            "chain_id",
            "uniprot_id",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )


def build_replacement_shortlist(
    discovered_candidates: pd.DataFrame,
    current_pool: pd.DataFrame,
    *,
    target_count: int,
    confidence_ranking: Mapping[str, Any],
    diversity_config: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    if not MIN_TARGET_COUNT <= int(target_count) <= MAX_TARGET_COUNT:
        raise ValueError(
            f"target_count must be between {MIN_TARGET_COUNT} and {MAX_TARGET_COUNT}"
        )

    required = {
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "length",
        "canonical_uniprot_length",
        "afdb_global_plddt",
        "discovery_status",
    }
    missing = sorted(required - set(discovered_candidates.columns))
    if missing:
        raise ValueError(f"Discovery candidates lack required fields: {missing}")

    source = discovered_candidates.loc[
        discovered_candidates["discovery_status"].eq("eligible")
    ].copy()
    source_eligible_count = len(source)
    source = source.loc[
        source[["pdb_id", "chain_id", "uniprot_id"]].notna().all(axis=1)
    ]
    source = _normalise_identifiers(source)

    current = current_pool.copy()
    if current.empty:
        current = pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"])
    else:
        missing_current = {
            "pdb_id",
            "chain_id",
            "uniprot_id",
        } - set(current.columns)
        if missing_current:
            raise ValueError(
                "Current pool lacks required fields: "
                f"{sorted(missing_current)}"
            )
        current = _normalise_identifiers(current)

    current_uniprots = set(current["uniprot_id"])
    current_pairs = set(
        current[["pdb_id", "chain_id", "uniprot_id"]].itertuples(
            index=False,
            name=None,
        )
    )
    source_pairs = list(
        source[["pdb_id", "chain_id", "uniprot_id"]].itertuples(
            index=False,
            name=None,
        )
    )
    existing_mask = source["uniprot_id"].isin(current_uniprots) | pd.Series(
        [pair in current_pairs for pair in source_pairs],
        index=source.index,
    )
    excluded_existing_pool_count = int(existing_mask.sum())
    source = source.loc[~existing_mask].copy()

    length_min = float(diversity_config.get("length_min", -np.inf))
    length_max = float(diversity_config.get("length_max", np.inf))
    min_ratio = float(
        diversity_config.get("min_pdb_to_uniprot_length_ratio", 0.90)
    )
    max_ratio = float(
        diversity_config.get("max_pdb_to_uniprot_length_ratio", 1.10)
    )
    pdb_length = pd.to_numeric(source["length"], errors="coerce")
    canonical_length = pd.to_numeric(
        source["canonical_uniprot_length"], errors="coerce"
    )
    global_plddt = pd.to_numeric(source["afdb_global_plddt"], errors="coerce")
    source["pdb_entity_length"] = pdb_length
    source["pdb_to_uniprot_length_ratio"] = pdb_length / canonical_length
    length_proxy_eligible = (
        pdb_length.between(length_min, length_max, inclusive="both")
        & canonical_length.gt(0)
        & source["pdb_to_uniprot_length_ratio"].between(
            min_ratio,
            max_ratio,
            inclusive="both",
        )
    )
    source = source.loc[length_proxy_eligible].copy()
    length_proxy_eligible_count = len(source)
    source = source.loc[global_plddt.loc[source.index].notna()].copy()
    confidence_ranked_count = len(source)

    source["organism"] = source.get(
        "organism",
        pd.Series(index=source.index, dtype=object),
    ).fillna("UNKNOWN_ORGANISM")
    source["organism"] = source["organism"].astype(str).str.strip()
    source.loc[source["organism"].eq(""), "organism"] = "UNKNOWN_ORGANISM"

    clusters = source.get(
        "sequence_cluster",
        pd.Series(index=source.index, dtype=object),
    )
    source["sequence_cluster"] = clusters
    missing_cluster = source["sequence_cluster"].isna() | source[
        "sequence_cluster"
    ].astype(str).str.strip().eq("")
    source.loc[missing_cluster, "sequence_cluster"] = source.loc[
        missing_cluster, "uniprot_id"
    ].map(lambda value: f"UNKNOWN_SEQUENCE_CLUSTER:{value}")

    source = _sort_candidates(
        source,
        confidence_ranking=confidence_ranking,
        seed=seed,
    )
    source = source.drop_duplicates(
        ["pdb_id", "chain_id", "uniprot_id"],
        keep="first",
    )
    if bool(diversity_config.get("unique_uniprot", True)):
        source = source.drop_duplicates("uniprot_id", keep="first")

    unique_cluster = bool(diversity_config.get("unique_sequence_cluster", False))
    max_per_organism = diversity_config.get("max_per_organism")
    organism_counts: Counter[str] = Counter()
    used_clusters: set[str] = set()
    selected_indices: list[Any] = []
    for index, row in source.iterrows():
        organism = str(row["organism"])
        cluster = str(row["sequence_cluster"])
        if (
            max_per_organism is not None
            and organism_counts[organism] >= int(max_per_organism)
        ):
            continue
        if unique_cluster and cluster in used_clusters:
            continue
        selected_indices.append(index)
        organism_counts[organism] += 1
        used_clusters.add(cluster)

    diversity_eligible_count = len(selected_indices)
    selected = source.loc[selected_indices].head(int(target_count)).copy()
    selected = selected.reset_index(drop=True)
    selected["discovery_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_reason"] = "full_length_proxy;confidence_ranked"
    selected["exclusion_reasons"] = ""
    selected["provisional_stratum"] = "lower_global_confidence_replacement"
    selected["selection_stage"] = "full_length_proxy_preflight"
    selected["duplicate_with_original_pool"] = False
    selected = selected.drop(
        columns=[column for column in selected.columns if column.startswith("_")],
        errors="ignore",
    )
    selected.attrs["selection_audit"] = {
        "source_eligible_count": int(source_eligible_count),
        "existing_pool_exclusions": excluded_existing_pool_count,
        "excluded_existing_pool_count": excluded_existing_pool_count,
        "length_proxy_eligible_count": int(length_proxy_eligible_count),
        "confidence_ranked_count": int(confidence_ranked_count),
        "diversity_eligible_count": int(diversity_eligible_count),
        "requested_replacement_count": int(target_count),
        "selected_shortlist_count": len(selected),
        "shortfall_count": max(int(target_count) - len(selected), 0),
    }
    return selected


def merge_replacement_preflight(
    shortlist: pd.DataFrame,
    preflight_results: pd.DataFrame,
) -> pd.DataFrame:
    for name, frame in [
        ("shortlist", shortlist),
        ("preflight_results", preflight_results),
    ]:
        missing = sorted(set(COMPOSITE_KEY) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks composite key fields: {missing}")

    left = _normalise_identifiers(shortlist)
    right = _normalise_identifiers(preflight_results)
    if left.duplicated(COMPOSITE_KEY).any():
        raise ValueError("Shortlist composite key is not unique.")
    if right.duplicated(COMPOSITE_KEY).any():
        raise ValueError("Preflight composite key is not unique.")

    left = left.copy()
    left["_replacement_order"] = np.arange(len(left))
    overlapping = [
        column
        for column in right.columns
        if column in left.columns and column not in COMPOSITE_KEY
    ]
    merged = left.merge(
        right,
        on=COMPOSITE_KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "__preflight"),
    )
    if not merged["_merge"].eq("both").all():
        missing_keys = merged.loc[
            merged["_merge"].ne("both"),
            COMPOSITE_KEY,
        ].to_dict("records")
        raise ValueError(
            "Preflight results do not match the full composite key: "
            f"{missing_keys}"
        )
    for column in overlapping:
        preflight_column = f"{column}__preflight"
        if column in PREFLIGHT_PROVENANCE_FIELDS:
            merged[column] = merged[preflight_column]
        else:
            merged[column] = merged[preflight_column].combine_first(
                merged[column]
            )
        merged = merged.drop(columns=preflight_column)
    return (
        merged.sort_values("_replacement_order")
        .drop(columns=["_replacement_order", "_merge"])
        .reset_index(drop=True)
    )


def _indices_for_status(
    candidates: pd.DataFrame,
    statuses: set[str],
) -> list[int]:
    return (
        candidates.loc[
            candidates["preflight_status"].isin(statuses),
            "screening_index",
        ]
        .astype(int)
        .sort_values()
        .tolist()
    )


def _thresholds_are_not_lowered(thresholds: Mapping[str, float]) -> bool:
    return (
        float(thresholds["min_pdb_to_uniprot_length_ratio"])
        >= HARD_QUALITY_THRESHOLDS["min_pdb_to_uniprot_length_ratio"]
        and float(thresholds["max_pdb_to_uniprot_length_ratio"])
        <= HARD_QUALITY_THRESHOLDS["max_pdb_to_uniprot_length_ratio"]
        and all(
            float(thresholds[name]) >= minimum
            for name, minimum in HARD_QUALITY_THRESHOLDS.items()
            if name.startswith("min_")
            and name != "min_pdb_to_uniprot_length_ratio"
        )
    )


def build_replacement_audit(
    candidates: pd.DataFrame,
    *,
    requested_count: int,
    thresholds: Mapping[str, float],
    seed: int,
    source_counts: Mapping[str, int] | None = None,
    ranking_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_counts = dict(source_counts or {})
    ranking_config = dict(ranking_config or {})
    status = candidates.get(
        "preflight_status",
        pd.Series(index=candidates.index, dtype=object),
    ).fillna("not_attempted")
    attempted_mask = status.ne("not_attempted")
    pass_mask = status.eq("pass_full_length")
    pass_quality_failures = [
        str(row["uniprot_id"])
        for _, row in candidates.loc[pass_mask].iterrows()
        if not evaluate_full_length_proxy(row, thresholds).eligible
    ]

    unique_uniprot_count = int(
        candidates.get(
            "uniprot_id",
            pd.Series(index=candidates.index, dtype=object),
        )
        .astype(str)
        .str.upper()
        .nunique()
    )
    duplicate_with_original_pool_count = int(
        candidates.get(
            "duplicate_with_original_pool",
            pd.Series(False, index=candidates.index),
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )
    thresholds_unchanged = _thresholds_are_not_lowered(thresholds)
    ranking_configuration_valid = (
        bool(ranking_config.get("enabled", False))
        and str(ranking_config.get("direction", "")).lower() == "ascending"
        and ranking_config.get("hard_max") is None
    )
    global_plddt = pd.to_numeric(
        candidates.get(
            "afdb_global_plddt",
            pd.Series(index=candidates.index, dtype=float),
        ),
        errors="coerce",
    )
    finite_global_plddt = global_plddt.dropna()
    ranking_order = candidates
    if "discovery_rank" in candidates:
        ranking_order = candidates.sort_values(
            "discovery_rank",
            kind="mergesort",
        )
    elif "screening_index" in candidates:
        ranking_order = candidates.sort_values(
            "screening_index",
            kind="mergesort",
        )
    ranked_global_plddt = pd.to_numeric(
        ranking_order.get(
            "afdb_global_plddt",
            pd.Series(index=ranking_order.index, dtype=float),
        ),
        errors="coerce",
    )
    ranking_order_valid = (
        candidates.empty
        or (
            ranked_global_plddt.notna().all()
            and ranked_global_plddt.is_monotonic_increasing
        )
    )
    prohibited_columns = sorted(
        {"is_low_conf_local", "primary_category"} & set(candidates.columns)
    )

    blocked_reasons: list[str] = []
    if len(candidates) < MIN_TARGET_COUNT:
        blocked_reasons.append("selected_shortlist_below_15")
    if int(pass_mask.sum()) < 5:
        blocked_reasons.append("fewer_than_5_full_length_passes")
    if pass_quality_failures:
        blocked_reasons.append("pass_samples_violate_hard_quality_thresholds")
    if unique_uniprot_count != len(candidates):
        blocked_reasons.append("duplicate_uniprot_candidates")
    if duplicate_with_original_pool_count:
        blocked_reasons.append("duplicates_original_screening_pool")
    if not thresholds_unchanged:
        blocked_reasons.append("quality_thresholds_lowered")
    if not ranking_configuration_valid:
        blocked_reasons.append("invalid_confidence_ranking_configuration")
    if not ranking_order_valid:
        blocked_reasons.append("confidence_ranking_order_violation")
    if prohibited_columns:
        blocked_reasons.append("mapped_confidence_or_primary_category_present")

    exclusion_counts: Counter[str] = Counter()
    if "exclusion_reasons" in candidates:
        for value in candidates["exclusion_reasons"].fillna("").astype(str):
            exclusion_counts.update(
                reason for reason in value.split(";") if reason
            )

    organism = candidates.get(
        "organism",
        pd.Series("UNKNOWN_ORGANISM", index=candidates.index),
    ).fillna("UNKNOWN_ORGANISM")
    clusters = candidates.get(
        "sequence_cluster",
        pd.Series(index=candidates.index, dtype=object),
    )
    warn_count = int(status.eq("warn_construct_difference").sum())
    fail_count = int(status.eq("fail_preflight").sum())
    unsupported_count = int(status.eq("unsupported_afdb_fragment").sum())
    runtime_failure_count = int(status.eq("failed_runtime").sum())
    direction = str(ranking_config.get("direction", "ascending")).lower()
    global_stats: dict[str, float | None] = {
        "global_plddt_min": None,
        "global_plddt_q25": None,
        "global_plddt_median": None,
        "global_plddt_q75": None,
        "global_plddt_max": None,
    }
    if not finite_global_plddt.empty:
        global_stats = {
            "global_plddt_min": float(finite_global_plddt.min()),
            "global_plddt_q25": float(finite_global_plddt.quantile(0.25)),
            "global_plddt_median": float(finite_global_plddt.median()),
            "global_plddt_q75": float(finite_global_plddt.quantile(0.75)),
            "global_plddt_max": float(finite_global_plddt.max()),
        }
    audit: dict[str, Any] = {
        "source_eligible_count": int(source_counts.get("source_eligible_count", 0)),
        "existing_pool_exclusions": int(
            source_counts.get(
                "existing_pool_exclusions",
                source_counts.get("excluded_existing_pool_count", 0),
            )
        ),
        "excluded_existing_pool_count": int(
            source_counts.get(
                "excluded_existing_pool_count",
                source_counts.get("existing_pool_exclusions", 0),
            )
        ),
        "length_proxy_eligible_count": int(
            source_counts.get("length_proxy_eligible_count", 0)
        ),
        "confidence_ranked_count": int(
            source_counts.get("confidence_ranked_count", 0)
        ),
        "diversity_eligible_count": int(
            source_counts.get("diversity_eligible_count", len(candidates))
        ),
        "requested_replacement_count": int(requested_count),
        "selected_shortlist_count": len(candidates),
        "shortfall_count": max(int(requested_count) - len(candidates), 0),
        "preflight_attempted_count": int(attempted_mask.sum()),
        "pass_full_length_count": int(pass_mask.sum()),
        "warn_count": warn_count,
        "fail_count": fail_count,
        "unsupported_count": unsupported_count,
        "runtime_failure_count": runtime_failure_count,
        "warn_construct_difference_count": warn_count,
        "fail_preflight_count": fail_count,
        "unsupported_afdb_fragment_count": unsupported_count,
        "failed_runtime_count": runtime_failure_count,
        "pass_indices": _indices_for_status(candidates, {"pass_full_length"}),
        "warning_indices": _indices_for_status(
            candidates,
            {"warn_construct_difference"},
        ),
        "failed_indices": _indices_for_status(
            candidates,
            {"fail_preflight", "failed_runtime"},
        ),
        "unsupported_indices": _indices_for_status(
            candidates,
            {"unsupported_afdb_fragment"},
        ),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "organism_counts": {
            str(key): int(value)
            for key, value in sorted(organism.value_counts().items())
        },
        "sequence_cluster_count": int(clusters.nunique(dropna=False)),
        "unique_uniprot_count": unique_uniprot_count,
        "duplicate_with_original_pool_count": duplicate_with_original_pool_count,
        "pass_quality_failure_uniprots": pass_quality_failures,
        "hard_thresholds": {
            key: float(value)
            for key, value in thresholds.items()
            if key in HARD_QUALITY_THRESHOLDS
        },
        "thresholds": {
            key: float(value)
            for key, value in thresholds.items()
            if key in HARD_QUALITY_THRESHOLDS
        },
        "thresholds_unchanged": thresholds_unchanged,
        "confidence_ranking_order_valid": ranking_order_valid,
        "ranking_features": [
            f"afdb_global_plddt:{direction}",
            "pdb_to_uniprot_length_ratio_distance:ascending",
            "full_length_mapping_coverage:descending",
            "observed_ca_fraction_of_mapped:descending",
            "resolution:ascending",
            "stable_sha256_tiebreak",
        ],
        "global_plddt_role": "ranking_prior_only",
        "mapped_region_confidence_implemented": False,
        "seed": int(seed),
        "audit_pass": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        **global_stats,
    }
    return audit
