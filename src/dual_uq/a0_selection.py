from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

PRIMARY_CATEGORIES = (
    "easy_control",
    "low_confidence_local",
    "high_pae_long_range",
    "high_confidence_state_disagreement",
)
MINIMUM_COMPLETE_QUALITY_PASS_DIAGNOSTICS = 24
DEFAULT_CATEGORY_REQUIREMENTS = {
    category: 3 for category in PRIMARY_CATEGORIES
}
PANEL_COLUMNS = (
    "panel_rank",
    "category_rank",
    "primary_category",
    "screening_index",
    "source",
    "pdb_id",
    "chain_id",
    "uniprot_id",
    "pair_name",
)
REQUIRED_COLUMNS = {
    "screening_index",
    "source",
    "pdb_id",
    "chain_id",
    "uniprot_id",
    "pair_name",
    "preflight_status",
    "classification_status",
    "full_diagnostic_complete",
    "main_quality_pass",
    "selection_eligible",
    "primary_category",
    "is_construct_difference",
    "is_missing_coordinate_stress",
}


@dataclass(frozen=True)
class A0SelectionResult:
    panel: pd.DataFrame
    audit: dict[str, Any]


def _boolean_series(table: pd.DataFrame, column: str) -> pd.Series:
    def convert(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"invalid boolean value in {column}: {value!r}")

    return table[column].map(convert).astype(bool)


def _validate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(summary.columns))
    if missing:
        raise ValueError("A0 summary missing columns: " + ",".join(missing))
    table = summary.copy()
    if table["pair_name"].isna().any() or table["pair_name"].duplicated().any():
        raise ValueError("A0 summary pair_name must be complete and unique")
    for column in (
        "full_diagnostic_complete",
        "main_quality_pass",
        "selection_eligible",
        "is_construct_difference",
        "is_missing_coordinate_stress",
    ):
        table[column] = _boolean_series(table, column)
    return table


def _validate_requirements(
    category_requirements: Mapping[str, int],
) -> dict[str, int]:
    if set(category_requirements) != set(PRIMARY_CATEGORIES):
        raise ValueError(
            "category requirements must cover exactly the four primary categories"
        )
    normalized = {
        category: int(category_requirements[category])
        for category in PRIMARY_CATEGORIES
    }
    if any(value < 1 for value in normalized.values()):
        raise ValueError("category requirements must be positive")
    return normalized


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=PANEL_COLUMNS)


def _eligible_category_candidates(table: pd.DataFrame) -> pd.DataFrame:
    non_reference = table["source"].astype(str).ne("reference_pair")
    complete_quality = (
        table["full_diagnostic_complete"] & table["main_quality_pass"]
    )
    eligible = (
        non_reference
        & complete_quality
        & table["selection_eligible"]
        & table["classification_status"].astype(str).eq(
            "complete_classification"
        )
        & table["primary_category"].astype(str).isin(PRIMARY_CATEGORIES)
        & ~table["is_construct_difference"]
        & ~table["is_missing_coordinate_stress"]
        & table["preflight_status"].astype(str).ne(
            "unsupported_afdb_fragment"
        )
    )
    return table.loc[eligible].copy()


def _select_panel(
    eligible: pd.DataFrame,
    requirements: Mapping[str, int],
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for category in PRIMARY_CATEGORIES:
        candidates = eligible.loc[
            eligible["primary_category"].astype(str).eq(category)
        ].copy()
        candidates["_screening_sort"] = pd.to_numeric(
            candidates["screening_index"],
            errors="coerce",
        )
        candidates = candidates.sort_values(
            ["_screening_sort", "pair_name"],
            na_position="last",
            kind="mergesort",
        ).head(requirements[category])
        candidates["category_rank"] = range(1, len(candidates) + 1)
        selected.append(candidates)
    panel = pd.concat(selected, ignore_index=True)
    panel.insert(0, "panel_rank", range(1, len(panel) + 1))
    panel["screening_index"] = pd.array(
        panel["screening_index"],
        dtype="Int64",
    )
    return panel.loc[:, PANEL_COLUMNS]


def select_final_a0_panel(
    summary: pd.DataFrame,
    *,
    minimum_complete_diagnostics: int = (
        MINIMUM_COMPLETE_QUALITY_PASS_DIAGNOSTICS
    ),
    category_requirements: Mapping[str, int] = (
        DEFAULT_CATEGORY_REQUIREMENTS
    ),
) -> A0SelectionResult:
    if minimum_complete_diagnostics < 1:
        raise ValueError("minimum_complete_diagnostics must be positive")
    requirements = _validate_requirements(category_requirements)
    table = _validate_summary(summary)

    non_reference = table["source"].astype(str).ne("reference_pair")
    complete_quality = (
        non_reference
        & table["full_diagnostic_complete"]
        & table["main_quality_pass"]
    )
    complete_quality_count = int(complete_quality.sum())
    diagnostic_gate_pass = (
        complete_quality_count >= minimum_complete_diagnostics
    )

    eligible = _eligible_category_candidates(table)
    category_counts = {
        category: int(
            eligible["primary_category"].astype(str).eq(category).sum()
        )
        for category in PRIMARY_CATEGORIES
    }
    category_gate_pass = all(
        category_counts[category] >= requirements[category]
        for category in PRIMARY_CATEGORIES
    )

    blocked_reasons: list[str] = []
    if not diagnostic_gate_pass:
        blocked_reasons.append(
            "fewer_than_24_complete_quality_pass_diagnostics"
        )
    for category in PRIMARY_CATEGORIES:
        if category_counts[category] < requirements[category]:
            blocked_reasons.append(
                f"insufficient_{category}_candidates"
            )

    audit_pass = diagnostic_gate_pass and category_gate_pass
    panel = (
        _select_panel(eligible, requirements)
        if audit_pass
        else _empty_panel()
    )
    ordinary_complete = int(
        (
            complete_quality
            & table["classification_status"].astype(str).eq(
                "complete_classification"
            )
            & table["primary_category"].astype(str).eq(
                "ordinary_or_unclassified"
            )
        ).sum()
    )
    audit = {
        "summary_rows": len(table),
        "reference_rows_excluded": int((~non_reference).sum()),
        "complete_classifications_excluding_reference": int(
            (
                non_reference
                & table["classification_status"].astype(str).eq(
                    "complete_classification"
                )
            ).sum()
        ),
        "complete_quality_pass_diagnostics": complete_quality_count,
        "minimum_complete_quality_pass_diagnostics": int(
            minimum_complete_diagnostics
        ),
        "diagnostic_gate_pass": diagnostic_gate_pass,
        "ordinary_complete_count": ordinary_complete,
        "eligible_primary_category_counts": category_counts,
        "required_primary_category_counts": requirements,
        "category_gate_pass": category_gate_pass,
        "target_panel_rows": int(sum(requirements.values())),
        "selected_panel_rows": len(panel),
        "selection_method": (
            "primary_category_order_then_screening_index_then_pair_name"
        ),
        "multi_label_quota_rule": "primary_category_only",
        "audit_pass": audit_pass,
        "blocked_reasons": blocked_reasons,
    }
    return A0SelectionResult(panel=panel, audit=audit)
