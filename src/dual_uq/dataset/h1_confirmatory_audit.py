"""Frozen Stage0 H1 confirmatory audit.

The audit is descriptive and offline.  It consumes immutable Stage0 releases,
does not execute ProteinMPNN, and permanently closes local analysis of the
frozen 1,790-position cohort after rendering its verdict.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.fixed_probe_sensitivity import (
    FixedProbeSensitivityError,
    write_immutable_json,
    write_immutable_parquet,
)

MECHANISM_COMMIT = "600f3cb473f1630fd31d4a5bafcd576dd8469024"
BOUNDARY_COMMIT = "e871c6cd5735c81758adb6b1c7ebb46d115b0aff"
STAGE0_ROOT = Path("experiments/p2_design_baseline/stage0")
MECHANISM_MANIFEST_PATH = STAGE0_ROOT / "fixed_probe_mechanism_manifest.json"
DECISION_MANIFEST_PATH = STAGE0_ROOT / "fixed_probe_decision_sensitivity_manifest.json"
BOUNDARY_MANIFEST_PATH = STAGE0_ROOT / "fixed_probe_boundary_audit_manifest.json"
MECHANISM_MANIFEST_SHA256 = "caab12fe4e290164d513c78decbfe07269ebce8ec207c0ca02c7cd1b2f85f14b"
DECISION_MANIFEST_SHA256 = "f7e0df9a6636864207a849e659d68aad077dbaf6aaaf5aa1fb2d943480986e98"
BOUNDARY_MANIFEST_SHA256 = "e1c434b31ce1262c3ded9a8b55739df6ec1d448145a350d044598cdf2ba49a0f"
EXPECTED_POSITION_COUNT = 1_790
EXPECTED_PROTEIN_COUNT = 8
EXPECTED_TECHNICAL_NULL_COUNT = 3_580
EXPECTED_POSITION_COUNTS = (272, 223, 224, 314, 125, 132, 114, 386)
TERTILES = ("LOW", "MID", "HIGH")
MATCH_CALIPER = 0.10
RECURRENCE_TARGET = 5
LEAVE_ONE_OUT_TARGET = 4
NUMERIC_TOLERANCE = 1.0e-12


class H1AuditError(ValueError):
    """Structured H1 input, analysis, or immutable-release failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class H1AuditInputs:
    project_root: Path
    protein_order: tuple[str, ...]
    mechanism_positions: pd.DataFrame
    mechanism_associations: pd.DataFrame
    decision_positions: pd.DataFrame
    technical_null: pd.DataFrame
    position_sensitivity: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class H1ConfirmatoryResult:
    project_root: Path
    protein_order: tuple[str, ...]
    input_provenance: dict[str, dict[str, Any]]
    position_audit: pd.DataFrame
    cells: pd.DataFrame
    matches: pd.DataFrame
    protein_summary: pd.DataFrame
    manifest_fields: dict[str, Any]


def within_protein_rank_pct(values: pd.Series) -> pd.Series:
    """Return the frozen average-rank empirical percentile in (0, 1)."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise H1AuditError("nonfinite_rank_variable", "Rank variable contains a nonfinite value")
    if len(numeric) == 0:
        raise H1AuditError("empty_rank_variable", "Rank variable is empty")
    return (numeric.rank(method="average") - 0.5) / len(numeric)


def assign_frozen_tertile(value: float) -> str:
    """Apply the fixed non-adaptive one-third boundaries."""
    if not np.isfinite(value) or not 0 < value < 1:
        raise H1AuditError("invalid_percentile_rank", f"Percentile rank is outside (0,1): {value}")
    if value < 1 / 3:
        return "LOW"
    if value < 2 / 3:
        return "MID"
    return "HIGH"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise H1AuditError("upstream_schema_mismatch", f"{label} is missing columns: {missing}")


def _require_unique_positions(frame: pd.DataFrame, label: str) -> None:
    if frame.duplicated(["protein_id", "position"]).any():
        raise H1AuditError("duplicate_position_key", f"{label} repeats a scientific key")


def build_position_audit(
    mechanism_positions: pd.DataFrame,
    technical_null: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Join frozen mechanism and technical-null values into the canonical cohort."""
    mechanism_required = {
        "protein_id",
        "position",
        "position_mean_abs_interaction",
        "margin_mean_both",
        "perturbation_to_margin",
        "top1_disagreement_fraction",
        "symmetric_regret_mean",
    }
    technical_required = {
        "protein_id",
        "position",
        "backbone_condition",
        "pairwise_top1_disagreement_rate",
        "pairwise_symmetric_regret_mean",
    }
    _require_columns(mechanism_positions, mechanism_required, "mechanism positions")
    _require_columns(technical_null, technical_required, "technical-reference table")
    _require_unique_positions(mechanism_positions, "mechanism positions")
    mechanism_order = tuple(dict.fromkeys(mechanism_positions["protein_id"].astype(str)))
    if mechanism_order != protein_order:
        raise H1AuditError(
            "protein_order_mismatch",
            "Frozen mechanism position order differs from the canonical protein order",
        )
    if technical_null.duplicated(["protein_id", "position", "backbone_condition"]).any():
        raise H1AuditError("duplicate_technical_reference", "Technical-reference keys repeat")
    if set(technical_null["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise H1AuditError(
            "technical_reference_condition_mismatch",
            "The technical-reference table must contain PDB and AFDB",
        )
    grouped = technical_null.groupby(["protein_id", "position"], sort=False)
    if not (grouped.size() == 2).all() or not grouped["backbone_condition"].agg(
        lambda values: set(values.astype(str)) == {"PDB", "AFDB"}
    ).all():
        raise H1AuditError(
            "technical_reference_grid_mismatch",
            "Every position requires exactly two technical-reference conditions",
        )
    technical = technical_null.pivot(
        index=["protein_id", "position"],
        columns="backbone_condition",
        values=["pairwise_top1_disagreement_rate", "pairwise_symmetric_regret_mean"],
    )
    technical.columns = [
        f"{metric}__{condition.lower()}" for metric, condition in technical.columns
    ]
    technical = technical.reset_index()
    joined = mechanism_positions.merge(
        technical,
        on=["protein_id", "position"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not (joined["_merge"] == "both").all():
        raise H1AuditError(
            "technical_reference_key_mismatch",
            "Mechanism and technical-reference position keys do not match",
        )
    p = pd.to_numeric(joined["position_mean_abs_interaction"], errors="coerce")
    m = pd.to_numeric(joined["margin_mean_both"], errors="coerce")
    sdfi = pd.to_numeric(joined["perturbation_to_margin"], errors="coerce")
    numeric = np.column_stack([p, m, sdfi])
    if not np.isfinite(numeric).all():
        raise H1AuditError("nonfinite_primary_variable", "P, M, or SDFI is nonfinite")
    if not (m > 0).all():
        raise H1AuditError("nonpositive_margin", "All frozen margins must be positive")
    expected_sdfi = p.to_numpy(dtype=np.float64) / m.to_numpy(dtype=np.float64)
    if not np.allclose(sdfi, expected_sdfi, rtol=0, atol=NUMERIC_TOLERANCE):
        raise H1AuditError("sdfi_algebra_mismatch", "Frozen SDFI does not equal P divided by M")
    output = pd.DataFrame(
        {
            "protein_id": joined["protein_id"].astype(str),
            "position": joined["position"].astype(int),
            "p": p.astype(float),
            "m": m.astype(float),
            "sdfi": sdfi.astype(float),
            "y_flip_struct": pd.to_numeric(joined["top1_disagreement_fraction"]),
            "y_regret_struct": pd.to_numeric(joined["symmetric_regret_mean"]),
            "y_flip_tech_pdb": pd.to_numeric(
                joined["pairwise_top1_disagreement_rate__pdb"]
            ),
            "y_flip_tech_afdb": pd.to_numeric(
                joined["pairwise_top1_disagreement_rate__afdb"]
            ),
            "y_regret_tech_pdb": pd.to_numeric(
                joined["pairwise_symmetric_regret_mean__pdb"]
            ),
            "y_regret_tech_afdb": pd.to_numeric(
                joined["pairwise_symmetric_regret_mean__afdb"]
            ),
        }
    )
    outcome_fields = [column for column in output if column.startswith("y_")]
    if not np.isfinite(output[outcome_fields].to_numpy(dtype=np.float64)).all():
        raise H1AuditError("nonfinite_outcome", "Structural or technical outcome is nonfinite")
    output["y_flip_tech_mean"] = (
        output["y_flip_tech_pdb"] + output["y_flip_tech_afdb"]
    ) / 2
    output["y_regret_tech_mean"] = (
        output["y_regret_tech_pdb"] + output["y_regret_tech_afdb"]
    ) / 2
    output["e_flip"] = output["y_flip_struct"] - output["y_flip_tech_mean"]
    output["e_regret"] = output["y_regret_struct"] - output["y_regret_tech_mean"]
    output["p_rank_pct"] = output.groupby("protein_id", sort=False)["p"].transform(
        within_protein_rank_pct
    )
    output["m_rank_pct"] = output.groupby("protein_id", sort=False)["m"].transform(
        within_protein_rank_pct
    )
    output["p_tertile"] = output["p_rank_pct"].map(assign_frozen_tertile)
    output["m_tertile"] = output["m_rank_pct"].map(assign_frozen_tertile)
    if set(output["protein_id"]) != set(protein_order):
        raise H1AuditError("protein_order_mismatch", "Position audit protein set differs")
    rank = {protein: index for index, protein in enumerate(protein_order)}
    output["_protein_order"] = output["protein_id"].map(rank)
    return output.sort_values(["_protein_order", "position"], kind="stable").drop(
        columns="_protein_order"
    ).reset_index(drop=True)


def materialize_conditional_cells(
    position_audit: pd.DataFrame, *, protein_order: tuple[str, ...]
) -> pd.DataFrame:
    """Materialize the fixed 3x3 grid for every protein, including empty cells."""
    summary_fields = (
        "p",
        "m",
        "sdfi",
        "y_flip_struct",
        "y_regret_struct",
        "e_flip",
        "e_regret",
    )
    _require_columns(
        position_audit,
        {"protein_id", "p_tertile", "m_tertile", *summary_fields},
        "position audit",
    )
    rows: list[dict[str, Any]] = []
    for protein_id, p_tertile, m_tertile in product(
        protein_order, TERTILES, TERTILES
    ):
        group = position_audit.loc[
            (position_audit["protein_id"] == protein_id)
            & (position_audit["p_tertile"] == p_tertile)
            & (position_audit["m_tertile"] == m_tertile)
        ]
        row: dict[str, Any] = {
            "protein_id": protein_id,
            "p_tertile": p_tertile,
            "m_tertile": m_tertile,
            "n_positions": len(group),
        }
        for field in summary_fields:
            row[f"{field}_mean"] = float(group[field].mean()) if len(group) else np.nan
            row[f"{field}_median"] = float(group[field].median()) if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _match_contract(match_type: str) -> dict[str, str]:
    if match_type == "C1":
        return {
            "exposure_field": "m_tertile",
            "exposed_value": "LOW",
            "comparator_value": "HIGH",
            "matching_field": "p_rank_pct",
        }
    if match_type == "C2":
        return {
            "exposure_field": "p_tertile",
            "exposed_value": "HIGH",
            "comparator_value": "LOW",
            "matching_field": "m_rank_pct",
        }
    raise H1AuditError("invalid_match_type", f"Unknown match type: {match_type}")


def _eligible_edges(
    frame: pd.DataFrame, *, match_type: str, caliper: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[float, int, int]]]:
    contract = _match_contract(match_type)
    exposed = frame.loc[
        frame[contract["exposure_field"]] == contract["exposed_value"]
    ].copy()
    comparator = frame.loc[
        frame[contract["exposure_field"]] == contract["comparator_value"]
    ].copy()
    edges: list[tuple[float, int, int]] = []
    for exposed_position, exposed_rank in zip(
        exposed["position"].astype(int), exposed[contract["matching_field"]], strict=True
    ):
        distances = (
            comparator[contract["matching_field"]].astype(float) - float(exposed_rank)
        ).abs()
        for comparator_position, distance in zip(
            comparator.loc[distances <= caliper, "position"].astype(int),
            distances.loc[distances <= caliper].astype(float),
            strict=True,
        ):
            edges.append((float(distance), int(exposed_position), int(comparator_position)))
    return exposed, comparator, sorted(edges, key=lambda edge: (edge[0], edge[1], edge[2]))


def deterministic_greedy_matches(
    position_audit: pd.DataFrame,
    *,
    match_type: str,
    caliper: float = MATCH_CALIPER,
    protein_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Perform frozen outcome-blind one-to-one greedy matching."""
    if caliper != MATCH_CALIPER:
        raise H1AuditError("matching_caliper_mismatch", "The primary matching caliper must equal 0.10")
    required = {
        "protein_id",
        "position",
        "p",
        "m",
        "p_rank_pct",
        "m_rank_pct",
        "p_tertile",
        "m_tertile",
        "y_flip_struct",
        "y_regret_struct",
        "e_flip",
        "e_regret",
    }
    _require_columns(position_audit, required, "position audit")
    order = protein_order or tuple(sorted(position_audit["protein_id"].astype(str).unique()))
    rows: list[dict[str, Any]] = []
    for protein_id in order:
        frame = position_audit.loc[position_audit["protein_id"] == protein_id].copy()
        _, _, edges = _eligible_edges(frame, match_type=match_type, caliper=caliper)
        by_position = frame.set_index("position")
        if not by_position.index.is_unique:
            raise H1AuditError("duplicate_position_key", "Matching positions repeat")
        used_exposed: set[int] = set()
        used_comparator: set[int] = set()
        for distance, exposed_position, comparator_position in edges:
            if exposed_position in used_exposed or comparator_position in used_comparator:
                continue
            used_exposed.add(exposed_position)
            used_comparator.add(comparator_position)
            exposed = by_position.loc[exposed_position]
            comparator = by_position.loc[comparator_position]
            rows.append(
                {
                    "protein_id": protein_id,
                    "match_type": match_type,
                    "exposed_position": exposed_position,
                    "comparator_position": comparator_position,
                    "matching_rank_distance": distance,
                    "p_exposed": float(exposed["p"]),
                    "p_comparator": float(comparator["p"]),
                    "m_exposed": float(exposed["m"]),
                    "m_comparator": float(comparator["m"]),
                    "p_rank_pct_exposed": float(exposed["p_rank_pct"]),
                    "p_rank_pct_comparator": float(comparator["p_rank_pct"]),
                    "m_rank_pct_exposed": float(exposed["m_rank_pct"]),
                    "m_rank_pct_comparator": float(comparator["m_rank_pct"]),
                    "delta_flip": float(exposed["y_flip_struct"] - comparator["y_flip_struct"]),
                    "delta_regret": float(
                        exposed["y_regret_struct"] - comparator["y_regret_struct"]
                    ),
                    "delta_e_flip": float(exposed["e_flip"] - comparator["e_flip"]),
                    "delta_e_regret": float(exposed["e_regret"] - comparator["e_regret"]),
                }
            )
    columns = [
        "protein_id",
        "match_type",
        "exposed_position",
        "comparator_position",
        "matching_rank_distance",
        "p_exposed",
        "p_comparator",
        "m_exposed",
        "m_comparator",
        "p_rank_pct_exposed",
        "p_rank_pct_comparator",
        "m_rank_pct_exposed",
        "m_rank_pct_comparator",
        "delta_flip",
        "delta_regret",
        "delta_e_flip",
        "delta_e_regret",
    ]
    result = pd.DataFrame(rows, columns=columns)
    order_index = {protein: index for index, protein in enumerate(order)}
    result["_protein_order"] = result["protein_id"].map(order_index)
    return result.sort_values(
        [
            "_protein_order",
            "match_type",
            "matching_rank_distance",
            "exposed_position",
            "comparator_position",
        ],
        kind="stable",
    ).drop(columns="_protein_order").reset_index(drop=True)


def _correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float | None, str | None]:
    values = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 2:
        return None, "insufficient_observations"
    if values["x"].nunique() < 2:
        return None, "constant_x"
    if values["y"].nunique() < 2:
        return None, "constant_y"
    value = float(values["x"].corr(values["y"], method=method))
    if not np.isfinite(value) or not -1 <= value <= 1:
        raise H1AuditError("invalid_descriptive_correlation", "Correlation is outside [-1,1]")
    return value, None


def _gradient_values(
    cells: pd.DataFrame,
    *,
    fixed_field: str,
    low_selector: tuple[str, str],
    high_selector: tuple[str, str],
    outcome: str,
) -> list[float]:
    values: list[float] = []
    for fixed in TERTILES:
        low = cells.loc[
            (cells[fixed_field] == fixed) & (cells[low_selector[0]] == low_selector[1])
        ]
        high = cells.loc[
            (cells[fixed_field] == fixed) & (cells[high_selector[0]] == high_selector[1])
        ]
        if len(low) == len(high) == 1 and int(low["n_positions"].iloc[0]) and int(
            high["n_positions"].iloc[0]
        ):
            values.append(float(high[f"{outcome}_mean"].iloc[0] - low[f"{outcome}_mean"].iloc[0]))
    return values


def _summarize_values(row: dict[str, Any], prefix: str, values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    row[f"{prefix}_valid_count"] = len(array)
    row[f"{prefix}_positive_count"] = int((array > 0).sum())
    row[f"{prefix}_mean"] = float(array.mean()) if len(array) else np.nan
    row[f"{prefix}_median"] = float(np.median(array)) if len(array) else np.nan


def _match_diagnostics(
    positions: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    match_type: str,
) -> dict[str, Any]:
    exposed, comparator, edges = _eligible_edges(
        positions, match_type=match_type, caliper=MATCH_CALIPER
    )
    output: dict[str, Any] = {
        f"{match_type.lower()}_exposed_count": len(exposed),
        f"{match_type.lower()}_comparator_count": len(comparator),
        f"{match_type.lower()}_matched_count": len(selected),
        f"{match_type.lower()}_matched_fraction_smaller_group": (
            len(selected) / min(len(exposed), len(comparator))
            if min(len(exposed), len(comparator))
            else np.nan
        ),
        f"{match_type.lower()}_eligible_edge_mean_rank_distance": (
            float(np.mean([edge[0] for edge in edges])) if edges else np.nan
        ),
        f"{match_type.lower()}_selected_mean_rank_distance": (
            float(selected["matching_rank_distance"].mean()) if len(selected) else np.nan
        ),
        f"{match_type.lower()}_selected_median_rank_distance": (
            float(selected["matching_rank_distance"].median()) if len(selected) else np.nan
        ),
    }
    if len(selected):
        if match_type == "C1":
            output["c1_mean_actual_p_difference"] = float(
                (selected["p_exposed"] - selected["p_comparator"]).abs().mean()
            )
            output["c1_mean_retained_m_contrast"] = float(
                (selected["m_comparator"] - selected["m_exposed"]).mean()
            )
        else:
            output["c2_mean_actual_m_difference"] = float(
                (selected["m_exposed"] - selected["m_comparator"]).abs().mean()
            )
            output["c2_mean_retained_p_contrast"] = float(
                (selected["p_exposed"] - selected["p_comparator"]).mean()
            )
    else:
        if match_type == "C1":
            output["c1_mean_actual_p_difference"] = np.nan
            output["c1_mean_retained_m_contrast"] = np.nan
        else:
            output["c2_mean_actual_m_difference"] = np.nan
            output["c2_mean_retained_p_contrast"] = np.nan
    for field in ("delta_flip", "delta_regret", "delta_e_flip", "delta_e_regret"):
        values = selected[field].to_numpy(dtype=np.float64)
        prefix = f"{match_type.lower()}_{field}"
        output[f"{prefix}_mean"] = float(values.mean()) if len(values) else np.nan
        output[f"{prefix}_median"] = float(np.median(values)) if len(values) else np.nan
        output[f"{prefix}_positive_fraction"] = float(np.mean(values > 0)) if len(values) else np.nan
        output[f"{prefix}_zero_fraction"] = float(np.mean(values == 0)) if len(values) else np.nan
        output[f"{prefix}_negative_fraction"] = float(np.mean(values < 0)) if len(values) else np.nan
    return output


def build_protein_summary(
    positions: pd.DataFrame,
    cells: pd.DataFrame,
    matches: pd.DataFrame,
    mechanism_associations: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Build the compact eight-protein evidence matrix."""
    rows: list[dict[str, Any]] = []
    mechanism_by_protein = mechanism_associations.set_index("protein_id")
    if not mechanism_by_protein.index.is_unique:
        raise H1AuditError("duplicate_protein_key", "Mechanism protein associations repeat")
    for protein_id in protein_order:
        protein_positions = positions.loc[positions["protein_id"] == protein_id]
        protein_cells = cells.loc[cells["protein_id"] == protein_id]
        row: dict[str, Any] = {
            "protein_id": protein_id,
            "position_count": len(protein_positions),
            "e_flip_mean": float(protein_positions["e_flip"].mean()),
            "e_flip_median": float(protein_positions["e_flip"].median()),
            "e_regret_mean": float(protein_positions["e_regret"].mean()),
            "e_regret_median": float(protein_positions["e_regret"].median()),
        }
        for outcome in ("y_flip_struct", "y_regret_struct"):
            p_gradients = _gradient_values(
                protein_cells,
                fixed_field="m_tertile",
                low_selector=("p_tertile", "LOW"),
                high_selector=("p_tertile", "HIGH"),
                outcome=outcome,
            )
            m_gradients = [
                -value
                for value in _gradient_values(
                    protein_cells,
                    fixed_field="p_tertile",
                    low_selector=("m_tertile", "LOW"),
                    high_selector=("m_tertile", "HIGH"),
                    outcome=outcome,
                )
            ]
            suffix = "flip" if outcome == "y_flip_struct" else "regret"
            _summarize_values(row, f"p_gradient_{suffix}", p_gradients)
            _summarize_values(row, f"m_gradient_{suffix}", m_gradients)
        for x_field in ("p", "m", "sdfi"):
            for y_field in ("e_flip", "e_regret"):
                for method in ("pearson", "spearman"):
                    value, reason = _correlation(
                        protein_positions[x_field], protein_positions[y_field], method
                    )
                    prefix = f"{x_field}_{y_field}_{method}"
                    row[prefix] = value
                    row[f"{prefix}_status"] = "defined" if value is not None else "undefined"
                    row[f"{prefix}_reason"] = reason
        for match_type in ("C1", "C2"):
            selected = matches.loc[
                (matches["protein_id"] == protein_id)
                & (matches["match_type"] == match_type)
            ]
            row.update(
                _match_diagnostics(
                    protein_positions, selected, match_type=match_type
                )
            )
        existing = mechanism_by_protein.loc[protein_id]
        row.update(
            {
                "existing_p_flip_positive": bool(
                    existing["perturbation_top1_disagreement_pearson"] > 0
                ),
                "existing_p_regret_positive": bool(
                    existing["perturbation_symmetric_regret_pearson"] > 0
                ),
                "existing_m_flip_protective": bool(
                    existing["margin_top1_disagreement_pearson"] < 0
                ),
                "existing_m_regret_protective": bool(
                    existing["margin_symmetric_regret_pearson"] < 0
                ),
                "existing_sdfi_top1_exceeds_p": bool(
                    abs(existing["ratio_top1_disagreement_pearson"])
                    >= abs(existing["perturbation_top1_disagreement_pearson"])
                    and abs(existing["ratio_top1_disagreement_spearman"])
                    >= abs(existing["perturbation_top1_disagreement_spearman"])
                ),
                "existing_sdfi_monotonic_regret_exceeds_p": bool(
                    abs(existing["ratio_symmetric_regret_spearman"])
                    >= abs(existing["perturbation_symmetric_regret_spearman"])
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def assess_h1_verdict(
    direction_evidence: dict[str, list[bool]], *, existing_evidence_supported: bool
) -> tuple[str, dict[str, Any]]:
    """Apply the frozen recurrence and leave-one-protein-out evidence gate."""
    evidence: dict[str, Any] = {}
    broad_all = True
    majority_contradiction = False
    for name, values in direction_evidence.items():
        if len(values) != EXPECTED_PROTEIN_COUNT:
            raise H1AuditError("evidence_count_mismatch", f"{name} does not contain eight proteins")
        support_count = sum(bool(value) for value in values)
        broad = support_count >= RECURRENCE_TARGET
        loo_counts = [support_count - int(bool(values[index])) for index in range(len(values))]
        loo_stable = broad and all(count >= LEAVE_ONE_OUT_TARGET for count in loo_counts)
        contradiction = len(values) - support_count >= RECURRENCE_TARGET
        evidence[name] = {
            "support_count": support_count,
            "contrary_or_nonpositive_count": len(values) - support_count,
            "broadly_preserved": broad,
            "leave_one_out_support_counts": loo_counts,
            "leave_one_out_stable": loo_stable,
        }
        broad_all = broad_all and broad and loo_stable
        majority_contradiction = majority_contradiction or contradiction
    if existing_evidence_supported and broad_all:
        verdict = "H1_STRONGLY_SUPPORTED_IN_STAGE0"
    elif not existing_evidence_supported or majority_contradiction:
        verdict = "H1_NOT_STABLE"
    else:
        verdict = "H1_PARTIALLY_SUPPORTED"
    return verdict, evidence


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H1AuditError("upstream_manifest_unreadable", f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise H1AuditError("upstream_manifest_schema_mismatch", f"{label} is not an object")
    return payload


def _validate_commit_history(project_root: Path) -> None:
    for commit in (MECHANISM_COMMIT, BOUNDARY_COMMIT):
        try:
            completed = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=project_root,
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise H1AuditError("git_history_unavailable", "Unable to validate Stage0 history") from exc
        if completed.returncode != 0:
            raise H1AuditError("stage0_history_mismatch", f"Required commit is not an ancestor: {commit}")


def _load_manifest(project_root: Path, logical_path: Path, expected_sha: str, label: str) -> tuple[Path, dict[str, Any]]:
    path = (project_root / logical_path).resolve()
    if not path.is_file():
        raise H1AuditError("upstream_manifest_missing", f"Missing {label}: {path}")
    if sha256_file(path) != expected_sha:
        raise H1AuditError("upstream_hash_mismatch", f"SHA mismatch for {label}")
    payload = _read_json(path, label=label)
    if payload.get("status") != "complete":
        raise H1AuditError("upstream_manifest_incomplete", f"{label} is not complete")
    return path, payload


def _load_parquet_record(
    project_root: Path,
    record: dict[str, Any],
    *,
    expected_rows: int,
    label: str,
) -> tuple[Path, pd.DataFrame]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise H1AuditError("upstream_manifest_schema_mismatch", f"Invalid record for {label}")
    path = (project_root / record["path"]).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise H1AuditError("upstream_path_escape", f"{label} escapes the project root") from exc
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise H1AuditError("upstream_hash_mismatch", f"Missing or changed {label}")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise H1AuditError("upstream_artifact_unreadable", f"Unable to parse {label}") from exc
    if len(frame) != expected_rows or record.get("rows") != expected_rows:
        raise H1AuditError("upstream_row_count_mismatch", f"Unexpected row count for {label}")
    return path, frame


def load_frozen_h1_inputs(project_root: Path) -> H1AuditInputs:
    """Validate the full frozen trust chain before scientific assembly."""
    root = project_root.resolve()
    _validate_commit_history(root)
    mechanism_path, mechanism_manifest = _load_manifest(
        root, MECHANISM_MANIFEST_PATH, MECHANISM_MANIFEST_SHA256, "mechanism manifest"
    )
    decision_path, decision_manifest = _load_manifest(
        root, DECISION_MANIFEST_PATH, DECISION_MANIFEST_SHA256, "decision manifest"
    )
    boundary_path, boundary_manifest = _load_manifest(
        root, BOUNDARY_MANIFEST_PATH, BOUNDARY_MANIFEST_SHA256, "boundary manifest"
    )
    if boundary_manifest.get("stage0_2b_status") != "HOLD_STAGE0_2B":
        raise H1AuditError("boundary_scope_mismatch", "Boundary release does not preserve HOLD_STAGE0_2B")
    protein_order = tuple(str(value) for value in mechanism_manifest.get("protein_order", []))
    if len(protein_order) != EXPECTED_PROTEIN_COUNT or tuple(
        boundary_manifest.get("protein_order", [])
    ) != protein_order:
        raise H1AuditError("protein_membership_mismatch", "Frozen manifests disagree on proteins")
    mechanism_position_path, mechanism_positions = _load_parquet_record(
        root,
        mechanism_manifest["outputs"]["position_mechanism"],
        expected_rows=EXPECTED_POSITION_COUNT,
        label="mechanism positions",
    )
    mechanism_association_path, mechanism_associations = _load_parquet_record(
        root,
        mechanism_manifest["outputs"]["protein_associations"],
        expected_rows=EXPECTED_PROTEIN_COUNT,
        label="mechanism associations",
    )
    decision_position_path, decision_positions = _load_parquet_record(
        root,
        decision_manifest["outputs"]["position_summary"],
        expected_rows=EXPECTED_POSITION_COUNT,
        label="decision positions",
    )
    technical_path, technical_null = _load_parquet_record(
        root,
        decision_manifest["outputs"]["technical_null"],
        expected_rows=EXPECTED_TECHNICAL_NULL_COUNT,
        label="local decision technical null",
    )
    sensitivity_record = mechanism_manifest["inputs"]["position_sensitivity"]
    sensitivity_path, position_sensitivity = _load_parquet_record(
        root,
        sensitivity_record,
        expected_rows=EXPECTED_POSITION_COUNT,
        label="position sensitivity",
    )
    provenance = {
        "mechanism_manifest": {"path": MECHANISM_MANIFEST_PATH.as_posix(), "sha256": MECHANISM_MANIFEST_SHA256},
        "decision_manifest": {"path": DECISION_MANIFEST_PATH.as_posix(), "sha256": DECISION_MANIFEST_SHA256},
        "boundary_manifest": {"path": BOUNDARY_MANIFEST_PATH.as_posix(), "sha256": BOUNDARY_MANIFEST_SHA256},
        "mechanism_positions": {**mechanism_manifest["outputs"]["position_mechanism"]},
        "mechanism_associations": {**mechanism_manifest["outputs"]["protein_associations"]},
        "decision_positions": {**decision_manifest["outputs"]["position_summary"]},
        "technical_null": {**decision_manifest["outputs"]["technical_null"]},
        "position_sensitivity": {**sensitivity_record},
    }
    del mechanism_path, decision_path, boundary_path
    del mechanism_position_path, mechanism_association_path, decision_position_path
    del technical_path, sensitivity_path
    return H1AuditInputs(
        project_root=root,
        protein_order=protein_order,
        mechanism_positions=mechanism_positions,
        mechanism_associations=mechanism_associations,
        decision_positions=decision_positions,
        technical_null=technical_null,
        position_sensitivity=position_sensitivity,
        input_provenance=provenance,
    )


def _validate_position_evidence(inputs: H1AuditInputs, positions: pd.DataFrame) -> None:
    keys = ["protein_id", "position"]
    expected_counts = tuple(
        int((positions["protein_id"] == protein).sum()) for protein in inputs.protein_order
    )
    if len(positions) != EXPECTED_POSITION_COUNT or expected_counts != EXPECTED_POSITION_COUNTS:
        raise H1AuditError("position_cohort_mismatch", "Frozen position counts differ")
    for label, frame in (
        ("decision positions", inputs.decision_positions),
        ("position sensitivity", inputs.position_sensitivity),
    ):
        _require_unique_positions(frame, label)
        joined = positions[keys].merge(frame[keys], on=keys, how="outer", indicator=True)
        if len(joined) != EXPECTED_POSITION_COUNT or not (joined["_merge"] == "both").all():
            raise H1AuditError("position_key_mismatch", f"{label} keys differ")
    mechanism = inputs.mechanism_positions.set_index(keys)
    decision = inputs.decision_positions.set_index(keys).reindex(mechanism.index)
    sensitivity = inputs.position_sensitivity.set_index(keys).reindex(mechanism.index)
    audit = positions.set_index(keys).reindex(mechanism.index)
    comparisons = (
        (mechanism["top1_disagreement_fraction"], decision["top1_disagreement_fraction"]),
        (mechanism["symmetric_regret_mean"], decision["symmetric_regret_mean"]),
        (mechanism["position_mean_abs_interaction"], sensitivity["position_mean_abs_interaction"]),
        (
            audit["y_flip_tech_pdb"],
            decision["pdb_same_state_pairwise_top1_disagreement_rate"],
        ),
        (
            audit["y_flip_tech_afdb"],
            decision["afdb_same_state_pairwise_top1_disagreement_rate"],
        ),
        (
            audit["y_regret_tech_pdb"],
            decision["pdb_same_state_pairwise_symmetric_regret_mean"],
        ),
        (
            audit["y_regret_tech_afdb"],
            decision["afdb_same_state_pairwise_symmetric_regret_mean"],
        ),
        (
            audit["e_flip"],
            decision["structural_minus_mean_same_state_top1_disagreement"],
        ),
        (
            audit["e_regret"],
            decision["structural_minus_mean_same_state_symmetric_regret"],
        ),
    )
    if any(
        not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0, atol=NUMERIC_TOLERANCE)
        for left, right in comparisons
    ):
        raise H1AuditError("position_evidence_mismatch", "Duplicated frozen evidence differs")


def _direction_matrix(proteins: pd.DataFrame) -> dict[str, list[bool]]:
    fields = {
        "audit_a_p_gradient_flip": "p_gradient_flip_mean",
        "audit_a_p_gradient_regret": "p_gradient_regret_mean",
        "audit_a_m_gradient_flip": "m_gradient_flip_mean",
        "audit_a_m_gradient_regret": "m_gradient_regret_mean",
        "audit_b_sdfi_e_flip_pearson": "sdfi_e_flip_pearson",
        "audit_b_sdfi_e_flip_spearman": "sdfi_e_flip_spearman",
        "audit_b_sdfi_e_regret_pearson": "sdfi_e_regret_pearson",
        "audit_b_sdfi_e_regret_spearman": "sdfi_e_regret_spearman",
        "audit_c_c1_flip": "c1_delta_flip_mean",
        "audit_c_c1_regret": "c1_delta_regret_mean",
        "audit_c_c1_e_flip": "c1_delta_e_flip_mean",
        "audit_c_c1_e_regret": "c1_delta_e_regret_mean",
        "audit_c_c2_flip": "c2_delta_flip_mean",
        "audit_c_c2_regret": "c2_delta_regret_mean",
        "audit_c_c2_e_flip": "c2_delta_e_flip_mean",
        "audit_c_c2_e_regret": "c2_delta_e_regret_mean",
    }
    return {
        name: [bool(np.isfinite(value) and value > 0) for value in proteins[field]]
        for name, field in fields.items()
    }


def _existing_evidence_supported(proteins: pd.DataFrame) -> tuple[bool, dict[str, int]]:
    fields = (
        "existing_p_flip_positive",
        "existing_p_regret_positive",
        "existing_m_flip_protective",
        "existing_m_regret_protective",
        "existing_sdfi_top1_exceeds_p",
        "existing_sdfi_monotonic_regret_exceeds_p",
    )
    counts = {field: int(proteins[field].astype(bool).sum()) for field in fields}
    return all(count >= RECURRENCE_TARGET for count in counts.values()), counts


def run_h1_confirmatory_audit(project_root: Path) -> H1ConfirmatoryResult:
    """Execute the canonical frozen-input confirmatory analysis."""
    inputs = load_frozen_h1_inputs(project_root)
    positions = build_position_audit(
        inputs.mechanism_positions,
        inputs.technical_null,
        protein_order=inputs.protein_order,
    )
    _validate_position_evidence(inputs, positions)
    cells = materialize_conditional_cells(positions, protein_order=inputs.protein_order)
    matches = pd.concat(
        [
            deterministic_greedy_matches(
                positions,
                match_type=match_type,
                protein_order=inputs.protein_order,
            )
            for match_type in ("C1", "C2")
        ],
        ignore_index=True,
    )
    proteins = build_protein_summary(
        positions,
        cells,
        matches,
        inputs.mechanism_associations,
        protein_order=inputs.protein_order,
    )
    direction_matrix = _direction_matrix(proteins)
    existing_supported, existing_counts = _existing_evidence_supported(proteins)
    verdict, evidence = assess_h1_verdict(
        direction_matrix, existing_evidence_supported=existing_supported
    )
    manifest_fields = {
        "schema_version": "stage0_h1_confirmatory_audit_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_h1_confirmatory_local_analysis_only",
        "required_history": {
            "mechanism_checkpoint": MECHANISM_COMMIT,
            "boundary_audit": BOUNDARY_COMMIT,
        },
        "inputs": inputs.input_provenance,
        "protein_order": list(inputs.protein_order),
        "analysis_protocol": {
            "identity": "stage0_h1_confirmatory_audit_v1",
            "primary_variables": {
                "P": "position_mean_abs_interaction",
                "M": "margin_mean_both",
                "SDFI": "P / M, algebraically reconciled to perturbation_to_margin",
                "Y_flip_struct": "top1_disagreement_fraction",
                "Y_regret_struct": "symmetric_regret_mean",
            },
            "rank_pct": "(average_rank_for_exact_ties - 0.5) / within_protein_position_count",
            "tertiles": {"LOW": "rank_pct < 1/3", "MID": "1/3 <= rank_pct < 2/3", "HIGH": "rank_pct >= 2/3"},
            "structural_excess": {
                "semantics": "descriptive structural-excess contrast",
                "E_flip": "Y_flip_struct - (Y_flip_tech_pdb + Y_flip_tech_afdb) / 2",
                "E_regret": "Y_regret_struct - (Y_regret_tech_pdb + Y_regret_tech_afdb) / 2",
                "negative_values_preserved": True,
                "causal_or_noise_corrected_effect": False,
            },
            "matching": {
                "caliper": MATCH_CALIPER,
                "distance_scale": "within_protein_empirical_percentile_rank",
                "algorithm": "deterministic_one_to_one_greedy_without_replacement",
                "edge_order": ["absolute_matching_rank_distance", "exposed_canonical_position", "comparator_canonical_position"],
                "outcome_informed": False,
                "C1": "LOW_M versus HIGH_M matched on p_rank_pct",
                "C2": "HIGH_P versus LOW_P matched on m_rank_pct",
            },
            "correlations": "within_protein descriptive Pearson and Spearman; null with structured reason when undefined",
            "hypothesis_testing": False,
            "p_values": False,
            "recurrence_target": RECURRENCE_TARGET,
            "recurrence_semantics": "descriptive cross-protein consistency gate, not a statistical threshold",
            "leave_one_out_target": LEAVE_ONE_OUT_TARGET,
        },
        "existing_evidence": {
            "supported": existing_supported,
            "direction_counts": existing_counts,
        },
        "h1_evidence_matrix_summary": evidence,
        "h1_verdict": verdict,
        "stage0_local_analysis_stop": True,
        "final_counts": {
            "position_audit_rows": len(positions),
            "conditional_cell_rows": len(cells),
            "matched_pair_rows": len(matches),
            "protein_summary_rows": len(proteins),
        },
        "outputs": {},
        "scope_declarations": {
            "proteinmpnn_executed": False,
            "new_proteins_added": False,
            "new_model_used": False,
            "new_data_acquired": False,
            "hypothesis_testing_performed": False,
            "p_values_computed": False,
            "binary_uq_labels_created": False,
            "stage0_2b_started": False,
            "arm_b_started": False,
            "evaluator_uq_started": False,
            "cdf_or_sdfi_predictor_trained": False,
            "publication_figures_created": False,
        },
    }
    return H1ConfirmatoryResult(
        project_root=inputs.project_root,
        protein_order=inputs.protein_order,
        input_provenance=inputs.input_provenance,
        position_audit=positions,
        cells=cells,
        matches=matches,
        protein_summary=proteins,
        manifest_fields=manifest_fields,
    )


def _rehash_inputs(result: H1ConfirmatoryResult) -> None:
    for label, record in result.input_provenance.items():
        path = result.project_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise H1AuditError("upstream_input_changed", f"Frozen input changed: {label}")


def _logical_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise H1AuditError("output_path_escape", f"Output is outside the project root: {path}") from exc


def _validate_h1_result(result: H1ConfirmatoryResult) -> None:
    """Validate the canonical structured result before any output is written."""
    positions = result.position_audit
    cells = result.cells
    matches = result.matches
    proteins = result.protein_summary
    if len(positions) != EXPECTED_POSITION_COUNT:
        raise H1AuditError("output_row_count_mismatch", "Position audit row count is invalid")
    if positions.duplicated(["protein_id", "position"]).any():
        raise H1AuditError("duplicate_output_key", "The position audit repeats a scientific key")
    observed_order = tuple(dict.fromkeys(positions["protein_id"].astype(str)))
    counts = tuple(
        int((positions["protein_id"] == protein).sum()) for protein in result.protein_order
    )
    if observed_order != result.protein_order or counts != EXPECTED_POSITION_COUNTS:
        raise H1AuditError("output_cohort_mismatch", "The position audit cohort differs")
    numeric = positions[["p", "m", "sdfi", "e_flip", "e_regret"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not (positions["m"] > 0).all():
        raise H1AuditError("invalid_output_value", "The position audit has invalid values")
    if not np.allclose(
        positions["sdfi"], positions["p"] / positions["m"], rtol=0, atol=NUMERIC_TOLERANCE
    ):
        raise H1AuditError("sdfi_algebra_mismatch", "Rendered SDFI differs from P divided by M")
    if len(cells) != EXPECTED_PROTEIN_COUNT * len(TERTILES) ** 2:
        raise H1AuditError("output_row_count_mismatch", "Conditional cell row count is invalid")
    if cells.duplicated(["protein_id", "p_tertile", "m_tertile"]).any():
        raise H1AuditError("duplicate_output_key", "Conditional cells repeat a scientific key")
    if int(cells["n_positions"].sum()) != EXPECTED_POSITION_COUNT:
        raise H1AuditError("cell_count_mismatch", "Conditional cells do not partition all positions")
    expected_cells = materialize_conditional_cells(
        positions, protein_order=result.protein_order
    )
    try:
        pd.testing.assert_frame_equal(cells, expected_cells, check_exact=True)
    except AssertionError as exc:
        raise H1AuditError(
            "conditional_cell_derivation_mismatch",
            "Conditional cells are not derived from the canonical position audit",
        ) from exc
    match_required = {
        "protein_id",
        "match_type",
        "exposed_position",
        "comparator_position",
        "matching_rank_distance",
        "delta_flip",
        "delta_regret",
        "delta_e_flip",
        "delta_e_regret",
    }
    _require_columns(matches, match_required, "matching audit")
    if set(matches["match_type"].astype(str)) != {"C1", "C2"}:
        raise H1AuditError("matching_type_mismatch", "Both frozen matching directions are required")
    distances = matches["matching_rank_distance"].to_numpy(float)
    if not np.isfinite(distances).all() or (distances < 0).any() or (
        distances > MATCH_CALIPER
    ).any():
        raise H1AuditError("matching_caliper_mismatch", "A matching distance exceeds 0.10")
    if matches.duplicated(["protein_id", "match_type", "exposed_position"]).any() or matches.duplicated(
        ["protein_id", "match_type", "comparator_position"]
    ).any():
        raise H1AuditError("matching_reuse", "Matching is not one-to-one within a protein")
    expected_matches = pd.concat(
        [
            deterministic_greedy_matches(
                positions,
                match_type=match_type,
                protein_order=result.protein_order,
            )
            for match_type in ("C1", "C2")
        ],
        ignore_index=True,
    )
    try:
        pd.testing.assert_frame_equal(matches, expected_matches, check_exact=True)
    except AssertionError as exc:
        raise H1AuditError(
            "matching_derivation_mismatch",
            "Matching output is not the frozen deterministic derivation",
        ) from exc
    if len(proteins) != EXPECTED_PROTEIN_COUNT or tuple(
        proteins["protein_id"].astype(str)
    ) != result.protein_order:
        raise H1AuditError("protein_summary_cohort_mismatch", "Protein summary order differs")
    for column in proteins.columns:
        if column.endswith(("_pearson", "_spearman")):
            finite = proteins[column].dropna().to_numpy(float)
            if not np.isfinite(finite).all() or (np.abs(finite) > 1).any():
                raise H1AuditError(
                    "invalid_descriptive_correlation",
                    f"Protein summary correlation is invalid: {column}",
                )
    all_columns = [
        column.lower()
        for frame in (positions, cells, matches, proteins)
        for column in frame.columns
    ]
    if any("pvalue" in column or "p_value" in column or "fdr" in column for column in all_columns):
        raise H1AuditError("forbidden_inference_field", "Inferential fields are forbidden")
    manifest = result.manifest_fields
    if manifest.get("h1_verdict") not in {
        "H1_STRONGLY_SUPPORTED_IN_STAGE0",
        "H1_PARTIALLY_SUPPORTED",
        "H1_NOT_STABLE",
    } or manifest.get("stage0_local_analysis_stop") is not True:
        raise H1AuditError("invalid_h1_terminal_state", "H1 verdict or stop state is invalid")


def materialize_h1_confirmatory_audit(
    result: H1ConfirmatoryResult, output_root: Path
) -> dict[str, Any]:
    """Render four immutable Parquet outputs and the manifest last."""
    _validate_h1_result(result)
    _rehash_inputs(result)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frames = {
        "position_audit": result.position_audit,
        "conditional_cells": result.cells,
        "matches": result.matches,
        "protein_summary": result.protein_summary,
    }
    paths = {
        "position_audit": output_root / "h1_confirmatory_position_audit.parquet",
        "conditional_cells": output_root / "h1_confirmatory_cells.parquet",
        "matches": output_root / "h1_confirmatory_matches.parquet",
        "protein_summary": output_root / "h1_confirmatory_protein_summary.parquet",
    }
    try:
        write_status = {
            label: write_immutable_parquet(paths[label], frames[label]) for label in paths
        }
    except FixedProbeSensitivityError as exc:
        raise H1AuditError(exc.code, str(exc), outcome=exc.outcome) from exc
    _rehash_inputs(result)
    manifest = dict(result.manifest_fields)
    manifest["outputs"] = {
        label: {
            "path": _logical_path(paths[label], result.project_root),
            "sha256": sha256_file(paths[label]),
            "rows": len(frames[label]),
            "bytes": paths[label].stat().st_size,
        }
        for label in paths
    }
    manifest_path = output_root / "h1_confirmatory_audit_manifest.json"
    try:
        write_status["manifest"] = write_immutable_json(manifest_path, manifest)
    except FixedProbeSensitivityError as exc:
        raise H1AuditError(exc.code, str(exc), outcome=exc.outcome) from exc
    _rehash_inputs(result)
    return {
        "status": "PASS",
        "h1_verdict": manifest["h1_verdict"],
        "manifest_path": manifest_path,
        "write_status": write_status,
        "stage0_local_analysis_stop": True,
    }


__all__ = [
    "H1AuditError",
    "H1ConfirmatoryResult",
    "assess_h1_verdict",
    "assign_frozen_tertile",
    "build_position_audit",
    "deterministic_greedy_matches",
    "load_frozen_h1_inputs",
    "materialize_conditional_cells",
    "materialize_h1_confirmatory_audit",
    "run_h1_confirmatory_audit",
    "within_protein_rank_pct",
]
