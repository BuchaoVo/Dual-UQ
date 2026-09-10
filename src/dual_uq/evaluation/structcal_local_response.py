"""Shared local-response equations and output contracts for StructCal v1.

The module is deliberately model-agnostic.  Model adapters produce native
20-amino-acid probability matrices; this module validates those matrices,
computes per-position Jensen--Shannon divergence, and aggregates only within a
frozen structural pair.  It does not load structures, select cohorts, or run a
model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
LOCAL_RESPONSE_RESIDUE_COLUMNS = (
    "model_id",
    "pair_id",
    "protein_id",
    "identity_cluster_id",
    "split",
    "track_or_diagnostic",
    "state_family",
    "canonical_position",
    "local_semantics_class",
    "probe_semantics",
    "js_bits",
    "condition_1_evaluable",
    "condition_2_evaluable",
)
LOCAL_RESPONSE_PAIR_COLUMNS = (
    "model_id",
    "pair_id",
    "protein_id",
    "identity_cluster_id",
    "split",
    "track_or_diagnostic",
    "state_family",
    "n_canonical_positions",
    "n_evaluable_positions",
    "coverage_fraction",
    "R_local_bits",
)


class StructCalLocalResponseError(ValueError):
    """Raised for a local-response schema, probability, or invariant failure."""


def validate_probability_matrix(value: Any, *, label: str) -> np.ndarray:
    """Validate and return a native standard-20-AA probability matrix."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise StructCalLocalResponseError(f"{label} is not numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(STANDARD_AMINO_ACIDS):
        raise StructCalLocalResponseError(f"{label} must have shape [L,20]")
    if matrix.shape[0] == 0:
        raise StructCalLocalResponseError(f"{label} must contain at least one residue")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise StructCalLocalResponseError(f"{label} must be finite and non-negative")
    row_sums = matrix.sum(axis=1)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise StructCalLocalResponseError(f"{label} rows must have positive mass")
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise StructCalLocalResponseError(f"{label} must be normalized")
    return matrix


def js_bits(left: Sequence[float], right: Sequence[float]) -> float:
    """Return symmetric Jensen--Shannon divergence in base-2 bits."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape or p.size == 0:
        raise StructCalLocalResponseError("JS inputs must be equal non-empty vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise StructCalLocalResponseError("JS inputs must be finite and non-negative")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        raise StructCalLocalResponseError("JS inputs must have positive mass")
    p = p / p_sum
    q = q / q_sum
    midpoint = 0.5 * (p + q)
    result = 0.0
    for distribution in (p, q):
        positive = distribution > 0
        result += float(
            0.5
            * np.sum(distribution[positive] * np.log2(distribution[positive] / midpoint[positive]))
        )
    if not np.isfinite(result) or result < -1e-12:
        raise StructCalLocalResponseError("JS divergence is invalid")
    return max(0.0, result)


def _metadata_value(group: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in group.columns:
        return default
    values = group[column].drop_duplicates()
    if len(values) > 1:
        raise StructCalLocalResponseError(f"pair metadata column is not constant: {column}")
    return values.iloc[0] if len(values) else default


def _required_case_columns(cases: pd.DataFrame) -> None:
    required = {"protein_id", "pair_id", "condition", "canonical_position"}
    missing = sorted(required.difference(cases.columns))
    if missing:
        raise StructCalLocalResponseError(f"local-response cases missing columns: {missing}")
    if cases.empty:
        raise StructCalLocalResponseError("local-response cases must not be empty")
    if cases.duplicated(["pair_id", "condition", "canonical_position"]).any():
        raise StructCalLocalResponseError("local-response cases contain duplicate positions")


def build_local_response_rows(
    cases: pd.DataFrame,
    distributions: Mapping[tuple[str, str], np.ndarray],
    *,
    model_id: str,
    semantic_class: str,
    probe_semantics: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build frozen residue and pair response rows from two condition matrices.

    ``cases`` contains one row per pair/condition/canonical position.  The
    distribution mapping contains one normalized ``[L,20]`` matrix per pair
    and condition, in the exact sorted canonical-position order of ``cases``.
    """

    _required_case_columns(cases)
    if not model_id or semantic_class not in {"L0", "L1", "L2"} or not probe_semantics:
        raise StructCalLocalResponseError("model response metadata is invalid")
    residue_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair_id, pair_group in cases.groupby("pair_id", sort=True):
        conditions = tuple(sorted(pair_group["condition"].astype(str).unique()))
        if len(conditions) != 2:
            raise StructCalLocalResponseError(f"pair must contain exactly two conditions: {pair_id}")
        condition_a, condition_b = conditions
        axes: dict[str, tuple[int, ...]] = {}
        matrices: dict[str, np.ndarray] = {}
        for condition in conditions:
            condition_rows = pair_group.loc[pair_group["condition"].astype(str).eq(condition)]
            ordered = condition_rows.sort_values("canonical_position", kind="mergesort")
            axes[condition] = tuple(int(value) for value in ordered["canonical_position"])
            key = (str(pair_id), condition)
            if key not in distributions:
                raise StructCalLocalResponseError(f"missing probability matrix: {key}")
            matrix = validate_probability_matrix(distributions[key], label=f"{key}")
            if matrix.shape[0] != len(axes[condition]):
                raise StructCalLocalResponseError(f"probability length differs from case axis: {key}")
            matrices[condition] = matrix
        if axes[condition_a] != axes[condition_b]:
            raise StructCalLocalResponseError(f"condition axes differ for pair: {pair_id}")

        metadata = {
            "model_id": model_id,
            "pair_id": str(pair_id),
            "protein_id": str(_metadata_value(pair_group, "protein_id", "")),
            "identity_cluster_id": _metadata_value(pair_group, "identity_cluster_id"),
            "split": _metadata_value(pair_group, "split"),
            "track_or_diagnostic": _metadata_value(pair_group, "track_or_diagnostic"),
            "state_family": _metadata_value(pair_group, "state_family"),
        }
        canonical_count = _metadata_value(pair_group, "n_canonical_positions", len(axes[condition_a]))
        try:
            canonical_count = int(canonical_count)
        except (TypeError, ValueError) as exc:
            raise StructCalLocalResponseError(f"invalid canonical position count: {pair_id}") from exc
        if canonical_count < len(axes[condition_a]) or canonical_count <= 0:
            raise StructCalLocalResponseError(f"canonical position count is inconsistent: {pair_id}")
        values = np.asarray(
            [js_bits(matrices[condition_a][i], matrices[condition_b][i]) for i in range(len(axes[condition_a]))],
            dtype=np.float64,
        )
        for position, value in zip(axes[condition_a], values, strict=True):
            residue_rows.append(
                {
                    **metadata,
                    "canonical_position": int(position),
                    "local_semantics_class": semantic_class,
                    "probe_semantics": probe_semantics,
                    "js_bits": float(value),
                    "condition_1_evaluable": True,
                    "condition_2_evaluable": True,
                }
            )
        pair_rows.append(
            {
                **metadata,
                "n_canonical_positions": canonical_count,
                "n_evaluable_positions": len(values),
                "coverage_fraction": float(len(values) / canonical_count),
                "R_local_bits": float(values.mean()),
            }
        )
    residue = pd.DataFrame(residue_rows, columns=LOCAL_RESPONSE_RESIDUE_COLUMNS)
    pair = pd.DataFrame(pair_rows, columns=LOCAL_RESPONSE_PAIR_COLUMNS)
    return residue.sort_values(["pair_id", "canonical_position"], kind="mergesort").reset_index(drop=True), pair.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
