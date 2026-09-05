"""Orchestration for StructCal cross-model representation sensitivity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.evaluation.cross_model_representation_sensitivity import build_response_rows

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "proteinmpnn": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L2",
        "probe_semantics": "fixed_order_native_autoregressive_context",
    },
    "esm_if1": {
        "atoms": ("N", "CA", "C"),
        "semantic_class": "L2",
        "probe_semantics": "teacher_forced_native_autoregressive_context",
    },
    "pifold": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L0",
        "probe_semantics": "geometry_only",
    },
    "dynamicmpnn": {
        "atoms": ("N", "CA", "C"),
        "semantic_class": "L2",
        "probe_semantics": "teacher_forced_natural_order_native_prefix",
    },
    "kwdesign": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L0_composite",
        "probe_semantics": "geometry_only_seeded_msa",
    },
}


def filter_dynamicmpnn_contiguous_cases(
    cases: pd.DataFrame, *, regime: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep whole pairs satisfying DynamicMPNN's native contiguous-axis contract."""

    excluded: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    for pair_id, group in cases.groupby("pair_id", sort=True):
        positions = tuple(sorted(group["canonical_position"].astype(int).unique()))
        if positions != tuple(range(positions[0], positions[-1] + 1)):
            excluded_ids.add(str(pair_id))
            excluded.append(
                {
                    "model": "dynamicmpnn",
                    "protein_id": str(group.iloc[0]["protein_id"]),
                    "pair_id": str(pair_id),
                    "structural_regime": regime,
                    "status": "DATA_UNRESOLVED",
                    "reason": "NONCONTIGUOUS_CANONICAL_AXIS",
                    "detail": "DynamicMPNN requires a complete contiguous canonical residue axis; the pair was not split or imputed.",
                }
            )
    evaluable = cases.loc[~cases["pair_id"].astype(str).isin(excluded_ids)].copy()
    return evaluable, pd.DataFrame(excluded)


def safe_output_path(output_root: Path, relative: str | Path) -> Path:
    """Resolve a task output while preventing writes outside its run root."""

    root = Path(output_root).resolve()
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError("output path would escape the task run root")
    return result


def response_shard_path(
    output_root: Path, *, model: str, checkpoint: str, regime: str
) -> Path:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown cross-model probe: {model}")
    if regime not in {"identical", "exact_se3", "controlled", "operational_pdb_afdb"}:
        raise ValueError(f"unknown representation regime: {regime}")
    return safe_output_path(output_root, Path("shards") / model / checkpoint / regime)


def _coordinates(group: pd.DataFrame) -> np.ndarray:
    for value in group.sort_values("canonical_position", kind="mergesort")["coordinates"]:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            result = np.asarray(
                [
                    [np.asarray(atom, dtype=np.float32) for atom in residue]
                    for residue in value
                ],
                dtype=np.float32,
            )
            if result.ndim != 3 or result.shape[-1] != 3 or not np.isfinite(result).all():
                raise ValueError("case coordinates must be finite with shape [L, atoms, 3]")
            return result
    raise ValueError("case condition has no coordinates")


def score_cases_with_callback(
    cases: pd.DataFrame,
    scorer: Callable[..., np.ndarray],
    *,
    model_id: str,
    checkpoint_id: str,
    semantic_class: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every paired view independently, then apply shared metrics."""

    distributions: dict[tuple[str, str], np.ndarray] = {}
    for (pair_id, condition), group in cases.groupby(["pair_id", "condition"], sort=True):
        ordered = group.sort_values("canonical_position", kind="mergesort")
        positions = tuple(int(value) for value in ordered["canonical_position"])
        sequences = ordered["wt_sequence_projection"].astype(str).drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"case sequence is not constant: {pair_id}/{condition}")
        distributions[(str(pair_id), str(condition))] = scorer(
            protein_id=str(ordered.iloc[0]["protein_id"]),
            pair_id=str(pair_id),
            condition=str(condition),
            sequence=str(sequences.iloc[0]),
            coordinates=_coordinates(ordered),
            positions=positions,
        )
    return build_response_rows(
        cases,
        distributions,
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        semantic_class=semantic_class,
        regime=regime,
    )
