"""Case transformations for cross-model representation-sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.models.dynamicmpnn import DynamicMPNNInputCase, validate_dynamic_input

from .structcal_local_response_cases import (
    build_structcal_local_cases,
    load_structcal_local_cohort,
)

EXACT_SE3_ROTATION = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
EXACT_SE3_TRANSLATION = np.asarray([10.0, -7.0, 3.0], dtype=np.float64)


def build_controlled_cross_model_cases(
    project_root: Path,
    cohort: pd.DataFrame,
    *,
    atom_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project controlled pairs using mutually consistent structural identities."""

    cases, exclusions = build_structcal_local_cases(
        project_root,
        cohort,
        atom_names=atom_names,
        allow_controlled_pair_consensus=True,
    )
    release = Path(project_root) / "artifacts/releases/structcal_v1"
    mappings = pd.read_parquet(release / "core/residue_mappings.parquet")
    mappings = mappings.loc[mappings["pair_id"].astype(str).isin(cohort["pair_id"].astype(str))]
    metadata: list[dict[str, Any]] = []
    for pair_id, group in mappings.groupby("pair_id", sort=True):
        mismatches = group.loc[
            group["condition_1_aa"].astype(str).str.upper()
            .ne(group["canonical_aa"].astype(str).str.upper())
            | group["condition_2_aa"].astype(str).str.upper()
            .ne(group["canonical_aa"].astype(str).str.upper())
        ].sort_values("canonical_position", kind="mergesort")
        details = [
            {
                "canonical_position": int(row.canonical_position),
                "canonical_aa": str(row.canonical_aa).upper(),
                "condition_1_aa": str(row.condition_1_aa).upper(),
                "condition_2_aa": str(row.condition_2_aa).upper(),
            }
            for row in mismatches.itertuples(index=False)
        ]
        metadata.append(
            {
                "pair_id": str(pair_id),
                "canonical_identity_mismatch_count": len(details),
                "canonical_identity_mismatch_positions_json": json.dumps(
                    [row["canonical_position"] for row in details], separators=(",", ":")
                ),
                "canonical_identity_mismatch_details_json": json.dumps(
                    details, sort_keys=True, separators=(",", ":")
                ),
                "sequence_context_basis": "CANONICAL_SEQUENCE",
            }
        )
    provenance = pd.DataFrame(metadata)
    if not cases.empty:
        cases = cases.merge(provenance, on="pair_id", how="left", validate="many_to_one")
    return cases, exclusions


def load_full_structcal_cohort(project_root: Any, *, cohort: str) -> pd.DataFrame:
    """Concatenate a frozen StructCal cohort without changing split membership."""

    frames = [
        load_structcal_local_cohort(project_root, cohort=cohort, split=split)  # type: ignore[arg-type]
        for split in ("TRAIN", "VALIDATION", "LOCKED_TEST")
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["split", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)


def _coordinates(value: Any, *, atoms: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[-1] != 3 or not len(result):
        raise ValueError("coordinates must have shape [L, atoms, 3]")
    if atoms is not None and result.shape[1] != atoms:
        raise ValueError(f"coordinates must contain exactly {atoms} atoms per residue")
    if not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all():
        raise ValueError("coordinates must be finite floating-point values")
    return result


def apply_exact_se3(coordinates: Any) -> np.ndarray:
    """Apply the fixed exactly representable rigid transform from the protocol."""

    source = _coordinates(coordinates)
    transformed = source.astype(np.float64) @ EXACT_SE3_ROTATION.T + EXACT_SE3_TRANSLATION
    return transformed.astype(source.dtype)


def build_exact_se3_cases(paired_cases: pd.DataFrame) -> pd.DataFrame:
    """Create one deterministic original/rigid pair per protein."""

    required = {"pair_id", "protein_id", "condition", "condition_label", "coordinates"}
    missing = sorted(required.difference(paired_cases.columns))
    if missing or paired_cases.empty:
        raise ValueError(f"exact-SE3 source cases missing columns: {missing}")
    first_pairs = (
        paired_cases[["protein_id", "pair_id"]]
        .drop_duplicates()
        .sort_values(["protein_id", "pair_id"], kind="mergesort")
        .drop_duplicates("protein_id", keep="first")
    )
    rows: list[pd.DataFrame] = []
    for protein_id, source_pair_id in first_pairs.itertuples(index=False, name=None):
        source = paired_cases.loc[
            paired_cases["pair_id"].astype(str).eq(str(source_pair_id))
            & paired_cases["condition"].astype(str).eq("CONDITION_1")
        ].copy()
        if source.empty:
            raise ValueError(f"exact-SE3 source lacks CONDITION_1: {source_pair_id}")
        pair_id = f"exact_se3::{protein_id}"
        original = source.copy()
        original["pair_id"] = pair_id
        original["condition_label"] = "ORIGINAL"
        transformed = source.copy()
        transformed["pair_id"] = pair_id
        transformed["condition"] = "CONDITION_2"
        transformed["condition_label"] = "RIGID_TRANSFORM"
        transformed["coordinates"] = transformed["coordinates"].map(
            lambda value: None
            if value is None or (isinstance(value, float) and np.isnan(value))
            else apply_exact_se3(np.asarray(value)).tolist()
        )
        for column in ("perturbation_family", "requested_dose", "requested_dose_unit"):
            if column in original:
                original[column] = None
                transformed[column] = None
        original["track_or_diagnostic"] = "EXACT_SE3_CONTROL"
        transformed["track_or_diagnostic"] = "EXACT_SE3_CONTROL"
        rows.extend((original, transformed))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["protein_id", "pair_id", "condition", "canonical_position"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_identical_input_cases(paired_cases: pd.DataFrame) -> pd.DataFrame:
    """Create two independently scored copies of one reference view per protein."""

    required = {"pair_id", "protein_id", "condition", "condition_label", "coordinates"}
    missing = sorted(required.difference(paired_cases.columns))
    if missing or paired_cases.empty:
        raise ValueError(f"identical-input source cases missing columns: {missing}")
    first_pairs = (
        paired_cases[["protein_id", "pair_id"]]
        .drop_duplicates()
        .sort_values(["protein_id", "pair_id"], kind="mergesort")
        .drop_duplicates("protein_id", keep="first")
    )
    rows: list[pd.DataFrame] = []
    for protein_id, source_pair_id in first_pairs.itertuples(index=False, name=None):
        source = paired_cases.loc[
            paired_cases["pair_id"].astype(str).eq(str(source_pair_id))
            & paired_cases["condition"].astype(str).eq("CONDITION_1")
        ].copy()
        if source.empty:
            raise ValueError(f"identical-input source lacks CONDITION_1: {source_pair_id}")
        pair_id = f"identical::{protein_id}"
        original = source.copy()
        original["pair_id"] = pair_id
        original["condition_label"] = "ORIGINAL"
        repeated = source.copy()
        repeated["pair_id"] = pair_id
        repeated["condition"] = "CONDITION_2"
        repeated["condition_label"] = "IDENTICAL_REPEAT"
        for column in ("perturbation_family", "requested_dose", "requested_dose_unit"):
            if column in original:
                original[column] = None
                repeated[column] = None
        original["track_or_diagnostic"] = "IDENTICAL_INPUT_CONTROL"
        repeated["track_or_diagnostic"] = "IDENTICAL_INPUT_CONTROL"
        rows.extend((original, repeated))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["protein_id", "pair_id", "condition", "canonical_position"],
        kind="mergesort",
    ).reset_index(drop=True)


def make_degenerate_dynamic_case(
    *,
    protein_id: str,
    pair_id: str,
    canonical_positions: tuple[int, ...],
    sequence: str,
    coordinates: Any,
    chain_id: str,
) -> DynamicMPNNInputCase:
    """Represent one structural view independently as DynamicMPNN ``(X, X)``."""

    source = np.ascontiguousarray(_coordinates(coordinates, atoms=3))
    digest = hashlib.sha256(source.tobytes()).hexdigest()
    case = DynamicMPNNInputCase(
        protein_id=protein_id,
        pair_id=pair_id,
        canonical_positions=canonical_positions,
        sequences=(sequence, sequence),
        coordinates=np.stack((source, source), axis=1),
        coordinate_present=np.ones((len(source), 2), dtype=bool),
        chain_ids=(chain_id, chain_id),
        structure_sha256=(digest, digest),
    )
    validate_dynamic_input(case)
    return case
