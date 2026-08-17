"""Deterministic identity-disjoint split construction for paired-state data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


class SplitPrerequisiteError(ValueError):
    """Raised when split inputs cannot satisfy the frozen identity contract."""


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Structured result for a cluster-preserving split attempt."""

    assignments: pd.DataFrame
    status: str
    source_metadata: dict[str, Any]
    summary: dict[str, Any]


_SPLITS = ("TRAIN", "VALIDATION", "LOCKED_TEST")
_TARGET_FRACTIONS = {"TRAIN": 0.70, "VALIDATION": 0.15, "LOCKED_TEST": 0.15}


def _require_columns(table: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise SplitPrerequisiteError(f"{name} is missing required columns: {missing}")


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _unresolved_result(candidates: pd.DataFrame, *, threshold: float, seed: int, reason: str) -> SplitResult:
    return SplitResult(
        assignments=pd.DataFrame(),
        status="SPLIT_PREREQUISITE_UNRESOLVED",
        source_metadata={"cluster_source": None, "identity_threshold": threshold, "seed": seed},
        summary={
            "status": "SPLIT_PREREQUISITE_UNRESOLVED",
            "reason": reason,
            "candidate_rows": int(len(candidates)),
            "candidate_protein_count": int(candidates["protein_id"].nunique()),
        },
    )


def build_identity_disjoint_split(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    threshold: float,
    seed: int,
) -> SplitResult:
    """Assign complete authoritative sequence-identity clusters to partitions.

    No sequence clustering is performed here.  An empty or unresolved cluster
    source produces a structured unresolved result instead of a local fallback.
    """

    _require_columns(candidates, ("protein_id", "pair_id"), "candidates")
    if candidates.empty:
        raise SplitPrerequisiteError("candidates is empty")
    candidate_rows = candidates.copy()
    candidate_rows["protein_id"] = candidate_rows["protein_id"].astype(str)
    candidate_rows["pair_id"] = candidate_rows["pair_id"].astype(str)
    if candidate_rows["pair_id"].duplicated().any():
        # Multiple state rows are valid; only the pair's identity must remain atomic.
        pass
    if clusters.empty:
        return _unresolved_result(candidate_rows, threshold=threshold, seed=seed, reason="authoritative cluster table is empty")

    _require_columns(clusters, ("protein_id",), "clusters")
    cluster_rows = clusters.copy()
    if "sequence_cluster_id" not in cluster_rows.columns and "cluster_id" in cluster_rows.columns:
        cluster_rows = cluster_rows.rename(columns={"cluster_id": "sequence_cluster_id"})
    _require_columns(cluster_rows, ("sequence_cluster_id",), "clusters")
    cluster_rows["protein_id"] = cluster_rows["protein_id"].astype(str)
    if cluster_rows["protein_id"].duplicated().any():
        raise SplitPrerequisiteError("clusters has duplicate protein_id assignments")
    if "identity_threshold" in cluster_rows:
        values = pd.to_numeric(cluster_rows["identity_threshold"], errors="coerce")
        if values.notna().any() and not values.dropna().eq(float(threshold)).all():
            raise SplitPrerequisiteError("clusters identity_threshold does not match requested threshold")
    cluster_values = cluster_rows["sequence_cluster_id"]
    if cluster_values.isna().any() or cluster_values.astype(str).str.strip().eq("").any():
        return _unresolved_result(
            candidate_rows,
            threshold=threshold,
            seed=seed,
            reason="candidate cluster assignments contain unresolved identifiers",
        )
    cluster_rows["sequence_cluster_id"] = cluster_values.astype(str)
    candidate_proteins = set(candidate_rows["protein_id"])
    available_proteins = set(cluster_rows["protein_id"])
    missing = sorted(candidate_proteins.difference(available_proteins))
    if missing:
        raise SplitPrerequisiteError(f"missing cluster assignments for proteins: {missing[:5]}")
    cluster_rows = cluster_rows.loc[cluster_rows["protein_id"].isin(candidate_proteins)].copy()
    source_values = cluster_rows.get("cluster_source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    source_metadata = {
        "cluster_source": source_values[0] if len(source_values) == 1 else source_values,
        "identity_threshold": float(threshold),
        "seed": int(seed),
    }
    assignments = candidate_rows.merge(
        cluster_rows[["protein_id", "sequence_cluster_id"]], on="protein_id", how="left", validate="many_to_one"
    )
    if assignments["sequence_cluster_id"].isna().any():
        raise SplitPrerequisiteError("missing cluster assignments after candidate join")
    pair_cluster_counts = assignments.groupby("pair_id")["sequence_cluster_id"].nunique()
    if (pair_cluster_counts > 1).any():
        raise SplitPrerequisiteError("a pair_id maps to multiple sequence clusters")

    cluster_sizes = assignments.groupby("sequence_cluster_id", sort=False).size().to_dict()
    ordered_clusters = sorted(
        cluster_sizes,
        key=lambda value: (-int(cluster_sizes[value]), _stable_digest(seed, str(value))),
    )
    target_rows = {name: len(assignments) * fraction for name, fraction in _TARGET_FRACTIONS.items()}
    current_rows = {name: 0 for name in _SPLITS}
    cluster_split: dict[str, str] = {}
    for cluster_id in ordered_clusters:
        size = int(cluster_sizes[cluster_id])
        selected = min(
            _SPLITS,
            key=lambda name: (
                current_rows[name] / max(target_rows[name], 1.0),
                _SPLITS.index(name),
            ),
        )
        cluster_split[str(cluster_id)] = selected
        current_rows[selected] += size
    assignments["split"] = assignments["sequence_cluster_id"].map(cluster_split)
    assignments["identity_threshold"] = float(threshold)
    assignments["cluster_source"] = source_metadata["cluster_source"]
    assignments = assignments.sort_values(["split", "protein_id", "pair_id"], kind="mergesort").reset_index(drop=True)
    result = SplitResult(
        assignments=assignments,
        status="READY",
        source_metadata=source_metadata,
        summary={
            "status": "READY",
            "candidate_rows": int(len(assignments)),
            "candidate_protein_count": int(assignments["protein_id"].nunique()),
            "cluster_count": int(assignments["sequence_cluster_id"].nunique()),
            "split_rows": {str(key): int(value) for key, value in assignments["split"].value_counts().to_dict().items()},
            "split_protein_counts": {
                str(key): int(value) for key, value in assignments.groupby("split")["protein_id"].nunique().to_dict().items()
            },
        },
    )
    validate_identity_disjoint_split(result)
    return result


def validate_identity_disjoint_split(result: SplitResult) -> None:
    """Validate atomicity and disjointness of a materialized split result."""

    if result.status == "SPLIT_PREREQUISITE_UNRESOLVED":
        if not result.assignments.empty:
            raise SplitPrerequisiteError("unresolved split must not contain assignments")
        return
    if result.status != "READY":
        raise SplitPrerequisiteError(f"unknown split status: {result.status}")
    _require_columns(result.assignments, ("protein_id", "pair_id", "sequence_cluster_id", "split"), "assignments")
    if result.assignments.empty:
        raise SplitPrerequisiteError("ready split has no assignments")
    if not result.assignments["split"].isin(_SPLITS).all():
        raise SplitPrerequisiteError("assignments contain an unknown split")
    pair_counts = result.assignments.groupby("pair_id")["split"].nunique()
    if (pair_counts > 1).any():
        raise SplitPrerequisiteError("pair states are split across partitions")
    cluster_counts = result.assignments.groupby("sequence_cluster_id")["split"].nunique()
    if (cluster_counts > 1).any():
        raise SplitPrerequisiteError("identity clusters are split across partitions")
