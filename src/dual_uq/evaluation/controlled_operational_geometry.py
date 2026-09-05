"""Structural-distribution comparison for Controlled and Operational pairs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pair_conditioned_local_mechanism import LOCAL_DESCRIPTORS

PAIR_FEATURES = tuple(
    f"{descriptor}_{statistic}"
    for descriptor in LOCAL_DESCRIPTORS
    for statistic in ("median", "q75", "q90")
) + ("aligned_ca_rmsd",)


def _require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{label} is missing required data: {missing}")


def pair_geometry_summaries(local_geometry: pd.DataFrame, *, regime: str) -> pd.DataFrame:
    """Reduce residue geometry to fixed pair-level median/q75/q90 summaries."""

    required = (
        "pair_id",
        "protein_id",
        "identity_cluster_id",
        "canonical_position",
        *LOCAL_DESCRIPTORS,
    )
    _require(local_geometry, required, "local geometry")
    if local_geometry.duplicated(["pair_id", "canonical_position"]).any():
        raise ValueError("local geometry must be unique by pair and canonical position")
    rows: list[dict[str, Any]] = []
    for pair_id, group in local_geometry.groupby("pair_id", sort=True):
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "protein_id": group["protein_id"].iloc[0],
            "identity_cluster_id": group["identity_cluster_id"].iloc[0],
            "identity_cluster_30": group["identity_cluster_id"].iloc[0],
            "structural_regime": regime,
            "mapped_residue_count": int(group["canonical_position"].nunique()),
        }
        for column in ("protein_id", "identity_cluster_id"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"pair has inconsistent {column}: {pair_id}")
        if "requested_dose" in group:
            doses = group["requested_dose"].dropna().unique()
            if len(doses) > 1:
                raise ValueError(f"pair has inconsistent requested dose: {pair_id}")
            row["requested_dose"] = float(doses[0]) if len(doses) else np.nan
        for descriptor in LOCAL_DESCRIPTORS:
            values = group[descriptor].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                raise ValueError(f"pair has no finite {descriptor}: {pair_id}")
            row[f"{descriptor}_median"] = float(np.quantile(values, 0.50))
            row[f"{descriptor}_q75"] = float(np.quantile(values, 0.75))
            row[f"{descriptor}_q90"] = float(np.quantile(values, 0.90))
            row[f"{descriptor}_valid_residue_count"] = len(values)
        displacement = group["ca_displacement"].to_numpy(dtype=np.float64)
        displacement = displacement[np.isfinite(displacement)]
        row["aligned_ca_rmsd"] = float(np.sqrt(np.mean(np.square(displacement))))
        row["aligned_ca_rmsd_source"] = "rms_of_global_kabsch_ca_displacement"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pair_id", kind="mergesort", ignore_index=True)


def controlled_signature_projection(
    controlled: pd.DataFrame,
    operational: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = PAIR_FEATURES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit standardization and PCA on Controlled pairs, then project both regimes."""

    metadata_columns = ("pair_id", "protein_id", "identity_cluster_id")
    _require(controlled, (*metadata_columns, *feature_columns), "controlled geometry")
    _require(operational, (*metadata_columns, *feature_columns), "operational geometry")
    controlled_values = controlled.loc[:, feature_columns].to_numpy(dtype=np.float64)
    operational_values = operational.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(controlled_values).all() or not np.isfinite(operational_values).all():
        raise ValueError("structural signature features must be finite")
    means = controlled_values.mean(axis=0)
    scales = controlled_values.std(axis=0, ddof=0)
    if np.any(scales == 0):
        raise ValueError("controlled structural signature contains a constant feature")
    controlled_z = (controlled_values - means) / scales
    operational_z = (operational_values - means) / scales
    _, singular_values, right_vectors = np.linalg.svd(controlled_z, full_matrices=False)
    components = right_vectors[:2].copy()
    for component in components:
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            component *= -1
    variance = np.square(singular_values) / (len(controlled_z) - 1)
    explained = variance[:2] / variance.sum()

    frames = []
    for regime, source, standardized in (
        ("controlled", controlled, controlled_z),
        ("operational_pdb_afdb", operational, operational_z),
    ):
        frame = source.copy()
        frame["structural_regime"] = regime
        for index, feature in enumerate(feature_columns):
            frame[f"standardized__{feature}"] = standardized[:, index]
        projected = standardized @ components.T
        frame["pc1"] = projected[:, 0]
        frame["pc2"] = projected[:, 1]
        frames.append(frame)
    fit = {
        "feature_columns": list(feature_columns),
        "standardization_fit_regime": "controlled",
        "controlled_mean": means.tolist(),
        "controlled_scale": scales.tolist(),
        "pca_fit_regime": "controlled",
        "components": components.tolist(),
        "explained_variance_ratio": explained.tolist(),
    }
    return pd.concat(frames, ignore_index=True), fit


def knn_coverage_distances(
    controlled_standardized: np.ndarray,
    operational_standardized: np.ndarray,
    *,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Controlled leave-one-out and Operational-to-Controlled kNN distances."""

    controlled = np.asarray(controlled_standardized, dtype=np.float64)
    operational = np.asarray(operational_standardized, dtype=np.float64)
    if (
        controlled.ndim != 2
        or operational.ndim != 2
        or controlled.shape[1] != operational.shape[1]
        or not np.isfinite(controlled).all()
        or not np.isfinite(operational).all()
        or k < 1
        or len(controlled) <= k
    ):
        raise ValueError("kNN coverage inputs or k are invalid")
    controlled_distances = np.linalg.norm(controlled[:, None, :] - controlled[None, :, :], axis=2)
    np.fill_diagonal(controlled_distances, np.inf)
    controlled_loo = np.partition(controlled_distances, k - 1, axis=1)[:, :k].mean(axis=1)
    operational_distances = np.linalg.norm(operational[:, None, :] - controlled[None, :, :], axis=2)
    operational_to_controlled = np.partition(operational_distances, k - 1, axis=1)[:, :k].mean(
        axis=1
    )
    return controlled_loo, operational_to_controlled


def nearest_global_rmsd_matches(
    controlled: pd.DataFrame,
    operational: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = PAIR_FEATURES[:-1],
) -> pd.DataFrame:
    """Match in-range Operational pairs to nearest Controlled global RMSD."""

    required = (
        "pair_id",
        "protein_id",
        "identity_cluster_id",
        "aligned_ca_rmsd",
        *feature_columns,
    )
    _require(controlled, required, "controlled geometry")
    _require(operational, required, "operational geometry")
    controlled = controlled.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    operational = operational.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    overlap_min = max(controlled["aligned_ca_rmsd"].min(), operational["aligned_ca_rmsd"].min())
    overlap_max = min(controlled["aligned_ca_rmsd"].max(), operational["aligned_ca_rmsd"].max())
    eligible = operational.loc[
        operational["aligned_ca_rmsd"].between(overlap_min, overlap_max, inclusive="both")
    ]
    rows: list[dict[str, Any]] = []
    controlled_rmsd = controlled["aligned_ca_rmsd"].to_numpy(dtype=np.float64)
    for operational_row in eligible.itertuples(index=False):
        index = int(np.argmin(np.abs(controlled_rmsd - operational_row.aligned_ca_rmsd)))
        controlled_row = controlled.iloc[index]
        for feature in feature_columns:
            controlled_value = float(controlled_row[feature])
            operational_value = float(getattr(operational_row, feature))
            rows.append(
                {
                    "operational_pair_id": operational_row.pair_id,
                    "operational_protein_id": operational_row.protein_id,
                    "operational_identity_cluster_id": operational_row.identity_cluster_id,
                    "controlled_pair_id": controlled_row["pair_id"],
                    "controlled_protein_id": controlled_row["protein_id"],
                    "controlled_identity_cluster_id": controlled_row["identity_cluster_id"],
                    "operational_aligned_ca_rmsd": operational_row.aligned_ca_rmsd,
                    "controlled_aligned_ca_rmsd": controlled_row["aligned_ca_rmsd"],
                    "absolute_rmsd_difference": abs(
                        operational_row.aligned_ca_rmsd - controlled_row["aligned_ca_rmsd"]
                    ),
                    "overlap_min": overlap_min,
                    "overlap_max": overlap_max,
                    "structural_feature": feature,
                    "controlled_feature_value": controlled_value,
                    "operational_feature_value": operational_value,
                    "feature_difference": operational_value - controlled_value,
                }
            )
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ranked_x = pd.Series(x).rank(method="average").to_numpy()
    ranked_y = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(ranked_x) == 0 or np.ptp(ranked_y) == 0:
        return np.nan
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def cluster_bootstrap_spearman(
    proteins: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    replicates: int = 10_000,
    seed: int = 2_026_09_07,
) -> dict[str, float | int | str]:
    """Bootstrap a protein-level Spearman correlation by identity cluster."""

    required = ("protein_id", "identity_cluster_id", x_column, y_column)
    _require(proteins, required, "protein association")
    if proteins["protein_id"].duplicated().any() or replicates < 1:
        raise ValueError("association input must contain one row per protein")
    values = proteins[[x_column, y_column]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("association input contains nonfinite values")
    groups = [
        group[[x_column, y_column]].to_numpy(dtype=np.float64)
        for _, group in proteins.groupby("identity_cluster_id", sort=True)
    ]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
    bootstrap = np.asarray(
        [
            _spearman(
                (sample := np.concatenate([groups[index] for index in draw]))[:, 0],
                sample[:, 1],
            )
            for draw in draws
        ]
    )
    finite = bootstrap[np.isfinite(bootstrap)]
    if not len(finite):
        raise ValueError("all bootstrap correlations are undefined")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "spearman_rho": _spearman(values[:, 0], values[:, 1]),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_proteins": len(proteins),
        "n_identity_clusters": len(groups),
        "bootstrap_replicates": replicates,
        "valid_bootstrap_replicates": len(finite),
        "bootstrap_unit": "identity_cluster_30",
        "bootstrap_statistic": "spearman",
        "seed": seed,
    }
