"""Pair-conditioned local mechanism checks for frozen StructCal responses."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .cross_model_representation_sensitivity import cluster_bootstrap_summary

LOCAL_DESCRIPTORS = (
    "ca_displacement",
    "fragment_7_rmsd",
    "neighborhood_distance_deformation",
    "torsion_phi_psi_change",
)

PAIR_KEYS = (
    "model_id",
    "checkpoint_id",
    "protein_id",
    "identity_cluster_id",
    "pair_id",
)


def _validate_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{label} is missing required data: {missing}")


def _groupby(frame: pd.DataFrame, columns: Sequence[str]):
    grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    return frame.groupby(grouper, dropna=False, sort=True)


def _pair_keys(frame: pd.DataFrame) -> tuple[str, ...]:
    return (*PAIR_KEYS, *(("requested_dose",) if "requested_dose" in frame else ()))


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, str, int, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    n_finite = int(finite.sum())
    n_nonfinite = int(len(x) - n_finite)
    if n_finite < 2:
        reason = "NONFINITE_VALUES" if n_finite == 0 else "TOO_FEW_FINITE_RESIDUES"
        return np.nan, reason, n_finite, n_nonfinite
    selected_x = x[finite]
    selected_y = y[finite]
    if np.ptp(selected_x) == 0:
        return np.nan, "CONSTANT_DESCRIPTOR", n_finite, n_nonfinite
    if np.ptp(selected_y) == 0:
        return np.nan, "CONSTANT_RESPONSE", n_finite, n_nonfinite
    ranked_x = pd.Series(selected_x).rank(method="average").to_numpy()
    ranked_y = pd.Series(selected_y).rank(method="average").to_numpy()
    rho = float(np.corrcoef(ranked_x, ranked_y)[0, 1])
    return rho, "", n_finite, n_nonfinite


def pair_conditioned_local_associations(
    local_response: pd.DataFrame,
    *,
    descriptors: tuple[str, ...] = LOCAL_DESCRIPTORS,
) -> pd.DataFrame:
    """Compute residue-level Spearman correlation separately in every pair."""

    pair_keys = _pair_keys(local_response)
    required = (*pair_keys, "canonical_position", "jsd_bits", *descriptors)
    _validate_columns(local_response, required, "local response")
    uniqueness = ["model_id", "checkpoint_id", "pair_id", "canonical_position"]
    if local_response.duplicated(uniqueness).any():
        raise ValueError("local response is not unique on model, pair, canonical position")

    rows: list[dict[str, Any]] = []
    for values, group in _groupby(local_response, pair_keys):
        metadata = dict(zip(pair_keys, values, strict=True))
        response = group["jsd_bits"].to_numpy(dtype=np.float64)
        for descriptor in descriptors:
            rho, reason, n_finite, n_nonfinite = _spearman(
                group[descriptor].to_numpy(dtype=np.float64), response
            )
            rows.append(
                {
                    **metadata,
                    "descriptor": descriptor,
                    "spearman_rho": rho,
                    "status": "VALID" if not reason else "UNDEFINED",
                    "reason": reason or None,
                    "n_total_residues": len(group),
                    "n_finite_residues": n_finite,
                    "n_nonfinite_residues": n_nonfinite,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [*pair_keys, "descriptor"], kind="mergesort", ignore_index=True
    )


def aggregate_pair_correlations_by_protein(
    pair_correlations: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Take the median valid pair correlation within each protein."""

    required = (*group_columns, "pair_id", "spearman_rho", "status")
    _validate_columns(pair_correlations, required, "pair correlations")
    rows: list[dict[str, Any]] = []
    for values, group in _groupby(pair_correlations, group_columns):
        values = values if isinstance(values, tuple) else (values,)
        valid = group.loc[group["status"].eq("VALID") & np.isfinite(group["spearman_rho"])]
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "spearman_rho": (
                    float(valid["spearman_rho"].median()) if not valid.empty else np.nan
                ),
                "status": "VALID" if not valid.empty else "UNDEFINED",
                "reason": None if not valid.empty else "NO_VALID_PAIRS",
                "n_total_pairs": int(group["pair_id"].nunique()),
                "n_valid_pairs": int(valid["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(list(group_columns), kind="mergesort", ignore_index=True)


def summarize_protein_correlations(
    protein_correlations: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    bootstrap_replicates: int = 10_000,
    seed: int = 2_026_09_05,
) -> pd.DataFrame:
    """Summarize protein medians using the frozen 30%-cluster bootstrap."""

    required = (
        *group_columns,
        "protein_id",
        "identity_cluster_id",
        "spearman_rho",
        "status",
        "n_valid_pairs",
    )
    _validate_columns(protein_correlations, required, "protein correlations")
    rows: list[dict[str, Any]] = []
    for offset, (values, group) in enumerate(_groupby(protein_correlations, group_columns)):
        values = values if isinstance(values, tuple) else (values,)
        valid = group.loc[group["status"].eq("VALID") & np.isfinite(group["spearman_rho"])]
        metadata = dict(zip(group_columns, values, strict=True))
        counts = {
            "n_total_pairs": int(group.get("n_total_pairs", group["n_valid_pairs"]).sum()),
            "n_valid_pairs": int(group["n_valid_pairs"].sum()),
            "n_total_proteins": int(group["protein_id"].nunique()),
        }
        if valid.empty:
            rows.append(
                {
                    **metadata,
                    **counts,
                    "mean": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "fraction_positive": np.nan,
                    "n_proteins": 0,
                    "n_identity_clusters": 0,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "identity_cluster_30",
                    "bootstrap_statistic": "median",
                }
            )
            continue
        summary = cluster_bootstrap_summary(
            valid[["protein_id", "identity_cluster_id", "spearman_rho"]],
            value_column="spearman_rho",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + offset,
            bootstrap_statistic="median",
        )
        rows.append({**metadata, **counts, **summary})
    return pd.DataFrame(rows).sort_values(list(group_columns), kind="mergesort", ignore_index=True)


def pair_fixed_effect_local_associations(
    local_response: pd.DataFrame,
    *,
    descriptors: tuple[str, ...] = LOCAL_DESCRIPTORS,
) -> pd.DataFrame:
    """Correlate pair-median-centered residue values within each protein."""

    pair_associations = pair_conditioned_local_associations(local_response, descriptors=descriptors)
    group_columns = (
        "model_id",
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "descriptor",
    )
    rows: list[dict[str, Any]] = []
    for values, correlations in _groupby(pair_associations, group_columns):
        values = values if isinstance(values, tuple) else (values,)
        valid_pairs = set(correlations.loc[correlations["status"].eq("VALID"), "pair_id"])
        model, checkpoint, protein, _cluster, descriptor = values
        selected = local_response.loc[
            local_response["model_id"].eq(model)
            & local_response["checkpoint_id"].eq(checkpoint)
            & local_response["protein_id"].eq(protein)
            & local_response["pair_id"].isin(valid_pairs),
            ["pair_id", descriptor, "jsd_bits"],
        ].copy()
        finite = np.isfinite(selected[descriptor]) & np.isfinite(selected["jsd_bits"])
        selected = selected.loc[finite]
        if selected.empty:
            rho, reason, n_finite, _ = np.nan, "NO_VALID_PAIRS", 0, 0
        else:
            selected["centered_descriptor"] = selected[descriptor] - selected.groupby("pair_id")[
                descriptor
            ].transform("median")
            selected["centered_response"] = selected["jsd_bits"] - selected.groupby("pair_id")[
                "jsd_bits"
            ].transform("median")
            rho, reason, n_finite, _ = _spearman(
                selected["centered_descriptor"].to_numpy(dtype=np.float64),
                selected["centered_response"].to_numpy(dtype=np.float64),
            )
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "spearman_rho": rho,
                "status": "VALID" if not reason else "UNDEFINED",
                "reason": reason or None,
                "n_total_pairs": int(correlations["pair_id"].nunique()),
                "n_valid_pairs": len(valid_pairs),
                "n_centered_residues": n_finite,
            }
        )
    return pd.DataFrame(rows).sort_values(list(group_columns), kind="mergesort", ignore_index=True)


def _model_frame(local_response: pd.DataFrame, model: str) -> pd.DataFrame:
    columns = [
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "pair_id",
        "canonical_position",
        "jsd_bits",
    ]
    if "requested_dose" in local_response:
        columns.insert(-1, "requested_dose")
    frame = local_response.loc[local_response["model_id"].eq(model), columns].copy()
    if frame.duplicated(["pair_id", "canonical_position"]).any():
        raise ValueError(f"{model} is not unique on pair and canonical position")
    return frame


def _shared_profiles(
    local_response: pd.DataFrame, left_model: str, right_model: str
) -> pd.DataFrame:
    left = _model_frame(local_response, left_model)
    right = _model_frame(local_response, right_model)
    shared = left.merge(
        right,
        on=["pair_id", "canonical_position"],
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    metadata_columns = ["protein_id", "identity_cluster_id"]
    if "requested_dose_left" in shared:
        metadata_columns.append("requested_dose")
    for column in metadata_columns:
        if not shared[f"{column}_left"].equals(shared[f"{column}_right"]):
            raise ValueError(f"cross-model {column} metadata disagree")
    shared["left_model_id"] = left_model
    shared["right_model_id"] = right_model
    return shared


def pair_conditioned_cross_model_hotspots(local_response: pd.DataFrame) -> pd.DataFrame:
    """Compare model response profiles separately within each structural pair."""

    required = (
        "model_id",
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "pair_id",
        "canonical_position",
        "jsd_bits",
    )
    _validate_columns(local_response, required, "local response")
    rows: list[dict[str, Any]] = []
    for left_model, right_model in combinations(sorted(local_response["model_id"].unique()), 2):
        shared = _shared_profiles(local_response, left_model, right_model)
        for pair_id, group in shared.groupby("pair_id", sort=True):
            rho, reason, n_finite, n_nonfinite = _spearman(
                group["jsd_bits_left"].to_numpy(dtype=np.float64),
                group["jsd_bits_right"].to_numpy(dtype=np.float64),
            )
            first = group.iloc[0]
            row = {
                "left_model_id": left_model,
                "left_checkpoint_id": first["checkpoint_id_left"],
                "right_model_id": right_model,
                "right_checkpoint_id": first["checkpoint_id_right"],
                "protein_id": first["protein_id_left"],
                "identity_cluster_id": first["identity_cluster_id_left"],
                "pair_id": pair_id,
                "spearman_rho": rho,
                "status": "VALID" if not reason else "UNDEFINED",
                "reason": reason or None,
                "n_shared_residues": len(group),
                "n_finite_residues": n_finite,
                "n_nonfinite_residues": n_nonfinite,
            }
            if "requested_dose_left" in group:
                row["requested_dose"] = first["requested_dose_left"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["left_model_id", "right_model_id", "pair_id"],
        kind="mergesort",
        ignore_index=True,
    )


def permuted_pair_values(
    values: np.ndarray, *, permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Independently permute one pair's values for each null replicate."""

    array = np.asarray(values)
    if array.ndim != 1 or permutations < 1:
        raise ValueError("values must be one-dimensional and permutations positive")
    return rng.permuted(np.broadcast_to(array, (permutations, len(array))), axis=1)


def hotspot_permutation_null(
    local_response: pd.DataFrame,
    pair_hotspots: pd.DataFrame,
    *,
    permutations: int = 1_000,
    seed: int = 2_026_09_05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute model-B positions independently in each pair and preserve hierarchy."""

    required = (
        "left_model_id",
        "right_model_id",
        "protein_id",
        "identity_cluster_id",
        "pair_id",
        "spearman_rho",
        "status",
    )
    _validate_columns(pair_hotspots, required, "pair hotspots")
    rng = np.random.default_rng(seed)
    null_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pair_columns = ("left_model_id", "right_model_id")
    for values, observed_pairs in _groupby(pair_hotspots, pair_columns):
        left_model, right_model = values
        valid = observed_pairs.loc[
            observed_pairs["status"].eq("VALID") & np.isfinite(observed_pairs["spearman_rho"])
        ].copy()
        if valid.empty:
            raise ValueError(f"no valid hotspots for {left_model}/{right_model}")
        shared = _shared_profiles(local_response, left_model, right_model)
        pair_rhos: list[np.ndarray] = []
        pair_proteins: list[str] = []
        for pair_id in valid["pair_id"]:
            group = shared.loc[shared["pair_id"].eq(pair_id)]
            x = group["jsd_bits_left"].to_numpy(dtype=np.float64)
            y = group["jsd_bits_right"].to_numpy(dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            ranked_x = pd.Series(x[finite]).rank(method="average").to_numpy()
            ranked_y = pd.Series(y[finite]).rank(method="average").to_numpy()
            centered_x = ranked_x - ranked_x.mean()
            centered_y = ranked_y - ranked_y.mean()
            denominator = np.sqrt(np.square(centered_x).sum() * np.square(centered_y).sum())
            permuted = permuted_pair_values(centered_y, permutations=permutations, rng=rng)
            pair_rhos.append((permuted @ centered_x) / denominator)
            pair_proteins.append(str(group.iloc[0]["protein_id_left"]))
        rho_matrix = np.column_stack(pair_rhos)
        proteins = np.asarray(pair_proteins)
        protein_rhos = np.column_stack(
            [
                np.median(rho_matrix[:, proteins == protein], axis=1)
                for protein in sorted(set(proteins))
            ]
        )
        null_values = np.median(protein_rhos, axis=1)
        observed_proteins = aggregate_pair_correlations_by_protein(
            valid,
            group_columns=(
                "left_model_id",
                "right_model_id",
                "protein_id",
                "identity_cluster_id",
            ),
        )
        observed = float(observed_proteins["spearman_rho"].median())
        null_low, null_high = np.quantile(null_values, [0.025, 0.975])
        tail_p = float((1 + np.count_nonzero(null_values >= observed)) / (permutations + 1))
        for replicate, value in enumerate(null_values):
            null_rows.append(
                {
                    "left_model_id": left_model,
                    "right_model_id": right_model,
                    "permutation": replicate,
                    "null_median_protein_rho": float(value),
                    "seed": seed,
                    "n_valid_pairs": len(valid),
                    "n_valid_proteins": len(observed_proteins),
                }
            )
        summary_rows.append(
            {
                "left_model_id": left_model,
                "right_model_id": right_model,
                "observed_median_protein_rho": observed,
                "null_median": float(np.median(null_values)),
                "null_ci_low": float(null_low),
                "null_ci_high": float(null_high),
                "observed_minus_null_median": observed - float(np.median(null_values)),
                "empirical_upper_tail_p": tail_p,
                "permutations": permutations,
                "seed": seed,
                "n_valid_pairs": len(valid),
                "n_valid_proteins": len(observed_proteins),
            }
        )
    return pd.DataFrame(null_rows), pd.DataFrame(summary_rows)
