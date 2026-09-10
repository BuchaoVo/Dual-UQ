"""Model-independent metrics for cross-model representation sensitivity."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .structcal_local_response import (
    STANDARD_AMINO_ACIDS,
    js_bits,
    validate_probability_matrix,
)


def common_pair_intersection(
    response: pd.DataFrame,
    *,
    models: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Restrict a model-response table to exactly shared pair identities.

    Pair identity, rather than protein identity, is the intersection unit because
    one protein may contribute more than one structural pair.
    """

    required = {"model_id", "pair_id", "protein_id", "identity_cluster_id"}
    missing = sorted(required.difference(response.columns))
    if missing or response.empty:
        raise ValueError(f"response is missing common-intersection columns: {missing}")
    selected = (
        tuple(sorted(response["model_id"].astype(str).unique()))
        if models is None
        else models
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("models must be a non-empty unique sequence")
    available = set(response["model_id"].astype(str))
    absent = sorted(set(selected).difference(available))
    if absent:
        raise ValueError(f"requested models are absent: {absent}")
    subset = response.loc[response["model_id"].astype(str).isin(selected)].copy()
    if subset.duplicated(["model_id", "pair_id"]).any():
        raise ValueError("response contains duplicate model/pair rows")
    pair_sets = [
        set(subset.loc[subset["model_id"].astype(str).eq(model), "pair_id"].astype(str))
        for model in selected
    ]
    shared = set.intersection(*pair_sets)
    if not shared:
        raise ValueError("models have no shared pair identities")
    result = subset.loc[subset["pair_id"].astype(str).isin(shared)].copy()
    counts = result.groupby("pair_id", sort=False)["model_id"].nunique()
    if not counts.eq(len(selected)).all():
        raise RuntimeError("common-pair intersection is incomplete")
    metadata = result[["pair_id", "protein_id", "identity_cluster_id"]].drop_duplicates()
    if metadata["pair_id"].duplicated().any():
        raise ValueError("pair metadata differs across models")
    return result.sort_values(["model_id", "pair_id"], kind="mergesort", ignore_index=True)


def attach_controlled_metadata(
    response: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Restore frozen controlled-pair descriptors by strict pair identity."""

    required = {
        "pair_id",
        "protein_id",
        "perturbation_family",
        "requested_dose",
        "requested_dose_unit",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing or metadata["pair_id"].duplicated().any():
        raise ValueError(f"controlled metadata is not one-to-one: {missing}")
    columns = [
        "pair_id",
        "protein_id",
        "perturbation_family",
        "requested_dose",
        "requested_dose_unit",
    ]
    base = response.drop(
        columns=[column for column in columns[2:] if column in response],
        errors="ignore",
    )
    enriched = base.merge(
        metadata[columns],
        on=["pair_id", "protein_id"],
        how="left",
        validate="many_to_one",
    )
    if enriched[columns[2:]].isna().any().any():
        raise ValueError("controlled metadata join is not complete")
    return enriched


def protein_level_response(
    pair_response: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("r_jsd_bits", "r_prob", "r_flip", "abs_delta_nll"),
) -> pd.DataFrame:
    """Average repeated pair observations before primary inference."""

    keys = [
        "model_id",
        "checkpoint_id",
        "regime",
        "protein_id",
        "identity_cluster_id",
    ]
    missing = sorted(set(keys + list(metrics)).difference(pair_response.columns))
    if missing:
        raise ValueError(f"pair response is missing protein aggregation columns: {missing}")
    grouped = pair_response.groupby(keys, as_index=False, dropna=False, sort=True)
    result = grouped[list(metrics)].mean()
    result["n_pairs"] = grouped["pair_id"].nunique()["pair_id"]
    return result


def cluster_bootstrap_summary(
    protein_response: pd.DataFrame,
    *,
    value_column: str,
    bootstrap_replicates: int = 10_000,
    seed: int = 2_026_09_04,
    bootstrap_statistic: str = "mean",
) -> dict[str, float | int | str]:
    """Summarize protein values with identity-cluster resampling."""

    required = {"protein_id", "identity_cluster_id", value_column}
    missing = sorted(required.difference(protein_response.columns))
    if missing or protein_response.empty:
        raise ValueError(f"protein response is missing bootstrap columns: {missing}")
    if protein_response["protein_id"].duplicated().any():
        raise ValueError("protein response must contain one row per protein")
    values = protein_response[value_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("protein response contains nonfinite values")
    grouped_clusters = protein_response.groupby("identity_cluster_id", sort=True)[value_column]
    clusters = grouped_clusters.agg(["sum", "count"])
    if bootstrap_statistic not in {"mean", "median"}:
        raise ValueError("bootstrap_statistic must be mean or median")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(clusters), size=(bootstrap_replicates, len(clusters))
    )
    if bootstrap_statistic == "mean":
        bootstrap = (
            clusters["sum"].to_numpy()[draws].sum(axis=1)
            / clusters["count"].to_numpy()[draws].sum(axis=1)
        )
    else:
        cluster_values = [group.to_numpy() for _, group in grouped_clusters]
        bootstrap = np.asarray(
            [
                np.median(np.concatenate([cluster_values[index] for index in draw]))
                for draw in draws
            ]
        )
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "ci_low": float(low),
        "ci_high": float(high),
        "fraction_positive": float((values > 0).mean()),
        "n_proteins": len(values),
        "n_identity_clusters": len(clusters),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_unit": "identity_cluster_30",
        "bootstrap_statistic": bootstrap_statistic,
    }


def paired_protein_differences(
    response: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    value_column: str,
    difference_column: str,
) -> pd.DataFrame:
    """Subtract a baseline after strict one-to-one protein matching."""

    required = {"protein_id", "identity_cluster_id", value_column}
    for label, frame in (("response", response), ("baseline", baseline)):
        missing = sorted(required.difference(frame.columns))
        if missing or frame["protein_id"].duplicated().any():
            raise ValueError(f"{label} is not one row per protein: {missing}")
    joined = response[list(required)].merge(
        baseline[list(required)],
        on="protein_id",
        how="inner",
        suffixes=("_response", "_baseline"),
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("response and baseline have no matched proteins")
    response_cluster = joined["identity_cluster_id_response"].fillna("")
    baseline_cluster = joined["identity_cluster_id_baseline"].fillna("")
    if not response_cluster.equals(baseline_cluster):
        raise ValueError("identity clusters differ for matched proteins")
    return pd.DataFrame(
        {
            "protein_id": joined["protein_id"],
            "identity_cluster_id": joined["identity_cluster_id_response"],
            "response_value": joined[f"{value_column}_response"],
            "baseline_value": joined[f"{value_column}_baseline"],
            difference_column: (
                joined[f"{value_column}_response"]
                - joined[f"{value_column}_baseline"]
            ),
        }
    ).sort_values("protein_id", kind="mergesort", ignore_index=True)


def controlled_severity_differences(
    controlled: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("r_jsd_bits", "r_prob", "r_flip", "abs_delta_nll"),
) -> pd.DataFrame:
    """Return matched high-minus-low controlled responses per protein."""

    keys = ["model_id", "checkpoint_id", "protein_id", "identity_cluster_id"]
    required = set(keys + ["pair_id", "requested_dose", *metrics])
    missing = sorted(required.difference(controlled.columns))
    if missing or controlled.empty:
        raise ValueError(f"controlled response is missing severity columns: {missing}")
    doses = sorted(float(value) for value in controlled["requested_dose"].unique())
    if len(doses) != 2:
        raise ValueError(f"expected exactly two controlled severity levels, found {doses}")
    grouped = controlled.groupby(
        keys + ["requested_dose"], as_index=False, dropna=False, sort=True
    )
    means = grouped[list(metrics)].mean()
    means["n_pairs"] = grouped["pair_id"].nunique()["pair_id"]
    low = means.loc[means["requested_dose"].eq(doses[0])]
    high = means.loc[means["requested_dose"].eq(doses[1])]
    joined = low.merge(
        high,
        on=keys,
        how="inner",
        suffixes=("_low", "_high"),
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("controlled severity levels have no matched proteins")
    result = joined[keys].copy()
    result["requested_dose_low"] = joined["requested_dose_low"]
    result["requested_dose_high"] = joined["requested_dose_high"]
    result["n_pairs_low"] = joined["n_pairs_low"]
    result["n_pairs_high"] = joined["n_pairs_high"]
    for metric in metrics:
        result[f"{metric}_low"] = joined[f"{metric}_low"]
        result[f"{metric}_high"] = joined[f"{metric}_high"]
        result[f"{metric}_high_minus_low"] = (
            joined[f"{metric}_high"] - joined[f"{metric}_low"]
        )
    return result.sort_values(keys, kind="mergesort", ignore_index=True)


def _native_indices(sequence: str, length: int) -> np.ndarray:
    if not isinstance(sequence, str) or len(sequence) != length:
        raise ValueError("native sequence length must match probability rows")
    lookup = {amino_acid: index for index, amino_acid in enumerate(STANDARD_AMINO_ACIDS)}
    if any(amino_acid not in lookup for amino_acid in sequence):
        raise ValueError("native sequence must use the standard-AA alphabet")
    return np.asarray([lookup[amino_acid] for amino_acid in sequence], dtype=np.int64)


def standard_probability_quality(probabilities: Any, native_sequence: str) -> dict[str, float]:
    """Return native recovery, mean NLL in nats, and perplexity."""

    matrix = validate_probability_matrix(probabilities, label="probabilities")
    target = _native_indices(native_sequence, len(matrix))
    native = matrix[np.arange(len(matrix)), target]
    if (native <= 0).any():
        raise ValueError("native-residue probabilities must be positive")
    nll = float(-np.log(native).mean())
    return {
        "recovery": float((np.argmax(matrix, axis=1) == target).mean()),
        "nll": nll,
        "perplexity": float(math.exp(nll)),
    }


def paired_probability_metrics(
    left: Any,
    right: Any,
    native_sequence: str,
) -> dict[str, Any]:
    """Compute the shared response metrics for two paired structural views."""

    first = validate_probability_matrix(left, label="left probabilities")
    second = validate_probability_matrix(right, label="right probabilities")
    if first.shape != second.shape:
        raise ValueError("paired probability matrices must have the same shape")
    target = _native_indices(native_sequence, len(first))
    native_left = first[np.arange(len(first)), target]
    native_right = second[np.arange(len(second)), target]
    if (native_left <= 0).any() or (native_right <= 0).any():
        raise ValueError("native-residue probabilities must be positive")
    residue_jsd = np.asarray(
        [js_bits(p, q) for p, q in zip(first, second, strict=True)],
        dtype=np.float64,
    )
    return {
        "residue_jsd_bits": residue_jsd,
        "r_jsd_bits": float(residue_jsd.mean()),
        "r_flip": float((np.argmax(first, axis=1) != np.argmax(second, axis=1)).mean()),
        "r_prob": float(np.abs(native_left - native_right).mean()),
        "nll_left": float(-np.log(native_left).mean()),
        "nll_right": float(-np.log(native_right).mean()),
        "abs_delta_nll": float(abs(np.log(native_left).mean() - np.log(native_right).mean())),
    }


def constant_pair_metadata(
    group: pd.DataFrame, column: str, default: Any = None
) -> Any:
    """Return one pair-level metadata value, rejecting inconsistent rows."""

    if column not in group:
        return default
    values = group[column].drop_duplicates()
    if len(values) > 1:
        raise ValueError(f"pair metadata is not constant: {column}")
    return values.iloc[0] if len(values) else default


def build_response_rows(
    cases: pd.DataFrame,
    distributions: dict[tuple[str, str], Any],
    *,
    model_id: str,
    checkpoint_id: str,
    semantic_class: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reduce model-native paired probabilities to residue, pair, and quality rows."""

    required = {
        "pair_id",
        "protein_id",
        "condition",
        "condition_label",
        "canonical_position",
        "wt_sequence_projection",
    }
    missing = sorted(required.difference(cases.columns))
    if missing or cases.empty:
        raise ValueError(f"response cases missing columns: {missing}")
    residue_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for pair_id, group in cases.groupby("pair_id", sort=True):
        conditions = tuple(sorted(group["condition"].astype(str).unique()))
        if conditions != ("CONDITION_1", "CONDITION_2"):
            raise ValueError(f"pair must preserve CONDITION_1/CONDITION_2: {pair_id}")
        axes = []
        matrices = []
        labels = []
        for condition in conditions:
            selected = group.loc[group["condition"].astype(str).eq(condition)].sort_values(
                "canonical_position", kind="mergesort"
            )
            axes.append(tuple(int(value) for value in selected["canonical_position"]))
            labels.append(
                str(constant_pair_metadata(selected, "condition_label", condition))
            )
            key = (str(pair_id), condition)
            if key not in distributions:
                raise ValueError(f"missing probability matrix: {key}")
            matrix = validate_probability_matrix(distributions[key], label=str(key))
            if len(matrix) != len(selected):
                raise ValueError(f"probability length differs from case axis: {key}")
            matrices.append(matrix)
        if axes[0] != axes[1]:
            raise ValueError(f"paired canonical axes differ: {pair_id}")
        sequences = group["wt_sequence_projection"].astype(str).drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"pair native sequence is not constant: {pair_id}")
        sequence = str(sequences.iloc[0])
        if len(sequence) != len(axes[0]):
            raise ValueError(f"pair sequence length differs from case axis: {pair_id}")
        metrics = paired_probability_metrics(matrices[0], matrices[1], sequence)
        quality = standard_probability_quality(matrices[0], sequence)
        metadata = {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "semantic_class": semantic_class,
            "regime": regime,
            "pair_id": str(pair_id),
            "protein_id": str(constant_pair_metadata(group, "protein_id", "")),
            "identity_cluster_id": constant_pair_metadata(group, "identity_cluster_id"),
            "split": constant_pair_metadata(group, "split"),
            "track_or_diagnostic": constant_pair_metadata(group, "track_or_diagnostic"),
            "state_family": constant_pair_metadata(group, "state_family"),
            "perturbation_family": constant_pair_metadata(group, "perturbation_family"),
            "requested_dose": constant_pair_metadata(group, "requested_dose"),
            "reference_label": labels[0],
            "comparison_label": labels[1],
        }
        target = _native_indices(sequence, len(sequence))
        top1_left = np.argmax(matrices[0], axis=1)
        top1_right = np.argmax(matrices[1], axis=1)
        for index, position in enumerate(axes[0]):
            residue_rows.append(
                {
                    **metadata,
                    "canonical_position": position,
                    "native_aa": sequence[index],
                    "jsd_bits": float(metrics["residue_jsd_bits"][index]),
                    "native_probability_left": float(matrices[0][index, target[index]]),
                    "native_probability_right": float(matrices[1][index, target[index]]),
                    "top1_left": STANDARD_AMINO_ACIDS[int(top1_left[index])],
                    "top1_right": STANDARD_AMINO_ACIDS[int(top1_right[index])],
                }
            )
        canonical_count = int(
            constant_pair_metadata(group, "n_canonical_positions", len(sequence))
        )
        pair_rows.append(
            {
                **metadata,
                "n_canonical_positions": canonical_count,
                "n_evaluable_positions": len(sequence),
                "coverage_fraction": len(sequence) / canonical_count,
                **{key: value for key, value in metrics.items() if key != "residue_jsd_bits"},
            }
        )
        quality_rows.append({**metadata, **quality, "n_evaluable_positions": len(sequence)})
    residue = pd.DataFrame(residue_rows).sort_values(
        ["model_id", "pair_id", "canonical_position"], kind="mergesort"
    ).reset_index(drop=True)
    pair = pd.DataFrame(pair_rows).sort_values(["model_id", "pair_id"], kind="mergesort").reset_index(drop=True)
    quality = pd.DataFrame(quality_rows).sort_values(["model_id", "pair_id"], kind="mergesort").reset_index(drop=True)
    return residue, pair, quality
