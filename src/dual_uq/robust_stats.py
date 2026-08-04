from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CorrelationResult:
    rho: float | None
    pvalue: float | None
    n: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {"rho": self.rho, "pvalue": self.pvalue, "n": self.n}


def safe_spearman(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return CorrelationResult(None, None, n)
    result = spearmanr(x[mask], y[mask])
    return CorrelationResult(float(result.statistic), float(result.pvalue), n)


def circular_shift_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 20260729,
) -> dict[str, float | int | None]:
    """Permutation test preserving one-dimensional autocorrelation by circular shifts."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    observed = safe_spearman(x, y)

    if observed.rho is None or len(x) < 4:
        return {
            **observed.as_dict(),
            "permutation_pvalue": None,
            "null_mean": None,
            "null_std": None,
            "n_permutations": 0,
        }

    rng = np.random.default_rng(seed)
    possible_shifts = np.arange(1, len(x))
    replace = n_permutations > len(possible_shifts)
    shifts = rng.choice(possible_shifts, size=n_permutations, replace=replace)

    null = np.empty(n_permutations, dtype=float)
    for index, shift in enumerate(shifts):
        null[index] = safe_spearman(np.roll(x, int(shift)), y).rho or 0.0

    pvalue = (1 + np.sum(np.abs(null) >= abs(observed.rho))) / (n_permutations + 1)
    return {
        **observed.as_dict(),
        "permutation_pvalue": float(pvalue),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_permutations": int(n_permutations),
    }


def matrix_label_permutation_test(
    predictor_matrix: np.ndarray,
    response_matrix: np.ndarray,
    positions: np.ndarray,
    *,
    min_sequence_separation: int = 6,
    n_permutations: int = 500,
    seed: int = 20260729,
) -> dict[str, float | int | None]:
    """Mantel-like label permutation for two residue-pair matrices.

    The same permutation is applied to rows and columns of the predictor matrix,
    preserving its internal matrix structure while breaking its alignment with
    the response matrix.
    """
    predictor = np.asarray(predictor_matrix, dtype=float)
    response = np.asarray(response_matrix, dtype=float)
    positions = np.asarray(positions, dtype=int)

    if predictor.shape != response.shape or predictor.shape[0] != len(positions):
        raise ValueError("Matrix and position dimensions are inconsistent.")

    separation = np.abs(positions[:, None] - positions[None, :])
    upper = np.triu(np.ones_like(predictor, dtype=bool), k=1)
    mask = upper & (separation >= min_sequence_separation)
    observed = safe_spearman(predictor[mask], response[mask])

    if observed.rho is None:
        return {
            **observed.as_dict(),
            "permutation_pvalue": None,
            "null_mean": None,
            "null_std": None,
            "n_permutations": 0,
        }

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    n = len(positions)
    for index in range(n_permutations):
        permutation = rng.permutation(n)
        permuted = predictor[np.ix_(permutation, permutation)]
        null[index] = safe_spearman(permuted[mask], response[mask]).rho or 0.0

    pvalue = (1 + np.sum(np.abs(null) >= abs(observed.rho))) / (n_permutations + 1)
    return {
        **observed.as_dict(),
        "permutation_pvalue": float(pvalue),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_permutations": int(n_permutations),
    }


def contiguous_segments(
    positions: np.ndarray,
    values: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, float | int]]:
    positions = np.asarray(positions, dtype=int)
    values = np.asarray(values, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    selected = np.where(np.isfinite(values) & (values >= threshold))[0]
    if len(selected) == 0:
        return []

    groups: list[list[int]] = [[int(selected[0])]]
    for index in selected[1:]:
        index = int(index)
        previous = groups[-1][-1]
        if positions[index] == positions[previous] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    segments = []
    for group in groups:
        segment_values = values[group]
        segment_confidence = confidence[group]
        segments.append(
            {
                "threshold": float(threshold),
                "start_position": int(positions[group[0]]),
                "end_position": int(positions[group[-1]]),
                "residue_count": len(group),
                "max_disagreement": float(np.max(segment_values)),
                "median_disagreement": float(np.median(segment_values)),
                "median_plddt": float(np.median(segment_confidence)),
                "min_plddt": float(np.min(segment_confidence)),
            }
        )
    return segments
