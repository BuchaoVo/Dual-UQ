from __future__ import annotations

import numpy as np


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align mobile coordinates to target coordinates using the Kabsch algorithm."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) arrays, got {mobile.shape} and {target.shape}"
        )
    if len(mobile) < 3:
        raise ValueError("At least three coordinate pairs are required for alignment.")

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center

    covariance = mobile_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    translation = target_center - mobile_center @ rotation
    aligned = mobile @ rotation + translation
    return aligned, rotation, translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shapes differ: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(delta**2, axis=-1))
