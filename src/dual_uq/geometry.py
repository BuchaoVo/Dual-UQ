from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

CA_CONTINUITY_MAX_ANGSTROM = 4.5
SEVERE_CLASH_DISTANCE_ANGSTROM = 1.5


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align mobile coordinates to target coordinates using the Kabsch algorithm."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) arrays, got {mobile.shape} and {target.shape}")
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


def obvious_geometry_counts(
    atom_coordinates: Sequence[np.ndarray],
    atom_names: Sequence[Sequence[str]],
    residue_positions: Sequence[int],
) -> dict[str, int]:
    """Count only obvious coordinate pathologies for operator calibration.

    This is a low-level, model-independent sanity summary.  It deliberately does
    not assess biological realism, model response, or perturbation descriptors.
    """

    if len(atom_coordinates) != len(atom_names) or len(atom_coordinates) != len(residue_positions):
        raise ValueError("coordinate, atom-name, and residue-position lengths differ")

    finite_atom_count = 0
    ca_coordinates: list[np.ndarray | None] = []
    finite_coordinates: list[np.ndarray] = []
    residue_indices: list[int] = []
    for residue_index, (coordinates, names) in enumerate(
        zip(atom_coordinates, atom_names, strict=True)
    ):
        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape != (len(names), 3):
            raise ValueError("each residue must contain one 3D coordinate per atom")
        finite = np.isfinite(values).all(axis=1)
        finite_atom_count += int((~finite).sum())
        finite_coordinates.extend(values[finite])
        residue_indices.extend([residue_index] * int(finite.sum()))
        ca_coordinates.append(
            values[tuple(str(name).strip().upper() for name in names).index("CA")]
            if "CA" in tuple(str(name).strip().upper() for name in names)
            else None
        )

    ca_chain_break_count = 0
    for index in range(len(ca_coordinates) - 1):
        left = ca_coordinates[index]
        right = ca_coordinates[index + 1]
        if (
            int(residue_positions[index + 1]) != int(residue_positions[index]) + 1
            or left is None
            or right is None
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or float(np.linalg.norm(right - left)) > CA_CONTINUITY_MAX_ANGSTROM
        ):
            ca_chain_break_count += 1

    severe_clash_count = 0
    if len(finite_coordinates) >= 2:
        pairs = np.asarray(
            cKDTree(np.asarray(finite_coordinates, dtype=float)).query_pairs(
                r=SEVERE_CLASH_DISTANCE_ANGSTROM,
                output_type="ndarray",
            ),
            dtype=int,
        )
        if pairs.size:
            residue_array = np.asarray(residue_indices, dtype=int)
            severe_clash_count = int(
                np.sum(np.abs(residue_array[pairs[:, 1]] - residue_array[pairs[:, 0]]) > 1)
            )

    return {
        "nonfinite_atom_count": finite_atom_count,
        "ca_chain_break_count": ca_chain_break_count,
        "severe_clash_count": severe_clash_count,
    }
