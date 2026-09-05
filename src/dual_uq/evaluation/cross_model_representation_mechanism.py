"""Local geometry and cross-model localization for frozen paired responses."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dual_uq.geometry import kabsch_align, pairwise_distances

LOCAL_DESCRIPTORS = (
    "ca_displacement",
    "fragment_7_rmsd",
    "torsion_phi_psi_change",
    "neighborhood_distance_deformation",
)


def _case_coordinates(group: pd.DataFrame) -> np.ndarray:
    payloads = group.loc[group["coordinates"].notna(), "coordinates"]
    if len(payloads) != 1:
        raise ValueError("each pair condition must contain exactly one coordinate payload")
    coordinates = np.asarray(
        [[np.asarray(atom, dtype=float) for atom in residue] for residue in payloads.iloc[0]],
        dtype=float,
    )
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
        raise ValueError("case coordinates must contain N/CA/C with shape [L, 3, 3]")
    return coordinates


def local_geometry_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen local-geometry protocol to paired N/CA/C cases."""

    required = {
        "pair_id",
        "protein_id",
        "identity_cluster_id",
        "condition",
        "canonical_position",
        "coordinates",
    }
    missing = sorted(required.difference(cases.columns))
    if missing or cases.empty:
        raise ValueError(f"paired cases are missing local geometry columns: {missing}")
    rows = []
    for pair_id, group in cases.groupby("pair_id", sort=True):
        for column in ("protein_id", "identity_cluster_id"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"paired case has inconsistent {column}: {pair_id}")
        positions: tuple[int, ...] | None = None
        coordinates = []
        for condition in ("CONDITION_1", "CONDITION_2"):
            selected = group.loc[group["condition"].eq(condition)].sort_values(
                "canonical_position", kind="mergesort"
            )
            current_positions = tuple(selected["canonical_position"].astype(int))
            if not current_positions or (positions is not None and current_positions != positions):
                raise ValueError(f"paired canonical axes differ or are empty: {pair_id}")
            positions = current_positions
            coordinates.append(_case_coordinates(selected))
        descriptors = paired_local_geometry(
            coordinates[0], coordinates[1], canonical_positions=positions or ()
        )
        descriptors.insert(0, "identity_cluster_id", group.iloc[0]["identity_cluster_id"])
        descriptors.insert(0, "protein_id", group.iloc[0]["protein_id"])
        descriptors.insert(0, "pair_id", pair_id)
        rows.append(descriptors)
    return pd.concat(rows, ignore_index=True)


def _dihedral(points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    first, second, third, fourth = points
    first_bond = second - first
    middle_bond = third - second
    last_bond = fourth - third
    first_normal = np.cross(first_bond, middle_bond)
    last_normal = np.cross(middle_bond, last_bond)
    if (
        min(
            np.linalg.norm(first_normal),
            np.linalg.norm(last_normal),
            np.linalg.norm(middle_bond),
        )
        == 0
    ):
        return float("nan")
    first_normal /= np.linalg.norm(first_normal)
    last_normal /= np.linalg.norm(last_normal)
    tangent = np.cross(first_normal, middle_bond / np.linalg.norm(middle_bond))
    return float(
        np.degrees(np.arctan2(np.dot(tangent, last_normal), np.dot(first_normal, last_normal)))
    )


def _circular_delta(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return float("nan")
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def _torsions(coordinates: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(coordinates)
    phi = np.full(count, np.nan)
    psi = np.full(count, np.nan)
    for index in range(count):
        if index > 0 and positions[index] == positions[index - 1] + 1:
            phi[index] = _dihedral(
                (
                    coordinates[index - 1, 2],
                    coordinates[index, 0],
                    coordinates[index, 1],
                    coordinates[index, 2],
                )
            )
        if index + 1 < count and positions[index + 1] == positions[index] + 1:
            psi[index] = _dihedral(
                (
                    coordinates[index, 0],
                    coordinates[index, 1],
                    coordinates[index, 2],
                    coordinates[index + 1, 0],
                )
            )
    return phi, psi


def paired_local_geometry(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
    *,
    canonical_positions: tuple[int, ...],
    fragment_size: int = 7,
    neighborhood_cutoff_angstrom: float = 12.0,
) -> pd.DataFrame:
    """Compute the four predeclared residue-level geometry descriptors."""

    reference = np.asarray(reference_coordinates, dtype=np.float64)
    comparison = np.asarray(comparison_coordinates, dtype=np.float64)
    positions = np.asarray(canonical_positions, dtype=int)
    if (
        reference.shape != comparison.shape
        or reference.ndim != 3
        or reference.shape[1:] != (3, 3)
        or len(positions) != len(reference)
        or not np.isfinite(reference).all()
        or not np.isfinite(comparison).all()
    ):
        raise ValueError("paired N/CA/C coordinates must be finite with shape [L, 3, 3]")
    if fragment_size != 7:
        raise ValueError("the mechanism protocol fixes fragment_size=7")
    if len(reference) < fragment_size or neighborhood_cutoff_angstrom <= 0:
        raise ValueError("local geometry requires at least seven residues and a positive cutoff")
    if len(set(positions)) != len(positions) or not np.all(np.diff(positions) > 0):
        raise ValueError("canonical positions must be unique and increasing")

    reference_ca = reference[:, 1]
    comparison_ca = comparison[:, 1]
    aligned, _, _ = kabsch_align(comparison_ca, reference_ca)
    displacement = np.linalg.norm(aligned - reference_ca, axis=1)
    reference_distances = pairwise_distances(reference_ca)
    comparison_distances = pairwise_distances(comparison_ca)
    neighborhood = reference_distances <= neighborhood_cutoff_angstrom
    np.fill_diagonal(neighborhood, False)
    deformation = np.asarray(
        [
            np.abs(comparison_distances[index] - reference_distances[index])[mask].mean()
            if mask.any()
            else np.nan
            for index, mask in enumerate(neighborhood)
        ]
    )

    half_width = fragment_size // 2
    fragment_rmsd = np.full(len(reference), np.nan)
    for center in range(half_width, len(reference) - half_width):
        bounds = slice(center - half_width, center + half_width + 1)
        fragment_positions = positions[bounds]
        if not np.array_equal(
            fragment_positions,
            np.arange(fragment_positions[0], fragment_positions[0] + fragment_size),
        ):
            continue
        fragment_aligned, _, _ = kabsch_align(comparison_ca[bounds], reference_ca[bounds])
        fragment_rmsd[center] = float(
            np.sqrt(np.mean(np.sum((fragment_aligned - reference_ca[bounds]) ** 2, axis=1)))
        )

    reference_phi, reference_psi = _torsions(reference, positions)
    comparison_phi, comparison_psi = _torsions(comparison, positions)
    phi_delta = np.asarray(
        [_circular_delta(left, right) for left, right in zip(reference_phi, comparison_phi)]
    )
    psi_delta = np.asarray(
        [_circular_delta(left, right) for left, right in zip(reference_psi, comparison_psi)]
    )
    torsion_change = np.sqrt((phi_delta**2 + psi_delta**2) / 2.0)
    return pd.DataFrame(
        {
            "canonical_position": positions,
            "ca_displacement": displacement,
            "fragment_7_rmsd": fragment_rmsd,
            "torsion_phi_psi_change": torsion_change,
            "neighborhood_distance_deformation": deformation,
            "neighborhood_cutoff_angstrom": neighborhood_cutoff_angstrom,
            "fragment_size": fragment_size,
        }
    )


def within_protein_spearman(
    local_response: pd.DataFrame,
    *,
    descriptors: tuple[str, ...] = LOCAL_DESCRIPTORS,
) -> pd.DataFrame:
    """Associate local geometry and JSD within each protein, retaining proteins as units."""

    keys = ["model_id", "checkpoint_id", "protein_id", "identity_cluster_id"]
    required = set(keys + ["jsd_bits", *descriptors])
    missing = sorted(required.difference(local_response.columns))
    if missing:
        raise ValueError(f"local response is missing association columns: {missing}")
    rows = []
    for values, group in local_response.groupby(keys, dropna=False, sort=True):
        prefix = dict(zip(keys, values, strict=True))
        for descriptor in descriptors:
            selected = group[[descriptor, "jsd_bits"]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = float("nan")
            if (
                len(selected) >= 3
                and selected[descriptor].nunique() > 1
                and selected["jsd_bits"].nunique() > 1
            ):
                rho = float(spearmanr(selected[descriptor], selected["jsd_bits"]).statistic)
            rows.append(
                {
                    **prefix,
                    "descriptor": descriptor,
                    "spearman_rho": rho,
                    "n_observations": len(selected),
                    "association_scope": "within_protein_pooled_controlled_pairs",
                }
            )
    return pd.DataFrame(rows)


def cross_model_localization(local_response: pd.DataFrame) -> pd.DataFrame:
    """Compare matched position-wise JSD profiles between every model pair."""

    required = {
        "model_id",
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "pair_id",
        "canonical_position",
        "jsd_bits",
    }
    missing = sorted(required.difference(local_response.columns))
    if missing:
        raise ValueError(f"local response is missing localization columns: {missing}")
    rows = []
    models = sorted(local_response["model_id"].unique())
    for left_model, right_model in combinations(models, 2):
        left = local_response.loc[local_response["model_id"].eq(left_model)]
        right = local_response.loc[local_response["model_id"].eq(right_model)]
        merge_keys = ["protein_id", "identity_cluster_id", "pair_id", "canonical_position"]
        matched = left.merge(
            right,
            on=merge_keys,
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for values, group in matched.groupby(
            ["protein_id", "identity_cluster_id"], dropna=False, sort=True
        ):
            rho = float("nan")
            if (
                len(group) >= 3
                and group["jsd_bits_left"].nunique() > 1
                and group["jsd_bits_right"].nunique() > 1
            ):
                rho = float(spearmanr(group["jsd_bits_left"], group["jsd_bits_right"]).statistic)
            rows.append(
                {
                    "left_model_id": left_model,
                    "right_model_id": right_model,
                    "left_checkpoint_id": str(group.iloc[0]["checkpoint_id_left"]),
                    "right_checkpoint_id": str(group.iloc[0]["checkpoint_id_right"]),
                    "protein_id": values[0],
                    "identity_cluster_id": values[1],
                    "spearman_rho": rho,
                    "n_matched_observations": len(group),
                    "association_scope": "within_protein_matched_pair_residue_profiles",
                }
            )
    return pd.DataFrame(rows)
