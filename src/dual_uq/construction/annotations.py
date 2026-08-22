"""Shared structural descriptors for an already oriented pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dual_uq.geometry import kabsch_align, pairwise_distances

_MAX_ABSOLUTE_CA_COORDINATE_ANGSTROM = 1_000_000.0


@dataclass(frozen=True, slots=True)
class StructuralDescriptorPolicy:
    """Generic cutoffs and provenance for aligned C-alpha descriptors."""

    contact_cutoff_angstrom: float = 8.0
    neighborhood_cutoff_angstrom: float = 12.0
    descriptor_version: str = "controlled_perturbation_geometry_v1"


def _validate_pair_inputs(
    *, pair_id: str, reference_ca: np.ndarray, perturbed_ca: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError("pair_id must be a non-empty string")

    reference = np.asarray(reference_ca, dtype=float)
    perturbed = np.asarray(perturbed_ca, dtype=float)
    if reference.shape != perturbed.shape or reference.ndim != 2 or reference.shape[1:] != (3,):
        raise ValueError("CA arrays must have equal shape [residue, 3]")
    if len(reference) < 3:
        raise ValueError("at least three coordinate pairs are required for Kabsch alignment")
    if not np.isfinite(reference).all() or not np.isfinite(perturbed).all():
        raise ValueError("CA coordinates must be finite")
    if (
        np.abs(reference).max() > _MAX_ABSOLUTE_CA_COORDINATE_ANGSTROM
        or np.abs(perturbed).max() > _MAX_ABSOLUTE_CA_COORDINATE_ANGSTROM
    ):
        raise ValueError(
            "CA coordinates must remain within the scientifically possible coordinate "
            "range of +/-1,000,000 angstrom"
        )
    return reference, perturbed


def _validate_policy(policy: StructuralDescriptorPolicy) -> None:
    cutoffs = (policy.contact_cutoff_angstrom, policy.neighborhood_cutoff_angstrom)
    if any(not np.isfinite(cutoff) or cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("descriptor cutoffs must be finite and positive")
    if not isinstance(policy.descriptor_version, str) or not policy.descriptor_version.strip():
        raise ValueError("descriptor_version must be a non-empty string")


def annotate_aligned_pair(
    *,
    pair_id: str,
    reference_ca: np.ndarray,
    perturbed_ca: np.ndarray,
    reference_distances: np.ndarray | None = None,
    policy: StructuralDescriptorPolicy | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Annotate one aligned coordinate pair using generic C-alpha relations.

    The returned residue table stores perturbation-projection values in
    ``attrs[\"perturbation_projection\"]``.  This non-tabular provenance channel
    deliberately keeps the pair table compatible with its frozen schema while
    exposing ``realized_pairwise_distance_change`` for a later projection to the
    perturbation-descriptor schema.
    """
    reference, perturbed = _validate_pair_inputs(
        pair_id=pair_id, reference_ca=reference_ca, perturbed_ca=perturbed_ca
    )
    effective_policy = policy or StructuralDescriptorPolicy()
    _validate_policy(effective_policy)

    aligned_perturbed, _, _ = kabsch_align(perturbed, reference)
    displacement = np.linalg.norm(aligned_perturbed - reference, axis=1)
    if reference_distances is None:
        reference_distances = pairwise_distances(reference)
    else:
        reference_distances = np.asarray(reference_distances, dtype=float)
        expected_shape = (len(reference), len(reference))
        if reference_distances.shape != expected_shape or not np.isfinite(reference_distances).all():
            raise ValueError("reference_distances must be a finite square CA distance matrix")
    perturbed_distances = pairwise_distances(aligned_perturbed)
    distance_change = np.abs(perturbed_distances - reference_distances)

    diagonal = np.eye(len(reference), dtype=bool)
    neighborhood = (
        reference_distances <= effective_policy.neighborhood_cutoff_angstrom
    ) & ~diagonal
    reference_contacts = (
        reference_distances <= effective_policy.contact_cutoff_angstrom
    ) & ~diagonal
    perturbed_contacts = (
        perturbed_distances <= effective_policy.contact_cutoff_angstrom
    ) & ~diagonal
    contact_changed = np.logical_xor(reference_contacts, perturbed_contacts)
    contact_union = np.logical_or(reference_contacts, perturbed_contacts)

    local_pairwise_change = np.array(
        [
            distance_change[index, mask].mean() if mask.any() else 0.0
            for index, mask in enumerate(neighborhood)
        ]
    )
    neighborhood_upper = np.triu(neighborhood, k=1)
    global_pairwise_distance_change = (
        float(distance_change[neighborhood_upper].mean())
        if neighborhood_upper.any()
        else None
    )
    changed_upper = np.triu(contact_changed, k=1)
    union_upper = np.triu(contact_union, k=1)
    global_contact_change = (
        float(changed_upper.sum() / union_upper.sum()) if union_upper.any() else 0.0
    )

    method = "kabsch_ca_pairwise_contacts"
    pair_descriptor = {
        "pair_id": pair_id,
        "aligned_ca_rmsd": float(np.sqrt(np.mean(displacement**2))),
        "median_residue_displacement": float(np.median(displacement)),
        "upper_tail_residue_displacement": float(np.quantile(displacement, 0.9)),
        "global_contact_change": global_contact_change,
        "descriptor_method": method,
        "descriptor_version": effective_policy.descriptor_version,
    }
    residue_descriptor = pd.DataFrame(
        {
            "pair_id": pair_id,
            "canonical_position": np.arange(1, len(reference) + 1, dtype=int),
            "ca_displacement": displacement,
            "local_pairwise_distance_change": local_pairwise_change,
            "neighborhood_geometry_change": local_pairwise_change,
            "contact_gain": np.any(perturbed_contacts & ~reference_contacts, axis=1),
            "contact_loss": np.any(reference_contacts & ~perturbed_contacts, axis=1),
            "descriptor_method": method,
            "descriptor_version": effective_policy.descriptor_version,
        }
    )
    residue_descriptor.attrs["perturbation_projection"] = {
        "realized_pairwise_distance_change": global_pairwise_distance_change
    }
    return pair_descriptor, residue_descriptor


def annotate_pair_geometry(
    pair_id: str, condition_1_ca: np.ndarray, condition_2_ca: np.ndarray
) -> dict[str, object]:
    """Return the legacy direct-coordinate annotation primitive."""
    left, right = np.asarray(condition_1_ca, dtype=float), np.asarray(condition_2_ca, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("aligned CA arrays must have equal shape [residue, 3]")
    displacement = np.linalg.norm(right - left, axis=1)
    return {
        "pair_id": pair_id,
        "aligned_ca_rmsd": float(np.sqrt(np.mean(displacement**2))),
        "median_residue_displacement": float(np.median(displacement)),
        "upper_tail_residue_displacement": float(np.quantile(displacement, 0.9)),
        "global_contact_change": None,
        "direction": "condition_2_minus_condition_1",
    }
