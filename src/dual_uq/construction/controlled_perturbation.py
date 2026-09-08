"""Controlled-perturbation construction adapter.

This module owns only the intervention-specific boundary.  Residue mapping,
pair comparability, and generic admission remain delegated to the canonical
construction owners.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.construction.admission import AdmissionPolicy, evaluate_pair
from dual_uq.construction.comparability import compare_mapped_pair
from dual_uq.construction.mapping import CANONICAL_PAIR_MAPPING_COLUMNS, map_condition_pair
from dual_uq.construction.structures import normalize_structure
from dual_uq.geometry import kabsch_align, obvious_geometry_counts, rmsd
from dual_uq.structcal.ids import pair_id as canonical_pair_id
from dual_uq.structcal.schema_registry import SchemaRegistry
from dual_uq.structcal.tables import validate_frame

CONTROLLED_PERTURBATION = "controlled_perturbation"
REFERENCE = "REFERENCE"
PERTURBED = "PERTURBED"
SUPPORTED_PERTURBATION_FAMILIES = frozenset(
    {"COORDINATE_NOISE", "SHEAR", "CONTACT_RELATIONAL_DEFORMATION"}
)
SUPPORTED_PERTURBATION_DOSES = (0.25, 0.50, 1.00, 2.00)
_CONTACT_BLOCK_FRACTION = 0.25
_NOISE_ANCHOR_SPACING = 8
_CONTACT_TRANSITION_MAX_WIDTH = 4
_CALIBRATION_TOLERANCE = 0.05
_CALIBRATION_VALIDITY_SEARCH_STEPS = 8
_CALIBRATION_BINARY_SEARCH_STEPS = 12


@dataclass(frozen=True, slots=True)
class CoordinateStructure:
    """Complete selected-chain coordinates with immutable structural identity."""

    protein_id: str
    source_structure_id: str
    residue_positions: tuple[int, ...]
    residue_names: tuple[str, ...]
    atom_names: tuple[tuple[str, ...], ...]
    atom_coordinates: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        residue_count = len(self.residue_positions)
        if residue_count < 3:
            raise ValueError("coordinate structures require at least three residues")
        if not str(self.protein_id).strip() or not str(self.source_structure_id).strip():
            raise ValueError("coordinate structures require protein and source structure identities")
        if len(set(self.residue_positions)) != residue_count:
            raise ValueError("coordinate structures require unique residue positions")
        if len(self.residue_names) != residue_count or len(self.atom_names) != residue_count:
            raise ValueError("coordinate structure residue identity lengths do not match")
        if len(self.atom_coordinates) != residue_count:
            raise ValueError("coordinate structure coordinate lengths do not match")

        normalized_atom_names: list[tuple[str, ...]] = []
        normalized_coordinates: list[np.ndarray] = []
        for residue_name, names, coordinates in zip(
            self.residue_names, self.atom_names, self.atom_coordinates, strict=True
        ):
            if not str(residue_name).strip() or not names or any(not str(name).strip() for name in names):
                raise ValueError("coordinate structures require non-empty residue and atom identities")
            coordinate_array = np.array(coordinates, dtype=float, copy=True)
            if coordinate_array.shape != (len(names), 3):
                raise ValueError("each residue coordinate array must have shape (atom_count, 3)")
            if not np.isfinite(coordinate_array).all():
                raise ValueError("coordinate structures require finite coordinates")
            coordinate_array.setflags(write=False)
            normalized_atom_names.append(tuple(str(name) for name in names))
            normalized_coordinates.append(coordinate_array)
        object.__setattr__(self, "residue_positions", tuple(int(position) for position in self.residue_positions))
        object.__setattr__(self, "residue_names", tuple(str(name) for name in self.residue_names))
        object.__setattr__(self, "atom_names", tuple(normalized_atom_names))
        object.__setattr__(self, "atom_coordinates", tuple(normalized_coordinates))


@dataclass(frozen=True, slots=True)
class OperatorResult:
    """A coordinate-only perturbation and its outcome-blind provenance."""

    perturbed: CoordinateStructure
    operator_metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ControlledPerturbationSpec:
    """Requested intervention semantics and provenance only."""

    perturbation_family: str
    requested_dose: float
    requested_dose_unit: str
    random_seed: int
    perturbation_method: str
    perturbation_version: str
    correlation_spacing_residues: int | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDescriptors:
    """Realized geometry, deliberately separate from admission inputs."""

    realized_ca_rmsd: float | None
    realized_pairwise_distance_change: float | None
    realized_contact_change: float | None


def _ca_coordinates(structure: CoordinateStructure) -> np.ndarray:
    coordinates: list[np.ndarray] = []
    for names, residue_coordinates in zip(
        structure.atom_names, structure.atom_coordinates, strict=True
    ):
        try:
            ca_index = names.index("CA")
        except ValueError as exc:
            raise ValueError("coordinate structures require one CA atom per residue") from exc
        coordinates.append(residue_coordinates[ca_index])
    return np.vstack(coordinates)


def _semantic_seed(reference: CoordinateStructure, spec: ControlledPerturbationSpec) -> int:
    fields = [
        str(spec.random_seed),
        reference.protein_id,
        reference.source_structure_id,
        str(spec.perturbation_family).strip().upper(),
        f"{float(spec.requested_dose):.8f}",
        str(spec.perturbation_version).strip(),
    ]
    if spec.correlation_spacing_residues is not None:
        fields.append(str(spec.correlation_spacing_residues))
    material = "|".join(fields)
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")


def _contact_selection_seed(reference: CoordinateStructure, spec: ControlledPerturbationSpec) -> int:
    material = "|".join(
        (
            str(spec.random_seed),
            reference.protein_id,
            reference.source_structure_id,
            "CONTACT_RELATIONAL_DEFORMATION",
            str(spec.perturbation_version).strip(),
            "length_relative_coordinate_complete_v1",
        )
    )
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")


def _replace_coordinates(
    reference: CoordinateStructure, coordinates: tuple[np.ndarray, ...]
) -> CoordinateStructure:
    return CoordinateStructure(
        protein_id=reference.protein_id,
        source_structure_id=reference.source_structure_id,
        residue_positions=reference.residue_positions,
        residue_names=reference.residue_names,
        atom_names=reference.atom_names,
        atom_coordinates=coordinates,
    )


def _assert_coordinate_identity(
    reference: CoordinateStructure, perturbed: CoordinateStructure
) -> None:
    if (
        perturbed.protein_id != reference.protein_id
        or perturbed.source_structure_id != reference.source_structure_id
        or perturbed.residue_positions != reference.residue_positions
        or perturbed.residue_names != reference.residue_names
        or perturbed.atom_names != reference.atom_names
    ):
        raise RuntimeError("perturbation changed structural identity")
    if any(not np.isfinite(coordinates).all() for coordinates in perturbed.atom_coordinates):
        raise RuntimeError("perturbation produced non-finite coordinates")


def _aligned_ca_rmsd(reference: CoordinateStructure, perturbed: CoordinateStructure) -> float:
    reference_ca = _ca_coordinates(reference)
    perturbed_ca = _ca_coordinates(perturbed)
    aligned, _, _ = kabsch_align(perturbed_ca, reference_ca)
    return rmsd(aligned, reference_ca)


def _calibrate(
    reference: CoordinateStructure,
    *,
    target_dose: float,
    transform: Any,
    geometry_valid: Callable[[CoordinateStructure], bool] | None = None,
) -> tuple[CoordinateStructure, float, float]:
    """Search geometry-valid native parameters near the shared RMSD target."""

    if geometry_valid is None:
        if target_dose == 0:
            unchanged = transform(0.0)
            return unchanged, 0.0, _aligned_ca_rmsd(reference, unchanged)
        upper = 1.0
        upper_structure = transform(upper)
        upper_rmsd = _aligned_ca_rmsd(reference, upper_structure)
        while upper_rmsd < target_dose and upper < 1_048_576.0:
            upper *= 2.0
            upper_structure = transform(upper)
            upper_rmsd = _aligned_ca_rmsd(reference, upper_structure)
        if upper_rmsd < target_dose:
            raise ValueError("SEVERITY_CALIBRATION_FAILURE")
        lower = 0.0
        for _ in range(_CALIBRATION_BINARY_SEARCH_STEPS):
            midpoint = (lower + upper) / 2.0
            candidate = transform(midpoint)
            candidate_rmsd = _aligned_ca_rmsd(reference, candidate)
            if candidate_rmsd < target_dose:
                lower = midpoint
            else:
                upper = midpoint
        calibrated = transform((lower + upper) / 2.0)
        return calibrated, (lower + upper) / 2.0, _aligned_ca_rmsd(reference, calibrated)

    validator = geometry_valid

    def evaluate(parameter: float) -> tuple[CoordinateStructure, float, bool]:
        candidate = transform(float(parameter))
        candidate_rmsd = _aligned_ca_rmsd(reference, candidate)
        return candidate, candidate_rmsd, bool(validator(candidate))

    if target_dose == 0:
        unchanged, realized, valid = evaluate(0.0)
        if not valid:
            raise ValueError("SEVERITY_CALIBRATION_FAILURE")
        return unchanged, 0.0, realized

    upper = 1.0
    upper_structure = transform(upper)
    upper_rmsd = _aligned_ca_rmsd(reference, upper_structure)
    while upper_rmsd < target_dose and upper < 1_048_576.0:
        upper *= 2.0
        upper_structure = transform(upper)
        upper_rmsd = _aligned_ca_rmsd(reference, upper_structure)
    if upper_rmsd < target_dose:
        raise ValueError("SEVERITY_CALIBRATION_FAILURE")

    lower = 0.0
    _, _, lower_valid = evaluate(lower)
    if not lower_valid:
        raise ValueError("SEVERITY_CALIBRATION_FAILURE")
    exact_parameter = upper
    for _ in range(_CALIBRATION_BINARY_SEARCH_STEPS):
        midpoint = (lower + upper) / 2.0
        candidate = transform(midpoint)
        candidate_rmsd = _aligned_ca_rmsd(reference, candidate)
        if candidate_rmsd < target_dose:
            lower = midpoint
        else:
            upper = midpoint
    exact_parameter = (lower + upper) / 2.0
    exact_candidate, exact_rmsd, exact_valid = evaluate(exact_parameter)
    if exact_valid and abs(exact_rmsd - target_dose) <= _CALIBRATION_TOLERANCE:
        return exact_candidate, exact_parameter, exact_rmsd

    base_candidate, base_rmsd, base_valid = evaluate(0.0)
    if not base_valid:
        raise ValueError("SEVERITY_CALIBRATION_FAILURE")
    best: tuple[float, CoordinateStructure, float] = (0.0, base_candidate, base_rmsd)
    valid_lower = 0.0
    invalid_upper = exact_parameter
    for _ in range(_CALIBRATION_VALIDITY_SEARCH_STEPS):
        midpoint = (valid_lower + invalid_upper) / 2.0
        candidate, realized, valid = evaluate(midpoint)
        if valid:
            valid_lower = midpoint
            if abs(realized - target_dose) < abs(best[2] - target_dose):
                best = (midpoint, candidate, realized)
        else:
            invalid_upper = midpoint

    if abs(best[2] - target_dose) > _CALIBRATION_TOLERANCE:
        raise ValueError("SEVERITY_CALIBRATION_FAILURE")
    return best[1], best[0], best[2]


def _smoothed_noise_displacements(
    residue_count: int,
    rng: np.random.Generator,
    *,
    anchor_spacing: int = _NOISE_ANCHOR_SPACING,
) -> np.ndarray:
    if residue_count < 2:
        raise ValueError("noise displacement field requires at least two residues")
    if isinstance(anchor_spacing, bool) or not isinstance(anchor_spacing, (int, np.integer)):
        raise ValueError("noise displacement anchor spacing must be a positive integer")  # noqa: TRY004
    if int(anchor_spacing) < 1:
        raise ValueError("noise displacement anchor spacing must be a positive integer")
    anchor_count = max(
        3,
        int(np.ceil((residue_count - 1) / int(anchor_spacing))) + 1,
    )
    anchor_positions = np.linspace(0.0, residue_count - 1, anchor_count)
    anchors = rng.normal(size=(len(anchor_positions), 3))
    residue_positions = np.arange(residue_count, dtype=float)
    return np.column_stack(
        [np.interp(residue_positions, anchor_positions, anchors[:, coordinate]) for coordinate in range(3)]
    )


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _contact_transition_weights(
    residue_count: int,
    block_start: int,
    block_length: int,
) -> tuple[np.ndarray, int]:
    block_end = block_start + block_length - 1
    left_flank = block_start
    right_flank = residue_count - block_end - 1
    transition_width = max(
        1,
        min(_CONTACT_TRANSITION_MAX_WIDTH, max(left_flank, right_flank)),
    )
    weights = np.zeros(residue_count, dtype=float)
    weights[block_start : block_end + 1] = 1.0
    for distance in range(1, transition_width + 1):
        weight = _smoothstep((transition_width + 1.0 - distance) / (transition_width + 1.0))
        left_index = block_start - distance
        right_index = block_end + distance
        if left_index >= 0:
            weights[left_index] = weight
        if right_index < residue_count:
            weights[right_index] = weight
    return weights, transition_width


def _project_ca_chain(
    reference_ca: np.ndarray,
    proposed_ca: np.ndarray,
    residue_positions: tuple[int, ...],
) -> np.ndarray:
    """Preserve existing consecutive Cα distances after a smooth field update."""

    projected = np.array(proposed_ca, dtype=float, copy=True)
    for index in range(1, len(projected)):
        if residue_positions[index] != residue_positions[index - 1] + 1:
            continue
        target_vector = reference_ca[index] - reference_ca[index - 1]
        target_distance = float(np.linalg.norm(target_vector))
        direction = projected[index] - projected[index - 1]
        direction_norm = float(np.linalg.norm(direction))
        if target_distance == 0.0:
            continue
        if direction_norm == 0.0:
            direction = target_vector
            direction_norm = target_distance
        projected[index] = projected[index - 1] + direction * (target_distance / direction_norm)
    return projected


def _apply_residue_displacement_field(
    reference: CoordinateStructure,
    displacements: np.ndarray,
    scale: float,
) -> CoordinateStructure:
    reference_ca = _ca_coordinates(reference)
    proposed_ca = reference_ca + scale * displacements
    projected_ca = _project_ca_chain(
        reference_ca,
        proposed_ca,
        reference.residue_positions,
    )
    coordinates: list[np.ndarray] = []
    for index, (residue_coordinates, displacement) in enumerate(
        zip(reference.atom_coordinates, displacements, strict=True)
    ):
        raw_shift = scale * displacement
        correction = projected_ca[index] - proposed_ca[index]
        coordinates.append(residue_coordinates + raw_shift + correction)
    return _replace_coordinates(reference, tuple(coordinates))


def _operator_geometry_valid(
    reference: CoordinateStructure,
    candidate: CoordinateStructure,
    *,
    reference_counts: dict[str, int] | None = None,
) -> bool:
    reference_counts = reference_counts or obvious_geometry_counts(
        reference.atom_coordinates,
        reference.atom_names,
        reference.residue_positions,
    )
    candidate_counts = obvious_geometry_counts(
        candidate.atom_coordinates,
        candidate.atom_names,
        candidate.residue_positions,
    )
    return (
        candidate_counts["nonfinite_atom_count"] == 0
        and candidate_counts["ca_chain_break_count"] <= reference_counts["ca_chain_break_count"]
        and candidate_counts["severe_clash_count"] <= reference_counts["severe_clash_count"]
    )


def _principal_frame(ca_coordinates: np.ndarray) -> np.ndarray:
    centered = ca_coordinates - ca_coordinates.mean(axis=0)
    _, _, frame = np.linalg.svd(centered, full_matrices=False)
    frame = frame.T
    for column in range(3):
        dominant = int(np.argmax(np.abs(frame[:, column])))
        if frame[dominant, column] < 0:
            frame[:, column] *= -1.0
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1.0
    return frame


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross_product = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3)
        + np.sin(angle) * cross_product
        + (1.0 - np.cos(angle)) * (cross_product @ cross_product)
    )


def _contact_block_indices(
    reference: CoordinateStructure, *, semantic_seed: int
) -> tuple[int, int]:
    residue_count = len(reference.residue_positions)
    block_length = max(3, round(residue_count * _CONTACT_BLOCK_FRACTION))
    if residue_count - block_length < 3:
        raise ValueError("CONTACT_RELATIONAL_DEFORMATION requires at least six residues")
    start = int(np.random.default_rng(semantic_seed).integers(0, residue_count - block_length + 1))
    return start, block_length


def _validate_operator_spec(spec: ControlledPerturbationSpec) -> tuple[str, float]:
    _validate_requested_dose(spec)
    family = str(spec.perturbation_family).strip().upper()
    if family not in SUPPORTED_PERTURBATION_FAMILIES:
        raise ValueError(f"unsupported perturbation family: {family}")
    dose = float(spec.requested_dose)
    if dose not in SUPPORTED_PERTURBATION_DOSES:
        raise ValueError(f"unsupported perturbation dose: {dose}")
    if str(spec.requested_dose_unit).strip().lower() != "angstrom":
        raise ValueError("operator requested_dose_unit must be angstrom")
    if spec.correlation_spacing_residues is not None and (
        isinstance(spec.correlation_spacing_residues, bool)
        or not isinstance(spec.correlation_spacing_residues, (int, np.integer))
        or int(spec.correlation_spacing_residues) < 1
    ):
        raise ValueError("correlation_spacing_residues must be a positive integer")
    return family, dose


def apply_perturbation(
    reference: CoordinateStructure, spec: ControlledPerturbationSpec
) -> OperatorResult:
    """Apply one calibrated perturbation using the stable public contract."""

    return _apply_perturbation(
        reference,
        spec,
        validate_geometry_during_calibration=True,
    )


def _apply_perturbation(
    reference: CoordinateStructure,
    spec: ControlledPerturbationSpec,
    *,
    validate_geometry_during_calibration: bool = True,
) -> OperatorResult:
    """Apply one calibrated, coordinate-only controlled perturbation."""

    family, target_dose = _validate_operator_spec(spec)
    reference_ca = _ca_coordinates(reference)
    semantic_seed = _semantic_seed(reference, spec)
    metadata: dict[str, object] = {
        "perturbation_family": family,
        "requested_dose": target_dose,
        "requested_dose_unit": "angstrom",
        "random_seed": spec.random_seed,
        "semantic_seed": semantic_seed,
        "calibration_target": "kabsch_aligned_ca_rmsd_angstrom",
    }

    if family == "COORDINATE_NOISE":
        correlation_spacing = (
            int(spec.correlation_spacing_residues)
            if spec.correlation_spacing_residues is not None
            else _NOISE_ANCHOR_SPACING
        )
        displacements = _smoothed_noise_displacements(
            len(reference.residue_positions),
            np.random.default_rng(semantic_seed),
            anchor_spacing=correlation_spacing,
        )

        def transform(scale: float) -> CoordinateStructure:
            return _apply_residue_displacement_field(reference, displacements, scale)

        metadata["operator_method"] = "smoothed_seeded_anchor_interpolation_chain_projected"
        metadata["correlation_spacing_residues"] = correlation_spacing
    elif family == "SHEAR":
        centroid = reference_ca.mean(axis=0)
        frame = _principal_frame(reference_ca)

        def transform(scale: float) -> CoordinateStructure:
            affine = np.eye(3)
            affine[0, 1] = scale
            return _replace_coordinates(
                reference,
                tuple(
                    ((coordinates - centroid) @ frame @ affine @ frame.T) + centroid
                    for coordinates in reference.atom_coordinates
                ),
            )

        metadata["operator_method"] = "centroided_principal_frame_affine_shear"
    else:
        frame = _principal_frame(reference_ca)
        selection_seed = _contact_selection_seed(reference, spec)
        block_start, block_length = _contact_block_indices(reference, semantic_seed=selection_seed)
        transition_weights, transition_width = _contact_transition_weights(
            len(reference.residue_positions), block_start, block_length
        )
        block_centroid = reference_ca[block_start : block_start + block_length].mean(axis=0)
        axis = frame[:, 2]
        translation_direction = frame[:, 0]

        def transform(scale: float) -> CoordinateStructure:
            angle = np.arctan(0.2 * scale)
            return _replace_coordinates(
                reference,
                tuple(
                    (
                        (coordinates - block_centroid)
                        @ _axis_angle_rotation(axis, transition_weights[index] * angle).T
                        + block_centroid
                        + transition_weights[index] * scale * translation_direction
                    )
                    if transition_weights[index] > 0.0
                    else coordinates.copy()
                    for index, coordinates in enumerate(reference.atom_coordinates)
                ),
            )

        metadata.update(
            {
                "operator_method": "frozen_length_relative_rigid_core_smooth_hinge_transform",
                "block_selection_rule": "length_relative_coordinate_complete_v1",
                "block_selection_seed": selection_seed,
                "block_start_position": reference.residue_positions[block_start],
                "block_end_position": reference.residue_positions[block_start + block_length - 1],
                "block_length": block_length,
                "transition_width_residues": transition_width,
            }
        )

    reference_counts = (
        obvious_geometry_counts(
            reference.atom_coordinates,
            reference.atom_names,
            reference.residue_positions,
        )
        if family != "SHEAR"
        else None
    )
    geometry_validator = (
        (
            lambda candidate: _operator_geometry_valid(
                reference,
                candidate,
                reference_counts=reference_counts,
            )
        )
        if family != "SHEAR" and validate_geometry_during_calibration
        else None
    )
    perturbed, native_parameter, realized_rmsd = _calibrate(
        reference,
        target_dose=target_dose,
        transform=transform,
        geometry_valid=geometry_validator,
    )
    if (
        family != "SHEAR"
        and not validate_geometry_during_calibration
        and not _operator_geometry_valid(
            reference,
            perturbed,
            reference_counts=reference_counts,
        )
    ):
        raise ValueError("SEVERITY_CALIBRATION_FAILURE")
    _assert_coordinate_identity(reference, perturbed)
    metadata["native_parameter"] = native_parameter
    metadata["realized_ca_rmsd"] = realized_rmsd
    return OperatorResult(perturbed=perturbed, operator_metadata=metadata)


def _inheritance_positions(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    positions = pd.to_numeric(frame["canonical_position"], errors="coerce")
    if positions.isna().any() or positions.mod(1).ne(0).any() or positions.le(0).any():
        raise ValueError(f"{context} canonical positions must be positive integers")
    if positions.duplicated().any():
        raise ValueError(f"{context} contains duplicate canonical positions")
    normalized = frame.copy()
    normalized["canonical_position"] = positions.astype(int)
    return normalized


def _require_inheritance_boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"parent mapping {field} must be boolean")
    return bool(value)


def _missing_value(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def _inherited_missing_reason(
    value: object, *, mapped: bool, coordinate_visible: bool
) -> str | None:
    if coordinate_visible:
        return None
    if not _missing_value(value) and str(value).strip():
        return str(value)
    return "COORDINATES_UNAVAILABLE" if mapped else "UNMAPPED"


def inherit_mapping(
    *,
    pair_id: str,
    protein_id: str,
    canonical: pd.DataFrame,
    parent_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Serialize one parent correspondence identically for reference and perturbed.

    Generated coordinates retain the parent's residue identities, so this function
    only validates and copies the existing correspondence.  It deliberately does
    not align sequences or structures, or invoke an external mapping source.
    """

    if not str(pair_id).strip() or not str(protein_id).strip():
        raise ValueError("inherited mapping requires non-empty pair_id and protein_id")
    canonical_required = {"canonical_position", "canonical_aa"}
    parent_required = {
        "canonical_position",
        "residue_id",
        "aa",
        "mapped",
        "coordinate_visible",
        "missing_reason",
    }
    if not canonical_required <= set(canonical.columns):
        raise ValueError("canonical mapping requires canonical_position and canonical_aa")
    if not parent_required <= set(parent_mapping.columns):
        raise ValueError("parent mapping lacks required inherited correspondence fields")

    canonical_frame = _inheritance_positions(canonical, context="canonical mapping")
    parent_frame = _inheritance_positions(parent_mapping, context="parent mapping")
    if canonical_frame["canonical_aa"].isna().any() or any(
        not str(amino_acid).strip() for amino_acid in canonical_frame["canonical_aa"]
    ):
        raise ValueError("canonical mapping requires non-empty canonical amino-acid identities")
    if len(parent_frame) != len(canonical_frame) or set(parent_frame["canonical_position"]) != set(
        canonical_frame["canonical_position"]
    ):
        raise ValueError("parent mapping must cover canonical positions exactly")

    parent_by_position = parent_frame.set_index("canonical_position").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for canonical_row in canonical_frame.to_dict(orient="records"):
        position = int(canonical_row["canonical_position"])
        canonical_aa = canonical_row["canonical_aa"]
        parent_row = parent_by_position[position]
        mapped = _require_inheritance_boolean(parent_row["mapped"], field="mapped")
        coordinate_visible = _require_inheritance_boolean(
            parent_row["coordinate_visible"], field="coordinate_visible"
        )
        if coordinate_visible and not mapped:
            raise ValueError("parent mapping cannot expose coordinates for an unmapped residue")

        parent_residue_id = parent_row["residue_id"]
        parent_aa = parent_row["aa"]
        if not mapped and (
            not _missing_value(parent_residue_id) or not _missing_value(parent_aa)
        ):
            raise ValueError("unmapped parent residues cannot have residue or amino-acid identities")
        if mapped and (
            _missing_value(parent_residue_id)
            or not str(parent_residue_id).strip()
            or _missing_value(parent_aa)
            or not str(parent_aa).strip()
        ):
            raise ValueError("mapped parent residues require non-empty residue and amino-acid identities")
        # The canonical amino acid identifies the inherited coordinate
        # position.  The observed parent amino acid is an independent,
        # persisted mapping fact and may legitimately be a variant.
        if coordinate_visible and (
            not _missing_value(parent_row["missing_reason"])
            and str(parent_row["missing_reason"]).strip()
        ):
            raise ValueError("coordinate-visible parent residues cannot have a missing reason")

        residue_id = None if not mapped else parent_residue_id
        parent_aa = None if not mapped else parent_aa
        missing_reason = _inherited_missing_reason(
            parent_row["missing_reason"],
            mapped=mapped,
            coordinate_visible=coordinate_visible,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "canonical_position": position,
                "canonical_aa": canonical_aa,
                "condition_1_residue_id": residue_id,
                "condition_2_residue_id": residue_id,
                "condition_1_aa": parent_aa,
                "condition_2_aa": parent_aa,
                "condition_1_mapped": mapped,
                "condition_2_mapped": mapped,
                "condition_1_coordinate_visible": coordinate_visible,
                "condition_2_coordinate_visible": coordinate_visible,
                "common_mapped": mapped,
                "common_coordinate_visible": coordinate_visible,
                "condition_1_missing_reason": missing_reason,
                "condition_2_missing_reason": missing_reason,
                "mapping_status": (
                    "COMMON_VISIBLE"
                    if coordinate_visible
                    else "MAPPED_NOT_VISIBLE"
                    if mapped
                    else "PARTIAL_OR_UNMAPPED"
                ),
            }
        )
    inherited = pd.DataFrame(rows, columns=CANONICAL_PAIR_MAPPING_COLUMNS)
    for field in (
        "condition_1_residue_id",
        "condition_2_residue_id",
        "condition_1_aa",
        "condition_2_aa",
        "condition_1_missing_reason",
        "condition_2_missing_reason",
    ):
        inherited[field] = inherited[field].astype(object).where(inherited[field].notna(), None)
    return inherited


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema_registry() -> SchemaRegistry:
    return SchemaRegistry(_repository_root() / "schemas")


def _required_text(record: Mapping[str, object], field: str, context: str) -> str:
    value = record.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{context} requires non-empty {field}")
    return str(value).strip()


def _text_or_none(value: object) -> str | None:
    if _missing_value(value):
        return None
    text = str(value).strip()
    return text or None


def _orient_inputs(
    reference: Mapping[str, object],
    perturbed: Mapping[str, object],
    reference_mapping: pd.DataFrame,
    perturbed_mapping: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], pd.DataFrame]:
    records = [
        (dict(reference), reference_mapping),
        (dict(perturbed), perturbed_mapping),
    ]
    labelled: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
    for record, mapping in records:
        label = str(record.get("condition_label") or "").strip().upper()
        if label not in {REFERENCE, PERTURBED} or label in labelled:
            raise ValueError(
                "controlled perturbation requires exactly one REFERENCE and one PERTURBED record"
            )
        record["condition_label"] = label
        labelled[label] = (record, mapping)
    if set(labelled) != {REFERENCE, PERTURBED}:
        raise ValueError(
            "controlled perturbation requires exactly one REFERENCE and one PERTURBED record"
        )
    reference_record, reference_frame = labelled[REFERENCE]
    perturbed_record, perturbed_frame = labelled[PERTURBED]
    return reference_record, reference_frame, perturbed_record, perturbed_frame


def _validate_pair_input(
    *,
    protein: Mapping[str, object],
    reference: Mapping[str, object],
    perturbed: Mapping[str, object],
) -> str:
    protein_id = _required_text(protein, "protein_id", "protein")
    for label, record in ((REFERENCE, reference), (PERTURBED, perturbed)):
        record_protein = record.get("protein_id")
        if record_protein is not None and str(record_protein) != protein_id:
            raise ValueError(f"{label} structure protein_id does not match protein")
        if _required_text(record, "source_structure_id", label) == "":
            raise ValueError(f"{label} source_structure_id is empty")
        source_type = _required_text(record, "source_type", label)
        if source_type != "CONTROLLED_PERTURBATION":
            raise ValueError(f"{label} source_type must be CONTROLLED_PERTURBATION")
        _required_text(record, "condition_type", label)
        _required_text(record, "source_file_ref", label)
    return protein_id


def _validate_requested_dose(spec: ControlledPerturbationSpec) -> None:
    if isinstance(spec.requested_dose, bool):
        raise TypeError("requested_dose must be a finite non-negative number")
    try:
        dose = float(spec.requested_dose)
    except (TypeError, ValueError) as exc:
        raise ValueError("requested_dose must be a finite non-negative number") from exc
    if not isfinite(dose) or dose < 0:
        raise ValueError("requested_dose must be a finite non-negative number")
    if isinstance(spec.random_seed, bool) or not isinstance(spec.random_seed, int):
        raise TypeError("random_seed must be an integer")


def _intervention_reasons(
    spec: ControlledPerturbationSpec,
    *,
    reference: Mapping[str, object],
    perturbed: Mapping[str, object],
    reference_structure_id: str,
) -> list[str]:
    reasons: list[str] = []
    family = str(spec.perturbation_family).strip().upper()
    if family not in SUPPORTED_PERTURBATION_FAMILIES:
        reasons.append("UNSUPPORTED_PERTURBATION_FAMILY")
    if not str(spec.requested_dose_unit).strip():
        reasons.append("MISSING_REQUESTED_DOSE_UNIT")
    if not str(spec.perturbation_method).strip() or not str(
        spec.perturbation_version
    ).strip():
        reasons.append("MISSING_PERTURBATION_PROVENANCE")
    declared_parent = perturbed.get("parent_structure_id")
    if declared_parent is not None and str(declared_parent) not in {
        reference_structure_id,
        str(reference.get("source_structure_id")),
    }:
        reasons.append("REFERENCE_PARENT_MISMATCH")
    return reasons


def _generic_reasons(decision: Any) -> list[str]:
    reason_map = {
        "SEQUENCE_NOT_COMPARABLE": "SEQUENCE_NOT_COMPARABLE",
        "MAPPING_NOT_COMPARABLE": "MAPPING_NOT_COMPARABLE",
        "COORDINATES_NOT_COMPARABLE": "COORDINATES_NOT_COMPARABLE",
        "CONSTRUCT_NOT_COMPARABLE": "CONSTRUCT_NOT_COMPARABLE",
        "ASSEMBLY_NOT_COMPARABLE": "ASSEMBLY_NOT_COMPARABLE",
    }
    return [
        reason_map[reason]
        for reason in (decision.primary_reason, *decision.secondary_reasons)
        if reason in reason_map
    ]


def _structure_row(
    record: Mapping[str, object],
    *,
    protein_id: str,
    parent_structure_id: str | None,
) -> dict[str, object]:
    normalized = normalize_structure(
        source_structure_id=_required_text(record, "source_structure_id", "structure"),
        protein_id=protein_id,
        structural_condition_semantics=CONTROLLED_PERTURBATION,
        condition_label=_required_text(record, "condition_label", "structure"),
        condition_type=_required_text(record, "condition_type", "structure"),
        source_type=_required_text(record, "source_type", "structure"),
        source_file_ref=_required_text(record, "source_file_ref", "structure"),
        source_accession=_required_text(record, "source_structure_id", "structure"),
        chain_id=_text_or_none(record.get("chain_id")),
        entity_id=_text_or_none(record.get("entity_id")),
        assembly_id=_text_or_none(record.get("assembly_id")),
        experimental_method=_text_or_none(record.get("experimental_method")),
        state_evidence_tier=_text_or_none(record.get("state_evidence_tier")),
        state_evidence_source=_text_or_none(record.get("state_evidence_source")),
        parent_structure_id=parent_structure_id,
    )
    return normalized


def build_controlled_perturbation_tables(
    *,
    protein: Mapping[str, object],
    reference: Mapping[str, object],
    perturbed: Mapping[str, object],
    canonical_residues: pd.DataFrame,
    reference_mapping: pd.DataFrame,
    perturbed_mapping: pd.DataFrame,
    sequence_identity: float | None,
    construct_overlap_fraction: float | None,
    assembly_comparable: bool,
    perturbation: ControlledPerturbationSpec,
    descriptors: PerturbationDescriptors,
) -> dict[str, pd.DataFrame]:
    """Build controlled-perturbation canonical fragments from normalized inputs."""

    (
        reference_record,
        reference_frame,
        perturbed_record,
        perturbed_frame,
    ) = _orient_inputs(reference, perturbed, reference_mapping, perturbed_mapping)
    protein_id = _validate_pair_input(
        protein=protein,
        reference=reference_record,
        perturbed=perturbed_record,
    )
    _validate_requested_dose(perturbation)

    reference_row = _structure_row(
        reference_record,
        protein_id=protein_id,
        parent_structure_id=None,
    )
    perturbed_row = _structure_row(
        perturbed_record,
        protein_id=protein_id,
        parent_structure_id=str(reference_row["structure_id"]),
    )
    reference_structure_id = str(reference_row["structure_id"])
    perturbed_structure_id = str(perturbed_row["structure_id"])
    pair_id = canonical_pair_id(
        protein_id,
        CONTROLLED_PERTURBATION,
        reference_structure_id,
        perturbed_structure_id,
    )

    residue_mappings = map_condition_pair(
        protein_id=protein_id,
        pair_id=pair_id,
        condition_1_label=REFERENCE,
        condition_2_label=PERTURBED,
        canonical=canonical_residues,
        condition_1=reference_frame,
        condition_2=perturbed_frame,
    )
    facts = compare_mapped_pair(
        mapping=residue_mappings,
        canonical_length=len(canonical_residues),
        sequence_identity=sequence_identity,
        construct_overlap_fraction=construct_overlap_fraction,
        assembly_comparable=bool(assembly_comparable),
    )
    generic_decision = evaluate_pair(facts, AdmissionPolicy())
    policy_reasons = _intervention_reasons(
        perturbation,
        reference=reference_record,
        perturbed=perturbed_record,
        reference_structure_id=reference_structure_id,
    )
    reasons = _generic_reasons(generic_decision) + policy_reasons
    admission_reason = ";".join(dict.fromkeys(reasons)) or None
    admitted = generic_decision.admitted and not policy_reasons

    condition_pairs = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": CONTROLLED_PERTURBATION,
                "condition_1_structure_id": reference_structure_id,
                "condition_1_label": REFERENCE,
                "condition_2_structure_id": perturbed_structure_id,
                "condition_2_label": PERTURBED,
                "pair_role": "PRIMARY",
                "admission_status": "ADMITTED" if admitted else "EXCLUDED",
                "sequence_comparable": facts.sequence_comparable,
                "construct_comparable": facts.construct_comparable,
                "assembly_comparable": facts.assembly_comparable,
                "mapping_comparable": facts.mapping_comparable,
                "pair_provenance": (
                    "controlled_perturbation:"
                    f"{perturbation.perturbation_method}:"
                    f"{perturbation.perturbation_version}"
                ),
                "admission_reason": admission_reason,
                "common_mapped_count": facts.common_mapped_count,
                "common_mapped_fraction": facts.mapping_fraction,
                "common_coordinate_visible_count": facts.common_coordinate_visible_count,
                "common_coordinate_visible_fraction": facts.coordinate_fraction,
                "ligand_context_class": None,
                "perturbation_family": str(perturbation.perturbation_family).strip().upper(),
                "perturbation_dose": float(perturbation.requested_dose),
                "perturbation_dose_unit": str(perturbation.requested_dose_unit).strip(),
            }
        ]
    )
    perturbation_descriptors = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "perturbation_family": str(perturbation.perturbation_family).strip().upper(),
                "requested_dose": float(perturbation.requested_dose),
                "requested_dose_unit": str(perturbation.requested_dose_unit).strip(),
                "realized_ca_rmsd": descriptors.realized_ca_rmsd,
                "realized_pairwise_distance_change": descriptors.realized_pairwise_distance_change,
                "realized_contact_change": descriptors.realized_contact_change,
                "random_seed": perturbation.random_seed,
                "perturbation_method": str(perturbation.perturbation_method).strip(),
                "perturbation_version": str(perturbation.perturbation_version).strip(),
            }
        ]
    )

    registry = _schema_registry()
    return {
        "structures": validate_frame(
            pd.DataFrame([reference_row, perturbed_row]), "structures", registry
        ),
        "condition_pairs": validate_frame(
            condition_pairs, "condition_pairs", registry
        ),
        "residue_mappings": validate_frame(
            residue_mappings, "residue_mappings", registry
        ),
        "perturbation_descriptors": validate_frame(
            perturbation_descriptors, "perturbation_descriptors", registry
        ),
    }
