"""Canonical condition labels, orientation, and directional semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_ORIENTATIONS = {
    ("representation_variation", None): ("PDB", "AFDB"),
    ("ligand_state", None): ("APO", "HOLO"),
    ("functional_state", "OPEN_CLOSED"): ("OPEN", "CLOSED"),
    ("functional_state", "INWARD_OUTWARD"): ("INWARD_FACING", "OUTWARD_FACING"),
    ("functional_state", "ACTIVE_INACTIVE"): ("ACTIVE", "INACTIVE"),
    ("functional_state", "RESTING_ACTIVATED"): ("RESTING", "ACTIVATED"),
    ("functional_state", "PRE_POST"): ("PRE_TRANSITION", "POST_TRANSITION"),
    ("controlled_perturbation", None): ("REFERENCE", "PERTURBED"),
}


@dataclass(frozen=True, slots=True)
class OrientedPair:
    condition_1_structure_id: str
    condition_1_label: str
    condition_2_structure_id: str
    condition_2_label: str
    structural_condition_semantics: str
    state_family: str | None = None


def orientation_for(semantics: str, state_family: str | None = None) -> tuple[str, str]:
    key = (str(semantics), state_family)
    if key not in _ORIENTATIONS:
        key = (str(semantics), None)
    try:
        return _ORIENTATIONS[key]
    except KeyError as exc:
        raise ValueError(f"no canonical orientation registered for {semantics}/{state_family}") from exc


def orient_pair(semantics: str, state_family: str | None, left: dict[str, Any], right: dict[str, Any]) -> OrientedPair:
    first, second = orientation_for(semantics, state_family)
    rows = {str(left.get("condition_label")): left, str(right.get("condition_label")): right}
    if first not in rows or second not in rows:
        raise ValueError(f"pair does not contain required labels {first} and {second}")
    return OrientedPair(str(rows[first]["structure_id"]), first, str(rows[second]["structure_id"]), second, semantics, state_family)


def directional_delta(condition_2_value: float, condition_1_value: float) -> float:
    return float(condition_2_value) - float(condition_1_value)
