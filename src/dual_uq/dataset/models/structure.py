from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from dual_uq.core.errors import PAEMappingError

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

_STANDARD_AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass(frozen=True)
class AFDBFragment:
    """Identity and inclusive UniProt interval of one selected AFDB model."""

    model_entity_id: str
    uniprot_start: int
    uniprot_end: int
    model_residue_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_entity_id, str) or not self.model_entity_id.strip():
            raise PAEMappingError(
                "invalid_afdb_fragment", "AFDB fragment model identity is required"
            )
        values = (self.uniprot_start, self.uniprot_end, self.model_residue_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise PAEMappingError(
                "invalid_afdb_fragment",
                "AFDB fragment interval and residue count must be integers",
            )
        if self.uniprot_start < 1 or self.uniprot_end < self.uniprot_start:
            raise PAEMappingError(
                "invalid_afdb_fragment", "AFDB fragment interval is invalid"
            )
        if self.model_residue_count != self.interval_length:
            raise PAEMappingError(
                "afdb_fragment_length_mismatch",
                "AFDB fragment residue count does not match its UniProt interval",
            )

    @property
    def interval_length(self) -> int:
        return self.uniprot_end - self.uniprot_start + 1

    @property
    def fragment_length(self) -> int:
        return self.model_residue_count


@dataclass(frozen=True)
class PAEMatrix:
    """Validated PAE matrix tied to one AFDB model identity."""

    model_entity_id: str
    values: np.ndarray

    @property
    def matrix_size(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class PAEResidueMapping:
    """Deterministically ordered output, UniProt, and PAE coordinates."""

    model_entity_id: str
    output_positions: tuple[int, ...]
    uniprot_positions: tuple[int, ...]
    model_residue_positions: tuple[int, ...]
    pae_indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.output_positions)


@dataclass(frozen=True)
class LongRangePAESummary:
    """Summary of unique symmetric PAE pairs passing a separation threshold."""

    count: int
    mean: float | None
    median: float | None
    maximum: float | None


def _identity_string(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    if "\0" in value or value != value.strip() or (not value and not allow_empty):
        qualifier = "possibly empty, " if allow_empty else "non-empty, "
        raise ValueError(
            f"{field_name} must be {qualifier}canonical, and contain no NUL."
        )
    return value


def _optional_chain_id(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string or None.")
    if value in {".", "?"}:
        return None
    return _identity_string(value, field_name, allow_empty=True)


def _sequence_id(value: object, field_name: str, *, optional: bool) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int:
        suffix = " or None" if optional else ""
        raise TypeError(f"{field_name} must be an integer{suffix}.")
    return value


def _missing_token(value: str | None, field_name: str) -> str:
    if value is None:
        return ""
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string or None.")
    token = value.strip()
    return "" if token in {"", ".", "?"} else token


def normalize_insertion_code(value: str | None) -> str:
    """Normalize format-level missing tokens while preserving a real code exactly."""
    token = _missing_token(value, "insertion_code")
    if len(token) > 1:
        raise ValueError("insertion_code must be empty or one character.")
    return token


def normalize_altloc(value: str | None) -> str:
    """Normalize format-level missing altloc tokens to the canonical empty token."""
    token = _missing_token(value, "altloc")
    if len(token) > 1:
        raise ValueError("altloc must be empty or one character.")
    return token


def canonical_amino_acid(raw_resname: str) -> str | None:
    """Return the supported one-letter code without inventing an unknown X."""
    checked = _identity_string(raw_resname, "raw_resname")
    return _STANDARD_AMINO_ACIDS.get(checked)


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Source residue identity used for grouping, independent of label numbering."""

    source_id: str
    auth_chain_id: str
    auth_seq_id: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _identity_string(self.source_id, "source_id")
        )
        object.__setattr__(
            self,
            "auth_chain_id",
            _identity_string(self.auth_chain_id, "auth_chain_id", allow_empty=True),
        )
        object.__setattr__(
            self,
            "auth_seq_id",
            _sequence_id(self.auth_seq_id, "auth_seq_id", optional=False),
        )
        object.__setattr__(
            self, "insertion_code", normalize_insertion_code(self.insertion_code)
        )


@dataclass(frozen=True)
class ResidueProvenance:
    """Full residue provenance attached to every selected source atom."""

    key: ResidueKey
    label_chain_id: str | None
    label_seq_id: int | None
    raw_resname: str
    record_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResidueKey):
            raise TypeError("key must be a ResidueKey.")
        object.__setattr__(
            self,
            "label_chain_id",
            _optional_chain_id(self.label_chain_id, "label_chain_id"),
        )
        object.__setattr__(
            self,
            "label_seq_id",
            _sequence_id(self.label_seq_id, "label_seq_id", optional=True),
        )
        object.__setattr__(
            self,
            "raw_resname",
            _identity_string(self.raw_resname, "raw_resname"),
        )
        record_type = _identity_string(self.record_type, "record_type")
        if record_type not in {"ATOM", "HETATM"}:
            raise ValueError("record_type must be ATOM or HETATM.")
        object.__setattr__(self, "record_type", record_type)

    @property
    def canonical_aa(self) -> str | None:
        return canonical_amino_acid(self.raw_resname)


def _finite_number(value: object, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a numeric value.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


@dataclass(frozen=True)
class AtomRecord:
    """Validated atom coordinates plus complete residue and altloc provenance."""

    residue: ResidueProvenance
    atom_name: str
    element: str | None
    altloc: str
    occupancy: float | None
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not isinstance(self.residue, ResidueProvenance):
            raise TypeError("residue must be ResidueProvenance.")
        if type(self.atom_name) is not str:
            raise TypeError("atom_name must be a string.")
        atom_name = self.atom_name.strip()
        object.__setattr__(
            self, "atom_name", _identity_string(atom_name, "atom_name")
        )
        if self.element is not None:
            object.__setattr__(
                self, "element", _identity_string(self.element, "element")
            )
        object.__setattr__(self, "altloc", normalize_altloc(self.altloc))
        if self.occupancy is not None:
            occupancy = _finite_number(self.occupancy, "occupancy")
            if not 0.0 <= occupancy <= 1.0:
                raise ValueError("occupancy must be between 0 and 1 inclusive.")
            object.__setattr__(self, "occupancy", occupancy)
        for coordinate in ("x", "y", "z"):
            object.__setattr__(
                self, coordinate, _finite_number(getattr(self, coordinate), coordinate)
            )

    @property
    def coordinates(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class ResidueGroup:
    residue: ResidueProvenance
    atoms: tuple[AtomRecord, ...]


@dataclass(frozen=True)
class BackboneSelection:
    residue: ResidueProvenance
    selected_atoms: tuple[AtomRecord, ...]
    missing_atoms: tuple[str, ...]

    def atom(self, atom_name: str) -> AtomRecord:
        for atom in self.selected_atoms:
            if atom.atom_name == atom_name:
                return atom
        raise KeyError(atom_name)


def altloc_rank(altloc: str | None) -> int:
    """Index36 tie rank: blank, then A, then another explicit conformer."""
    token = normalize_altloc(altloc)
    if token == "":
        return 2
    if token == "A":
        return 1
    return 0


def _selection_key(atom: AtomRecord) -> tuple[int, float, int, str]:
    occupancy_present = atom.occupancy is not None
    occupancy = atom.occupancy if atom.occupancy is not None else 0.0
    return int(occupancy_present), occupancy, altloc_rank(atom.altloc), atom.altloc


def select_preferred_atom(records: Iterable[AtomRecord]) -> AtomRecord:
    """Select one atom by occupancy and Index36 altloc rules, or reject ambiguity."""
    candidates = tuple(records)
    if not candidates:
        raise ValueError("At least one atom candidate is required.")
    if any(not isinstance(candidate, AtomRecord) for candidate in candidates):
        raise TypeError("Atom candidates must be AtomRecord values.")
    identity = (candidates[0].residue, candidates[0].atom_name)
    if any((candidate.residue, candidate.atom_name) != identity for candidate in candidates):
        raise ValueError("Atom candidates must share the same residue and atom identity.")

    unique_candidates = frozenset(candidates)
    best_key = max(_selection_key(candidate) for candidate in unique_candidates)
    winners = tuple(
        candidate
        for candidate in unique_candidates
        if _selection_key(candidate) == best_key
    )
    if len(winners) != 1:
        raise ValueError(
            "ambiguous duplicate atom: equal occupancy and altloc provenance "
            "have different scientific records."
        )
    return winners[0]


def _atom_sort_key(atom: AtomRecord) -> tuple[object, ...]:
    return (
        atom.atom_name,
        atom.altloc,
        atom.occupancy is None,
        -1.0 if atom.occupancy is None else atom.occupancy,
        atom.element or "",
        atom.x,
        atom.y,
        atom.z,
    )


def group_residue_records(records: Iterable[AtomRecord]) -> tuple[ResidueGroup, ...]:
    """Group records by source auth key with stable ordering and no fallback."""
    grouped: dict[ResidueKey, list[AtomRecord]] = defaultdict(list)
    provenance: dict[ResidueKey, ResidueProvenance] = {}
    for atom in records:
        if not isinstance(atom, AtomRecord):
            raise TypeError("Structure records must be AtomRecord values.")
        key = atom.residue.key
        previous = provenance.setdefault(key, atom.residue)
        if previous != atom.residue:
            raise ValueError(f"conflicting residue provenance for source key {key!r}.")
        grouped[key].append(atom)
    return tuple(
        ResidueGroup(
            residue=provenance[key],
            atoms=tuple(sorted(set(grouped[key]), key=_atom_sort_key)),
        )
        for key in sorted(grouped)
    )


def select_backbone_atoms(records: Iterable[AtomRecord]) -> BackboneSelection:
    """Select at most one N/CA/C/O atom while retaining incomplete residues."""
    groups = group_residue_records(records)
    if len(groups) != 1:
        raise ValueError("Backbone selection requires exactly one source residue.")
    group = groups[0]
    by_name: dict[str, list[AtomRecord]] = defaultdict(list)
    for atom in group.atoms:
        if atom.atom_name in BACKBONE_ATOM_NAMES:
            by_name[atom.atom_name].append(atom)
    selected = tuple(
        select_preferred_atom(by_name[name])
        for name in BACKBONE_ATOM_NAMES
        if by_name[name]
    )
    missing = tuple(name for name in BACKBONE_ATOM_NAMES if not by_name[name])
    return BackboneSelection(group.residue, selected, missing)
