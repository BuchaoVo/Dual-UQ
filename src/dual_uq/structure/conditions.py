"""Concrete structure-condition and neutral paired-intervention identity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _required_text(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\0" in value
    ):
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


@dataclass(frozen=True)
class StructureCondition:
    """Scientific identity and optional locator for one concrete structure."""

    protein_id: str
    condition_id: str
    source: str
    structure_sha256: str
    structure_locator: str | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protein_id", _required_text(self.protein_id, "protein_id")
        )
        object.__setattr__(
            self,
            "condition_id",
            _required_text(self.condition_id, "condition_id"),
        )
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        digest = _required_text(self.structure_sha256, "structure_sha256")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("structure_sha256 must be a lowercase SHA-256 digest")
        if self.structure_locator is not None:
            object.__setattr__(
                self,
                "structure_locator",
                _required_text(self.structure_locator, "structure_locator"),
            )

    @property
    def scientific_identity(self) -> tuple[str, str, str, str]:
        """Return identity independent of the mutable artifact locator."""
        return (
            self.protein_id,
            self.condition_id,
            self.source,
            self.structure_sha256,
        )


@dataclass(frozen=True)
class StructuralIntervention:
    """A neutral relationship between two concrete conditions of one protein."""

    intervention_id: str
    condition_a: StructureCondition
    condition_b: StructureCondition

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention_id",
            _required_text(self.intervention_id, "intervention_id"),
        )
        if self.condition_a.protein_id != self.condition_b.protein_id:
            raise ValueError("Structural intervention conditions must share the same protein")
        if self.condition_a == self.condition_b:
            raise ValueError("Structural intervention requires two distinct conditions")

    @property
    def protein_id(self) -> str:
        """Derive protein identity without storing a duplicate field."""
        return self.condition_a.protein_id

    @property
    def conditions(self) -> tuple[StructureCondition, StructureCondition]:
        return (self.condition_a, self.condition_b)
