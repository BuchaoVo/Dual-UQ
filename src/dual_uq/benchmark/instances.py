"""Model-independent benchmark instance eligibility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkInstance:
    instance_id: str
    pair_id: str
    protein_id: str
    arm: str
    local_sensitivity_eligible: bool
    generation_eligible: bool
    compatibility_eligible: bool
    multistate_eligible: bool
    geometry_evaluable: bool
    ligand_evaluable: bool
    eligibility_reason: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        if self.model_id is not None:
            raise ValueError("BenchmarkInstance cannot carry model-specific eligibility")
