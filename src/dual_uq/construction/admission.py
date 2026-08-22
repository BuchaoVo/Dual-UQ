"""Outcome-blind deterministic admission primitives."""

from dataclasses import dataclass

from dual_uq.construction.comparability import ComparabilityFacts


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    require_sequence: bool = True
    require_mapping: bool = True
    require_coordinates: bool = True
    require_construct: bool = True
    require_assembly: bool = True


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    sequence_comparable: bool
    mapping_comparable: bool
    coordinate_comparable: bool
    construct_comparable: bool
    assembly_comparable: bool
    primary_reason: str
    secondary_reasons: tuple[str, ...] = ()


def evaluate_pair(facts: ComparabilityFacts, policy: AdmissionPolicy) -> AdmissionDecision:
    if not isinstance(facts, ComparabilityFacts):
        raise TypeError("evaluate_pair accepts comparability facts only; model results are forbidden")
    failures = []
    if policy.require_sequence and not facts.sequence_comparable: failures.append("SEQUENCE_NOT_COMPARABLE")
    if policy.require_mapping and not facts.mapping_comparable: failures.append("MAPPING_NOT_COMPARABLE")
    if policy.require_coordinates and not facts.coordinate_comparable: failures.append("COORDINATES_NOT_COMPARABLE")
    if policy.require_construct and not facts.construct_comparable: failures.append("CONSTRUCT_NOT_COMPARABLE")
    if policy.require_assembly and not facts.assembly_comparable: failures.append("ASSEMBLY_NOT_COMPARABLE")
    return AdmissionDecision(
        admitted=not failures,
        sequence_comparable=facts.sequence_comparable,
        mapping_comparable=facts.mapping_comparable,
        coordinate_comparable=facts.coordinate_comparable,
        construct_comparable=facts.construct_comparable,
        assembly_comparable=facts.assembly_comparable,
        primary_reason="ADMITTED" if not failures else failures[0],
        secondary_reasons=tuple(failures[1:]),
    )
