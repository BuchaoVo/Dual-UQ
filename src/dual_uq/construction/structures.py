"""Structural-condition instance normalization."""

from dual_uq.structcal.condition_semantics import orientation_for
from dual_uq.structcal.ids import structure_id


def normalize_structure(
    *,
    source_structure_id: str,
    protein_id: str,
    structural_condition_semantics: str,
    condition_label: str,
    condition_type: str,
    source_type: str,
    source_file_ref: str,
    state_family: str | None = None,
    source_accession: str | None = None,
    chain_id: str | None = None,
    entity_id: str | None = None,
    assembly_id: str | None = None,
    experimental_method: str | None = None,
    state_evidence_tier: str | None = None,
    state_evidence_source: str | None = None,
    parent_structure_id: str | None = None,
) -> dict[str, object]:
    first, second = orientation_for(structural_condition_semantics, state_family)
    if condition_label not in {first, second}:
        raise ValueError(f"invalid condition label {condition_label} for {structural_condition_semantics}/{state_family}")
    result: dict[str, object] = {
        "structure_id": structure_id(source_structure_id, structural_condition_semantics, condition_label, protein_id),
        "source_structure_id": source_structure_id,
        "protein_id": protein_id,
        "arm": structural_condition_semantics,
        "condition_type": condition_type,
        "condition_label": condition_label,
        "state_family": state_family,
        "source_type": source_type,
        "source_accession": source_accession or source_structure_id,
        "chain_id": chain_id,
        "entity_id": entity_id,
        "assembly_id": assembly_id,
        "experimental_method": experimental_method,
        "source_file_ref": source_file_ref,
        "state_evidence_tier": state_evidence_tier,
        "state_evidence_source": state_evidence_source,
        "parent_structure_id": parent_structure_id,
    }
    return result
