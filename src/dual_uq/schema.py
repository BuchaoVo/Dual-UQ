from __future__ import annotations
from pydantic import BaseModel, Field

class ProteinRecord(BaseModel):
    protein_id: str
    uniprot_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    length: int = Field(gt=0)
    sequence_cluster: str | None = None
    domain_count: int | None = Field(default=None, ge=1)
    secondary_structure_class: str | None = None
    split: str = "unassigned"
    mapping_coverage: float = Field(ge=0.0, le=1.0)
    sequence_identity: float = Field(ge=0.0, le=1.0)
    quality_flag: str = "pending"

class StructureRecord(BaseModel):
    structure_id: str
    protein_id: str
    source: str
    parent_structure_id: str | None = None
    coordinate_path: str
    plddt_path: str | None = None
    pae_path: str | None = None
    perturbation_type: str | None = None
    perturbation_strength: float | None = None
    perturbed_residues: str | None = None
    random_seed: int | None = None
    mapping_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    structure_valid: bool = False

class SequenceRecord(BaseModel):
    sequence_id: str
    protein_id: str
    generation_structure_id: str
    generator: str
    checkpoint: str
    temperature: float
    sampling_seed: int
    sequence: str
    sequence_identity_to_native: float | None = Field(default=None, ge=0.0, le=1.0)
    generation_status: str = "pending"

class ScoreRecord(BaseModel):
    protein_id: str
    structure_id: str
    sequence_id: str
    evaluator_id: str
    property_id: str
    score: float | None = None
    score_mean: float | None = None
    score_variance: float | None = Field(default=None, ge=0.0)
    ood_score: float | None = None
    run_status: str = "pending"
    error_message: str | None = None
    runtime: float | None = Field(default=None, ge=0.0)
