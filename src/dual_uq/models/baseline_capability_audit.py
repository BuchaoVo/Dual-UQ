"""StructCal v1 baseline capability and TRAIN-only smoke contracts.

The models remain responsible for their native inference semantics.  This
module validates only audit metadata and model-independent output invariants;
it does not redefine a model's logits, decoder, or score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

CANONICAL_AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")

CapabilityStatus = Literal[
    "SUPPORTED",
    "SUPPORTED_WITH_CANONICALIZATION",
    "LIMITED",
    "UNSUPPORTED",
    "BLOCKED_ENVIRONMENT",
    "BLOCKED_PROVENANCE",
    "NOT_APPLICABLE",
]
HistoricalReuseStatus = Literal[
    "REUSABLE_EXACT",
    "REUSABLE_AFTER_CANONICAL_REINDEX",
    "REUSABLE_DIAGNOSTIC_ONLY",
    "NOT_REUSABLE",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(_StrictModel):
    official_repository: str
    source_revision: str
    source_revision_type: str
    checkpoint_id: str
    checkpoint_source: str
    checkpoint_configuration: str
    checkpoint_sha256: str | None
    license: str
    access_constraints: str
    implementation_language: str
    primary_framework: str
    accelerator_assumptions: str


class Environment(_StrictModel):
    environment_id: str
    environment_path: str
    status: CapabilityStatus
    python_version: str | None
    pytorch_version: str | None
    framework_cuda: str | None
    gpu_detected: bool | None
    official_model_import: CapabilityStatus
    checkpoint_load: CapabilityStatus
    minimal_inference: CapabilityStatus
    critical_packages: dict[str, str]
    limitations: tuple[str, ...] = ()


class StructureConditioning(_StrictModel):
    status: CapabilityStatus
    explanation: str
    type: str
    required_atoms: tuple[str, ...]
    tokenization: str
    supports_missing_coordinates: bool | None
    supports_chain_breaks: bool | None
    supports_multi_chain: bool | None
    supports_partial_structure: bool | None
    uses_sequence_context_during_structure_encoding: bool | None
    uses_sidechain_information: bool | None
    structure_tokenization_required: bool | None
    limitations: tuple[str, ...] = ()


class LocalResponse(_StrictModel):
    status: CapabilityStatus
    explanation: str
    semantic_class: Literal["L0", "L1", "L2", "L3"]
    probe_state: str
    sequence_context_type: str
    iteration_index: int | None
    decoding_context: str
    temperature_if_relevant: float | None
    native_sequence_leakage: bool | None
    output_shape: str | None
    aa_order: tuple[str, ...] | None
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_semantics(self) -> LocalResponse:
        if self.semantic_class in {"L0", "L1"} and self.native_sequence_leakage is not False:
            raise ValueError("L0/L1 probes must explicitly exclude native-sequence leakage")
        if self.status in {"SUPPORTED", "SUPPORTED_WITH_CANONICALIZATION"}:
            if self.output_shape != "[L,20]":
                raise ValueError("supported local response must declare output_shape [L,20]")
            if self.aa_order != CANONICAL_AA_ORDER:
                raise ValueError("supported local response must use the canonical amino-acid order")
        return self


class Generation(_StrictModel):
    status: CapabilityStatus
    explanation: str
    semantics: Literal[
        "AUTOREGRESSIVE",
        "NON_AUTOREGRESSIVE_ONE_SHOT",
        "ITERATIVE_REFINEMENT",
        "MASKED_ITERATIVE",
        "DISCRETE_DIFFUSION",
        "MRF_OR_RECYCLING",
        "OTHER",
        "UNSUPPORTED",
    ]
    native_decoding_steps: str
    requires_initial_sequence: bool | None
    supports_all_mask_initialization: bool | None
    supports_fixed_positions: bool | None
    supports_variable_length: bool | None
    supports_exact_structural_length: bool | None
    outputs_canonical_amino_acids_only: bool | None
    temperature_semantics: Literal[
        "NATIVE_SEQUENCE_SAMPLING_TEMPERATURE",
        "MODEL_SPECIFIC_TEMPERATURE",
        "NOT_APPLICABLE",
        "UNSUPPORTED",
    ]
    temperature_acts_on: str
    temperature_stage: str
    temperature_applied_every_iterative_step: bool | None
    official_inference_exposes_temperature: bool | None
    seed_reproducibility: Literal[
        "DETERMINISTIC_EXACT",
        "DETERMINISTIC_WITH_LIMITATIONS",
        "STOCHASTIC_NONDETERMINISTIC_KERNEL",
        "NO_EXPLICIT_SEED_CONTROL",
        "NOT_APPLICABLE",
    ]
    rng_sources: tuple[str, ...]
    native_decoding_preserved: bool
    limitations: tuple[str, ...] = ()


class SequenceScoring(_StrictModel):
    status: CapabilityStatus
    explanation: str
    semantics: Literal[
        "EXACT_NATIVE_LOG_LIKELIHOOD",
        "AUTOREGRESSIVE_LOG_LIKELIHOOD",
        "PSEUDO_LIKELIHOOD",
        "MASKED_TOKEN_SCORE",
        "ITERATIVE_SURROGATE_SCORE",
        "ENERGY_OR_COMPATIBILITY_SCORE",
        "UNSUPPORTED",
    ]
    score_direction: str
    length_normalization: str
    uses_native_sequence_context: bool | None
    decoding_order_matters: bool | None
    additive_across_residues: bool | None
    comparable_across_proteins: bool | None
    comparable_across_models: bool
    limitations: tuple[str, ...] = ()


class CrossEvaluator(_StrictModel):
    proteinmpnn: CapabilityStatus
    proteinmpnn_explanation: str
    esm_if1: CapabilityStatus
    esm_if1_explanation: str


class HistoricalReuseDecision(_StrictModel):
    status: HistoricalReuseStatus
    explanation: str


class HistoricalReuse(_StrictModel):
    local: HistoricalReuseDecision
    generation: HistoricalReuseDecision
    scoring: HistoricalReuseDecision


class CapabilityDecision(_StrictModel):
    status: CapabilityStatus
    explanation: str


class StructCalTasks(_StrictModel):
    track_i_local: CapabilityDecision
    track_ii_local: CapabilityDecision
    controlled_local: CapabilityDecision
    generation: CapabilityDecision
    sequence_distribution: CapabilityDecision
    sequence_scoring: CapabilityDecision
    cross_evaluation: CapabilityDecision
    quality_guardrail: CapabilityDecision


class CapabilityCard(_StrictModel):
    """One model's frozen, performance-free StructCal capability card."""

    schema_version: Literal["structcal_baseline_capability_v1"]
    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    tier: Literal["A", "B", "C", "D"]
    architecture_class: str = Field(min_length=1)
    provenance: Provenance
    environment: Environment
    structure_conditioning: StructureConditioning
    local_response: LocalResponse
    generation: Generation
    sequence_scoring: SequenceScoring
    cross_evaluator: CrossEvaluator
    historical_reuse: HistoricalReuse
    structcal_tasks: StructCalTasks
    overall_v1_class: Literal[
        "FULL_PRIMARY",
        "PRIMARY_PARTIAL",
        "GENERATION_ONLY",
        "EXCLUDED_FROM_V1_BASELINE",
    ]
    limitations: tuple[str, ...] = ()


def validate_capability_registry(
    cards: Sequence[CapabilityCard],
) -> tuple[CapabilityCard, ...]:
    """Reject ambiguous model identities and return the immutable card order."""

    result = tuple(cards)
    model_ids = [card.model_id for card in result]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("capability registry contains duplicate model_id values")
    return result


def validate_local_preferences(
    probabilities: np.ndarray,
    aa_order: Sequence[str],
) -> np.ndarray:
    """Validate the shared numerical boundary without redefining model semantics."""

    values = np.asarray(probabilities, dtype=np.float64)
    if tuple(aa_order) != CANONICAL_AA_ORDER:
        raise ValueError("local preference amino-acid order is not canonical")
    if values.ndim != 2 or values.shape[1] != len(CANONICAL_AA_ORDER):
        raise ValueError("local preferences must have shape [L,20]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("local preferences must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("local preferences must be normalized per residue")
    return values


def guard_train_only_inference(
    split_by_protein: Mapping[str, str],
    protein_ids: Sequence[str],
) -> None:
    """Fail before inference unless every selected protein belongs to TRAIN."""

    for protein_id in protein_ids:
        split = split_by_protein.get(str(protein_id))
        if split == "LOCKED_TEST":
            raise ValueError(f"LOCKED_TEST inference is forbidden: {protein_id}")
        if split != "TRAIN":
            raise ValueError(f"StructCal capability smoke is TRAIN-only: {protein_id} ({split})")
