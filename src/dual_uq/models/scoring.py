"""Normalized successful score measurements with explicit scientific identity."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Protocol, runtime_checkable

from dual_uq.core.hashing import sha256_bytes
from dual_uq.structure import StructureCondition

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class VariantKind(str, Enum):
    """Explicit measurement variant, never encoded by mutation sentinels."""

    WT = "WT"
    PROBE = "PROBE"


class ScoreDispatchError(ValueError):
    """Structured model-independent request/result contract failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ScoringVariant:
    """One model-independent full-sequence member of a scoring collection."""

    variant_kind: VariantKind
    variant_id: str | None
    sequence_hash: str
    sequence: str
    position: int | None
    wt_aa: str | None
    mut_aa: str | None

    def __post_init__(self) -> None:
        try:
            kind = VariantKind(self.variant_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("variant_kind must be WT or PROBE") from exc
        object.__setattr__(self, "variant_kind", kind)
        sequence = _required_text(self.sequence, "sequence")
        if set(sequence).difference(_STANDARD_AMINO_ACIDS):
            raise ValueError("sequence must use the standard uppercase 20-AA alphabet")
        digest = _sha256(self.sequence_hash, "sequence_hash")
        if sha256_bytes(sequence.encode("ascii")) != digest:
            raise ValueError("sequence_hash does not match sequence content")

        probe_values = (self.variant_id, self.position, self.wt_aa, self.mut_aa)
        if kind is VariantKind.WT:
            if any(value is not None for value in probe_values):
                raise ValueError("WT variants cannot carry probe identity")
            return
        if any(value is None for value in probe_values):
            raise ValueError("PROBE variants require complete substitution identity")
        object.__setattr__(
            self, "variant_id", _required_text(self.variant_id, "variant_id")
        )
        if self.variant_id != digest:
            raise ValueError("probe variant_id must equal its full sequence hash")
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position <= 0
            or self.wt_aa not in _STANDARD_AMINO_ACIDS
            or self.mut_aa not in _STANDARD_AMINO_ACIDS
            or self.wt_aa == self.mut_aa
            or self.position > len(sequence)
            or sequence[self.position - 1] != self.mut_aa
        ):
            raise ValueError("PROBE variants require a valid single substitution")


@dataclass(frozen=True, slots=True)
class CandidateCollection:
    """One WT plus the fixed candidates measured by a logical request."""

    collection_id: str
    wt: ScoringVariant
    probes: tuple[ScoringVariant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "collection_id", _sha256(self.collection_id, "collection_id")
        )
        if self.wt.variant_kind is not VariantKind.WT:
            raise ValueError("candidate collection requires one WT variant")
        probes = tuple(self.probes)
        if not probes or any(
            probe.variant_kind is not VariantKind.PROBE for probe in probes
        ):
            raise ValueError("candidate collection requires PROBE variants")
        identifiers = [probe.variant_id for probe in probes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate collection probe identities must be unique")
        wt_sequence = self.wt.sequence
        for probe in probes:
            position = probe.position
            wt_aa = probe.wt_aa
            if (
                position is None
                or wt_aa is None
                or len(probe.sequence) != len(wt_sequence)
                or position > len(wt_sequence)
                or wt_sequence[position - 1] != wt_aa
                or probe.sequence[: position - 1]
                + wt_aa
                + probe.sequence[position:]
                != wt_sequence
            ):
                raise ValueError(
                    "candidate collection probes must be one declared single substitution from WT"
                )
        object.__setattr__(self, "probes", probes)

    @property
    def variants(self) -> tuple[ScoringVariant, ...]:
        """Return deterministic WT-then-probe scoring order."""
        return (self.wt, *self.probes)


@dataclass(frozen=True, slots=True)
class ScorerBinding:
    """External implementation identity for a scorer capability."""

    scorer_id: str
    implementation_id: str
    checkpoint_id: str | None
    score_contract_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scorer_id", _required_text(self.scorer_id, "scorer_id"))
        object.__setattr__(
            self,
            "implementation_id",
            _required_text(self.implementation_id, "implementation_id"),
        )
        if self.checkpoint_id is not None:
            object.__setattr__(
                self, "checkpoint_id", _sha256(self.checkpoint_id, "checkpoint_id")
            )
        object.__setattr__(
            self,
            "score_contract_id",
            _required_text(self.score_contract_id, "score_contract_id"),
        )


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """One collection-level, model-independent scientific scoring request."""

    condition: StructureCondition
    scoring_domain_id: str
    canonical_positions: tuple[int, ...]
    candidate_collection: CandidateCollection
    repeat_index: int
    seed: int
    realization_id: str
    realization_algorithm: str
    score_contract_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.condition, StructureCondition):
            raise TypeError("condition must be a StructureCondition")
        object.__setattr__(
            self,
            "scoring_domain_id",
            _sha256(self.scoring_domain_id, "scoring_domain_id"),
        )
        positions = tuple(self.canonical_positions)
        if (
            not positions
            or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
            or any(value <= 0 for value in positions)
            or any(right <= left for left, right in pairwise(positions))
        ):
            raise ValueError("canonical_positions must be positive and strictly increasing")
        object.__setattr__(self, "canonical_positions", positions)
        if not isinstance(self.candidate_collection, CandidateCollection):
            raise TypeError("candidate_collection must be a CandidateCollection")
        if positions[-1] > len(self.candidate_collection.wt.sequence):
            raise ValueError("candidate collection does not cover canonical_positions")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("repeat_index and seed must be non-negative integers")
        object.__setattr__(
            self, "realization_id", _sha256(self.realization_id, "realization_id")
        )
        object.__setattr__(
            self,
            "realization_algorithm",
            _required_text(self.realization_algorithm, "realization_algorithm"),
        )
        object.__setattr__(
            self,
            "score_contract_id",
            _required_text(self.score_contract_id, "score_contract_id"),
        )

    @property
    def protein_id(self) -> str:
        """Derive protein identity from the concrete structure condition."""
        return self.condition.protein_id

    @property
    def result_count(self) -> int:
        """One normalized result is expected for every collection member."""
        return len(self.candidate_collection.variants)


@runtime_checkable
class SequenceScorer(Protocol):
    """Capability that scores one logical collection-level request."""

    @property
    def binding(self) -> ScorerBinding:
        """Return implementation identity external to request identity."""
        ...

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        """Return normalized WT-then-candidate measurements."""
        ...


def _required_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _sha256(value: object, field_name: str) -> str:
    digest = _required_text(value, field_name)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class ScoreRecord:
    """One normalized successful score under an explicit scorer contract.

    The envelope is model-independent, while the precise ProteinMPNN quantities
    deliberately retain their scientific names and score-contract binding.
    """

    protein_id: str
    condition_id: str
    structure_sha256: str

    variant_kind: VariantKind
    variant_id: str | None
    sequence_hash: str | None
    position: int | None
    wt_aa: str | None
    mut_aa: str | None

    repeat_index: int
    seed: int
    realization_id: str

    scorer_id: str
    implementation_id: str
    checkpoint_id: str | None
    score_contract_id: str

    score_sum_logp_mask: float
    score_mean_logp_mask: float
    scored_residue_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "protein_id", _required_text(self.protein_id, "protein_id"))
        object.__setattr__(
            self, "condition_id", _required_text(self.condition_id, "condition_id")
        )
        object.__setattr__(
            self,
            "structure_sha256",
            _sha256(self.structure_sha256, "structure_sha256"),
        )
        try:
            kind = VariantKind(self.variant_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("variant_kind must be WT or PROBE") from exc
        object.__setattr__(self, "variant_kind", kind)

        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("repeat_index and seed must be non-negative integers")
        object.__setattr__(
            self, "realization_id", _sha256(self.realization_id, "realization_id")
        )
        object.__setattr__(self, "scorer_id", _required_text(self.scorer_id, "scorer_id"))
        object.__setattr__(
            self,
            "implementation_id",
            _required_text(self.implementation_id, "implementation_id"),
        )
        if self.checkpoint_id is not None:
            object.__setattr__(
                self,
                "checkpoint_id",
                _sha256(self.checkpoint_id, "checkpoint_id"),
            )
        object.__setattr__(
            self,
            "score_contract_id",
            _required_text(self.score_contract_id, "score_contract_id"),
        )

        score_sum = float(self.score_sum_logp_mask)
        score_mean = float(self.score_mean_logp_mask)
        if not math.isfinite(score_sum) or not math.isfinite(score_mean):
            raise ValueError("scores must be finite")
        if (
            isinstance(self.scored_residue_count, bool)
            or not isinstance(self.scored_residue_count, int)
            or self.scored_residue_count <= 0
        ):
            raise ValueError("scored_residue_count must be a positive integer")
        if not math.isclose(
            score_mean,
            score_sum / self.scored_residue_count,
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise ValueError("mean score must equal sum / residue count")
        object.__setattr__(self, "score_sum_logp_mask", score_sum)
        object.__setattr__(self, "score_mean_logp_mask", score_mean)

        probe_values = (
            self.variant_id,
            self.sequence_hash,
            self.position,
            self.wt_aa,
            self.mut_aa,
        )
        if kind is VariantKind.WT:
            if any(value is not None for value in probe_values):
                raise ValueError("WT measurements cannot carry probe identity")
            return
        if any(value is None for value in probe_values):
            raise ValueError("PROBE measurements require complete probe identity")
        object.__setattr__(
            self, "variant_id", _required_text(self.variant_id, "variant_id")
        )
        object.__setattr__(
            self, "sequence_hash", _sha256(self.sequence_hash, "sequence_hash")
        )
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position <= 0
            or self.wt_aa not in _STANDARD_AMINO_ACIDS
            or self.mut_aa not in _STANDARD_AMINO_ACIDS
            or self.wt_aa == self.mut_aa
        ):
            raise ValueError("PROBE measurements require a valid single substitution")


def execute_score_request(
    scorer: SequenceScorer, request: ScoreRequest
) -> tuple[ScoreRecord, ...]:
    """Invoke one scorer and validate normalized results against the request.

    Scheduling, model-native batching, persistence, and resume are deliberately
    outside this boundary.
    """
    binding = scorer.binding
    if binding.score_contract_id != request.score_contract_id:
        raise ScoreDispatchError(
            "score_contract_mismatch",
            "Scorer score contract does not match the request score contract",
        )
    records = tuple(scorer.score(request))
    expected_variants = request.candidate_collection.variants
    if len(records) != len(expected_variants):
        raise ScoreDispatchError(
            "score_result_cardinality_mismatch",
            "Scorer must return exactly one WT and one result per declared probe",
        )
    for index, (record, variant) in enumerate(
        zip(records, expected_variants, strict=True)
    ):
        if not isinstance(record, ScoreRecord):
            raise ScoreDispatchError(
                "invalid_score_result_type",
                f"Scorer result {index} is not a ScoreRecord",
            )
        if (
            record.protein_id != request.protein_id
            or record.condition_id != request.condition.condition_id
            or record.structure_sha256 != request.condition.structure_sha256
            or record.repeat_index != request.repeat_index
            or record.seed != request.seed
            or record.realization_id != request.realization_id
            or record.score_contract_id != request.score_contract_id
            or record.scored_residue_count != len(request.canonical_positions)
        ):
            raise ScoreDispatchError(
                "score_result_request_identity_mismatch",
                f"Scorer result {index} does not match request identity",
            )
        if (
            record.scorer_id != binding.scorer_id
            or record.implementation_id != binding.implementation_id
            or record.checkpoint_id != binding.checkpoint_id
        ):
            raise ScoreDispatchError(
                "score_result_scorer_binding_mismatch",
                f"Scorer result {index} does not match scorer binding",
            )
        expected_sequence_hash = (
            None if variant.variant_kind is VariantKind.WT else variant.sequence_hash
        )
        if (
            record.variant_kind is not variant.variant_kind
            or record.variant_id != variant.variant_id
            or record.sequence_hash != expected_sequence_hash
            or record.position != variant.position
            or record.wt_aa != variant.wt_aa
            or record.mut_aa != variant.mut_aa
        ):
            raise ScoreDispatchError(
                "score_result_variant_association_mismatch",
                f"Scorer result {index} differs from declared WT/probe order",
            )
    return records
