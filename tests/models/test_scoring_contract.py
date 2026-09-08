from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.scoring import (
    CandidateCollection,
    ScoreRecord,
    ScoreRequest,
    ScorerBinding,
    ScoringVariant,
    VariantKind,
    execute_score_request,
)
from dual_uq.structure import StructuralIntervention, StructureCondition


def _variant(sequence: str, *, position: int | None = None) -> ScoringVariant:
    digest = sha256_bytes(sequence.encode("ascii"))
    if position is None:
        return ScoringVariant(VariantKind.WT, None, digest, sequence, None, None, None)
    return ScoringVariant(
        VariantKind.PROBE,
        digest,
        digest,
        sequence,
        position,
        "C",
        sequence[position - 1],
    )


def _request() -> ScoreRequest:
    return ScoreRequest(
        condition=StructureCondition("protein", "PDB", "PDB", "a" * 64),
        scoring_domain_id="b" * 64,
        canonical_positions=(1, 2, 3),
        candidate_collection=CandidateCollection(
            "c" * 64,
            _variant("ACD"),
            (_variant("AED", position=2),),
        ),
        repeat_index=0,
        seed=7,
        realization_id="d" * 64,
        realization_algorithm="test",
        score_contract_id="test-contract",
    )


def _record(request: ScoreRequest, variant: ScoringVariant) -> ScoreRecord:
    return ScoreRecord(
        protein_id=request.protein_id,
        condition_id=request.condition.condition_id,
        structure_sha256=request.condition.structure_sha256,
        variant_kind=variant.variant_kind,
        variant_id=variant.variant_id,
        sequence_hash=None if variant.variant_kind is VariantKind.WT else variant.sequence_hash,
        position=variant.position,
        wt_aa=variant.wt_aa,
        mut_aa=variant.mut_aa,
        repeat_index=request.repeat_index,
        seed=request.seed,
        realization_id=request.realization_id,
        scorer_id="fixture",
        implementation_id="fixture-v1",
        checkpoint_id=None,
        score_contract_id=request.score_contract_id,
        score_sum_logp_mask=-3.0,
        score_mean_logp_mask=-1.0,
        scored_residue_count=3,
    )


class _Scorer:
    binding = ScorerBinding("fixture", "fixture-v1", None, "test-contract")

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        return tuple(_record(request, variant) for variant in request.candidate_collection.variants)


def test_structure_and_scoring_contracts_are_live_without_legacy_artifacts() -> None:
    request = _request()
    relocated = StructureCondition("protein", "PDB", "PDB", "a" * 64, "/new/path")

    assert request.condition == relocated
    assert request.result_count == 2
    assert len(execute_score_request(_Scorer(), request)) == 2
    with pytest.raises(FrozenInstanceError):
        request.condition.condition_id = "AFDB"  # type: ignore[misc]


def test_structural_intervention_rejects_cross_protein_pairs() -> None:
    with pytest.raises(ValueError, match="same protein"):
        StructuralIntervention(
            "pair",
            StructureCondition("first", "PDB", "PDB", "a" * 64),
            StructureCondition("second", "AFDB", "AFDB", "b" * 64),
        )
