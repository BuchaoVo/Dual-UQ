from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.proteinmpnn import DECODING_REALIZATION_ALGORITHM
from dual_uq.models.scoring import (
    CandidateCollection,
    ScoreDispatchError,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
    execute_score_request,
)
from dual_uq.structure import StructuralIntervention, StructureCondition
from dual_uq.workflows.final_confirmatory_protocol import adapt_frozen_scoring_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPOSITORY_ROOT / "experiments/p2_design_baseline/scale1/scale1b_v2"
SCORE_CONTRACT = "stage0_fixed_sequence_autoregressive_mask_logp_v1"


def _condition(
    condition_id: str = "state-a", source: str = "PDB", digest: str = "1" * 64
) -> StructureCondition:
    return StructureCondition(
        protein_id="fixture_A__P00001",
        condition_id=condition_id,
        source=source,
        structure_sha256=digest,
    )


def _request(condition: StructureCondition | None = None) -> ScoreRequest:
    wt_hash = sha256_bytes(b"ACD")
    probe_hash = sha256_bytes(b"AED")
    return ScoreRequest(
        condition=condition or _condition(),
        scoring_domain_id="2" * 64,
        canonical_positions=(1, 2, 3),
        candidate_collection=CandidateCollection(
            collection_id="3" * 64,
            wt=ScoringVariant(
                variant_kind=VariantKind.WT,
                variant_id=None,
                sequence_hash=wt_hash,
                sequence="ACD",
                position=None,
                wt_aa=None,
                mut_aa=None,
            ),
            probes=(
                ScoringVariant(
                    variant_kind=VariantKind.PROBE,
                    variant_id=probe_hash,
                    sequence_hash=probe_hash,
                    sequence="AED",
                    position=2,
                    wt_aa="C",
                    mut_aa="E",
                ),
            ),
        ),
        repeat_index=0,
        seed=0,
        realization_id="4" * 64,
        realization_algorithm=DECODING_REALIZATION_ALGORITHM,
        score_contract_id=SCORE_CONTRACT,
    )


def _records(request: ScoreRequest) -> tuple[ScoreRecord, ...]:
    binding = ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id=SCORE_CONTRACT,
    )
    rows = []
    for index, variant in enumerate(request.candidate_collection.variants):
        score_sum = -3.0 - index * 0.3
        rows.append(
            ScoreRecord(
                protein_id=request.protein_id,
                condition_id=request.condition.condition_id,
                structure_sha256=request.condition.structure_sha256,
                variant_kind=variant.variant_kind,
                variant_id=variant.variant_id,
                sequence_hash=(
                    None
                    if variant.variant_kind is VariantKind.WT
                    else variant.sequence_hash
                ),
                position=variant.position,
                wt_aa=variant.wt_aa,
                mut_aa=variant.mut_aa,
                repeat_index=request.repeat_index,
                seed=request.seed,
                realization_id=request.realization_id,
                scorer_id=binding.scorer_id,
                implementation_id=binding.implementation_id,
                checkpoint_id=binding.checkpoint_id,
                score_contract_id=binding.score_contract_id,
                score_sum_logp_mask=score_sum,
                score_mean_logp_mask=score_sum / len(request.canonical_positions),
                scored_residue_count=len(request.canonical_positions),
            )
        )
    return tuple(rows)


class _FakeSequenceScorer:
    binding = ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id=SCORE_CONTRACT,
    )

    def __init__(self, records=None) -> None:
        self._records = records
        self.calls = 0

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        self.calls += 1
        if self._records is not None:
            return tuple(self._records)
        return _records(request)


def test_execute_score_request_validates_and_preserves_declared_order() -> None:
    request = _request()
    scorer = _FakeSequenceScorer()

    records = execute_score_request(scorer, request)

    assert scorer.calls == 1
    assert len(records) == request.result_count
    assert [record.variant_kind for record in records] == [
        VariantKind.WT,
        VariantKind.PROBE,
    ]
    assert records[1].variant_id == request.candidate_collection.probes[0].variant_id


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_protein",
        "wrong_condition",
        "wrong_structure",
        "wrong_repeat",
        "wrong_seed",
        "wrong_realization",
        "wrong_score_contract",
        "missing_wt",
        "duplicate_wt",
        "missing_probe",
        "duplicate_probe",
        "unexpected_probe",
    ],
)
def test_execute_score_request_rejects_invalid_result_contract(mutation: str) -> None:
    request = _request()
    records = list(_records(request))
    if mutation == "wrong_protein":
        records[0] = replace(records[0], protein_id="other_A__P00002")
    elif mutation == "wrong_condition":
        records[0] = replace(records[0], condition_id="state-b")
    elif mutation == "wrong_structure":
        records[0] = replace(records[0], structure_sha256="8" * 64)
    elif mutation == "wrong_repeat":
        records[0] = replace(records[0], repeat_index=1)
    elif mutation == "wrong_seed":
        records[0] = replace(records[0], seed=1)
    elif mutation == "wrong_realization":
        records[0] = replace(records[0], realization_id="8" * 64)
    elif mutation == "wrong_score_contract":
        records[0] = replace(records[0], score_contract_id="other-v1")
    elif mutation == "missing_wt":
        records = records[1:]
    elif mutation == "duplicate_wt":
        records[1] = records[0]
    elif mutation == "missing_probe":
        records = records[:1]
    elif mutation == "duplicate_probe":
        records.append(records[1])
    elif mutation == "unexpected_probe":
        records[1] = replace(
            records[1], variant_id="8" * 64, sequence_hash="8" * 64
        )

    with pytest.raises(ScoreDispatchError):
        execute_score_request(_FakeSequenceScorer(records), request)


def test_execute_score_request_rejects_scorer_contract_before_invocation() -> None:
    request = _request()
    scorer = _FakeSequenceScorer()
    scorer.binding = replace(scorer.binding, score_contract_id="other-v1")

    with pytest.raises(ScoreDispatchError, match="score contract"):
        execute_score_request(scorer, request)

    assert scorer.calls == 0


def test_frozen_and_same_source_conditions_use_identical_dispatch() -> None:
    full_plan = pd.read_parquet(
        V2_ROOT / "scale1b_v2_proteinmpnn_scoring_plan.parquet"
    )
    protein_id = str(full_plan.iloc[0]["protein_id"])
    plan = full_plan.loc[
        full_plan["protein_id"].eq(protein_id) & full_plan["repeat"].eq(0)
    ]
    cohort = pd.read_parquet(V2_ROOT / "scale1b_v2_primary_cohort.parquet").query(
        "protein_id == @protein_id"
    )
    masks = pd.read_parquet(
        V2_ROOT / "scale1b_v2_primary_common_masks.parquet"
    ).query("protein_id == @protein_id")
    probes = pd.read_parquet(V2_ROOT / "scale1b_v2_fixed_probes.parquet").query(
        "protein_id == @protein_id"
    )
    frozen_view = adapt_frozen_scoring_plan(
        plan=plan,
        primary_cohort=cohort,
        common_masks=masks,
        fixed_probes=probes,
    )[0]
    frozen_records = execute_score_request(
        _FakeSequenceScorer(), frozen_view.request
    )

    intervention = StructuralIntervention(
        intervention_id="fixture::apo-like__holo-like",
        condition_a=_condition("apo-like", "PDB", "6" * 64),
        condition_b=_condition("holo-like", "PDB", "7" * 64),
    )
    synthetic_records = tuple(
        execute_score_request(_FakeSequenceScorer(), _request(condition))
        for condition in intervention.conditions
    )

    assert len(frozen_records) == frozen_view.request.result_count
    assert all(len(records) == 2 for records in synthetic_records)
    assert {condition.source for condition in intervention.conditions} == {"PDB"}
