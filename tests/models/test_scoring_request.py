from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.models.proteinmpnn import (
    DECODING_REALIZATION_ALGORITHM,
    ProteinMPNNScore,
    ProteinMPNNScorer,
    ProteinMPNNScoringError,
    ProteinMPNNStructureInput,
)
from dual_uq.models.scoring import (
    CandidateCollection,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    SequenceScorer,
    VariantKind,
)
from dual_uq.structure import StructuralIntervention, StructureCondition
from dual_uq.workflows.final_confirmatory_protocol import (
    adapt_frozen_scoring_plan,
    frozen_plan_scientific_fingerprint,
    frozen_score_request_scientific_view,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPOSITORY_ROOT / "experiments/p2_design_baseline/scale1/scale1b_v2"
SCORE_CONTRACT = "stage0_fixed_sequence_autoregressive_mask_logp_v1"


def _condition(
    condition_id: str = "state_a", source: str = "PDB", digest: str = "1" * 64
) -> StructureCondition:
    return StructureCondition(
        protein_id="fixture_A__P00001",
        condition_id=condition_id,
        source=source,
        structure_sha256=digest,
    )


def _collection() -> CandidateCollection:
    return CandidateCollection(
        collection_id="2" * 64,
        wt=ScoringVariant(
            variant_kind=VariantKind.WT,
            variant_id=None,
            sequence_hash=(
                "00e66854ddc46722ac3db985136265f4a24bcbbf0b45103d80cfea510e9217bf"
            ),
            sequence="ACD",
            position=None,
            wt_aa=None,
            mut_aa=None,
        ),
        probes=(
            ScoringVariant(
                variant_kind=VariantKind.PROBE,
                variant_id=(
                    "cb3f8a65058da0b7c8524a4fb6c27f8456b7df01949b26f671a8e16622ec65b9"
                ),
                sequence_hash=(
                    "cb3f8a65058da0b7c8524a4fb6c27f8456b7df01949b26f671a8e16622ec65b9"
                ),
                sequence="AED",
                position=2,
                wt_aa="C",
                mut_aa="E",
            ),
        ),
    )


def _request(condition: StructureCondition | None = None) -> ScoreRequest:
    return ScoreRequest(
        condition=condition or _condition(),
        scoring_domain_id="5" * 64,
        canonical_positions=(1, 2, 3),
        candidate_collection=_collection(),
        repeat_index=0,
        seed=0,
        realization_id=(
            "0f004f117335020e1d19c25b8767278bf1edb2fa6ff3fac943d843b6003d0eb5"
        ),
        realization_algorithm=DECODING_REALIZATION_ALGORITHM,
        score_contract_id=SCORE_CONTRACT,
    )


def test_score_request_is_minimal_immutable_and_collection_level() -> None:
    request = _request()

    assert tuple(field.name for field in fields(ScoreRequest)) == (
        "condition",
        "scoring_domain_id",
        "canonical_positions",
        "candidate_collection",
        "repeat_index",
        "seed",
        "realization_id",
        "realization_algorithm",
        "score_contract_id",
    )
    assert request.protein_id == request.condition.protein_id
    assert request.result_count == 2
    assert request.candidate_collection.wt.variant_kind is VariantKind.WT
    assert len(request.candidate_collection.probes) == 1
    with pytest.raises(FrozenInstanceError):
        request.seed = 1  # type: ignore[misc]

    forbidden = {
        "cluster_id_30",
        "source_stratum",
        "logical_shard_id",
        "batch_size",
        "device",
        "worker",
        "output_path",
        "checkpoint_id",
        "implementation_id",
    }
    assert forbidden.isdisjoint({field.name for field in fields(ScoreRequest)})


def test_candidate_collection_rejects_non_singleton_sequence_drift() -> None:
    drifted = ScoringVariant(
        variant_kind=VariantKind.PROBE,
        variant_id=sha256_bytes(b"EEE"),
        sequence_hash=sha256_bytes(b"EEE"),
        sequence="EEE",
        position=2,
        wt_aa="C",
        mut_aa="E",
    )

    with pytest.raises(ValueError, match="single substitution from WT"):
        CandidateCollection(
            collection_id="2" * 64,
            wt=_collection().wt,
            probes=(drifted,),
        )


class _FakeSequenceScorer:
    binding = ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id=SCORE_CONTRACT,
    )

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        count = len(request.canonical_positions)
        records = []
        for index, variant in enumerate(request.candidate_collection.variants):
            score_sum = float(-(index + 1) * count)
            records.append(
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
                    scorer_id=self.binding.scorer_id,
                    implementation_id=self.binding.implementation_id,
                    checkpoint_id=self.binding.checkpoint_id,
                    score_contract_id=request.score_contract_id,
                    score_sum_logp_mask=score_sum,
                    score_mean_logp_mask=score_sum / count,
                    scored_residue_count=count,
                )
            )
        return tuple(records)


def test_sequence_scorer_contract_is_one_request_to_wt_plus_probe_records() -> None:
    scorer: SequenceScorer = _FakeSequenceScorer()
    request = _request()

    records = scorer.score(request)

    assert len(records) == request.result_count == 2
    assert [record.variant_kind for record in records] == [
        VariantKind.WT,
        VariantKind.PROBE,
    ]
    assert str(inspect.signature(SequenceScorer.score)) == (
        "(self, request: 'ScoreRequest') -> 'tuple[ScoreRecord, ...]'"
    )


def test_same_source_intervention_composes_with_same_request_and_scorer_path() -> None:
    condition_a = _condition("apo-like", "PDB", "7" * 64)
    condition_b = _condition("holo-like", "PDB", "8" * 64)
    intervention = StructuralIntervention(
        intervention_id="fixture::apo-like__holo-like",
        condition_a=condition_a,
        condition_b=condition_b,
    )
    scorer: SequenceScorer = _FakeSequenceScorer()

    requests = tuple(_request(condition) for condition in intervention.conditions)
    outputs = tuple(scorer.score(request) for request in requests)

    assert {request.condition.source for request in requests} == {"PDB"}
    assert {request.condition.condition_id for request in requests} == {
        "apo-like",
        "holo-like",
    }
    assert all(len(records) == 2 for records in outputs)


class _NoForwardAdapter:
    implementation_id = "fake-proteinmpnn-v1"
    checkpoint_id = "9" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[ProteinMPNNStructureInput, tuple[str, ...], object, int]] = []

    def score_sequences(
        self,
        structure: ProteinMPNNStructureInput,
        sequences: tuple[str, ...],
        realization: object,
        *,
        batch_size: int,
    ) -> tuple[ProteinMPNNScore, ...]:
        self.calls.append((structure, sequences, realization, batch_size))
        count = structure.residue_count
        return tuple(
            ProteinMPNNScore(
                score_sum_logp_mask=float(-(index + 1) * count),
                score_mean_logp_mask=float(-(index + 1)),
            )
            for index, _sequence in enumerate(sequences)
        )


def test_proteinmpnn_scorer_maps_generic_request_without_forward() -> None:
    request = _request()
    projection = ProteinMPNNStructureInput(
        protein_id=request.protein_id,
        backbone_condition=request.condition.condition_id,
        uniprot_positions=request.canonical_positions,
        wt_sequence_projection="ACD",
        coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        structure_sha256=request.condition.structure_sha256,
    )
    adapter = _NoForwardAdapter()
    scorer = ProteinMPNNScorer(
        adapter=adapter,  # type: ignore[arg-type]
        projection_resolver=lambda observed: projection,
        score_contract_id=SCORE_CONTRACT,
        batch_size=8,
    )

    records = scorer.score(request)

    assert len(records) == 2
    assert [record.variant_kind for record in records] == [
        VariantKind.WT,
        VariantKind.PROBE,
    ]
    assert records[1].sequence_hash == request.candidate_collection.probes[0].sequence_hash
    assert records[0].scorer_id == "ProteinMPNN"
    assert records[0].implementation_id == adapter.implementation_id
    assert records[0].checkpoint_id == adapter.checkpoint_id
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1] == ("ACD", "AED")
    assert adapter.calls[0][3] == 8


def test_proteinmpnn_scorer_rejects_contract_or_realization_mismatch() -> None:
    projection = ProteinMPNNStructureInput(
        protein_id="fixture_A__P00001",
        backbone_condition="state_a",
        uniprot_positions=(1, 2, 3),
        wt_sequence_projection="ACD",
        coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        structure_sha256="1" * 64,
    )
    scorer = ProteinMPNNScorer(
        adapter=_NoForwardAdapter(),  # type: ignore[arg-type]
        projection_resolver=lambda observed: projection,
        score_contract_id=SCORE_CONTRACT,
        batch_size=8,
    )

    with pytest.raises(ProteinMPNNScoringError) as contract_error:
        scorer.score(
            replace(_request(), score_contract_id="other-v1")
        )
    assert contract_error.value.code == "score_contract_mismatch"

    with pytest.raises(ProteinMPNNScoringError) as realization_error:
        scorer.score(replace(_request(), realization_id="6" * 64))
    assert realization_error.value.code == "decoding_realization_mismatch"


def test_proteinmpnn_scorer_rejects_projection_structure_identity_mismatch() -> None:
    projection = ProteinMPNNStructureInput(
        protein_id="fixture_A__P00001",
        backbone_condition="state_a",
        uniprot_positions=(1, 2, 3),
        wt_sequence_projection="ACD",
        coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        structure_sha256="8" * 64,
    )
    scorer = ProteinMPNNScorer(
        adapter=_NoForwardAdapter(),  # type: ignore[arg-type]
        projection_resolver=lambda observed: projection,
        score_contract_id=SCORE_CONTRACT,
        batch_size=8,
    )

    with pytest.raises(ProteinMPNNScoringError) as caught:
        scorer.score(_request())

    assert caught.value.code == "scoring_projection_structure_mismatch"


def test_all_frozen_rows_map_to_requests_with_exact_r0_fingerprints() -> None:
    plan = pd.read_parquet(V2_ROOT / "scale1b_v2_proteinmpnn_scoring_plan.parquet")
    cohort = pd.read_parquet(V2_ROOT / "scale1b_v2_primary_cohort.parquet")
    masks = pd.read_parquet(V2_ROOT / "scale1b_v2_primary_common_masks.parquet")
    probes = pd.read_parquet(V2_ROOT / "scale1b_v2_fixed_probes.parquet")

    views = adapt_frozen_scoring_plan(
        plan=plan,
        primary_cohort=cohort,
        common_masks=masks,
        fixed_probes=probes,
    )
    request_fingerprints = [
        frozen_plan_scientific_fingerprint(frozen_score_request_scientific_view(view))
        for view in views
    ]
    row_fingerprints = [
        frozen_plan_scientific_fingerprint(row) for row in plan.to_dict("records")
    ]

    assert len(views) == 7_620
    assert request_fingerprints == row_fingerprints
    assert sha256_canonical({"fingerprints": request_fingerprints}) == (
        "378c15043e5b46e743c799a42e574524189348de474ff052f387de48ebde49f9"
    )
    assert len({id(view.request.candidate_collection) for view in views}) == 127
    assert all(view.request.result_count > 1 for view in views)
    assert all(view.request.condition in view.intervention.conditions for view in views)


def test_frozen_adapter_contains_historical_metadata_outside_score_request() -> None:
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
    masks = pd.read_parquet(V2_ROOT / "scale1b_v2_primary_common_masks.parquet").query(
        "protein_id == @protein_id"
    )
    probes = pd.read_parquet(V2_ROOT / "scale1b_v2_fixed_probes.parquet").query(
        "protein_id == @protein_id"
    )

    views = adapt_frozen_scoring_plan(
        plan=plan,
        primary_cohort=cohort,
        common_masks=masks,
        fixed_probes=probes,
    )

    assert len(views) == 2
    assert views[0].historical_metadata["logical_shard_id"].endswith("::r00")
    assert views[0].historical_metadata["cluster_id_30"]
    assert views[0].historical_metadata["source_stratum"]
    records = _FakeSequenceScorer().score(views[0].request)
    assert len(records) == views[0].request.result_count
    assert records[0].variant_kind is VariantKind.WT
    request_fields = {field.name for field in fields(ScoreRequest)}
    assert {"logical_shard_id", "cluster_id_30", "source_stratum"}.isdisjoint(
        request_fields
    )
