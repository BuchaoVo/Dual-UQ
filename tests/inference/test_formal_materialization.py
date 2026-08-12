from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dual_uq.core.hashing import sha256_bytes
from dual_uq.inference.materialization import (
    ArtifactInspectionState,
    FormalMaterializationError,
    bind_formal_artifact,
    build_formal_shard_payload,
    inspect_formal_artifact,
    materialize_formal_artifact,
    validate_formal_shard_payload,
)
from dual_uq.models.scoring import (
    CandidateCollection,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
)
from dual_uq.structure import StructureCondition


def _request() -> ScoreRequest:
    wt = "ACD"
    probe = "AED"
    wt_hash = sha256_bytes(wt.encode("ascii"))
    probe_hash = sha256_bytes(probe.encode("ascii"))
    return ScoreRequest(
        condition=StructureCondition(
            protein_id="fixture_A__P00001",
            condition_id="PDB",
            source="PDB",
            structure_sha256="1" * 64,
        ),
        scoring_domain_id="2" * 64,
        canonical_positions=(1, 2, 3),
        candidate_collection=CandidateCollection(
            collection_id=sha256_bytes(
                ('{"sequence_hashes":["' + probe_hash + '"]}').encode("utf-8")
            ),
            wt=ScoringVariant(
                variant_kind=VariantKind.WT,
                variant_id=None,
                sequence_hash=wt_hash,
                sequence=wt,
                position=None,
                wt_aa=None,
                mut_aa=None,
            ),
            probes=(
                ScoringVariant(
                    variant_kind=VariantKind.PROBE,
                    variant_id=probe_hash,
                    sequence_hash=probe_hash,
                    sequence=probe,
                    position=2,
                    wt_aa="C",
                    mut_aa="E",
                ),
            ),
        ),
        repeat_index=0,
        seed=0,
        realization_id="3" * 64,
        realization_algorithm="fixture_order_v1",
        score_contract_id="fixture_mask_logp_v1",
    )


def _binding() -> ScorerBinding:
    return ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id="fixture_mask_logp_v1",
    )


def _records(request: ScoreRequest) -> tuple[ScoreRecord, ...]:
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
                scorer_id="FakeSequenceScorer",
                implementation_id="fake-v1",
                checkpoint_id=None,
                score_contract_id=request.score_contract_id,
                score_sum_logp_mask=score_sum,
                score_mean_logp_mask=score_sum / 3,
                scored_residue_count=3,
            )
        )
    return tuple(rows)


def test_artifact_binding_is_deterministic_portable_and_prefix_sharded() -> None:
    first = bind_formal_artifact("a" * 64)
    second = bind_formal_artifact("a" * 64)

    assert first == second
    assert first.orchestration_id == "a" * 64
    assert first.relative_path == Path("shards/aa") / f"{'a' * 64}.json"
    assert not first.relative_path.is_absolute()


def test_valid_artifact_is_atomic_reusable_and_content_validated(tmp_path: Path) -> None:
    request = _request()
    scorer = _binding()
    fingerprint = "a" * 64
    artifact = bind_formal_artifact(fingerprint)
    payload = build_formal_shard_payload(
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
        orchestration_id=artifact.orchestration_id,
        records=_records(request),
        execution_provenance={"mode": "fake-test"},
    )

    assert materialize_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        payload=payload,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
    ) == "created"
    assert materialize_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        payload=payload,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
    ) == "reused_identical"

    inspected = inspect_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
    )
    assert inspected.state is ArtifactInspectionState.VALID
    assert inspected.artifact_sha256 is not None
    assert validate_formal_shard_payload(
        payload,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
        orchestration_id=artifact.orchestration_id,
    ) == payload


def test_missing_invalid_and_binding_conflict_are_distinct(tmp_path: Path) -> None:
    request = _request()
    scorer = _binding()
    artifact = bind_formal_artifact("a" * 64)
    missing = inspect_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint="a" * 64,
    )
    assert missing.state is ArtifactInspectionState.MISSING

    path = artifact.resolve(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    invalid = inspect_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint="a" * 64,
    )
    assert invalid.state is ArtifactInspectionState.INVALID

    path.unlink()
    payload = build_formal_shard_payload(
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint="b" * 64,
        orchestration_id="b" * 64,
        records=_records(request),
        execution_provenance={"mode": "fake-test"},
    )
    path.write_bytes(__import__("json").dumps(payload).encode("utf-8"))
    conflict = inspect_formal_artifact(
        root=tmp_path,
        artifact=artifact,
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint="a" * 64,
    )
    assert conflict.state is ArtifactInspectionState.CONFLICT


def test_immutable_conflict_is_never_overwritten(tmp_path: Path) -> None:
    request = _request()
    scorer = _binding()
    fingerprint = "a" * 64
    artifact = bind_formal_artifact(fingerprint)
    payload = build_formal_shard_payload(
        request=request,
        scorer_binding=scorer,
        scientific_fingerprint=fingerprint,
        orchestration_id=artifact.orchestration_id,
        records=_records(request),
        execution_provenance={"mode": "fake-test"},
    )
    path = artifact.resolve(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("occupied", encoding="utf-8")

    with pytest.raises(FormalMaterializationError, match="immutable"):
        materialize_formal_artifact(
            root=tmp_path,
            artifact=artifact,
            payload=payload,
            request=request,
            scorer_binding=scorer,
            scientific_fingerprint=fingerprint,
        )

    assert path.read_text(encoding="utf-8") == "occupied"


def test_score_record_binding_mismatch_is_rejected() -> None:
    request = _request()
    records = list(_records(request))
    records[1] = replace(records[1], realization_id="9" * 64)

    with pytest.raises(FormalMaterializationError, match="request identity"):
        build_formal_shard_payload(
            request=request,
            scorer_binding=_binding(),
            scientific_fingerprint="a" * 64,
            orchestration_id="a" * 64,
            records=tuple(records),
            execution_provenance={"mode": "fake-test"},
        )
