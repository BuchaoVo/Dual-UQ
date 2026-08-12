from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.inference.formal import (
    FormalInventoryError,
    FormalRequestDefinition,
    HistoricalReuseDecision,
    HistoricalReuseStatus,
    MaterializationState,
    derive_execution_inventory,
    derive_orchestration_id,
    discover_valid_materializations,
    materialize_fresh_request,
    materialize_historical_reuse,
    materialize_inventory_snapshot,
    select_fresh_work,
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


def _request(condition: str = "PDB") -> ScoreRequest:
    wt = "ACD"
    probe = "AED"
    wt_hash = sha256_bytes(wt.encode("ascii"))
    probe_hash = sha256_bytes(probe.encode("ascii"))
    return ScoreRequest(
        condition=StructureCondition(
            protein_id="fixture_A__P00001",
            condition_id=condition,
            source=condition,
            structure_sha256=("1" if condition == "PDB" else "2") * 64,
        ),
        scoring_domain_id="3" * 64,
        canonical_positions=(1, 2, 3),
        candidate_collection=CandidateCollection(
            collection_id=sha256_canonical({"sequence_hashes": [probe_hash]}),
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
        realization_id="4" * 64,
        realization_algorithm="fixture_order_v1",
        score_contract_id="fixture_mask_logp_v1",
    )


def _scorer_binding() -> ScorerBinding:
    return ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id="fixture_mask_logp_v1",
    )


def _definition(condition: str = "PDB") -> FormalRequestDefinition:
    fingerprint = ("a" if condition == "PDB" else "b") * 64
    return FormalRequestDefinition(
        scientific_fingerprint=fingerprint,
        orchestration_id=derive_orchestration_id(fingerprint),
        workflow_request_id=f"fixture::{condition}::r00",
        request=_request(condition),
        scorer_binding=_scorer_binding(),
        workflow_metadata={"cluster_id_30": "fixture-cluster"},
    )


class _FakeSequenceScorer:
    binding = _scorer_binding()

    def __init__(self) -> None:
        self.calls = 0

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        self.calls += 1
        rows = []
        for index, variant in enumerate(request.candidate_collection.variants):
            score_sum = -3.0 - 0.3 * index
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
                    scorer_id=self.binding.scorer_id,
                    implementation_id=self.binding.implementation_id,
                    checkpoint_id=self.binding.checkpoint_id,
                    score_contract_id=self.binding.score_contract_id,
                    score_sum_logp_mask=score_sum,
                    score_mean_logp_mask=score_sum / 3,
                    scored_residue_count=3,
                )
            )
        return tuple(rows)


def test_orchestration_identity_is_derived_but_role_distinct() -> None:
    fingerprint = "a" * 64
    definition = _definition()

    assert derive_orchestration_id(fingerprint) == fingerprint
    assert definition.scientific_fingerprint == definition.orchestration_id
    assert definition.artifact.relative_path.as_posix().startswith("shards/aa/")
    assert definition.artifact.relative_path.as_posix() not in str(
        definition.request
    )


def test_inventory_snapshot_is_derived_from_artifact_validity_and_selects_fresh(
    tmp_path: Path,
) -> None:
    definitions = (_definition("PDB"), _definition("AFDB"))
    initial = derive_execution_inventory(definitions, artifact_root=tmp_path)

    assert initial.total_requests == 2
    assert initial.state_counts == {MaterializationState.FRESH_EXECUTION_REQUIRED: 2}
    assert [row.orchestration_id for row in select_fresh_work(initial, definitions)] == [
        definitions[0].orchestration_id,
        definitions[1].orchestration_id,
    ]
    assert initial.expected_wt_rows == 2
    assert initial.expected_probe_rows == 2


def test_fake_end_to_end_materializes_validates_resumes_and_discovers(
    tmp_path: Path,
) -> None:
    definition = _definition()
    scorer = _FakeSequenceScorer()
    initial = derive_execution_inventory((definition,), artifact_root=tmp_path)
    assert initial.entries[0].state is MaterializationState.FRESH_EXECUTION_REQUIRED

    assert materialize_fresh_request(
        definition,
        scorer=scorer,
        artifact_root=tmp_path,
        execution_provenance={"mode": "fake-test"},
    ) == "created"
    assert scorer.calls == 1

    resumed = derive_execution_inventory((definition,), artifact_root=tmp_path)
    assert resumed.entries[0].state is MaterializationState.VALID_CANONICAL_COMPLETE
    assert select_fresh_work(resumed, (definition,)) == ()
    discovered = discover_valid_materializations(
        (definition,), artifact_root=tmp_path
    )
    assert tuple(discovered) == (definition.orchestration_id,)
    assert discovered[definition.orchestration_id]["binding"][
        "scientific_fingerprint"
    ] == definition.scientific_fingerprint


def test_duplicate_identity_and_binding_collision_fail_loudly(tmp_path: Path) -> None:
    first = _definition("PDB")
    duplicate = replace(first, workflow_request_id="different-row")
    with pytest.raises(FormalInventoryError, match="duplicate orchestration"):
        derive_execution_inventory((first, duplicate), artifact_root=tmp_path)

    second = _definition("AFDB")
    collision = replace(second, artifact=first.artifact)
    with pytest.raises(FormalInventoryError, match="artifact-binding collision"):
        derive_execution_inventory((first, collision), artifact_root=tmp_path)


def test_invalid_and_conflicting_artifacts_are_not_fresh_work(tmp_path: Path) -> None:
    definition = _definition()
    path = definition.artifact.resolve(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    invalid = derive_execution_inventory((definition,), artifact_root=tmp_path)
    assert invalid.entries[0].state is MaterializationState.INVALID_EXISTING
    assert select_fresh_work(invalid, (definition,)) == ()

    path.write_text(
        '{"schema_version":"stage0_fixed_probe_scoring_shard_v1",'
        '"binding":{"orchestration_id":"different"}}',
        encoding="utf-8",
    )
    conflict = derive_execution_inventory((definition,), artifact_root=tmp_path)
    assert conflict.entries[0].state is MaterializationState.CONFLICT


def test_historical_authorization_and_compatibility_are_separate(tmp_path: Path) -> None:
    definition = _definition()
    authorized_but_rejected = replace(
        definition,
        historical_reuse=HistoricalReuseDecision(
            status=HistoricalReuseStatus.REJECTED,
            reuse_contract_identity="5" * 64,
            source_execution_identity="stage0-formal",
            source_artifact_reference="experiments/stage0/scores.parquet",
            source_artifact_sha256="6" * 64,
            compatibility_validation_identity="7" * 64,
            rejection_reason="structure_sha_mismatch",
        ),
    )
    rejected = derive_execution_inventory(
        (authorized_but_rejected,), artifact_root=tmp_path
    )
    assert rejected.entries[0].state is MaterializationState.FRESH_EXECUTION_REQUIRED
    assert rejected.entries[0].historical_reuse_status is (
        HistoricalReuseStatus.REJECTED
    )

    accepted = replace(
        definition,
        historical_reuse=replace(
            authorized_but_rejected.historical_reuse,
            status=HistoricalReuseStatus.ACCEPTED,
            rejection_reason=None,
        ),
    )
    inventory = derive_execution_inventory((accepted,), artifact_root=tmp_path)
    assert inventory.entries[0].state is MaterializationState.HISTORICAL_REUSE_ACCEPTED


def test_accepted_historical_records_materialize_canonically_with_provenance(
    tmp_path: Path,
) -> None:
    base = _definition()
    scorer = _FakeSequenceScorer()
    records = scorer.score(base.request)
    accepted = replace(
        base,
        historical_reuse=HistoricalReuseDecision(
            status=HistoricalReuseStatus.ACCEPTED,
            reuse_contract_identity="5" * 64,
            source_execution_identity="stage0-formal",
            source_artifact_reference="experiments/stage0/scores.parquet",
            source_artifact_sha256="6" * 64,
            compatibility_validation_identity="7" * 64,
            rejection_reason=None,
        ),
    )

    assert materialize_historical_reuse(
        accepted, records=records, artifact_root=tmp_path
    ) == "created"
    refreshed = derive_execution_inventory((accepted,), artifact_root=tmp_path)
    assert refreshed.entries[0].state is MaterializationState.VALID_CANONICAL_COMPLETE
    payload = discover_valid_materializations(
        (accepted,), artifact_root=tmp_path
    )[accepted.orchestration_id]
    provenance = payload["execution_environment"]["historical_reuse"]
    assert provenance["target_scientific_fingerprint"] == (
        accepted.scientific_fingerprint
    )
    assert provenance["source_artifact_sha256"] == "6" * 64
    assert provenance["compatibility_validation_identity"] == "7" * 64


def test_inventory_snapshot_is_content_addressed_atomic_and_reusable(
    tmp_path: Path,
) -> None:
    definitions = (_definition("PDB"), _definition("AFDB"))
    snapshot = derive_execution_inventory(definitions, artifact_root=tmp_path)
    provenance = {
        "workflow_id": "fixture-formal-v1",
        "frozen_request_aggregate_sha256": "8" * 64,
        "execution_code_identity": {"git_head": "fixture-head"},
        "proteinmpnn_forward_executions": 0,
    }

    first = materialize_inventory_snapshot(
        snapshot, output_root=tmp_path / "inventory", provenance=provenance
    )
    second = materialize_inventory_snapshot(
        snapshot, output_root=tmp_path / "inventory", provenance=provenance
    )

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["write_status"] == {
        "inventory": "created",
        "manifest": "created",
    }
    assert second["write_status"] == {
        "inventory": "reused_identical",
        "manifest": "reused_identical",
    }
    inventory = pd.read_parquet(first["inventory_path"])
    manifest = __import__("json").loads(first["manifest_path"].read_text())
    assert len(inventory) == 2
    assert inventory["orchestration_id"].is_unique
    assert set(inventory["materialization_state"]) == {
        MaterializationState.FRESH_EXECUTION_REQUIRED.value
    }
    assert manifest["snapshot_id"] == first["snapshot_id"]
    assert manifest["counts"]["total_requests"] == 2
    assert manifest["provenance"] == provenance
