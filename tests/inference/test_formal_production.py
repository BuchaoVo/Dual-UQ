from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.inference.formal import (
    FormalRequestDefinition,
    MaterializationState,
    derive_execution_inventory,
    derive_orchestration_id,
    execute_formal_partition,
    materialize_fresh_request,
    partition_formal_work,
    select_deterministic_preflight_pair,
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


def test_worker_import_path_does_not_require_pyarrow() -> None:
    script = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'pyarrow' or name.startswith('pyarrow.'):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import dual_uq.inference.formal
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _binding() -> ScorerBinding:
    return ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id=None,
        score_contract_id="fixture-mask-logp-v1",
    )


def _definition(condition: str, repeat: int) -> FormalRequestDefinition:
    wt = "ACD"
    probe = "AED"
    probe_hash = sha256_bytes(probe.encode("ascii"))
    request = ScoreRequest(
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
                sequence_hash=sha256_bytes(wt.encode("ascii")),
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
        repeat_index=repeat,
        seed=repeat,
        realization_id=(str(repeat + 4) * 64)[:64],
        realization_algorithm="fixture-order-v1",
        score_contract_id="fixture-mask-logp-v1",
    )
    fingerprint = sha256_canonical(
        {"condition": condition, "repeat": repeat, "fixture": True}
    )
    return FormalRequestDefinition(
        scientific_fingerprint=fingerprint,
        orchestration_id=derive_orchestration_id(fingerprint),
        workflow_request_id=f"fixture::{condition}::r{repeat:02d}",
        request=request,
        scorer_binding=_binding(),
        workflow_metadata={"cluster_id_30": "fixture-cluster"},
    )


class _FakeScorer:
    binding = _binding()

    def __init__(self) -> None:
        self.calls: list[str] = []

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        self.calls.append(
            f"{request.condition.condition_id}:r{request.repeat_index:02d}"
        )
        rows = []
        for index, variant in enumerate(request.candidate_collection.variants):
            score_sum = -3.0 - index
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


def test_preflight_selects_smallest_fully_pending_realization_pair(
    tmp_path: Path,
) -> None:
    definitions = tuple(
        _definition(condition, repeat)
        for repeat in (0, 1)
        for condition in ("PDB", "AFDB")
    )
    scorer = _FakeScorer()
    materialize_fresh_request(
        definitions[0],
        scorer=scorer,
        artifact_root=tmp_path,
        execution_provenance={"mode": "fixture"},
    )
    snapshot = derive_execution_inventory(definitions, artifact_root=tmp_path)

    selected = select_deterministic_preflight_pair(snapshot, definitions)

    assert {row.request.condition.condition_id for row in selected} == {
        "PDB",
        "AFDB",
    }
    assert {row.request.repeat_index for row in selected} == {1}
    assert {row.request.realization_id for row in selected} == {
        definitions[2].request.realization_id
    }


def test_static_partitions_are_sorted_disjoint_and_complete() -> None:
    definitions = tuple(
        _definition(condition, repeat)
        for repeat in range(4)
        for condition in ("PDB", "AFDB")
    )
    partitions = [
        partition_formal_work(definitions, worker_count=3, worker_index=index)
        for index in range(3)
    ]

    observed = [
        row.orchestration_id for partition in partitions for row in partition
    ]
    assert len(observed) == len(set(observed)) == len(definitions)
    assert set(observed) == {row.orchestration_id for row in definitions}
    for index, partition in enumerate(partitions):
        expected = tuple(sorted(definitions, key=lambda row: row.orchestration_id))[
            index::3
        ]
        assert partition == expected


def test_partition_execution_validates_resume_without_duplicate_forward(
    tmp_path: Path,
) -> None:
    definitions = (_definition("PDB", 0), _definition("AFDB", 0))
    scorer = _FakeScorer()

    first = execute_formal_partition(
        definitions,
        scorer=scorer,
        artifact_root=tmp_path,
        execution_provenance={"mode": "fixture-production"},
    )
    second = execute_formal_partition(
        definitions,
        scorer=scorer,
        artifact_root=tmp_path,
        execution_provenance={"mode": "fixture-production"},
    )

    assert first == {"selected": 2, "executed": 2, "reused_valid": 0}
    assert second == {"selected": 2, "executed": 0, "reused_valid": 2}
    assert len(scorer.calls) == 2
    assert derive_execution_inventory(
        definitions, artifact_root=tmp_path
    ).state_counts == {MaterializationState.VALID_CANONICAL_COMPLETE: 2}
