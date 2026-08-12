from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.inference.formal import (
    FormalRequestDefinition,
    derive_orchestration_id,
    materialize_fresh_request,
)
from dual_uq.inference.materialization import consolidate_formal_measurements
from dual_uq.models.scoring import (
    CandidateCollection,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
)
from dual_uq.structure import StructureCondition

WT_COLUMNS = [
    "protein_id",
    "backbone_condition",
    "backbone_sha256",
    "repeat_index",
    "seed",
    "decoding_realization_sha256",
    "score_sum_logp_mask",
    "score_mean_logp_mask",
    "scored_residue_count",
    "model_checkpoint_sha256",
    "scoring_protocol",
]
PROBE_COLUMNS = [
    "protein_id",
    "sequence_hash",
    "position",
    "wt_aa",
    "mut_aa",
    "backbone_condition",
    "backbone_sha256",
    "repeat_index",
    "seed",
    "decoding_realization_sha256",
    "score_sum_logp_mask",
    "score_mean_logp_mask",
    "delta_score_vs_wt",
    "scored_residue_count",
    "model_checkpoint_sha256",
    "scoring_protocol",
]


def _binding() -> ScorerBinding:
    return ScorerBinding(
        scorer_id="FakeSequenceScorer",
        implementation_id="fake-v1",
        checkpoint_id="f" * 64,
        score_contract_id="fixture-mask-logp-v1",
    )


def _definition(condition: str) -> FormalRequestDefinition:
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
        repeat_index=0,
        seed=0,
        realization_id="4" * 64,
        realization_algorithm="fixture-order-v1",
        score_contract_id="fixture-mask-logp-v1",
    )
    fingerprint = sha256_canonical({"condition": condition, "fixture": True})
    return FormalRequestDefinition(
        scientific_fingerprint=fingerprint,
        orchestration_id=derive_orchestration_id(fingerprint),
        workflow_request_id=f"fixture::{condition}::r00",
        request=request,
        scorer_binding=_binding(),
        workflow_metadata={"cluster_id_30": "fixture-cluster"},
    )


class _FakeScorer:
    binding = _binding()

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
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
                    repeat_index=0,
                    seed=0,
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


def test_consolidation_uses_expected_universe_and_preserves_frozen_schema(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    definitions = (_definition("AFDB"), _definition("PDB"))
    scorer = _FakeScorer()
    for definition in definitions:
        materialize_fresh_request(
            definition,
            scorer=scorer,
            artifact_root=artifact_root,
            execution_provenance={"mode": "fixture"},
        )
    extra = artifact_root / "shards/ff/unexpected.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}", encoding="utf-8")

    first = consolidate_formal_measurements(
        definitions,
        artifact_root=artifact_root,
        output_root=tmp_path / "consolidated",
        provenance={"r5_snapshot_id": "a" * 64},
    )
    second = consolidate_formal_measurements(
        definitions,
        artifact_root=artifact_root,
        output_root=tmp_path / "consolidated",
        provenance={"r5_snapshot_id": "a" * 64},
    )

    wt = pd.read_parquet(first["wt_path"])
    probes = pd.read_parquet(first["probe_path"])
    manifest = json.loads(first["manifest_path"].read_text(encoding="utf-8"))
    assert list(wt.columns) == WT_COLUMNS
    assert list(probes.columns) == PROBE_COLUMNS
    assert len(wt) == len(probes) == 2
    assert wt["backbone_condition"].tolist() == ["AFDB", "PDB"]
    assert probes["backbone_condition"].tolist() == ["AFDB", "PDB"]
    assert probes["delta_score_vs_wt"].tolist() == pytest.approx([-1 / 3, -1 / 3])
    assert manifest["counts"] == {
        "canonical_shards": 2,
        "probe_rows": 2,
        "wt_rows": 2,
    }
    assert manifest["provenance"] == {"r5_snapshot_id": "a" * 64}
    assert first["write_status"] == {
        "manifest": "created",
        "probe": "created",
        "wt": "created",
    }
    assert second["write_status"] == {
        "manifest": "reused_identical",
        "probe": "reused_identical",
        "wt": "reused_identical",
    }


def test_consolidation_resumes_after_manifest_last_interruption(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    definitions = (_definition("AFDB"), _definition("PDB"))
    for definition in definitions:
        materialize_fresh_request(
            definition,
            scorer=_FakeScorer(),
            artifact_root=artifact_root,
            execution_provenance={"mode": "fixture"},
        )
    output_root = tmp_path / "consolidated"
    first = consolidate_formal_measurements(
        definitions,
        artifact_root=artifact_root,
        output_root=output_root,
        provenance={"r5_snapshot_id": "a" * 64},
    )
    Path(first["manifest_path"]).unlink()

    resumed = consolidate_formal_measurements(
        definitions,
        artifact_root=artifact_root,
        output_root=output_root,
        provenance={"r5_snapshot_id": "a" * 64},
    )

    assert resumed["write_status"] == {
        "manifest": "created",
        "probe": "reused_identical",
        "wt": "reused_identical",
    }
