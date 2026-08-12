from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from dual_uq.dataset.services.proteinmpnn_scoring import (
    build_paired_scoring_projection,
)
from dual_uq.inference.formal import (
    HistoricalReuseStatus,
    MaterializationState,
    derive_execution_inventory,
)
from dual_uq.workflows.final_confirmatory_protocol import (
    BLOCKED_REUSE_POLICY,
    FinalConfirmatoryProtocolError,
    HistoricalReusePolicyStatus,
    build_final_confirmatory_projection_resolver,
    load_final_confirmatory_formal_bundle,
    resolve_historical_reuse_policy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPOSITORY_ROOT / "experiments/p2_design_baseline/scale1/scale1b_v2"


def _json(name: str) -> dict:
    return json.loads((V2_ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def formal_bundle():
    return load_final_confirmatory_formal_bundle(REPOSITORY_ROOT)


def test_frozen_protocol_authorizes_reuse_before_candidate_audit() -> None:
    protocol = _json("scale1b_v2_scoring_protocol.json")
    manifest = _json("scale1b_v2_freeze_manifest.json")

    decision = resolve_historical_reuse_policy(protocol, manifest)

    assert decision.status is HistoricalReusePolicyStatus.AUTHORIZED
    assert len(decision.policy_identity) == 64
    assert decision.evidence == (
        "historical_result_reuse_contract",
        "HISTORICAL_SCORE_RESULT_REUSE_RULE_FROZEN",
        "UNEXECUTED_V1_PLAN_NOT_COUNTED_AS_SCORE_RESULT",
    )


def test_ambiguous_reuse_policy_blocks_before_candidate_audit() -> None:
    protocol = deepcopy(_json("scale1b_v2_scoring_protocol.json"))
    manifest = deepcopy(_json("scale1b_v2_freeze_manifest.json"))
    protocol.pop("historical_result_reuse_contract")

    with pytest.raises(FinalConfirmatoryProtocolError) as caught:
        resolve_historical_reuse_policy(protocol, manifest)

    assert caught.value.status == BLOCKED_REUSE_POLICY


def test_real_bundle_has_exact_requests_bindings_and_derived_historical_reuse(
    formal_bundle,
) -> None:
    bundle = formal_bundle

    assert bundle.reuse_policy.status is HistoricalReusePolicyStatus.AUTHORIZED
    assert len(bundle.definitions) == 7_620
    assert len({row.orchestration_id for row in bundle.definitions}) == 7_620
    assert len({row.artifact.relative_path for row in bundle.definitions}) == 7_620
    assert bundle.historical_reuse_candidates_reconstructed == 240
    assert bundle.historical_reuse_accepted == 240
    assert bundle.historical_reuse_rejected == 0
    assert bundle.frozen_request_aggregate_sha256 == (
        "378c15043e5b46e743c799a42e574524189348de474ff052f387de48ebde49f9"
    )
    assert sum(
        row.historical_reuse.status is HistoricalReuseStatus.ACCEPTED
        for row in bundle.definitions
    ) == 240
    assert all(
        row.orchestration_id == row.scientific_fingerprint
        for row in bundle.definitions
    )


def test_real_empty_target_inventory_closes_without_hard_coded_candidate_count(
    tmp_path: Path,
    formal_bundle,
) -> None:
    bundle = formal_bundle
    snapshot = derive_execution_inventory(
        bundle.definitions, artifact_root=tmp_path
    )

    assert snapshot.total_requests == 7_620
    assert snapshot.expected_wt_rows == 7_620
    assert snapshot.expected_probe_rows == 31_111_740
    assert snapshot.state_counts == {
        MaterializationState.HISTORICAL_REUSE_ACCEPTED: 240,
        MaterializationState.FRESH_EXECUTION_REQUIRED: 7_380,
    }
    assert sum(snapshot.state_counts.values()) == 7_620


def test_projection_resolver_preserves_stage0_overlap_coordinates_and_hashes(
    formal_bundle,
) -> None:
    resolver = build_final_confirmatory_projection_resolver(REPOSITORY_ROOT)
    stage0 = json.loads(
        (
            REPOSITORY_ROOT
            / "experiments/p2_design_baseline/stage0/protein_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source_by_id = {
        row["protein_id"]: row for row in stage0["proteins"]
    }
    overlap_ids = set(source_by_id).intersection(
        definition.request.protein_id for definition in formal_bundle.definitions
    )
    assert len(overlap_ids) == 4

    definitions = {
        (
            definition.request.protein_id,
            definition.request.condition.condition_id,
        ): definition
        for definition in formal_bundle.definitions
        if definition.request.protein_id in overlap_ids
        and definition.request.repeat_index == 0
    }
    assert len(definitions) == 8
    for protein_id in sorted(overlap_ids):
        historical = build_paired_scoring_projection(
            source_by_id[protein_id], REPOSITORY_ROOT
        )
        for condition_id, expected in (
            ("PDB", historical.pdb),
            ("AFDB", historical.afdb),
        ):
            definition = definitions[(protein_id, condition_id)]
            observed = resolver(definition.request)
            assert observed.structure_sha256 == (
                definition.request.condition.structure_sha256
            )
            assert observed.uniprot_positions == expected.uniprot_positions
            assert observed.wt_sequence_projection == (
                expected.wt_sequence_projection
            )
            assert np.array_equal(observed.coordinates, expected.coordinates)
