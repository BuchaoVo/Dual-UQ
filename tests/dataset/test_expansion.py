from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.expansion import (
    ADMITTED_COVERED_CLUSTER,
    FAILURE_ONLY_RESCUE,
    MIXED_UNADMITTED_RESCUE,
    NEW_EXTERNAL_CLUSTER,
    PENDING_ONLY_RESCUE,
    ExpansionError,
    build_expansion_design,
    classify_target_cluster,
    discover_remaining_query_candidates,
    estimate_cluster_yield,
    fetch_prospective_candidate_metadata,
    load_expansion_inputs,
    materialize_expansion_design,
    plan_capacity_scenarios,
    plan_expansion_waves,
    rank_prospective_candidates,
    reconstruct_cluster_state,
    select_primary_and_reserves,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "a", "sequence_cluster": "30:1", "formal_admission_status": "FORMALLY_ADMITTED"},
            {"pair_id": "b", "sequence_cluster": "30:1", "formal_admission_status": "PENDING_HUMAN_VARIANT_REVIEW"},
            {"pair_id": "c", "sequence_cluster": "30:2", "formal_admission_status": "PENDING_HUMAN_VARIANT_REVIEW"},
            {"pair_id": "d", "sequence_cluster": "30:3", "formal_admission_status": "MAPPING_FAIL"},
            {"pair_id": "e", "sequence_cluster": "30:4", "formal_admission_status": "PENDING_HUMAN_VARIANT_REVIEW"},
            {"pair_id": "f", "sequence_cluster": "30:4", "formal_admission_status": "PROVENANCE_FAIL"},
        ]
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "2bbb_A__P00002",
                "pair_id": "2bbb_A__P00002",
                "polymer_entity_id": "2BBB_1",
                "canonical_accession": "P00002",
                "pdb_id": "2bbb",
                "pdb_chain": "A",
                "sequence_cluster": "30:9",
                "target_class": NEW_EXTERNAL_CLUSTER,
                "identity_resolved": True,
                "identifier_unambiguous": True,
                "afdb_exact_record_resolved": True,
                "canonical_sequence_available": True,
                "supported_length": True,
                "auth_chain_count": 1,
                "resolution": 2.0,
            },
            {
                "candidate_id": "1aaa_A__P00001",
                "pair_id": "1aaa_A__P00001",
                "polymer_entity_id": "1AAA_1",
                "canonical_accession": "P00001",
                "pdb_id": "1aaa",
                "pdb_chain": "A",
                "sequence_cluster": "30:9",
                "target_class": NEW_EXTERNAL_CLUSTER,
                "identity_resolved": True,
                "identifier_unambiguous": True,
                "afdb_exact_record_resolved": True,
                "canonical_sequence_available": True,
                "supported_length": True,
                "auth_chain_count": 1,
                "resolution": 1.5,
            },
            {
                "candidate_id": "3ccc_A__P00003",
                "pair_id": "3ccc_A__P00003",
                "polymer_entity_id": "3CCC_1",
                "canonical_accession": "P00003",
                "pdb_id": "3ccc",
                "pdb_chain": "A",
                "sequence_cluster": "30:10",
                "target_class": FAILURE_ONLY_RESCUE,
                "identity_resolved": True,
                "identifier_unambiguous": True,
                "afdb_exact_record_resolved": True,
                "canonical_sequence_available": True,
                "supported_length": True,
                "auth_chain_count": 1,
                "resolution": 2.5,
            },
        ]
    )


def test_reconstructs_disjoint_original_cluster_states() -> None:
    state = reconstruct_cluster_state(_frame()).set_index("sequence_cluster")

    assert state.loc["30:1", "cluster_state"] == "ADMITTED_COVERED"
    assert state.loc["30:2", "cluster_state"] == "PENDING_ONLY"
    assert state.loc["30:3", "cluster_state"] == "FAILURE_ONLY"
    assert state.loc["30:4", "cluster_state"] == "MIXED_UNADMITTED"
    assert bool(state.loc["30:4", "contains_pending_no_admitted"])


@pytest.mark.parametrize(
    ("cluster_id", "expected"),
    [
        ("30:1", ADMITTED_COVERED_CLUSTER),
        ("30:2", PENDING_ONLY_RESCUE),
        ("30:3", FAILURE_ONLY_RESCUE),
        ("30:4", MIXED_UNADMITTED_RESCUE),
        ("30:999", NEW_EXTERNAL_CLUSTER),
    ],
)
def test_classifies_target_clusters_relative_to_original_frame(
    cluster_id: str, expected: str
) -> None:
    assert classify_target_cluster(cluster_id, reconstruct_cluster_state(_frame())) == expected


def test_ranking_is_input_order_independent_and_rejects_outcomes() -> None:
    candidates = _candidates()
    ranked = rank_prospective_candidates(candidates)
    shuffled = rank_prospective_candidates(
        candidates.sample(frac=1.0, random_state=17).reset_index(drop=True)
    )

    assert ranked["pair_id"].tolist() == shuffled["pair_id"].tolist()
    assert ranked.loc[ranked["sequence_cluster"].eq("30:9"), "pair_id"].tolist() == [
        "1aaa_A__P00001",
        "2bbb_A__P00002",
    ]

    with pytest.raises(ExpansionError, match="outcome"):
        rank_prospective_candidates(candidates.assign(common_mask_count=100))


def test_one_cluster_first_assigns_primary_and_reserves() -> None:
    selected = select_primary_and_reserves(_candidates())
    roles = selected.set_index("pair_id")["selection_role"].to_dict()

    assert roles == {
        "1aaa_A__P00001": "PRIMARY_CANDIDATE",
        "2bbb_A__P00002": "RESERVE_CANDIDATE",
        "3ccc_A__P00003": "PRIMARY_CANDIDATE",
    }
    assert selected.loc[selected["selection_role"].eq("PRIMARY_CANDIDATE"), "sequence_cluster"].nunique() == 2


def test_historical_yield_selects_primary_before_joining_outcome() -> None:
    candidates = _candidates().drop(columns=["target_class"])
    outcomes = pd.DataFrame(
        {
            "pair_id": ["1aaa_A__P00001", "2bbb_A__P00002", "3ccc_A__P00003"],
            "formal_admission_status": ["MAPPING_FAIL", "FORMALLY_ADMITTED", "FORMALLY_ADMITTED"],
        }
    )
    evidence = estimate_cluster_yield(candidates, outcomes)

    assert evidence["retrospective_primary_count"] == 2
    assert evidence["retrospective_primary_admitted_count"] == 1
    assert evidence["primary_candidate_admission_fraction"] == 0.5


def test_capacity_scenarios_and_wave_plan_use_unique_cluster_capacity() -> None:
    evidence = {
        "cluster_level_admission_success_fraction": 0.5,
        "primary_candidate_admission_fraction": 0.4,
        "candidate_level_admission_fraction": 0.8,
    }
    scenarios = plan_capacity_scenarios(
        starting_capacity=63,
        targets=(100, 120, 150),
        evidence=evidence,
        actionable_cluster_count=200,
    )

    assert len(scenarios) == 9
    assert scenarios.groupby("desired_capacity")["required_new_admitted_clusters"].first().to_dict() == {
        100: 37,
        120: 57,
        150: 87,
    }
    primary = scenarios.loc[scenarios["desired_capacity"].eq(120)].set_index("planning_scenario")
    assert primary.loc["CONSERVATIVE_REALIZED_CLUSTER_YIELD", "estimated_primary_candidate_requirement"] == math.ceil(57 / 0.5)
    assert primary.loc["CLUSTER_AWARE_PRIMARY_YIELD", "estimated_primary_candidate_requirement"] == math.ceil(57 / 0.4)
    assert primary.loc["OPTIMISTIC_CANDIDATE_YIELD", "estimated_primary_candidate_requirement"] == math.ceil(57 / 0.8)

    wave = plan_expansion_waves(scenarios, primary_target=120, actionable_cluster_count=200)
    assert wave["recommended_wave1_primary_n"] == math.ceil(57 / 0.4)
    assert wave["wave2_outcome_independent"] is True


def test_admitted_covered_candidates_never_become_primary() -> None:
    candidates = _candidates()
    covered = candidates.iloc[[0]].assign(
        sequence_cluster="30:1", target_class=ADMITTED_COVERED_CLUSTER
    )
    selected = select_primary_and_reserves(pd.concat([candidates, covered], ignore_index=True))

    assert not selected.loc[
        selected["target_class"].eq(ADMITTED_COVERED_CLUSTER), "primary_eligible"
    ].any()


def test_repository_inputs_reconstruct_frozen_state_and_remaining_query() -> None:
    paths = ProjectPaths.discover(anchor=Path(__file__))
    inputs = load_expansion_inputs(paths)

    assert len(inputs.original_candidates) == 213
    assert inputs.original_candidates["pair_id"].nunique() == 213
    assert len(inputs.query_identifiers) == 500
    assert len(inputs.remaining_query_identifiers) == 280
    assert len(inputs.cluster_state) == 87
    assert int(inputs.cluster_state["contains_admitted"].sum()) == 63
    assert int(inputs.cluster_state["contains_pending_no_admitted"].sum()) == 15


def test_historical_a3_implementation_binding_is_retained_after_move() -> None:
    paths = ProjectPaths.discover(anchor=Path(__file__))
    inputs = load_expansion_inputs(paths)

    record = next(
        item
        for item in inputs.input_artifacts
        if item.get("label") == "effective Scale-1A3 scientific implementation"
    )
    assert record["path"] == "src/dual_uq/dataset/scale1a3_redundancy_diversity.py"


def test_metadata_discovery_is_record_isolated_and_exact_identity_bound() -> None:
    def provider(identifier: str) -> dict[str, object]:
        if identifier == "FAIL_1":
            raise RuntimeError("transport unavailable")
        accession = "P00001" if identifier == "GOOD_1" else "P00002"
        returned = accession if identifier == "GOOD_1" else f"{accession}-2"
        return {
            "polymer_entity_id": identifier,
            "pdb_id": identifier[:4].lower(),
            "entity_id": "1",
            "chain_id": "A",
            "auth_asym_ids": "A",
            "uniprot_id": accession,
            "all_uniprot_ids": accession,
            "length": 150,
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution": 2.0,
            "organism": "Test organism",
            "taxonomy_id": 1,
            "sequence_cluster": f"30:{identifier}",
            "afdb_model_entity_id": f"AF-{accession}-F1",
            "afdb_returned_accession": returned,
            "afdb_exact_record_count": int(returned == accession),
            "afdb_record_count": 1,
            "afdb_sequence_present": True,
        }

    eligible, audit = discover_remaining_query_candidates(
        ("GOOD_1", "MISMATCH_1", "FAIL_1"), provider
    )

    assert eligible["polymer_entity_id"].tolist() == ["GOOD_1"]
    assert audit["attempted_count"] == 3
    assert audit["eligible_count"] == 1
    assert audit["failed_count"] == 2
    assert audit["failure_code_counts"] == {
        "exact_afdb_identity_unresolved": 1,
        "metadata_provider_failure": 1,
    }


def _external_candidates(template: pd.Series, count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        row = template.to_dict()
        row.update(
            {
                "candidate_id": f"x{index:03d}_A__U{index:05d}",
                "pair_id": f"x{index:03d}_A__U{index:05d}",
                "polymer_entity_id": f"X{index:03d}_1",
                "canonical_accession": f"U{index:05d}",
                "pdb_id": f"x{index:03d}",
                "sequence_cluster": f"30:EXT{index:05d}",
                "candidate_origin": "EXPANSION_SAME_QUERY_UNPROBED",
                "current_frame_overlap": False,
                "resolution": 1.0 + index / 1000,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_builds_one_canonical_result_and_immutable_outputs(tmp_path: Path) -> None:
    repository_paths = ProjectPaths.discover(anchor=Path(__file__))
    inputs = load_expansion_inputs(repository_paths)
    external = _external_candidates(inputs.original_candidates.iloc[0], 100)
    audit = {
        "attempted_count": 280,
        "eligible_count": 100,
        "failed_count": 180,
        "failure_code_counts": {"metadata_provider_failure": 180},
        "failures": [],
    }
    result = build_expansion_design(inputs, external, audit)

    assert result.status == "SCALE1_EXPANSION_DESIGN_READY"
    assert len(result.cluster_state) == 87
    assert len(result.expansion_universe) == 313
    assert result.target_clusters["sequence_cluster"].is_unique
    assert len(result.planning_scenarios) == 9
    assert result.summary["current_capacity"]["admitted_30pct_clusters"] == 63
    assert result.summary["interpretation"]["combined_prevalence_estimation_forbidden"] is True

    paths = replace(repository_paths, runs_root=tmp_path / "runs")
    config = replace(
        inputs.config,
        output_root_ref="runs/test-scale1-expansion",
    )
    first = materialize_expansion_design(result, paths, config=config)
    second = materialize_expansion_design(result, paths, config=config)

    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert set(first["outputs"]) == {
        "cluster_state",
        "expansion_universe",
        "cluster_targets",
        "planning_scenarios",
        "design_summary",
        "manifest",
    }
    manifest_path = paths.resolve_logical(
        "runs/test-scale1-expansion/scale1_expansion_design_manifest.json"
    )
    manifest = __import__("json").loads(manifest_path.read_text())
    assert manifest["ORIGINAL_213_FRAME_IMMUTABLE"] is True
    assert manifest["NO_COORDINATE_ACQUISITION"] is True
    assert manifest["outputs"]["expansion_universe"]["rows"] == 313


def test_production_provider_partitions_exact_afdb_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "polymer_entity_id": "1ABC_1",
        "pdb_id": "1abc",
        "entity_id": "1",
        "chain_id": "A",
        "auth_asym_ids": "A",
        "uniprot_id": "P12345",
        "all_uniprot_ids": "P12345",
        "length": 150,
        "experimental_method": "X-RAY DIFFRACTION",
        "resolution": 1.5,
        "organism": "Example",
        "taxonomy_id": 1,
        "sequence_cluster": "30:42",
    }
    records = [
        {
            "uniprotAccession": "P12345-2",
            "modelEntityId": "AF-P12345-2-F1",
            "uniprotSequence": "AAA",
        },
        {
            "uniprotAccession": "P12345",
            "modelEntityId": "AF-P12345-F1",
            "uniprotSequence": "A" * 150,
            "globalMetricValue": 90.0,
        },
        {
            "uniprotAccession": "P12345",
            "modelEntityId": "AF-P12345-F2",
            "uniprotSequence": "A" * 150,
            "globalMetricValue": 88.0,
        },
    ]
    monkeypatch.setattr(
        "dual_uq.dataset.expansion.fetch_candidate_metadata",
        lambda identifier, require_single_uniprot: dict(base),
    )
    monkeypatch.setattr(
        "dual_uq.dataset.expansion.get_afdb_prediction_records",
        lambda accession: records,
    )

    result = fetch_prospective_candidate_metadata("1ABC_1")

    assert result["afdb_returned_accession"] == "P12345"
    assert result["afdb_exact_record_count"] == 2
    assert result["afdb_record_count"] == 3
    assert result["afdb_model_entity_id"] is None
    assert result["afdb_exact_model_ids"] == "AF-P12345-F1;AF-P12345-F2"
    assert result["afdb_fragment_resolution_deferred"] is True
    assert result["afdb_sequence_present"] is True


def test_discovery_accepts_multiple_exact_accession_fragments_without_selecting() -> None:
    def provider(identifier: str) -> dict[str, object]:
        return {
            "polymer_entity_id": identifier,
            "pdb_id": "1abc",
            "entity_id": "1",
            "chain_id": "A",
            "auth_asym_ids": "A",
            "uniprot_id": "P0DTD1",
            "all_uniprot_ids": "P0DTD1",
            "length": 400,
            "experimental_method": "ELECTRON MICROSCOPY",
            "resolution": 2.5,
            "organism": "Example",
            "taxonomy_id": 1,
            "sequence_cluster": "30:999",
            "afdb_model_entity_id": None,
            "afdb_exact_model_ids": "AF-P0DTD1-F1;AF-P0DTD1-F2",
            "afdb_returned_accession": "P0DTD1",
            "afdb_exact_record_count": 2,
            "afdb_record_count": 3,
            "afdb_sequence_present": True,
            "afdb_fragment_resolution_deferred": True,
        }

    eligible, audit = discover_remaining_query_candidates(("1ABC_1",), provider)

    assert audit["eligible_count"] == 1
    assert eligible.loc[0, "afdb_exact_record_count"] == 2
    assert pd.isna(eligible.loc[0, "afdb_model_entity_id"])
    assert bool(eligible.loc[0, "afdb_fragment_resolution_deferred"])
