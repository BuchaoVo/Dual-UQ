from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.redundancy_diversity import (
    ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED,
    BLOCKED_INPUT_INTEGRITY,
    BLOCKED_REDUNDANCY_BINDING,
    CATH_METADATA_COMPLETE,
    CATH_METADATA_PARTIAL,
    CATH_METADATA_UNAVAILABLE,
    CORE_SCALE_REACHED,
    RedundancyDiversityConfig,
    RedundancyDiversityError,
    assign_admitted_representatives,
    build_redundancy_diversity_census,
    compute_capacity_snapshot,
    materialize_redundancy_diversity_census,
    validate_decision_invariance,
    validate_diversity_independence,
    validate_redundancy_diversity_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = (
    REPOSITORY_ROOT
    / "scripts/dataset/assess_redundancy_diversity.py"
)


@pytest.fixture(scope="module")
def repository_paths() -> ProjectPaths:
    return ProjectPaths.discover(project_root=REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def repository_inputs(repository_paths: ProjectPaths):
    return validate_redundancy_diversity_inputs(repository_paths, RedundancyDiversityConfig())


@pytest.fixture(scope="module")
def repository_result(repository_inputs):
    return build_redundancy_diversity_census(repository_inputs)


def test_input_gate_binds_complete_frame_and_scale1b_v1(
    repository_inputs,
) -> None:
    assert len(repository_inputs.census) == 213
    assert repository_inputs.census["formal_admission_status"].value_counts().to_dict() == {
        "FORMALLY_ADMITTED": 135,
        "PENDING_HUMAN_VARIANT_REVIEW": 51,
        "MAPPING_FAIL": 12,
        "IDENTITY_CONTRACT_FAIL": 9,
        "PROVENANCE_FAIL": 6,
    }
    assert len(repository_inputs.admitted_metadata) == 135
    assert len(repository_inputs.pending_metadata) == 51
    assert repository_inputs.admitted_metadata["sequence_cluster"].str.match(
        r"^30:.+"
    ).all()
    canonical_axis_lengths = repository_inputs.common_masks.groupby(
        "candidate_id"
    )["canonical_position"].size()
    observed_lengths = repository_inputs.admitted_metadata.set_index("pair_id")[
        "sequence_length"
    ]
    assert observed_lengths.astype(int).to_dict() == canonical_axis_lengths.loc[
        observed_lengths.index
    ].astype(int).to_dict()
    artifact_paths = {record["path"] for record in repository_inputs.input_artifacts}
    assert {
        "experiments/p2_design_baseline/scale1/scale1a2/scale1_full_frame_census.parquet",
        "experiments/p2_design_baseline/scale1/scale1b_freeze/scale1b_cohort_freeze_manifest.json",
        "experiments/p2_design_baseline/scale1/scale1b_protocol/scale1b_protocol_freeze_manifest.json",
        "artifacts/dataset/reports/census/candidate_inventory_v1.tsv",
        "data/processed/discovery/discovered_candidates.parquet",
        "src/dual_uq/dataset/redundancy_diversity.py",
    }.issubset(artifact_paths)
    implementation_record = next(
        record
        for record in repository_inputs.input_artifacts
        if record["path"]
        == "src/dual_uq/dataset/redundancy_diversity.py"
    )
    assert implementation_record["sha256"] == sha256_file(
        REPOSITORY_ROOT
        / "src/dual_uq/dataset/redundancy_diversity.py"
    )


def test_input_sha_drift_is_structured_block(
    repository_paths: ProjectPaths,
) -> None:
    config = replace(
        RedundancyDiversityConfig(), expected_scale1a2_census_sha256="0" * 64
    )
    with pytest.raises(RedundancyDiversityError) as exc:
        validate_redundancy_diversity_inputs(repository_paths, config)
    assert exc.value.status == BLOCKED_INPUT_INTEGRITY
    assert exc.value.code == "upstream_sha256_mismatch"


def test_pending_capacity_counts_unique_resolved_clusters_not_proteins() -> None:
    snapshot = compute_capacity_snapshot(
        ["30:a", "30:b"],
        ["30:b", "30:c", "30:c", "30:d"],
    )
    assert snapshot.n_nr_admitted == 2
    assert snapshot.n_pending_proteins == 4
    assert snapshot.n_pending_cluster_resolved == 4
    assert snapshot.n_pending_cluster_unresolved == 0
    assert snapshot.n_unique_pending_resolved_clusters == 3
    assert snapshot.n_pending_clusters_already_represented == 1
    assert snapshot.n_pending_new_clusters == 2
    assert snapshot.n_nr_upper == 4
    assert snapshot.pending_upper_bound_exact is True


def test_unresolved_pending_is_not_imputed_and_blocks_below_core_scale() -> None:
    snapshot = compute_capacity_snapshot(
        [f"30:{index}" for index in range(60)],
        ["30:100", None, pd.NA],
    )
    assert snapshot.n_pending_cluster_resolved == 1
    assert snapshot.n_pending_cluster_unresolved == 2
    assert snapshot.n_pending_new_clusters == 1
    assert snapshot.pending_upper_bound_exact is False
    assert snapshot.capacity_decision == BLOCKED_REDUNDANCY_BINDING


def test_core_scale_short_circuits_pending_exactness_requirement() -> None:
    snapshot = compute_capacity_snapshot(
        [f"30:{index}" for index in range(100)],
        [None],
    )
    assert snapshot.pending_upper_bound_exact is False
    assert snapshot.capacity_decision == CORE_SCALE_REACHED


@pytest.mark.parametrize(
    "diversity_status",
    [
        CATH_METADATA_COMPLETE,
        CATH_METADATA_PARTIAL,
        CATH_METADATA_UNAVAILABLE,
    ],
)
def test_capacity_decision_is_independent_of_optional_diversity(
    diversity_status: str,
) -> None:
    snapshot = compute_capacity_snapshot(
        [f"30:{index}" for index in range(63)],
        [f"30:{index}" for index in range(55, 78)],
    )
    audit = validate_diversity_independence(snapshot, diversity_status)
    assert audit["capacity_decision_pre_diversity"] == (
        ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED
    )
    assert audit["capacity_decision_post_diversity_validation"] == (
        ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED
    )
    assert audit["decision_independent_of_diversity_metadata"] is True


def test_capacity_decision_disagreement_is_invariant_failure() -> None:
    with pytest.raises(RedundancyDiversityError) as exc:
        validate_decision_invariance(
            CORE_SCALE_REACHED,
            ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED,
        )
    assert exc.value.code == "decision_invariance_failure"


def test_frozen_rule_reproduces_scale1b_v1_on_original_subset(
    repository_inputs,
) -> None:
    member_ids = set(repository_inputs.scale1b_panel["pair_id"])
    subset = repository_inputs.admitted_metadata.loc[
        repository_inputs.admitted_metadata["pair_id"].isin(member_ids)
    ].copy()
    assigned = assign_admitted_representatives(subset).set_index("pair_id")
    frozen = repository_inputs.scale1b_panel.set_index("pair_id")
    assert assigned.index.isin(frozen.index).all()
    assert assigned["is_cluster_representative"].to_dict() == frozen[
        "is_primary_representative"
    ].astype(bool).to_dict()


def test_repository_result_has_expected_set_capacity_and_decision(
    repository_result,
) -> None:
    result = repository_result
    assert result.status == ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED
    assert len(result.admitted_census) == 135
    assert len(result.nonredundant_capacity) == 63
    assert len(result.pending_capacity) == 51
    assert len(result.domain_annotations) == 63
    assert result.capacity_snapshot.n_nr_admitted == 63
    assert result.capacity_snapshot.n_unique_pending_resolved_clusters == 23
    assert result.capacity_snapshot.n_pending_clusters_already_represented == 8
    assert result.capacity_snapshot.n_pending_new_clusters == 15
    assert result.capacity_snapshot.n_nr_upper == 78
    assert result.capacity_snapshot.pending_upper_bound_exact is True
    assert result.summary["capacity_decision_pre_diversity"] == result.status
    assert result.summary["capacity_decision_post_diversity_validation"] == result.status
    assert result.summary["decision_independent_of_diversity_metadata"] is True
    assert result.summary["cath_metadata_status"] == CATH_METADATA_UNAVAILABLE
    required_capacity_fields = {
        "n_pending_proteins": 51,
        "n_pending_cluster_resolved": 51,
        "n_pending_cluster_unresolved": 0,
        "n_unique_pending_resolved_clusters": 23,
        "n_pending_clusters_already_represented": 8,
        "n_pending_new_clusters": 15,
        "pending_upper_bound_exact": True,
    }
    for field, expected in required_capacity_fields.items():
        assert result.summary[field] == expected
        assert result.manifest["capacity_statistics"][field] == expected


def test_repository_result_is_order_independent(
    repository_inputs, repository_result
) -> None:
    reversed_inputs = replace(
        repository_inputs,
        admitted_metadata=repository_inputs.admitted_metadata.iloc[::-1].reset_index(
            drop=True
        ),
        pending_metadata=repository_inputs.pending_metadata.iloc[::-1].reset_index(
            drop=True
        ),
    )
    reversed_result = build_redundancy_diversity_census(reversed_inputs)
    pd.testing.assert_frame_equal(
        repository_result.admitted_census,
        reversed_result.admitted_census,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        repository_result.pending_capacity,
        reversed_result.pending_capacity,
        check_exact=True,
    )


def test_outputs_contain_no_scale1_outcome_dependency(repository_result) -> None:
    forbidden = {
        "p",
        "m",
        "sdfi",
        "top1_disagreement",
        "regret",
        "structural_excess",
        "proteinmpnn_score",
        "scale1_outcome",
    }
    for frame in (
        repository_result.admitted_census,
        repository_result.nonredundant_capacity,
        repository_result.pending_capacity,
        repository_result.domain_annotations,
    ):
        assert not forbidden.intersection(column.lower() for column in frame.columns)
    assert repository_result.manifest["NO_OUTCOME_DEPENDENT_SELECTION"] is True
    assert repository_result.manifest["SCALE1_SCORING_NOT_STARTED"] is True


def test_immutable_materialization_is_portable_and_conflict_safe(
    tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = RedundancyDiversityConfig(output_root_ref="reports/scale1a3")
    first = materialize_redundancy_diversity_census(repository_result, paths, config=config)
    second = materialize_redundancy_diversity_census(repository_result, paths, config=config)
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert all(
        not Path(record["path"]).is_absolute()
        for record in first["outputs"].values()
    )
    assert first["outputs"]["admitted_redundancy_census"]["rows"] == 135
    assert first["outputs"]["nonredundant_capacity"]["rows"] == 63
    assert first["outputs"]["pending_cluster_capacity"]["rows"] == 51

    target = tmp_path / "reports/scale1a3/scale1a3_nonredundant_capacity.parquet"
    target.write_bytes(b"conflict")
    with pytest.raises(RedundancyDiversityError) as exc:
        materialize_redundancy_diversity_census(repository_result, paths, config=config)
    assert exc.value.code == "immutable_scale1a3_artifact_conflict"


def test_cli_reports_structured_project_root_failure(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--project-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["scale1a3_status"] == BLOCKED_INPUT_INTEGRITY
    assert payload["failure_code"] == "project_root_unresolved"
