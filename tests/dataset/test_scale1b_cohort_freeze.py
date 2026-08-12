from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.scale1b_cohort_freeze import (
    BLOCKED_DIVERSITY_CAPACITY,
    BLOCKED_INPUT_INTEGRITY,
    BLOCKED_REDUNDANCY_DEFINITION,
    FREEZE_COMPLETE,
    Scale1BCohortFreezeConfig,
    Scale1BCohortFreezeError,
    assign_redundancy_roles,
    build_scale1b_cohort_freeze,
    materialize_scale1b_cohort_freeze,
    validate_scale1b_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts/dataset/freeze_cohort.py"


@pytest.fixture(scope="module")
def repository_inputs():
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    return validate_scale1b_inputs(paths, Scale1BCohortFreezeConfig())


def test_input_gate_binds_frozen_admission_planning_and_cluster_provenance(
    repository_inputs,
) -> None:
    assert len(repository_inputs.admitted) == 52
    assert len(repository_inputs.metadata) == 52
    assert repository_inputs.metadata["sequence_cluster"].notna().all()
    assert repository_inputs.metadata["sequence_cluster"].str.startswith("30:").all()
    assert repository_inputs.metadata["redundancy_metadata_resolved"].all()
    assert repository_inputs.planning_summary["recommended_mechanical_rule"] == (
        "NO_ADDITIONAL_COMMON_MASK_FEASIBILITY_FILTER_REQUIRED"
    )
    assert repository_inputs.planning_summary["additional_admitted_needed"] == 0
    assert repository_inputs.planning_summary["diversity_capacity"] == (
        "DIVERSITY_CAPACITY_NOT_YET_ASSESSED"
    )
    artifact_paths = {record["path"] for record in repository_inputs.input_artifacts}
    assert {
        "artifacts/dataset/reports/census/candidate_inventory_v1.tsv",
        "data/processed/discovery/discovered_candidates.parquet",
        "src/dual_uq/rcsb_discovery.py",
    }.issubset(artifact_paths)


def test_input_gate_structures_sha_drift_as_input_integrity_block() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = replace(
        Scale1BCohortFreezeConfig(),
        expected_planning_manifest_sha256="0" * 64,
    )
    with pytest.raises(Scale1BCohortFreezeError) as exc:
        validate_scale1b_inputs(paths, config)
    assert exc.value.status == BLOCKED_INPUT_INTEGRITY
    assert exc.value.code == "upstream_sha256_mismatch"


def test_missing_canonical_cluster_assignment_blocks_without_reclustering(
    repository_inputs,
) -> None:
    metadata = repository_inputs.metadata.copy()
    metadata.loc[metadata.index[0], "sequence_cluster"] = None

    with pytest.raises(Scale1BCohortFreezeError) as exc:
        assign_redundancy_roles(metadata)
    assert exc.value.status == BLOCKED_REDUNDANCY_DEFINITION
    assert exc.value.code == "missing_frozen_redundancy_assignment"

    invalid = repository_inputs.metadata.copy()
    invalid.loc[invalid.index[0], "sequence_cluster"] = "90:unfrozen"
    with pytest.raises(Scale1BCohortFreezeError) as invalid_exc:
        assign_redundancy_roles(invalid)
    assert invalid_exc.value.status == BLOCKED_REDUNDANCY_DEFINITION
    assert invalid_exc.value.code == "invalid_frozen_redundancy_assignment"


def test_all_consumed_canonical_sequences_are_hash_bound(repository_inputs) -> None:
    metadata = repository_inputs.metadata
    assert metadata["canonical_sequence"].str.len().eq(
        metadata["canonical_sequence_length"]
    ).all()
    source_paths = set(metadata["canonical_sequence_source_relative"])
    artifact_paths = {record["path"] for record in repository_inputs.input_artifacts}
    assert source_paths.issubset(artifact_paths)


def test_representative_priority_is_lexicographic_and_stage0_neutral() -> None:
    frame = pd.DataFrame(
        [
            {
                "pair_id": "z_pair",
                "canonical_accession": "P00002",
                "pdb_id": "2bbb",
                "pdb_chain": "A",
                "sequence_cluster": "30:1",
                "redundancy_metadata_resolved": False,
                "common_mask_count": 500,
                "common_mask_fraction": 1.0,
                "sampling_frame_index": 1,
                "stage0_overlap": False,
            },
            {
                "pair_id": "b_pair",
                "canonical_accession": "P00001",
                "pdb_id": "1bbb",
                "pdb_chain": "A",
                "sequence_cluster": "30:1",
                "redundancy_metadata_resolved": True,
                "common_mask_count": 100,
                "common_mask_fraction": 0.8,
                "sampling_frame_index": 4,
                "stage0_overlap": False,
            },
            {
                "pair_id": "a_pair",
                "canonical_accession": "P00001",
                "pdb_id": "1aaa",
                "pdb_chain": "A",
                "sequence_cluster": "30:1",
                "redundancy_metadata_resolved": True,
                "common_mask_count": 100,
                "common_mask_fraction": 0.8,
                "sampling_frame_index": 4,
                "stage0_overlap": True,
            },
            {
                "pair_id": "c_pair",
                "canonical_accession": "P00003",
                "pdb_id": "3aaa",
                "pdb_chain": "A",
                "sequence_cluster": "30:2",
                "redundancy_metadata_resolved": True,
                "common_mask_count": 90,
                "common_mask_fraction": 1.0,
                "sampling_frame_index": 8,
                "stage0_overlap": False,
            },
        ]
    )

    assigned = assign_redundancy_roles(frame)
    representatives = assigned.loc[assigned["is_primary_representative"]]
    assert representatives["pair_id"].tolist() == ["a_pair", "c_pair"]
    assert set(assigned["cohort_role"]) == {
        "PRIMARY_CORE",
        "SECONDARY_RELATED_REPLICATION",
    }

    swapped = frame.copy()
    swapped["stage0_overlap"] = ~swapped["stage0_overlap"]
    swapped_representatives = assign_redundancy_roles(swapped).loc[
        lambda table: table["is_primary_representative"], "pair_id"
    ]
    assert swapped_representatives.tolist() == ["a_pair", "c_pair"]


@pytest.fixture(scope="module")
def repository_result(repository_inputs):
    return build_scale1b_cohort_freeze(repository_inputs)


def test_real_freeze_assigns_all_52_into_46_preoutcome_clusters(
    repository_result,
) -> None:
    result = repository_result
    assert result.status == FREEZE_COMPLETE
    assert len(result.scoring_panel) == 52
    assert len(result.redundancy_clusters) == 52
    assert len(result.primary_core) == 46
    assert result.scoring_panel["redundancy_cluster_id"].nunique() == 46
    assert result.scoring_panel["is_primary_representative"].sum() == 46
    assert result.scoring_panel["cohort_role"].value_counts().to_dict() == {
        "PRIMARY_CORE": 46,
        "SECONDARY_RELATED_REPLICATION": 6,
    }
    assert result.summary["cluster_statistics"] == {
        "total_clusters": 46,
        "singleton_clusters": 40,
        "multi_member_clusters": 6,
        "largest_cluster_size": 2,
        "cluster_size_distribution": {"1": 40, "2": 6},
    }
    assert result.summary["capacity_status"] == "PRIMARY_CORE_CAPACITY_ADEQUATE"
    assert result.summary["capacity_checks"] == {
        "ge_16": True,
        "ge_24": True,
        "ge_30": True,
        "ge_40": True,
    }


def test_representatives_are_exact_and_stage0_membership_does_not_override_rule(
    repository_result,
) -> None:
    panel = repository_result.scoring_panel.set_index("redundancy_cluster_id")
    representatives = repository_result.primary_core.set_index(
        "redundancy_cluster_id"
    )["pair_id"]
    assert representatives.loc["30:1944"] == "1nwz_A__P16113"
    assert representatives.loc["30:228"] == "6s2s_A__P02689"
    assert representatives.loc["30:2732"] == "1x8p_A__Q94734"
    assert representatives.loc["30:3653"] == "3w5h_A__P83686"
    assert representatives.loc["30:6115"] == "1pjx_A__Q7SIG4"
    assert representatives.loc["30:71"] == "4i8h_A__P00760"
    assert panel.loc["30:1944", "stage0_overlap"].sum() == 1
    assert panel.loc["30:1944", "is_primary_representative"].sum() == 1


def test_stage0_and_future_analysis_subsets_are_frozen_without_outcome_fields(
    repository_result,
) -> None:
    panel = repository_result.scoring_panel
    assert panel["stage0_overlap"].sum() == 8
    assert repository_result.primary_core["stage0_overlap"].sum() == 5
    assert panel["in_full_scoring_panel"].all()
    assert panel["in_primary_nonredundant_core"].sum() == 46
    assert panel["in_new_protein_primary_core"].sum() == 41
    assert repository_result.summary["stage0_overlap_counts"] == {
        "full_scoring_panel": 8,
        "primary_core": 5,
        "secondary_replication": 3,
        "new_protein_primary_core": 41,
    }
    forbidden = {
        "p",
        "m",
        "sdfi",
        "top1_disagreement",
        "regret",
        "structural_excess",
        "scale1_outcome",
    }
    assert not forbidden.intersection({column.lower() for column in panel.columns})
    assert repository_result.manifest["NO_OUTCOME_DEPENDENT_SELECTION"] is True
    assert repository_result.manifest["SCALE1_OUTCOMES_NOT_OBSERVED"] is True


def test_freeze_is_input_order_independent(repository_inputs, repository_result) -> None:
    reversed_inputs = replace(
        repository_inputs,
        metadata=repository_inputs.metadata.iloc[::-1].reset_index(drop=True),
    )
    reversed_result = build_scale1b_cohort_freeze(reversed_inputs)
    columns = [
        "pair_id",
        "redundancy_cluster_id",
        "is_primary_representative",
        "cohort_role",
        "analysis_origin",
    ]
    pd.testing.assert_frame_equal(
        repository_result.scoring_panel[columns],
        reversed_result.scoring_panel[columns],
        check_exact=True,
    )


def test_capacity_below_16_is_blocked_without_changing_cluster_rule(
    repository_inputs,
) -> None:
    metadata = repository_inputs.metadata.copy()
    metadata["sequence_cluster"] = [f"30:{index % 15}" for index in range(52)]
    constrained = replace(repository_inputs, metadata=metadata)

    result = build_scale1b_cohort_freeze(constrained)
    assert result.status == BLOCKED_DIVERSITY_CAPACITY
    assert len(result.primary_core) == 15
    assert result.summary["capacity_status"] == "PRIMARY_CORE_CAPACITY_INSUFFICIENT"


def test_immutable_materialization_has_portable_manifest_and_conflict_gate(
    tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = Scale1BCohortFreezeConfig(output_root_ref="reports/scale1b-freeze")
    first = materialize_scale1b_cohort_freeze(repository_result, paths, config=config)
    second = materialize_scale1b_cohort_freeze(repository_result, paths, config=config)

    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert all(not Path(record["path"]).is_absolute() for record in first["outputs"].values())
    assert first["outputs"]["scoring_panel"]["rows"] == 52
    assert first["outputs"]["primary_core"]["rows"] == 46
    assert first["outputs"]["redundancy_clusters"]["rows"] == 52

    target = tmp_path / "reports/scale1b-freeze/scale1b_primary_core.parquet"
    target.write_bytes(b"conflict")
    with pytest.raises(Scale1BCohortFreezeError) as exc:
        materialize_scale1b_cohort_freeze(repository_result, paths, config=config)
    assert exc.value.code == "immutable_cohort_freeze_artifact_conflict"


def test_cli_reports_structured_input_block(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--project-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["freeze_status"] == BLOCKED_INPUT_INTEGRITY
    assert payload["failure_code"] == "project_root_unresolved"
