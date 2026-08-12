from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.workflows.final_confirmatory_protocol import (
    BLOCKED_INPUT_INTEGRITY,
    FINAL_PROTOCOL_FROZEN,
    FROZEN_PLAN_SCIENTIFIC_FIELDS,
    FinalConfirmatoryProtocolConfig,
    FinalConfirmatoryProtocolError,
    build_final_confirmatory_protocol,
    frozen_plan_scientific_fingerprint,
    frozen_plan_scientific_view,
    materialize_final_confirmatory_protocol,
    validate_final_confirmatory_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SCORING_PLAN_PATH = (
    REPOSITORY_ROOT
    / "experiments/p2_design_baseline/scale1/scale1b_v2/"
    "scale1b_v2_proteinmpnn_scoring_plan.parquet"
)


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.discover(project_root=REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def frozen_inputs(paths: ProjectPaths):
    return validate_final_confirmatory_inputs(
        paths, FinalConfirmatoryProtocolConfig()
    )


@pytest.fixture(scope="module")
def frozen_result(frozen_inputs):
    return build_final_confirmatory_protocol(frozen_inputs)


@pytest.fixture(scope="module")
def frozen_scoring_plan() -> pd.DataFrame:
    return pd.read_parquet(FROZEN_SCORING_PLAN_PATH)


def test_upstream_gate_reconciles_exact_admitted_union(frozen_inputs) -> None:
    assert len(frozen_inputs.original_admitted) == 135
    assert len(frozen_inputs.wave1_admitted) == 49
    assert len(frozen_inputs.wave2_admitted) == 15
    assert frozen_inputs.wave2_summary["final_N_NR"] == 127
    assert frozen_inputs.wave2_manifest["capacity_status"] == (
        "PRIMARY_CONFIRMATORY_CAPACITY_REACHED"
    )
    assert frozen_inputs.proteinmpnn_forward_executions == 0


def test_upstream_sha_drift_is_structured_block(paths: ProjectPaths) -> None:
    config = replace(
        FinalConfirmatoryProtocolConfig(),
        expected_wave2_manifest_sha256="0" * 64,
    )
    with pytest.raises(FinalConfirmatoryProtocolError) as caught:
        validate_final_confirmatory_inputs(paths, config)
    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "upstream_sha256_mismatch"


def test_final_selector_reuses_frozen_representative_semantics(
    frozen_inputs, frozen_result
) -> None:
    original_expected = set(
        frozen_inputs.original_redundancy.loc[
            frozen_inputs.original_redundancy["is_cluster_representative"],
            "pair_id",
        ]
    )
    original_observed = set(
        frozen_result.primary_cohort.loc[
            frozen_result.primary_cohort["source_stratum"].eq(
                "ORIGINAL_213_FRAME"
            ),
            "protein_id",
        ]
    )
    assert original_observed == original_expected
    assert len(frozen_result.primary_cohort) == 127
    assert frozen_result.primary_cohort["cluster_id_30"].nunique() == 127
    assert not frozen_result.primary_cohort["protein_id"].duplicated().any()
    assert len(frozen_result.secondary_pool) == 72
    assert len(frozen_result.primary_cohort) + len(
        frozen_result.secondary_pool
    ) == 199
    assert frozen_result.primary_cohort["stage0_overlap"].sum() == 4
    assert (
        frozen_result.primary_cohort["stage0_overlap"].sum()
        + frozen_result.secondary_pool["stage0_overlap"].sum()
        == 8
    )


def test_common_masks_and_fixed_probes_have_exact_cardinality(
    frozen_result,
) -> None:
    masks = frozen_result.primary_common_masks
    probes = frozen_result.fixed_probes
    assert len(masks) == 27_291
    assert masks["protein_id"].nunique() == 127
    assert masks["common_mask"].all()
    assert len(probes) == 518_529 == 19 * len(masks)
    assert probes.groupby(["protein_id", "canonical_position"]).size().eq(19).all()
    assert not probes.duplicated(
        ["protein_id", "canonical_position", "candidate_aa"]
    ).any()
    assert probes["candidate_aa"].ne(probes["wt_aa"]).all()
    assert set(probes["probe_origin"]) == {
        "REUSED_EXACT_SCIENTIFIC_DEFINITION",
        "GENERATED_FROZEN_V2_CONTRACT",
    }


def test_probe_reuse_is_scientific_identity_only(frozen_result) -> None:
    audit = frozen_result.probe_reuse_audit
    assert len(audit) == 127
    assert audit["probe_reuse_status"].value_counts().to_dict() == {
        "PROBE_REUSE_INCOMPATIBLE": 86,
        "PROBE_REUSE_COMPATIBLE": 41,
    }
    compatible = audit["probe_reuse_status"].eq("PROBE_REUSE_COMPATIBLE")
    assert audit.loc[compatible, "n_reused_probe_definitions"].gt(0).all()
    assert audit.loc[~compatible, "n_reused_probe_definitions"].eq(0).all()
    assert not any("structure" in column for column in audit.columns)


def test_scoring_plan_is_logical_only_and_paired(frozen_result) -> None:
    plan = frozen_result.scoring_plan
    assert len(plan) == 7_620
    assert plan["protein_id"].nunique() == 127
    assert set(plan["structure_condition"]) == {"PDB", "AFDB"}
    assert set(plan["repeat"]) == set(range(30))
    assert plan["seed"].eq(plan["repeat"]).all()
    assert plan.groupby(["protein_id", "repeat"])[
        "explicit_realization_id"
    ].nunique().eq(1).all()
    forbidden = {
        "execution_action",
        "score",
        "P",
        "M",
        "SDFI",
        "regret",
        "top1",
    }
    assert forbidden.isdisjoint(plan.columns)


def test_all_frozen_plan_rows_have_unique_scientific_fingerprints(
    frozen_scoring_plan: pd.DataFrame,
) -> None:
    plan = frozen_scoring_plan
    views = [
        frozen_plan_scientific_view(row)
        for row in plan.to_dict("records")
    ]
    fingerprints = [
        frozen_plan_scientific_fingerprint(row)
        for row in plan.to_dict("records")
    ]
    assert len(views) == len(fingerprints) == 7_620
    assert all(tuple(view) == FROZEN_PLAN_SCIENTIFIC_FIELDS for view in views)
    assert len(set(fingerprints)) == 7_620


def test_frozen_plan_fingerprint_excludes_operational_representation(
    frozen_scoring_plan: pd.DataFrame,
) -> None:
    row = frozen_scoring_plan.iloc[0].to_dict()
    changed = dict(row)
    for field in (
        "logical_shard_id",
        "cluster_id_30",
        "source_stratum",
        "structure_artifact_reference",
        "probe_manifest_binding",
    ):
        changed[field] = f"operational-change::{field}"
    assert frozen_plan_scientific_view(changed) == frozen_plan_scientific_view(row)
    assert frozen_plan_scientific_fingerprint(changed) == (
        frozen_plan_scientific_fingerprint(row)
    )


def test_each_frozen_plan_scientific_field_changes_the_fingerprint(
    frozen_scoring_plan: pd.DataFrame,
) -> None:
    row = frozen_scoring_plan.iloc[0].to_dict()
    baseline = frozen_plan_scientific_fingerprint(row)
    for field in FROZEN_PLAN_SCIENTIFIC_FIELDS:
        changed = dict(row)
        if field in {"repeat", "seed"}:
            changed[field] = int(row[field]) + 10_000
        else:
            changed[field] = f"scientific-change::{field}"
        assert frozen_plan_scientific_fingerprint(changed) != baseline, field


def test_summary_and_manifest_freeze_closure_without_execution(
    frozen_result,
) -> None:
    assert frozen_result.status == FINAL_PROTOCOL_FROZEN
    assert frozen_result.summary["n_unique_admitted_proteins"] == 199
    assert frozen_result.summary["n_primary_clusters"] == 127
    assert frozen_result.summary["n_primary_proteins"] == 127
    assert frozen_result.summary["n_secondary_proteins"] == 72
    assert frozen_result.summary["n_common_mask_positions"] == 27_291
    assert frozen_result.summary["n_fixed_probes"] == 518_529
    assert frozen_result.summary["logical_scoring_shards"] == 7_620
    assert frozen_result.summary["candidate_score_rows"] == 31_111_740
    assert frozen_result.summary["WT_score_rows"] == 7_620
    for key in (
        "FINAL_COHORT_MEMBERSHIP_FROZEN",
        "FINAL_PROBE_SPACE_FROZEN",
        "FINAL_SCORING_PROTOCOL_FROZEN",
        "PROTOCOL_ENGINEERING_CLOSED",
        "ARM_A_DATASET_EXPANSION_CLOSED",
    ):
        assert frozen_result.manifest[key] is True
    assert frozen_result.manifest["PROTEINMPNN_FORWARD_EXECUTIONS"] == 0


def test_materialization_is_immutable_and_manifest_last(
    frozen_inputs, tmp_path: Path
) -> None:
    isolated_paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        runs_root=tmp_path,
    )
    config = replace(
        FinalConfirmatoryProtocolConfig(),
        output_root_ref="runs/scale1b-v2-test",
    )
    isolated_result = build_final_confirmatory_protocol(
        replace(frozen_inputs, output_root_ref=config.output_root_ref)
    )
    first = materialize_final_confirmatory_protocol(
        isolated_paths, config, isolated_result
    )
    second = materialize_final_confirmatory_protocol(
        isolated_paths, config, isolated_result
    )
    assert set(first.values()) == {"created"}
    assert set(second.values()) == {"reused_identical"}
    manifest_path = (
        tmp_path / "scale1b-v2-test" / "scale1b_v2_freeze_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["outputs"].items():
        if name == "freeze_manifest":
            continue
        path = isolated_paths.resolve_logical(record["path"])
        assert path.is_file()
        if path.suffix == ".parquet":
            assert len(pd.read_parquet(path)) == record["rows"]
