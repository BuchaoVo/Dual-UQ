from __future__ import annotations

import builtins
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset import scale1b_protocol_freeze as protocol_module
from dual_uq.dataset.scale1b_protocol_freeze import (
    BLOCKED_INPUT_INTEGRITY,
    BLOCKED_SCORER_INTEGRITY,
    Scale1BProtocolFreezeConfig,
    Scale1BProtocolFreezeError,
    build_scale1b_fixed_probe_candidates,
    build_scale1b_protocol_freeze,
    build_scale1b_scoring_plan,
    materialize_scale1b_protocol_freeze,
    summarize_scale1b_workload,
    validate_scale1b_protocol_inputs,
    validate_stage0_overlap_regression,
)
from dual_uq.dataset.storage.proteinmpnn import STANDARD_AMINO_ACIDS, sequence_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts/dataset/freeze_scoring_protocol.py"


def test_input_gate_binds_frozen_cohort_masks_and_stage0_contracts() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)

    inputs = validate_scale1b_protocol_inputs(
        paths, Scale1BProtocolFreezeConfig()
    )

    assert len(inputs.scoring_panel) == 52
    assert inputs.scoring_panel["is_primary_representative"].sum() == 46
    assert (~inputs.scoring_panel["is_primary_representative"]).sum() == 6
    assert inputs.scoring_panel["stage0_overlap"].sum() == 8
    assert inputs.scoring_panel["in_new_protein_primary_core"].sum() == 41
    assert len(inputs.stage0_fixed_probes) == 34_010
    assert inputs.stage0_protein_manifest["summary"]["total_mask_positions"] == 1_790
    assert inputs.stage0_scoring_manifest["status"] == "complete"
    assert inputs.stage0_scoring_manifest["scoring_protocol"]["identity"] == (
        "stage0_fixed_sequence_autoregressive_mask_logp_v1"
    )


def test_input_sha_drift_is_a_structured_input_integrity_block() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = replace(
        Scale1BProtocolFreezeConfig(),
        expected_scoring_panel_sha256="0" * 64,
    )

    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        validate_scale1b_protocol_inputs(paths, config)

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "upstream_sha256_mismatch"


def test_scorer_is_verified_statically_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    real_import = builtins.__import__

    def reject_torch(name: str, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("protocol freeze must not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    inputs = validate_scale1b_protocol_inputs(
        paths, Scale1BProtocolFreezeConfig()
    )

    assert inputs.model_identity.implementation_commit == (
        "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
    )
    assert inputs.model_identity.checkpoint_sha256 == (
        "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
    )
    assert inputs.proteinmpnn_forward_executions == 0
    artifacts = {record["path"]: record for record in inputs.input_artifacts}
    assert artifacts[
        "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    ]["sha256"] == inputs.model_identity.checkpoint_sha256
    assert artifacts["third_party/ProteinMPNN"]["kind"] == "git_repository"
    assert artifacts["third_party/ProteinMPNN"]["commit"] == (
        inputs.model_identity.implementation_commit
    )


def test_scorer_sha_drift_is_a_structured_scorer_integrity_block() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = replace(
        Scale1BProtocolFreezeConfig(),
        expected_checkpoint_sha256="0" * 64,
    )

    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        validate_scale1b_protocol_inputs(paths, config)

    assert caught.value.status == BLOCKED_SCORER_INTEGRITY
    assert caught.value.code == "checkpoint_hash_mismatch"


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/raw/pdb/2vb1.cif",
        "experiments/p2_design_baseline/stage0/fixed_probe_scores.parquet",
    ],
)
def test_static_structure_and_stage0_score_payload_drift_is_blocked(
    monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    target = (REPOSITORY_ROOT / relative_path).resolve()
    real_sha256 = protocol_module.sha256_file

    def drift_target(path: Path) -> str:
        if path.resolve() == target:
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(protocol_module, "sha256_file", drift_target)
    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        validate_scale1b_protocol_inputs(paths, Scale1BProtocolFreezeConfig())

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code in {
        "static_backbone_sha256_mismatch",
        "stage0_score_artifact_sha256_mismatch",
    }


@pytest.fixture(scope="module")
def repository_inputs():
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    return validate_scale1b_protocol_inputs(paths, Scale1BProtocolFreezeConfig())


def test_all_eight_stage0_overlap_proteins_are_exactly_reuse_compatible(
    repository_inputs,
) -> None:
    admission = repository_inputs.admission_census.set_index("pair_id")
    stage0_manifest = {
        row["protein_id"]: row
        for row in repository_inputs.stage0_protein_manifest["proteins"]
    }
    assert admission.loc[
        "1fn8_A__P35049", "afdb_file_path_relative"
    ] != stage0_manifest["1fn8_A__P35049"]["afdb_backbone_path"]
    assert admission.loc[
        "1fn8_A__P35049", "afdb_file_sha256"
    ] == stage0_manifest["1fn8_A__P35049"]["afdb_backbone_sha256"]

    regression = validate_stage0_overlap_regression(repository_inputs)

    assert len(regression) == 8
    assert regression["protein_id"].tolist() == repository_inputs.scoring_panel.loc[
        repository_inputs.scoring_panel["stage0_overlap"], "pair_id"
    ].tolist()
    for field in (
        "canonical_sequence_compatible",
        "common_mask_compatible",
        "probe_compatible",
        "score_definition_compatible",
        "scorer_compatible",
        "repeat_set_compatible",
        "realization_semantics_compatible",
        "structural_inputs_compatible",
    ):
        assert regression[field].all()
    assert set(regression["probe_reuse_status"]) == {"PROBE_REUSE_COMPATIBLE"}
    assert set(regression["score_reuse_status"]) == {"SCORE_REUSE_COMPATIBLE"}
    assert set(regression["scoring_source"]) == {"REUSE_FROZEN_STAGE0"}


def test_stage0_mask_drift_blocks_before_full_probe_materialization(
    repository_inputs,
) -> None:
    masks = repository_inputs.common_masks.copy()
    protein_id = str(
        repository_inputs.scoring_panel.loc[
            repository_inputs.scoring_panel["stage0_overlap"], "pair_id"
        ].iloc[0]
    )
    index = masks.index[
        masks["pair_id"].eq(protein_id)
        & masks["common_mask"].fillna(False).astype(bool)
    ][0]
    masks.loc[index, "common_mask"] = False

    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        validate_stage0_overlap_regression(
            replace(repository_inputs, common_masks=masks)
        )

    assert caught.value.status == "SCALE1B_BLOCKED_STAGE0_REGRESSION"
    assert caught.value.code == "stage0_common_mask_mismatch"


def test_stage0_realization_drift_blocks_score_reuse(repository_inputs) -> None:
    manifest = deepcopy(repository_inputs.stage0_scoring_manifest)
    manifest["decoding_realizations"]["records"][0][
        "decoding_realization_sha256"
    ] = "0" * 64

    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        validate_stage0_overlap_regression(
            replace(repository_inputs, stage0_scoring_manifest=manifest)
        )

    assert caught.value.status == "SCALE1B_BLOCKED_STAGE0_REGRESSION"
    assert caught.value.code == "stage0_realization_mismatch"


@pytest.fixture(scope="module")
def repository_probe_space(repository_inputs):
    regression = validate_stage0_overlap_regression(repository_inputs)
    return build_scale1b_fixed_probe_candidates(repository_inputs, regression)


def test_full_probe_space_is_exactly_19_per_frozen_common_mask_position(
    repository_inputs, repository_probe_space
) -> None:
    probes = repository_probe_space
    panel = repository_inputs.scoring_panel
    census = repository_inputs.admission_census.set_index("pair_id")
    expected_positions = int(panel["common_mask_count"].sum())

    assert expected_positions == 11_437
    assert len(probes) == 217_303 == 19 * expected_positions
    assert probes["protein_id"].nunique() == 52
    assert not probes.duplicated(
        ["protein_id", "mutation_position", "mutant_aa"]
    ).any()
    per_position = probes.groupby(["protein_id", "mutation_position"]).size()
    assert per_position.eq(19).all()
    per_protein = probes.groupby("protein_id").size()
    expected_per_protein = (
        panel.set_index("pair_id")["common_mask_count"].astype(int) * 19
    )
    assert per_protein.to_dict() == expected_per_protein.to_dict()
    assert (int(per_protein.min()), int(per_protein.max())) == (1_805, 9_462)
    assert probes["polymer_entity_id"].notna().all()
    assert probes["afdb_model_id"].notna().all()
    observed_models = probes.groupby("protein_id", sort=False)[
        "afdb_model_id"
    ].first()
    expected_models = census.loc[observed_models.index, "selected_model_entity_id"]
    assert observed_models.to_dict() == expected_models.astype(str).to_dict()


def test_probe_identity_order_and_single_mutation_contract_are_deterministic(
    repository_inputs, repository_probe_space
) -> None:
    probes = repository_probe_space
    panel = repository_inputs.scoring_panel.set_index("pair_id")

    assert probes["candidate_index_global"].tolist() == list(
        range(1, len(probes) + 1)
    )
    assert probes["candidate_id"].is_unique
    assert not probes.duplicated(["protein_id", "sequence_hash"]).any()
    assert not {
        "score_sum_logp_mask",
        "score_mean_logp_mask",
        "delta_score",
        "regret",
        "top1_disagreement",
        "P",
        "M",
        "SDFI",
    }.intersection(probes.columns)

    for protein_id, group in probes.groupby("protein_id", sort=False):
        canonical = str(panel.loc[protein_id, "canonical_sequence"])
        assert group["candidate_index_within_protein"].tolist() == list(
            range(1, len(group) + 1)
        )
        assert group["mutation_position"].tolist() == sorted(
            group["mutation_position"].tolist()
        )
        for position, position_rows in group.groupby("mutation_position", sort=False):
            wt_aa = canonical[int(position) - 1]
            assert position_rows["mutant_aa"].tolist() == [
                aa for aa in STANDARD_AMINO_ACIDS if aa != wt_aa
            ]
        sample = group.iloc[[0, len(group) // 2, len(group) - 1]]
        for row in sample.itertuples(index=False):
            differences = [
                index
                for index, (left, right) in enumerate(
                    zip(canonical, row.full_sequence, strict=True), start=1
                )
                if left != right
            ]
            assert differences == [row.mutation_position]
            assert row.sequence_hash == sequence_sha256(row.full_sequence)


@pytest.fixture(scope="module")
def repository_scoring_plan(repository_inputs, repository_probe_space):
    regression = validate_stage0_overlap_regression(repository_inputs)
    return build_scale1b_scoring_plan(
        repository_inputs, repository_probe_space, regression
    )


def test_scoring_plan_has_complete_shards_and_exact_reuse_labels(
    repository_scoring_plan,
) -> None:
    plan = repository_scoring_plan

    assert len(plan) == 3_120
    assert plan["logical_shard_id"].is_unique
    assert plan.groupby("protein_id").size().eq(60).all()
    assert plan.groupby(["protein_id", "backbone_type"]).size().eq(30).all()
    assert set(plan["backbone_type"]) == {"PDB", "AFDB"}
    assert set(plan["repeat_index"]) == set(range(30))
    assert (plan["repeat_index"] == plan["seed"]).all()
    assert plan.loc[
        plan["scoring_source"].eq("REUSE_FROZEN_STAGE0")
    ].shape[0] == 480
    assert plan.loc[
        plan["scoring_source"].eq("NEW_SCALE1B_SCORING")
    ].shape[0] == 2_640
    assert (~plan.loc[
        plan["scoring_source"].eq("REUSE_FROZEN_STAGE0"),
        "execution_required",
    ]).all()
    assert plan.loc[
        plan["scoring_source"].eq("NEW_SCALE1B_SCORING"),
        "execution_required",
    ].all()


def test_one_explicit_realization_is_shared_across_backbones_wt_and_probes(
    repository_scoring_plan,
) -> None:
    plan = repository_scoring_plan
    paired = plan.groupby(["protein_id", "repeat_index"], sort=False)

    assert paired["decoding_realization_sha256"].nunique().eq(1).all()
    assert paired["decoding_realization_algorithm"].nunique().eq(1).all()
    assert paired["candidate_identity_sha256"].nunique().eq(1).all()
    assert paired["candidate_count"].nunique().eq(1).all()
    assert paired["expected_wt_count"].nunique().eq(1).all()
    assert plan["expected_wt_count"].eq(1).all()
    assert plan["realization_applies_to"].eq(
        "WT_AND_ALL_PROBES_PAIRED_ACROSS_PDB_AFDB"
    ).all()


def test_workload_arithmetic_separates_full_panel_from_new_execution(
    repository_inputs, repository_probe_space, repository_scoring_plan
) -> None:
    workload = summarize_scale1b_workload(
        repository_inputs, repository_probe_space, repository_scoring_plan
    )

    assert workload == {
        "protein_count": 52,
        "new_protein_count": 44,
        "total_common_mask_positions": 11_437,
        "new_protein_common_mask_positions": 9_647,
        "total_fixed_probes": 217_303,
        "new_protein_fixed_probes": 183_293,
        "per_protein_probe_min": 1_805,
        "per_protein_probe_max": 9_462,
        "full_candidate_score_rows": 13_038_180,
        "full_wt_rows": 3_120,
        "new_candidate_score_rows": 10_997_580,
        "new_wt_rows": 2_640,
        "total_logical_shards": 3_120,
        "execution_required_logical_shards": 2_640,
        "reuse_frozen_stage0_logical_shards": 480,
    }


@pytest.fixture(scope="module")
def repository_result(
    repository_inputs, repository_probe_space, repository_scoring_plan
):
    regression = validate_stage0_overlap_regression(repository_inputs)
    return build_scale1b_protocol_freeze(
        repository_inputs,
        overlap_regression=regression,
        fixed_probes=repository_probe_space,
        scoring_plan=repository_scoring_plan,
    )


def test_single_structured_result_freezes_science_without_computing_outcomes(
    repository_result,
) -> None:
    assert repository_result.status == "SCALE1B_PROTOCOL_FREEZE_COMPLETE"
    assert repository_result.proteinmpnn_forward_executions == 0
    assert repository_result.probe_manifest["protein_count"] == 52
    assert repository_result.probe_manifest["total_common_mask_positions"] == 11_437
    assert repository_result.probe_manifest["total_fixed_probes"] == 217_303
    assert repository_result.probe_manifest["probes_per_position"] == 19
    assert repository_result.probe_manifest["alphabet"] == "ACDEFGHIKLMNPQRSTVWY"
    assert len(repository_result.probe_manifest["stage0_overlap_regression"]) == 8
    protocol = repository_result.scoring_protocol
    assert protocol["model"]["implementation_commit"] == (
        "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
    )
    assert protocol["model"]["checkpoint_sha256"] == (
        "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
    )
    assert protocol["score_semantics"]["canonical_outputs"] == [
        "score_sum_logp_mask",
        "score_mean_logp_mask",
    ]
    assert protocol["stochasticity"]["seeds"] == list(range(30))
    assert protocol["batch_chunk_semantics"]["frozen_batch_size"] == 128
    assert protocol["future_endpoints"]["computed"] is False
    assert repository_result.manifest["SCALE1B_COHORT_FROZEN"] is True
    assert repository_result.manifest["FIXED_PROBE_SPACE_FROZEN"] is True
    assert repository_result.manifest["SCORING_PROTOCOL_FROZEN"] is True
    assert repository_result.manifest["SCALE1B_SCORING_NOT_STARTED"] is True
    assert repository_result.manifest["STAGE0_SCORE_REUSE_POLICY_FROZEN"] is True
    assert repository_result.manifest["NO_OUTCOME_DEPENDENT_CHANGES"] is True
    assert repository_result.manifest["STAGE0_LOCAL_ANALYSIS_STOP"] is True


def test_materialization_is_immutable_manifest_last_and_portable(
    tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        artifacts_root=tmp_path / "artifacts",
    )
    config = replace(
        Scale1BProtocolFreezeConfig(),
        output_root_ref="artifacts/scale1b_protocol",
    )

    first = materialize_scale1b_protocol_freeze(
        repository_result, paths, config=config
    )
    second = materialize_scale1b_protocol_freeze(
        repository_result, paths, config=config
    )

    assert first["protocol_status"] == "SCALE1B_PROTOCOL_FREEZE_COMPLETE"
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    output_root = tmp_path / "artifacts/scale1b_protocol"
    manifest_path = output_root / "scale1b_protocol_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["fixed_probe_candidates"]["rows"] == 217_303
    assert manifest["outputs"]["scoring_plan"]["rows"] == 3_120
    assert all(not record["path"].startswith(("/home/", "/mnt/")) for record in manifest["upstream_artifacts"])
    assert b"/home/" not in manifest_path.read_bytes()
    assert b"/mnt/" not in manifest_path.read_bytes()
    nonmanifest_mtime = max(
        (output_root / name).stat().st_mtime_ns
        for name in (
            "scale1b_fixed_probe_candidates.parquet",
            "scale1b_probe_manifest.json",
            "scale1b_scoring_plan.parquet",
            "scale1b_scoring_protocol.json",
        )
    )
    assert manifest_path.stat().st_mtime_ns >= nonmanifest_mtime


def test_checkpoint_drift_between_validation_and_materialization_is_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        artifacts_root=tmp_path / "artifacts",
    )
    checkpoint = (
        REPOSITORY_ROOT
        / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    ).resolve()
    real_sha256 = protocol_module.sha256_file

    def drift_checkpoint(path: Path) -> str:
        if path.resolve() == checkpoint:
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(protocol_module, "sha256_file", drift_checkpoint)
    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        materialize_scale1b_protocol_freeze(
            repository_result,
            paths,
            config=replace(
                Scale1BProtocolFreezeConfig(),
                output_root_ref="artifacts/checkpoint-drift",
            ),
        )

    assert caught.value.status == BLOCKED_SCORER_INTEGRITY
    assert caught.value.code == "checkpoint_changed_before_materialization"
    assert not (tmp_path / "artifacts/checkpoint-drift").exists()


def test_implementation_drift_between_validation_and_materialization_is_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        artifacts_root=tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        protocol_module, "implementation_worktree_is_clean", lambda _path: False
    )

    with pytest.raises(Scale1BProtocolFreezeError) as caught:
        materialize_scale1b_protocol_freeze(
            repository_result,
            paths,
            config=replace(
                Scale1BProtocolFreezeConfig(),
                output_root_ref="artifacts/implementation-drift",
            ),
        )

    assert caught.value.status == BLOCKED_SCORER_INTEGRITY
    assert caught.value.code == "implementation_changed_before_materialization"
    assert not (tmp_path / "artifacts/implementation-drift").exists()


def test_cli_reports_project_root_failure_without_model_execution(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--project-root",
            str(tmp_path / "not-a-repository"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["protocol_status"] == "SCALE1B_BLOCKED_INPUT_INTEGRITY"
    assert payload["failure_code"] == "project_root_unresolved"
    assert payload["proteinmpnn_forward_executions"] == 0
