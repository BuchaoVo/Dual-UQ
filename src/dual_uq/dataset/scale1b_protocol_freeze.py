"""Offline Scale-1B fixed-probe and scoring-protocol freeze."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from dual_uq.core.hashing import sha256_canonical, sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.fixed_probe_scoring import (
    SCORING_PROTOCOL_VERSION,
    FixedProbeScoringError,
    VerifiedModelIdentity,
    load_stage0_scoring_config,
    verify_authorized_model,
)
from dual_uq.dataset.fixed_probes import build_fixed_probe_candidates
from dual_uq.dataset.services.proteinmpnn_scoring import (
    DECODING_REALIZATION_ALGORITHM,
    ProteinMPNNScoringError,
    implementation_worktree_is_clean,
    make_decoding_realization,
)

BLOCKED_INPUT_INTEGRITY = "SCALE1B_BLOCKED_INPUT_INTEGRITY"
BLOCKED_SCORER_INTEGRITY = "SCALE1B_BLOCKED_SCORER_INTEGRITY"
BLOCKED_STAGE0_REGRESSION = "SCALE1B_BLOCKED_STAGE0_REGRESSION"
PROTOCOL_FREEZE_COMPLETE = "SCALE1B_PROTOCOL_FREEZE_COMPLETE"


class Scale1BProtocolFreezeError(ValueError):
    """A structured protocol-freeze blocker."""

    def __init__(
        self,
        status: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class Scale1BProtocolFreezeConfig:
    scoring_panel_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_scoring_panel.parquet"
    )
    primary_core_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_primary_core.parquet"
    )
    redundancy_clusters_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_redundancy_clusters.parquet"
    )
    cohort_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_summary.json"
    )
    cohort_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_freeze_manifest.json"
    )
    admission_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_census.parquet"
    )
    common_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_common_masks.parquet"
    )
    stage0_config_ref: str = "configs/experiments/design_baseline/stage0.yaml"
    stage0_protein_manifest_ref: str = (
        "experiments/p2_design_baseline/stage0/protein_manifest.json"
    )
    stage0_fixed_probes_ref: str = (
        "experiments/p2_design_baseline/stage0/fixed_probe_candidates.parquet"
    )
    stage0_scoring_manifest_ref: str = (
        "experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json"
    )
    output_root_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol"
    )

    expected_scoring_panel_sha256: str = (
        "6fdc074d8dd56bb502592b51830efdcafcb34dcb2b530e8fe26e95d657684072"
    )
    expected_primary_core_sha256: str = (
        "4adbb8ccd07036df9243d26fdaf10b1d5421a1a345d441fced00f3937b7b89a7"
    )
    expected_redundancy_clusters_sha256: str = (
        "1e7ff6990a20eae5f373e36f1648c903c355cdb8e31adf5ad2a3fb2a27bac944"
    )
    expected_cohort_summary_sha256: str = (
        "064f6b04e39244fe8d85d0b0b8e67b81062e654d0007eb1903503680bf727965"
    )
    expected_cohort_manifest_sha256: str = (
        "275f7373f4f5de9b0a1f12c2b5585bc7ef91e88d11dc2b61b387326798e6e730"
    )
    expected_admission_census_sha256: str = (
        "ca01e17f55637d0f4584aa7bbb8940e50ea4aaf25b9c7ee7f741bb51a4095a7f"
    )
    expected_common_masks_sha256: str = (
        "face3d6f211585d452f474b090adb618d121aada849a4cbd492e92e9b60a03b9"
    )
    expected_stage0_protein_manifest_sha256: str = (
        "fd34ae871c3d5feebba1dbe38bce24141634764882a9d51db1ce79cd7581d30b"
    )
    expected_stage0_fixed_probes_sha256: str = (
        "26dc56745c005c01e78007ad3c3b6dbc59708da23087ac7bd2337acc2ea27ef0"
    )
    expected_stage0_scoring_manifest_sha256: str = (
        "ec9882604cdfad0d61fbd30c1314466fbc7cbbea1fe7173d84c9913309783b00"
    )
    expected_checkpoint_sha256: str = (
        "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
    )


@dataclass(frozen=True)
class Scale1BProtocolFreezeInputs:
    scoring_panel: pd.DataFrame
    common_masks: pd.DataFrame
    admission_census: pd.DataFrame
    stage0_protein_manifest: dict[str, Any]
    stage0_fixed_probes: pd.DataFrame
    stage0_scoring_manifest: dict[str, Any]
    model_identity: VerifiedModelIdentity
    input_artifacts: tuple[dict[str, Any], ...]
    proteinmpnn_forward_executions: int = 0


@dataclass(frozen=True)
class Scale1BProtocolFreezeResult:
    status: str
    fixed_probes: pd.DataFrame
    scoring_plan: pd.DataFrame
    overlap_regression: pd.DataFrame
    workload: dict[str, int]
    probe_manifest: dict[str, Any]
    scoring_protocol: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]
    proteinmpnn_forward_executions: int = 0


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_input_path",
            f"Invalid logical path: {logical_ref}",
        ) from exc


def _require_hash(
    paths: ProjectPaths,
    logical_ref: str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            expected_sha256=expected_sha256,
            observed_sha256=observed,
        )
    return path, {"path": logical_ref, "sha256": observed, "label": label}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not isinstance(value, dict):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must be a JSON object",
        )
    return value


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def validate_scale1b_protocol_inputs(
    paths: ProjectPaths,
    config: Scale1BProtocolFreezeConfig,
) -> Scale1BProtocolFreezeInputs:
    """Validate frozen cohort, mask, Stage-0, and scorer trust roots."""
    bindings = (
        (config.scoring_panel_ref, config.expected_scoring_panel_sha256, "Scale-1B scoring panel"),
        (config.primary_core_ref, config.expected_primary_core_sha256, "Scale-1B primary core"),
        (config.redundancy_clusters_ref, config.expected_redundancy_clusters_sha256, "Scale-1B redundancy clusters"),
        (config.cohort_summary_ref, config.expected_cohort_summary_sha256, "Scale-1B cohort summary"),
        (config.cohort_manifest_ref, config.expected_cohort_manifest_sha256, "Scale-1B cohort manifest"),
        (config.admission_census_ref, config.expected_admission_census_sha256, "Scale-1A1 admission census"),
        (config.common_masks_ref, config.expected_common_masks_sha256, "Scale-1A1 common masks"),
        (config.stage0_protein_manifest_ref, config.expected_stage0_protein_manifest_sha256, "Stage-0 protein manifest"),
        (config.stage0_fixed_probes_ref, config.expected_stage0_fixed_probes_sha256, "Stage-0 fixed probes"),
        (config.stage0_scoring_manifest_ref, config.expected_stage0_scoring_manifest_sha256, "Stage-0 scoring manifest"),
    )
    resolved: dict[str, Path] = {}
    artifact_records: list[dict[str, Any]] = []
    for logical_ref, expected_sha, label in bindings:
        path, record = _require_hash(paths, logical_ref, expected_sha, label)
        resolved[logical_ref] = path
        artifact_records.append(record)

    scoring_panel = _read_parquet(
        resolved[config.scoring_panel_ref], "Scale-1B scoring panel"
    )
    primary_core = _read_parquet(
        resolved[config.primary_core_ref], "Scale-1B primary core"
    )
    redundancy = _read_parquet(
        resolved[config.redundancy_clusters_ref], "Scale-1B redundancy clusters"
    )
    cohort_summary = _read_json(
        resolved[config.cohort_summary_ref], "Scale-1B cohort summary"
    )
    cohort_manifest = _read_json(
        resolved[config.cohort_manifest_ref], "Scale-1B cohort manifest"
    )
    admission_census = _read_parquet(
        resolved[config.admission_census_ref], "Scale-1A1 admission census"
    )
    common_masks = _read_parquet(
        resolved[config.common_masks_ref], "Scale-1A1 common masks"
    )
    stage0_protein_manifest = _read_json(
        resolved[config.stage0_protein_manifest_ref], "Stage-0 protein manifest"
    )
    stage0_fixed_probes = _read_parquet(
        resolved[config.stage0_fixed_probes_ref], "Stage-0 fixed probes"
    )
    stage0_scoring_manifest = _read_json(
        resolved[config.stage0_scoring_manifest_ref], "Stage-0 scoring manifest"
    )

    if (
        len(scoring_panel) != 52
        or len(primary_core) != 46
        or len(redundancy) != 52
        or int(scoring_panel["stage0_overlap"].sum()) != 8
        or int(scoring_panel["in_new_protein_primary_core"].sum()) != 41
        or int((~scoring_panel["is_primary_representative"]).sum()) != 6
        or cohort_summary.get("freeze_status")
        != "SCALE1B_COHORT_FREEZE_COMPLETE"
        or cohort_manifest.get("SCALE1B_SCORING_PANEL_FROZEN") is not True
    ):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_cohort_count_mismatch",
            "Frozen Scale-1B cohort counts differ",
        )
    admitted_ids = set(
        admission_census.loc[
            admission_census["admission_status"].eq("FORMALLY_ADMITTED"), "pair_id"
        ].astype(str)
    )
    if admitted_ids != set(scoring_panel["pair_id"].astype(str)):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_cohort_identity_mismatch",
            "Scale-1B panel differs from the formally admitted identities",
        )
    seen_artifacts = {
        (str(record["path"]), str(record["sha256"]))
        for record in artifact_records
    }
    admitted_rows = admission_census.loc[
        admission_census["pair_id"].astype(str).isin(admitted_ids)
    ]
    for admission in admitted_rows.to_dict("records"):
        for side in ("pdb", "afdb"):
            logical_ref = str(admission[f"{side}_file_path_relative"])
            expected_sha = str(admission[f"{side}_file_sha256"])
            path = _resolve(paths, logical_ref)
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise Scale1BProtocolFreezeError(
                    BLOCKED_INPUT_INTEGRITY,
                    "static_backbone_sha256_mismatch",
                    f"Frozen {side.upper()} backbone differs for {admission['pair_id']}",
                )
            key = (logical_ref, expected_sha)
            if key not in seen_artifacts:
                artifact_records.append(
                    {
                        "path": logical_ref,
                        "sha256": expected_sha,
                        "label": f"Scale-1B {side.upper()} backbone",
                    }
                )
                seen_artifacts.add(key)
    if (
        len(stage0_fixed_probes) != 34_010
        or stage0_protein_manifest.get("summary", {}).get("total_mask_positions")
        != 1_790
        or stage0_scoring_manifest.get("status") != "complete"
        or stage0_scoring_manifest.get("upstream", {})
        .get("fixed_probes", {})
        .get("sha256")
        != config.expected_stage0_fixed_probes_sha256
    ):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "stage0_contract_mismatch",
            "Canonical Stage-0 contracts differ",
        )
    stage0_payload_records = [
        *stage0_scoring_manifest.get("outputs", {}).values(),
        stage0_scoring_manifest.get("upstream", {}).get("g2_audit", {}),
    ]
    for record in stage0_payload_records:
        logical_ref = record.get("path") if isinstance(record, dict) else None
        expected_sha = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(logical_ref, str) or not isinstance(expected_sha, str):
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "stage0_score_artifact_binding_missing",
                "Stage-0 scoring manifest lacks a required artifact binding",
            )
        path = _resolve(paths, logical_ref)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "stage0_score_artifact_sha256_mismatch",
                f"Stage-0 score artifact differs: {logical_ref}",
            )
        key = (logical_ref, expected_sha)
        if key not in seen_artifacts:
            artifact_records.append(
                {
                    "path": logical_ref,
                    "sha256": expected_sha,
                    "label": "Stage-0 immutable scoring artifact",
                }
            )
            seen_artifacts.add(key)

    for protein in stage0_protein_manifest.get("proteins", []):
        for side in ("pdb", "afdb"):
            logical_ref = str(protein[f"{side}_backbone_path"])
            expected_sha = str(protein[f"{side}_backbone_sha256"])
            path = _resolve(paths, logical_ref)
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise Scale1BProtocolFreezeError(
                    BLOCKED_INPUT_INTEGRITY,
                    "stage0_score_artifact_sha256_mismatch",
                    f"Stage-0 {side.upper()} backbone differs for {protein['protein_id']}",
                )
            key = (logical_ref, expected_sha)
            if key not in seen_artifacts:
                artifact_records.append(
                    {
                        "path": logical_ref,
                        "sha256": expected_sha,
                        "label": f"Stage-0 {side.upper()} scoring backbone",
                    }
                )
                seen_artifacts.add(key)

    stage0_config_path = _resolve(paths, config.stage0_config_ref)
    try:
        scoring_config = load_stage0_scoring_config(
            stage0_config_path, paths.repository_root
        )
        model_identity = verify_authorized_model(scoring_config)
    except FixedProbeScoringError as exc:
        raise Scale1BProtocolFreezeError(
            BLOCKED_SCORER_INTEGRITY,
            exc.code,
            str(exc),
        ) from exc
    if model_identity.checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise Scale1BProtocolFreezeError(
            BLOCKED_SCORER_INTEGRITY,
            "checkpoint_hash_mismatch",
            "Authorized checkpoint SHA differs",
        )
    artifact_records.append(
        {
            "path": config.stage0_config_ref,
            "sha256": sha256_file(stage0_config_path),
            "label": "Stage-0 scoring config",
        }
    )
    artifact_records.extend(
        [
            {
                "path": paths.logical_ref(model_identity.checkpoint_path),
                "sha256": model_identity.checkpoint_sha256,
                "label": "Authorized ProteinMPNN checkpoint",
                "kind": "model_checkpoint",
            },
            {
                "path": paths.logical_ref(model_identity.implementation_path),
                "commit": model_identity.implementation_commit,
                "label": "Authorized ProteinMPNN implementation",
                "kind": "git_repository",
                "tracked_worktree_clean": True,
            },
        ]
    )
    return Scale1BProtocolFreezeInputs(
        scoring_panel=scoring_panel,
        common_masks=common_masks,
        admission_census=admission_census,
        stage0_protein_manifest=stage0_protein_manifest,
        stage0_fixed_probes=stage0_fixed_probes,
        stage0_scoring_manifest=stage0_scoring_manifest,
        model_identity=model_identity,
        input_artifacts=tuple(artifact_records),
    )


def _stage0_regression_block(code: str, protein_id: str) -> None:
    raise Scale1BProtocolFreezeError(
        BLOCKED_STAGE0_REGRESSION,
        code,
        f"Stage-0 reuse compatibility failed for {protein_id}",
        protein_id=protein_id,
    )


def _records_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    left_values = left.astype(object).where(pd.notna(left), None).values.tolist()
    right_values = right.astype(object).where(pd.notna(right), None).values.tolist()
    return left_values == right_values


def validate_stage0_overlap_regression(
    inputs: Scale1BProtocolFreezeInputs,
) -> pd.DataFrame:
    """Fail fast unless all eight Stage-0 proteins are exactly reusable."""
    overlap = inputs.scoring_panel.loc[
        inputs.scoring_panel["stage0_overlap"].astype(bool)
    ].copy()
    if len(overlap) != 8:
        _stage0_regression_block("stage0_overlap_count_mismatch", "ALL")
    stage0_proteins = {
        str(record["protein_id"]): record
        for record in inputs.stage0_protein_manifest.get("proteins", [])
    }
    scoring_proteins = {
        str(record["protein_id"]): record
        for record in inputs.stage0_scoring_manifest.get("proteins", [])
    }
    realization_contract = inputs.stage0_scoring_manifest.get(
        "decoding_realizations", {}
    )
    realization_records = realization_contract.get("records", [])
    if (
        realization_contract.get("protocol") != DECODING_REALIZATION_ALGORITHM
        or realization_contract.get("pairing")
        != "one fingerprint per protein and repeat shared across PDB/AFDB/WT/candidates"
    ):
        _stage0_regression_block("stage0_realization_contract_mismatch", "ALL")
    scoring_protocol = inputs.stage0_scoring_manifest.get("scoring_protocol", {})
    model = inputs.stage0_scoring_manifest.get("model", {})
    global_score_compatible = (
        scoring_protocol.get("identity") == SCORING_PROTOCOL_VERSION
        and scoring_protocol.get("mode")
        == "fixed_sequence_autoregressive_mask_logp"
        and scoring_protocol.get("domain") == "frozen_common_mask_projection"
        and scoring_protocol.get("score_direction") == "higher_is_better"
        and scoring_protocol.get("sequence_generation") is False
    )
    global_scorer_compatible = (
        model.get("implementation_commit")
        == inputs.model_identity.implementation_commit
        and model.get("checkpoint_sha256")
        == inputs.model_identity.checkpoint_sha256
    )
    if not global_score_compatible:
        _stage0_regression_block("stage0_score_definition_mismatch", "ALL")
    if not global_scorer_compatible:
        _stage0_regression_block("stage0_scorer_mismatch", "ALL")

    census = inputs.admission_census.set_index("pair_id", drop=False)
    rows: list[dict[str, Any]] = []
    for panel in overlap.to_dict("records"):
        protein_id = str(panel["pair_id"])
        protein = stage0_proteins.get(protein_id)
        score_record = scoring_proteins.get(protein_id)
        if protein is None or score_record is None or protein_id not in census.index:
            _stage0_regression_block("stage0_identity_missing", protein_id)
        canonical_sequence_compatible = (
            str(panel["canonical_sequence"])
            == str(protein["canonical_wt_sequence"])
            and str(panel["canonical_sequence_sha256"])
            == str(protein["canonical_sequence_sha256"])
            and str(panel["canonical_accession"])
            == str(protein["uniprot_accession"])
        )
        if not canonical_sequence_compatible:
            _stage0_regression_block("stage0_canonical_sequence_mismatch", protein_id)

        mask_positions = (
            inputs.common_masks.loc[
                inputs.common_masks["pair_id"].eq(protein_id)
                & inputs.common_masks["common_mask"].fillna(False).astype(bool),
                "canonical_position",
            ]
            .astype(int)
            .tolist()
        )
        common_mask_compatible = mask_positions == [
            int(value) for value in protein["mask_positions"]
        ]
        if not common_mask_compatible:
            _stage0_regression_block("stage0_common_mask_mismatch", protein_id)

        generated = build_fixed_probe_candidates(
            {
                "proteins": [
                    {
                        "protein_id": protein_id,
                        "uniprot_accession": str(panel["canonical_accession"]),
                        "canonical_wt_sequence": str(panel["canonical_sequence"]),
                        "mask_positions": mask_positions,
                    }
                ]
            }
        ).reset_index(drop=True)
        frozen = inputs.stage0_fixed_probes.loc[
            inputs.stage0_fixed_probes["protein_id"].eq(protein_id)
        ].reset_index(drop=True)
        probe_compatible = _records_equal(generated, frozen)
        if not probe_compatible:
            _stage0_regression_block("stage0_probe_identity_mismatch", protein_id)

        protein_realizations = [
            record
            for record in realization_records
            if record.get("protein_id") == protein_id
        ]
        repeat_set_compatible = (
            len(protein_realizations) == 30
            and [record.get("repeat_index") for record in protein_realizations]
            == list(range(30))
            and [record.get("seed") for record in protein_realizations]
            == list(range(30))
        )
        if not repeat_set_compatible:
            _stage0_regression_block("stage0_repeat_set_mismatch", protein_id)
        realization_semantics_compatible = True
        for record in protein_realizations:
            repeat_index = int(record["repeat_index"])
            realization = make_decoding_realization(
                protein_id=protein_id,
                mask_length=len(mask_positions),
                repeat_index=repeat_index,
                seed=int(record["seed"]),
                protocol_version=SCORING_PROTOCOL_VERSION,
            )
            if (
                record.get("algorithm") != realization.algorithm
                or record.get("decoding_realization_sha256")
                != realization.fingerprint
            ):
                realization_semantics_compatible = False
                break
        if not realization_semantics_compatible:
            _stage0_regression_block("stage0_realization_mismatch", protein_id)

        admission = census.loc[protein_id]
        structural_inputs_compatible = (
            str(protein["pdb_backbone_path"])
            == str(score_record["pdb_backbone_path"])
            and str(protein["afdb_backbone_path"])
            == str(score_record["afdb_backbone_path"])
            and str(admission["pdb_file_sha256"])
            == str(protein["pdb_backbone_sha256"])
            == str(score_record["pdb_backbone_sha256"])
            and str(admission["afdb_file_sha256"])
            == str(protein["afdb_backbone_sha256"])
            == str(score_record["afdb_backbone_sha256"])
        )
        if not structural_inputs_compatible:
            _stage0_regression_block("stage0_structural_input_mismatch", protein_id)

        rows.append(
            {
                "protein_id": protein_id,
                "canonical_sequence_compatible": True,
                "common_mask_compatible": True,
                "probe_compatible": True,
                "score_definition_compatible": True,
                "scorer_compatible": True,
                "repeat_set_compatible": True,
                "realization_semantics_compatible": True,
                "structural_inputs_compatible": True,
                "probe_reuse_status": "PROBE_REUSE_COMPATIBLE",
                "score_reuse_status": "SCORE_REUSE_COMPATIBLE",
                "scoring_source": "REUSE_FROZEN_STAGE0",
            }
        )
    return pd.DataFrame(rows)


def build_scale1b_fixed_probe_candidates(
    inputs: Scale1BProtocolFreezeInputs,
    overlap_regression: pd.DataFrame,
) -> pd.DataFrame:
    """Adapt the frozen Stage-0 generator to all 52 Scale-1B proteins."""
    if (
        len(overlap_regression) != 8
        or not overlap_regression["probe_reuse_status"]
        .eq("PROBE_REUSE_COMPATIBLE")
        .all()
        or not overlap_regression["score_reuse_status"]
        .eq("SCORE_REUSE_COMPATIBLE")
        .all()
    ):
        _stage0_regression_block("stage0_regression_not_complete", "ALL")

    generator_proteins: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    census = inputs.admission_census.set_index("pair_id", drop=False)
    for panel in inputs.scoring_panel.to_dict("records"):
        protein_id = str(panel["pair_id"])
        if protein_id not in census.index:
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "probe_metadata_identity_mismatch",
                f"Admission metadata is missing for {protein_id}",
            )
        admission = census.loc[protein_id]
        canonical = str(panel["canonical_sequence"])
        protein_masks = inputs.common_masks.loc[
            inputs.common_masks["pair_id"].eq(protein_id)
            & inputs.common_masks["common_mask"].fillna(False).astype(bool)
        ].copy()
        if protein_masks["canonical_position"].duplicated().any():
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "duplicate_common_mask_position",
                f"Common-mask positions repeat for {protein_id}",
            )
        protein_masks = protein_masks.sort_values(
            "canonical_position", kind="stable"
        )
        positions = protein_masks["canonical_position"].astype(int).tolist()
        if (
            len(positions) != int(panel["common_mask_count"])
            or any(position < 1 or position > len(canonical) for position in positions)
            or protein_masks["canonical_aa"].astype(str).tolist()
            != [canonical[position - 1] for position in positions]
        ):
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "common_mask_sequence_mismatch",
                f"Common mask differs from the canonical sequence for {protein_id}",
            )
        generator_proteins.append(
            {
                "protein_id": protein_id,
                "uniprot_accession": str(panel["canonical_accession"]),
                "canonical_wt_sequence": canonical,
                "mask_positions": positions,
            }
        )
        metadata_rows.append(
            {
                "protein_id": protein_id,
                "canonical_accession": str(panel["canonical_accession"]),
                "pdb_id": str(panel["pdb_id"]),
                "pdb_chain": str(panel["pdb_chain"]),
                "polymer_entity_id": str(panel["polymer_entity_id"]),
                "afdb_model_id": str(admission["selected_model_entity_id"]),
                "canonical_length": int(panel["canonical_sequence_length"]),
                "common_mask_count": int(panel["common_mask_count"]),
                "cohort_role": str(panel["cohort_role"]),
                "redundancy_cluster_id": str(panel["redundancy_cluster_id"]),
                "stage0_overlap": bool(panel["stage0_overlap"]),
                "scale1_new_protein": not bool(panel["stage0_overlap"]),
                "analysis_origin": str(panel["analysis_origin"]),
                "in_primary_nonredundant_core": bool(
                    panel["in_primary_nonredundant_core"]
                ),
                "in_new_protein_primary_core": bool(
                    panel["in_new_protein_primary_core"]
                ),
                "scoring_source": (
                    "REUSE_FROZEN_STAGE0"
                    if bool(panel["stage0_overlap"])
                    else "NEW_SCALE1B_SCORING"
                ),
            }
        )

    generated = build_fixed_probe_candidates(
        {"proteins": generator_proteins}
    ).rename(
        columns={
            "position": "mutation_position",
            "mut_aa": "mutant_aa",
        }
    )
    generated = generated.drop(columns=["uniprot_accession"])
    metadata = pd.DataFrame(metadata_rows)
    output = generated.merge(
        metadata,
        on="protein_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output.insert(
        0,
        "candidate_id",
        output.apply(
            lambda row: (
                f"{row['protein_id']}::{row['wt_aa']}"
                f"{int(row['mutation_position'])}{row['mutant_aa']}"
            ),
            axis=1,
        ),
    )
    output.insert(1, "candidate_index_global", range(1, len(output) + 1))
    output.insert(
        3,
        "candidate_index_within_protein",
        output.groupby("protein_id", sort=False).cumcount() + 1,
    )
    if (
        output["candidate_id"].duplicated().any()
        or output.duplicated(
            ["protein_id", "mutation_position", "mutant_aa"]
        ).any()
        or not output.groupby(["protein_id", "mutation_position"]).size().eq(19).all()
    ):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "fixed_probe_identity_mismatch",
            "Scale-1B fixed-probe identities are incomplete or duplicated",
        )
    return output.reset_index(drop=True)


def build_scale1b_scoring_plan(
    inputs: Scale1BProtocolFreezeInputs,
    fixed_probes: pd.DataFrame,
    overlap_regression: pd.DataFrame,
) -> pd.DataFrame:
    """Build static protein × backbone × repeat scientific shard bindings."""
    if (
        len(overlap_regression) != 8
        or not overlap_regression["score_reuse_status"]
        .eq("SCORE_REUSE_COMPATIBLE")
        .all()
    ):
        _stage0_regression_block("stage0_score_reuse_not_complete", "ALL")
    stage0_proteins = {
        str(record["protein_id"]): record
        for record in inputs.stage0_protein_manifest["proteins"]
    }
    census = inputs.admission_census.set_index("pair_id", drop=False)
    grouped = {
        protein_id: group.reset_index(drop=True)
        for protein_id, group in fixed_probes.groupby("protein_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for panel in inputs.scoring_panel.to_dict("records"):
        protein_id = str(panel["pair_id"])
        candidates = grouped.get(protein_id)
        if candidates is None or candidates.empty or protein_id not in census.index:
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "scoring_plan_candidate_identity_mismatch",
                f"No fixed-probe candidate set for {protein_id}",
            )
        candidate_hashes = candidates["sequence_hash"].astype(str).tolist()
        candidate_identity_sha = sha256_canonical(
            {"sequence_hashes": candidate_hashes}
        )
        candidate_start = int(candidates["candidate_index_global"].iloc[0])
        candidate_end = int(candidates["candidate_index_global"].iloc[-1])
        candidate_count = len(candidates)
        mask_length = int(panel["common_mask_count"])
        source = (
            "REUSE_FROZEN_STAGE0"
            if bool(panel["stage0_overlap"])
            else "NEW_SCALE1B_SCORING"
        )
        admission = census.loc[protein_id]
        if source == "REUSE_FROZEN_STAGE0":
            stage0 = stage0_proteins[protein_id]
            backbone_bindings = (
                (
                    "PDB",
                    str(stage0["pdb_backbone_path"]),
                    str(stage0["pdb_backbone_sha256"]),
                ),
                (
                    "AFDB",
                    str(stage0["afdb_backbone_path"]),
                    str(stage0["afdb_backbone_sha256"]),
                ),
            )
        else:
            backbone_bindings = (
                (
                    "PDB",
                    str(admission["pdb_file_path_relative"]),
                    str(admission["pdb_file_sha256"]),
                ),
                (
                    "AFDB",
                    str(admission["afdb_file_path_relative"]),
                    str(admission["afdb_file_sha256"]),
                ),
            )
        realizations = {
            repeat_index: make_decoding_realization(
                protein_id=protein_id,
                mask_length=mask_length,
                repeat_index=repeat_index,
                seed=repeat_index,
                protocol_version=SCORING_PROTOCOL_VERSION,
            )
            for repeat_index in range(30)
        }
        for backbone_type, backbone_path, backbone_sha in backbone_bindings:
            for repeat_index in range(30):
                realization = realizations[repeat_index]
                rows.append(
                    {
                        "logical_shard_id": (
                            f"{protein_id}::{backbone_type}::r{repeat_index:02d}"
                        ),
                        "protein_id": protein_id,
                        "canonical_accession": str(panel["canonical_accession"]),
                        "pdb_id": str(panel["pdb_id"]),
                        "pdb_chain": str(panel["pdb_chain"]),
                        "afdb_model_id": str(admission["selected_model_entity_id"]),
                        "backbone_type": backbone_type,
                        "backbone_path": backbone_path,
                        "backbone_sha256": backbone_sha,
                        "repeat_index": repeat_index,
                        "seed": repeat_index,
                        "decoding_realization_sha256": realization.fingerprint,
                        "decoding_realization_algorithm": realization.algorithm,
                        "realization_applies_to": (
                            "WT_AND_ALL_PROBES_PAIRED_ACROSS_PDB_AFDB"
                        ),
                        "candidate_index_global_start": candidate_start,
                        "candidate_index_global_end": candidate_end,
                        "candidate_count": candidate_count,
                        "candidate_identity_sha256": candidate_identity_sha,
                        "expected_wt_count": 1,
                        "common_mask_count": mask_length,
                        "cohort_role": str(panel["cohort_role"]),
                        "redundancy_cluster_id": str(
                            panel["redundancy_cluster_id"]
                        ),
                        "stage0_overlap": bool(panel["stage0_overlap"]),
                        "analysis_origin": str(panel["analysis_origin"]),
                        "scoring_source": source,
                        "execution_required": source == "NEW_SCALE1B_SCORING",
                        "frozen_batch_size": 128,
                        "scoring_protocol": SCORING_PROTOCOL_VERSION,
                        "checkpoint_sha256": (
                            inputs.model_identity.checkpoint_sha256
                        ),
                        "implementation_commit": (
                            inputs.model_identity.implementation_commit
                        ),
                    }
                )
    plan = pd.DataFrame(rows)
    if (
        len(plan) != 3_120
        or plan["logical_shard_id"].duplicated().any()
        or not plan.groupby(["protein_id", "repeat_index"])[
            "decoding_realization_sha256"
        ]
        .nunique()
        .eq(1)
        .all()
    ):
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "logical_shard_plan_mismatch",
            "Scale-1B logical shard plan is incomplete or unpaired",
        )
    return plan


def summarize_scale1b_workload(
    inputs: Scale1BProtocolFreezeInputs,
    fixed_probes: pd.DataFrame,
    scoring_plan: pd.DataFrame,
) -> dict[str, int]:
    """Return exact full-panel and execution-required static workloads."""
    new_probes = fixed_probes.loc[fixed_probes["scale1_new_protein"]]
    per_protein = fixed_probes.groupby("protein_id", sort=False).size()
    total_positions = int(inputs.scoring_panel["common_mask_count"].sum())
    new_positions = int(
        inputs.scoring_panel.loc[
            ~inputs.scoring_panel["stage0_overlap"], "common_mask_count"
        ].sum()
    )
    return {
        "protein_count": len(inputs.scoring_panel),
        "new_protein_count": int((~inputs.scoring_panel["stage0_overlap"]).sum()),
        "total_common_mask_positions": total_positions,
        "new_protein_common_mask_positions": new_positions,
        "total_fixed_probes": len(fixed_probes),
        "new_protein_fixed_probes": len(new_probes),
        "per_protein_probe_min": int(per_protein.min()),
        "per_protein_probe_max": int(per_protein.max()),
        "full_candidate_score_rows": len(fixed_probes) * 2 * 30,
        "full_wt_rows": len(inputs.scoring_panel) * 2 * 30,
        "new_candidate_score_rows": len(new_probes) * 2 * 30,
        "new_wt_rows": int((~inputs.scoring_panel["stage0_overlap"]).sum())
        * 2
        * 30,
        "total_logical_shards": len(scoring_plan),
        "execution_required_logical_shards": int(
            scoring_plan["execution_required"].sum()
        ),
        "reuse_frozen_stage0_logical_shards": int(
            (~scoring_plan["execution_required"]).sum()
        ),
    }


def build_scale1b_protocol_freeze(
    inputs: Scale1BProtocolFreezeInputs,
    *,
    overlap_regression: pd.DataFrame | None = None,
    fixed_probes: pd.DataFrame | None = None,
    scoring_plan: pd.DataFrame | None = None,
) -> Scale1BProtocolFreezeResult:
    """Build one canonical static protocol-freeze result without scoring."""
    regression = (
        validate_stage0_overlap_regression(inputs)
        if overlap_regression is None
        else overlap_regression
    )
    probes = (
        build_scale1b_fixed_probe_candidates(inputs, regression)
        if fixed_probes is None
        else fixed_probes
    )
    plan = (
        build_scale1b_scoring_plan(inputs, probes, regression)
        if scoring_plan is None
        else scoring_plan
    )
    workload = summarize_scale1b_workload(inputs, probes, plan)
    per_protein: list[dict[str, Any]] = []
    for panel in inputs.scoring_panel.to_dict("records"):
        protein_id = str(panel["pair_id"])
        candidates = probes.loc[probes["protein_id"].eq(protein_id)]
        per_protein.append(
            {
                "protein_id": protein_id,
                "canonical_accession": str(panel["canonical_accession"]),
                "common_mask_count": int(panel["common_mask_count"]),
                "fixed_probe_count": len(candidates),
                "candidate_index_global_start": int(
                    candidates["candidate_index_global"].iloc[0]
                ),
                "candidate_index_global_end": int(
                    candidates["candidate_index_global"].iloc[-1]
                ),
                "cohort_role": str(panel["cohort_role"]),
                "redundancy_cluster_id": str(panel["redundancy_cluster_id"]),
                "analysis_origin": str(panel["analysis_origin"]),
                "scoring_source": (
                    "REUSE_FROZEN_STAGE0"
                    if bool(panel["stage0_overlap"])
                    else "NEW_SCALE1B_SCORING"
                ),
            }
        )
    probe_manifest = {
        "schema_version": "dual-uq.scale1b-fixed-probe-manifest.v1",
        "status": PROTOCOL_FREEZE_COMPLETE,
        "protein_count": 52,
        "total_common_mask_positions": workload["total_common_mask_positions"],
        "total_fixed_probes": workload["total_fixed_probes"],
        "new_protein_common_mask_positions": workload[
            "new_protein_common_mask_positions"
        ],
        "new_protein_fixed_probes": workload["new_protein_fixed_probes"],
        "probes_per_position": 19,
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "candidate_sequence_materialized": True,
        "candidate_sequence_materialization_reason": (
            "required_by_frozen_stage0_scoring_adapter"
        ),
        "candidate_ordering": [
            "frozen_scale1b_scoring_panel_order",
            "canonical_mutation_position_ascending",
            "frozen_amino_acid_alphabet_order_with_wt_omitted",
        ],
        "identity_invariants": {
            "hamming_distance": 1,
            "mutation_position_domain": "frozen_common_mask",
            "backbone_independent": True,
            "duplicate_probe_identity_count": 0,
        },
        "per_protein": per_protein,
        "stage0_overlap_regression": regression.to_dict("records"),
        "candidate_artifact": {},
    }
    scoring_protocol = {
        "schema_version": "dual-uq.scale1b-scoring-protocol.v1",
        "status": PROTOCOL_FREEZE_COMPLETE,
        "model": {
            "family": "ProteinMPNN",
            "role": "stage0_internal_inverse_folding_scorer",
            "implementation_path": "third_party/ProteinMPNN",
            "implementation_commit": inputs.model_identity.implementation_commit,
            "checkpoint_path": (
                "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
            ),
            "checkpoint_sha256": inputs.model_identity.checkpoint_sha256,
        },
        "score_semantics": {
            "identity": SCORING_PROTOCOL_VERSION,
            "formula": (
                "S(k,B,r)=mean_{j in common_mask} log p_theta("
                "x_j^(k) | B,x^(k),r)"
            ),
            "canonical_outputs": [
                "score_sum_logp_mask",
                "score_mean_logp_mask",
            ],
            "direction": "higher_is_greater_internal_sequence_compatibility",
            "domain": "frozen_common_mask_projection",
            "official_proteinmpnn_scores_nll_is_canonical": False,
            "target_position_conditional_probability": False,
            "experimental_fitness": False,
            "design_success_probability": False,
        },
        "wt_semantics": {
            "expected_wt_rows_per_logical_shard": 1,
            "same_backbone_common_mask_realization_scorer_as_probes": True,
            "future_condition_local_normalization": (
                "DeltaS(k,B,r)=S(k,B,r)-S(WT,B,r)"
            ),
            "normalization_computed_in_this_task": False,
        },
        "stochasticity": {
            "classification": "stochastic_due_to_decoding_order",
            "repeat_count": 30,
            "repeat_indices": list(range(30)),
            "seeds": list(range(30)),
            "realization_algorithm": DECODING_REALIZATION_ALGORITHM,
            "pairing": (
                "one explicit realization per protein/repeat shared across "
                "PDB/AFDB/WT/all probes"
            ),
        },
        "batch_chunk_semantics": {
            "logical_shard": "protein_x_backbone_x_repeat",
            "frozen_batch_size": 128,
            "execution_chunking_may_subdivide_logical_shard": True,
            "chunking_must_preserve_candidate_order": True,
            "chunking_must_preserve_realization_identity": True,
            "chunking_must_preserve_wt_probe_and_backbone_pairing": True,
        },
        "score_reuse_policy": {
            "REUSE_FROZEN_STAGE0_proteins": 8,
            "NEW_SCALE1B_SCORING_proteins": 44,
            "reuse_requires_exact_probe_score_realization_and_structure_identity": True,
            "silent_rescoring_fallback": False,
        },
        "technical_null": {
            "future_formula": "Y_tech_mean=(Y_tech,PDB+Y_tech,AFDB)/2",
            "future_structural_excess": "E=Y_struct-Y_tech_mean",
            "interpretation": "descriptive structural-excess contrast",
            "causal_correction": False,
            "variance_decomposition": False,
            "computed_in_this_task": False,
        },
        "future_endpoints": {
            "H1a": [
                "conditional P gradient for Top-1",
                "conditional P gradient for regret",
                "M-protective gradient for Top-1",
                "M-protective gradient for regret",
            ],
            "H1b": [
                "Spearman P vs E_flip",
                "Spearman P vs E_regret",
                "supporting C2 M-matched HIGH-P minus LOW-P excess contrasts",
            ],
            "H1c_descriptive_unresolved": [
                "M vs E",
                "C1 P-matched LOW-M minus HIGH-M excess contrasts",
            ],
            "composite_h1_score": False,
            "computed": False,
        },
        "workload": workload,
        "proteinmpnn_forward_executions": 0,
    }
    manifest = {
        "schema_version": "dual-uq.scale1b-protocol-freeze-manifest.v1",
        "protocol_status": PROTOCOL_FREEZE_COMPLETE,
        "upstream_artifacts": list(inputs.input_artifacts),
        "stage0_overlap_regression": regression.to_dict("records"),
        "workload": workload,
        "outputs": {},
        "manifest_self_hash_policy": (
            "reported_by_cli_after_write_to_avoid_recursive_self_hash"
        ),
        "SCALE1B_COHORT_FROZEN": True,
        "FIXED_PROBE_SPACE_FROZEN": True,
        "SCORING_PROTOCOL_FROZEN": True,
        "SCALE1B_SCORING_NOT_STARTED": True,
        "STAGE0_SCORE_REUSE_POLICY_FROZEN": True,
        "NO_OUTCOME_DEPENDENT_CHANGES": True,
        "STAGE0_LOCAL_ANALYSIS_STOP": True,
        "scope_confirmations": {
            "proteinmpnn_forward_executions": 0,
            "scale1_outcomes_computed": False,
            "sequence_generation_performed": False,
            "network_access_used": False,
            "acquisition_performed": False,
        },
    }
    return Scale1BProtocolFreezeResult(
        status=PROTOCOL_FREEZE_COMPLETE,
        fixed_probes=probes,
        scoring_plan=plan,
        overlap_regression=regression,
        workload=workload,
        probe_manifest=probe_manifest,
        scoring_protocol=scoring_protocol,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )


def _render_json(payload: dict[str, Any]) -> bytes:
    rendered = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if b"/home/" in rendered or b"/mnt/" in rendered:
        raise Scale1BProtocolFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "absolute_path_leakage",
            "Protocol output contains a machine absolute path",
        )
    return rendered


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        if path.exists():
            if path.read_bytes() == payload:
                return "reused_identical"
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_protocol_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_protocol_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False)
        if path.exists():
            if (
                path.stat().st_size == temporary.stat().st_size
                and sha256_file(path) == sha256_file(temporary)
            ):
                return "reused_identical"
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_protocol_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_protocol_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(
    result: Scale1BProtocolFreezeResult, paths: ProjectPaths
) -> None:
    for record in result.input_artifacts:
        path = _resolve(paths, str(record["path"]))
        kind = str(record.get("kind", "file"))
        if kind == "git_repository":
            try:
                observed_commit = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                clean = implementation_worktree_is_clean(path)
            except (
                OSError,
                subprocess.CalledProcessError,
                ProteinMPNNScoringError,
            ) as exc:
                raise Scale1BProtocolFreezeError(
                    BLOCKED_SCORER_INTEGRITY,
                    "implementation_changed_before_materialization",
                    "Cannot revalidate the authorized ProteinMPNN implementation",
                ) from exc
            if (
                not path.is_dir()
                or observed_commit != str(record["commit"])
                or not clean
            ):
                raise Scale1BProtocolFreezeError(
                    BLOCKED_SCORER_INTEGRITY,
                    "implementation_changed_before_materialization",
                    "Authorized ProteinMPNN implementation changed before materialization",
                )
            continue
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            if kind == "model_checkpoint":
                raise Scale1BProtocolFreezeError(
                    BLOCKED_SCORER_INTEGRITY,
                    "checkpoint_changed_before_materialization",
                    "Authorized ProteinMPNN checkpoint changed before materialization",
                )
            raise Scale1BProtocolFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "upstream_input_changed",
                f"Frozen input changed: {record['label']}",
            )


def materialize_scale1b_protocol_freeze(
    result: Scale1BProtocolFreezeResult,
    paths: ProjectPaths,
    *,
    config: Scale1BProtocolFreezeConfig | None = None,
) -> dict[str, Any]:
    """Write four canonical outputs, then the immutable freeze manifest last."""
    config = config or Scale1BProtocolFreezeConfig()
    _rehash_inputs(result, paths)
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "fixed_probe_candidates": output_root
        / "scale1b_fixed_probe_candidates.parquet",
        "probe_manifest": output_root / "scale1b_probe_manifest.json",
        "scoring_plan": output_root / "scale1b_scoring_plan.parquet",
        "scoring_protocol": output_root / "scale1b_scoring_protocol.json",
        "manifest": output_root / "scale1b_protocol_freeze_manifest.json",
    }
    write_status = {
        "fixed_probe_candidates": _write_immutable_parquet(
            targets["fixed_probe_candidates"], result.fixed_probes
        ),
        "scoring_plan": _write_immutable_parquet(
            targets["scoring_plan"], result.scoring_plan
        ),
    }
    probe_manifest = {
        **result.probe_manifest,
        "candidate_artifact": {
            "path": paths.logical_ref(targets["fixed_probe_candidates"]),
            "rows": len(result.fixed_probes),
            "sha256": sha256_file(targets["fixed_probe_candidates"]),
        },
    }
    write_status["probe_manifest"] = _write_immutable_bytes(
        targets["probe_manifest"], _render_json(probe_manifest)
    )
    write_status["scoring_protocol"] = _write_immutable_bytes(
        targets["scoring_protocol"], _render_json(result.scoring_protocol)
    )
    outputs = {
        "fixed_probe_candidates": {
            "path": paths.logical_ref(targets["fixed_probe_candidates"]),
            "rows": len(result.fixed_probes),
            "sha256": sha256_file(targets["fixed_probe_candidates"]),
        },
        "probe_manifest": {
            "path": paths.logical_ref(targets["probe_manifest"]),
            "sha256": sha256_file(targets["probe_manifest"]),
        },
        "scoring_plan": {
            "path": paths.logical_ref(targets["scoring_plan"]),
            "rows": len(result.scoring_plan),
            "sha256": sha256_file(targets["scoring_plan"]),
        },
        "scoring_protocol": {
            "path": paths.logical_ref(targets["scoring_protocol"]),
            "sha256": sha256_file(targets["scoring_protocol"]),
        },
    }
    manifest = {**result.manifest, "outputs": outputs}
    write_status["manifest"] = _write_immutable_bytes(
        targets["manifest"], _render_json(manifest)
    )
    outputs["manifest"] = {
        "path": paths.logical_ref(targets["manifest"]),
        "sha256": sha256_file(targets["manifest"]),
    }
    return {
        "protocol_status": result.status,
        "proteinmpnn_forward_executions": 0,
        "write_status": write_status,
        "outputs": outputs,
    }


def run_scale1b_protocol_freeze(
    paths: ProjectPaths,
    *,
    config: Scale1BProtocolFreezeConfig | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and materialize without executing ProteinMPNN."""
    config = config or Scale1BProtocolFreezeConfig()
    inputs = validate_scale1b_protocol_inputs(paths, config)
    result = build_scale1b_protocol_freeze(inputs)
    return materialize_scale1b_protocol_freeze(result, paths, config=config)
