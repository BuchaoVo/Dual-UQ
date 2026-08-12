"""Redundancy capacity and descriptive diversity census."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths

SCHEMA_VERSION = "dual-uq.scale1a3-redundancy-diversity.v1"
DEFAULT_OUTPUT_ROOT = "experiments/p2_design_baseline/scale1/scale1a3"

CORE_SCALE_REACHED = "CORE_SCALE_REACHED"
PENDING_VARIANT_REVIEW_PRIORITY = "PENDING_VARIANT_REVIEW_PRIORITY"
ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED = (
    "ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED"
)
BLOCKED_INPUT_INTEGRITY = "BLOCKED_INPUT_INTEGRITY"
BLOCKED_REDUNDANCY_BINDING = "BLOCKED_REDUNDANCY_BINDING"
BLOCKED_DECISION_INVARIANT = "BLOCKED_DECISION_INVARIANT"
BLOCKED_REDUNDANCY_DEFINITION = "BLOCKED_REDUNDANCY_DEFINITION"

CATH_METADATA_COMPLETE = "CATH_METADATA_COMPLETE"
CATH_METADATA_PARTIAL = "CATH_METADATA_PARTIAL"
CATH_METADATA_UNAVAILABLE = "CATH_METADATA_UNAVAILABLE"

_CLUSTER_PATTERN = re.compile(r"^30:.+$")


class RedundancyDiversityError(RuntimeError):
    """Structured Scale-1A3 blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class RedundancyDiversityConfig:
    scale1a2_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_census.parquet"
    )
    scale1a2_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_common_masks.parquet"
    )
    scale1a2_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_attrition_summary.json"
    )
    scale1a2_ledger_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1a2_acquisition_ledger.parquet"
    )
    inventory_tsv_ref: str = (
        "artifacts/dataset/reports/census/candidate_inventory_v1.tsv"
    )
    inventory_json_ref: str = (
        "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    )
    discovery_ref: str = "data/processed/discovery/discovered_candidates.parquet"
    cluster_implementation_ref: str = "src/dual_uq/rcsb_discovery.py"
    cohort_implementation_ref: str = (
        "src/dual_uq/dataset/scale1b_cohort_freeze.py"
    )
    scale1b_panel_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_scoring_panel.parquet"
    )
    scale1b_primary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_primary_core.parquet"
    )
    scale1b_clusters_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_redundancy_clusters.parquet"
    )
    scale1b_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_summary.json"
    )
    scale1b_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_freeze_manifest.json"
    )
    scale1b_probe_candidates_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_fixed_probe_candidates.parquet"
    )
    scale1b_probe_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_probe_manifest.json"
    )
    scale1b_scoring_plan_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_scoring_plan.parquet"
    )
    scale1b_scoring_protocol_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_scoring_protocol.json"
    )
    scale1b_protocol_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_protocol_freeze_manifest.json"
    )
    output_root_ref: str = DEFAULT_OUTPUT_ROOT

    expected_scale1a2_census_sha256: str = (
        "fc69e13d3d94def1a751b5528230fa5f15dd1ae741d8361b4187236167120793"
    )
    expected_scale1a2_masks_sha256: str = (
        "4a6d89a9368c3349f0598616e3dda670b17164fe6dff7fd37e131a71e4acaae0"
    )
    expected_scale1a2_summary_sha256: str = (
        "36ee351614ed3e0f9ba57161bbdc2a4410acc3ca5cb5ad8d1b2de54427ae7002"
    )
    expected_scale1a2_ledger_sha256: str = (
        "a6310505eb8ba397b835ccaa9fb5fece2ed6309792d6af07368bfa517959869f"
    )
    expected_inventory_tsv_sha256: str = (
        "e7b1d70a00c069c16c49dd701bc60ac1a11711bb4b859b0ebc9a8e80e1214996"
    )
    expected_inventory_json_sha256: str = (
        "05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf"
    )
    expected_discovery_sha256: str = (
        "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436"
    )
    expected_cluster_implementation_sha256: str = (
        "db70dfcecf21588be1a8ea84da36cac1e06ea4de1985598a6e9221ef073960e8"
    )
    expected_cohort_implementation_sha256: str = (
        "4d4e62a4ee019ce5612ec2f366bee8934b19e353fa38973f8703cf4b48bc89aa"
    )
    expected_scale1b_panel_sha256: str = (
        "6fdc074d8dd56bb502592b51830efdcafcb34dcb2b530e8fe26e95d657684072"
    )
    expected_scale1b_primary_sha256: str = (
        "4adbb8ccd07036df9243d26fdaf10b1d5421a1a345d441fced00f3937b7b89a7"
    )
    expected_scale1b_clusters_sha256: str = (
        "1e7ff6990a20eae5f373e36f1648c903c355cdb8e31adf5ad2a3fb2a27bac944"
    )
    expected_scale1b_summary_sha256: str = (
        "064f6b04e39244fe8d85d0b0b8e67b81062e654d0007eb1903503680bf727965"
    )
    expected_scale1b_manifest_sha256: str = (
        "275f7373f4f5de9b0a1f12c2b5585bc7ef91e88d11dc2b61b387326798e6e730"
    )
    expected_scale1b_probe_candidates_sha256: str = (
        "af405b7dafae84d27e92d3654d173eef29f268b68ad241c9934c6920fcd99126"
    )
    expected_scale1b_probe_manifest_sha256: str = (
        "5701462be2ebe30b713723bd09bae2a70a980f8cc47eb6d0772884b0bea1fa13"
    )
    expected_scale1b_scoring_plan_sha256: str = (
        "7d0f2aa7353f68d4c78736fafc40aa1ffe46269af7e4f5bfe8d86ee718898cea"
    )
    expected_scale1b_scoring_protocol_sha256: str = (
        "e5ede85840192ae2191ba2c4523a00c1464f11759492a87e77ae2c2a7595039a"
    )
    expected_scale1b_protocol_manifest_sha256: str = (
        "485a22a89d0fc532cdee72c09ed7faa36c5fdc54079aa2a971479d41c53af918"
    )


@dataclass(frozen=True)
class CapacitySnapshot:
    admitted_clusters: tuple[str, ...]
    pending_resolved_clusters: tuple[str, ...]
    n_nr_admitted: int
    n_pending_proteins: int
    n_pending_cluster_resolved: int
    n_pending_cluster_unresolved: int
    n_unique_pending_resolved_clusters: int
    n_pending_clusters_already_represented: int
    n_pending_new_clusters: int
    n_nr_upper: int
    pending_upper_bound_exact: bool
    capacity_decision: str


@dataclass(frozen=True)
class RedundancyDiversityInputs:
    census: pd.DataFrame
    common_masks: pd.DataFrame
    admitted_metadata: pd.DataFrame
    pending_metadata: pd.DataFrame
    scale1b_panel: pd.DataFrame
    input_artifacts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RedundancyDiversityResult:
    status: str
    admitted_census: pd.DataFrame
    nonredundant_capacity: pd.DataFrame
    pending_capacity: pd.DataFrame
    domain_annotations: pd.DataFrame
    capacity_snapshot: CapacitySnapshot
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_input_path",
            f"Invalid logical path: {logical_ref}",
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            path=logical_ref,
            expected_sha256=expected,
            observed_sha256=observed,
        )
    return path, {"path": logical_ref, "sha256": observed, "label": label}


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def _read_tsv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not isinstance(value, dict):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must be a JSON object",
        )
    return value


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} is missing fields: {missing}",
        )


def _bindings(config: RedundancyDiversityConfig) -> tuple[tuple[str, str, str], ...]:
    return (
        (config.scale1a2_census_ref, config.expected_scale1a2_census_sha256, "Scale-1A2 census"),
        (config.scale1a2_masks_ref, config.expected_scale1a2_masks_sha256, "Scale-1A2 common masks"),
        (config.scale1a2_summary_ref, config.expected_scale1a2_summary_sha256, "Scale-1A2 summary"),
        (config.scale1a2_ledger_ref, config.expected_scale1a2_ledger_sha256, "Scale-1A2 acquisition ledger"),
        (config.inventory_tsv_ref, config.expected_inventory_tsv_sha256, "candidate inventory TSV"),
        (config.inventory_json_ref, config.expected_inventory_json_sha256, "candidate inventory JSON"),
        (config.discovery_ref, config.expected_discovery_sha256, "canonical discovery source"),
        (config.cluster_implementation_ref, config.expected_cluster_implementation_sha256, "frozen RCSB cluster implementation"),
        (config.cohort_implementation_ref, config.expected_cohort_implementation_sha256, "frozen Scale-1B cohort implementation"),
        (config.scale1b_panel_ref, config.expected_scale1b_panel_sha256, "Scale-1B-v1 scoring panel"),
        (config.scale1b_primary_ref, config.expected_scale1b_primary_sha256, "Scale-1B-v1 primary core"),
        (config.scale1b_clusters_ref, config.expected_scale1b_clusters_sha256, "Scale-1B-v1 redundancy clusters"),
        (config.scale1b_summary_ref, config.expected_scale1b_summary_sha256, "Scale-1B-v1 cohort summary"),
        (config.scale1b_manifest_ref, config.expected_scale1b_manifest_sha256, "Scale-1B-v1 cohort manifest"),
        (config.scale1b_probe_candidates_ref, config.expected_scale1b_probe_candidates_sha256, "Scale-1B-v1 fixed probes"),
        (config.scale1b_probe_manifest_ref, config.expected_scale1b_probe_manifest_sha256, "Scale-1B-v1 probe manifest"),
        (config.scale1b_scoring_plan_ref, config.expected_scale1b_scoring_plan_sha256, "Scale-1B-v1 scoring plan"),
        (config.scale1b_scoring_protocol_ref, config.expected_scale1b_scoring_protocol_sha256, "Scale-1B-v1 scoring protocol"),
        (config.scale1b_protocol_manifest_ref, config.expected_scale1b_protocol_manifest_sha256, "Scale-1B-v1 protocol manifest"),
    )


def _build_metadata(
    census: pd.DataFrame,
    inventory: pd.DataFrame,
    discovery: pd.DataFrame,
    scale1b_panel: pd.DataFrame,
) -> pd.DataFrame:
    inventory_fields = [
        "candidate_index",
        "canonical_source_row",
        "polymer_entity_id",
        "pair_id",
        "PDB",
        "chain",
        "UniProt",
        "sequence_cluster",
        "protein_family_if_available",
        "canonical_uniprot_length",
        "sampling_stratum_prior",
    ]
    _require_columns(inventory, set(inventory_fields), "candidate inventory")
    base = census.merge(
        inventory[inventory_fields],
        left_on="candidate_id",
        right_on="pair_id",
        how="left",
        validate="one_to_one",
    )
    discovery_rows = discovery.reset_index().rename(
        columns={
            "index": "discovery_source_row",
            "polymer_entity_id": "discovery_polymer_entity_id",
            "pdb_id": "discovery_pdb_id",
            "chain_id": "discovery_chain_id",
            "uniprot_id": "discovery_uniprot_id",
            "sequence_cluster": "discovery_sequence_cluster",
            "length": "discovery_sequence_length",
            "experimental_method": "discovery_experimental_method",
            "resolution": "discovery_resolution",
            "organism": "discovery_organism",
            "taxonomy_id": "discovery_taxonomy_id",
        }
    )
    discovery_fields = [
        "discovery_source_row",
        "discovery_polymer_entity_id",
        "discovery_pdb_id",
        "discovery_chain_id",
        "discovery_uniprot_id",
        "discovery_sequence_cluster",
        "discovery_sequence_length",
        "discovery_experimental_method",
        "discovery_resolution",
        "discovery_organism",
        "discovery_taxonomy_id",
    ]
    _require_columns(discovery_rows, set(discovery_fields), "canonical discovery")
    base = base.merge(
        discovery_rows[discovery_fields],
        left_on="canonical_source_row",
        right_on="discovery_source_row",
        how="left",
        validate="many_to_one",
    )
    identity_ok = (
        base["sampling_frame_index"].eq(base["candidate_index"])
        & base["candidate_id"].eq(base["pair_id"])
        & base["canonical_accession"].eq(base["UniProt"])
        & base["pdb_id"].str.lower().eq(base["PDB"].str.lower())
        & base["pdb_chain"].eq(base["chain"])
        & base["canonical_source_row"].eq(base["discovery_source_row"])
        & base["polymer_entity_id"].eq(base["discovery_polymer_entity_id"])
        & base["pdb_id"].str.lower().eq(base["discovery_pdb_id"].str.lower())
        & base["pdb_chain"].eq(base["discovery_chain_id"])
        & base["canonical_accession"].eq(base["discovery_uniprot_id"])
        & base["sequence_cluster"].eq(base["discovery_sequence_cluster"])
    )
    if len(base) != 213 or not identity_ok.all():
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_identity_binding_mismatch",
            "Scale-1A2, inventory, and discovery identities do not agree",
        )
    base["redundancy_metadata_resolved"] = base["sequence_cluster"].map(
        lambda value: isinstance(value, str) and bool(_CLUSTER_PATTERN.fullmatch(value))
    )
    base["scale1b_v1_member"] = base["pair_id"].isin(set(scale1b_panel["pair_id"]))
    base["stage0_overlap"] = False
    base["common_mask_fraction"] = pd.to_numeric(
        base["common_mask_fraction"], errors="coerce"
    )
    return base.sort_values("sampling_frame_index", kind="stable").reset_index(
        drop=True
    )


def validate_redundancy_diversity_inputs(
    paths: ProjectPaths, config: RedundancyDiversityConfig
) -> RedundancyDiversityInputs:
    """Validate the complete frame and frozen Scale-1B-v1 trust roots."""
    resolved: dict[str, Path] = {}
    artifacts: list[dict[str, Any]] = []
    for logical_ref, expected, label in _bindings(config):
        path, record = _require_hash(paths, logical_ref, expected, label)
        resolved[logical_ref] = path
        artifacts.append(record)

    census = _read_parquet(resolved[config.scale1a2_census_ref], "Scale-1A2 census")
    common_masks = _read_parquet(
        resolved[config.scale1a2_masks_ref], "Scale-1A2 common masks"
    )
    summary = _read_json(resolved[config.scale1a2_summary_ref], "Scale-1A2 summary")
    inventory = _read_tsv(resolved[config.inventory_tsv_ref], "candidate inventory")
    discovery = _read_parquet(resolved[config.discovery_ref], "canonical discovery")
    scale1b_panel = _read_parquet(
        resolved[config.scale1b_panel_ref], "Scale-1B-v1 scoring panel"
    )
    scale1b_primary = _read_parquet(
        resolved[config.scale1b_primary_ref], "Scale-1B-v1 primary core"
    )
    scale1b_manifest = _read_json(
        resolved[config.scale1b_manifest_ref], "Scale-1B-v1 cohort manifest"
    )
    protocol_manifest = _read_json(
        resolved[config.scale1b_protocol_manifest_ref],
        "Scale-1B-v1 protocol manifest",
    )

    _require_columns(
        census,
        {
            "sampling_frame_index",
            "candidate_id",
            "canonical_accession",
            "pdb_id",
            "pdb_chain",
            "formal_admission_status",
            "common_mask_count",
            "common_mask_fraction",
        },
        "Scale-1A2 census",
    )
    expected_counts = {
        "FORMALLY_ADMITTED": 135,
        "PENDING_HUMAN_VARIANT_REVIEW": 51,
        "MAPPING_FAIL": 12,
        "IDENTITY_CONTRACT_FAIL": 9,
        "PROVENANCE_FAIL": 6,
    }
    observed_counts = census["formal_admission_status"].value_counts().to_dict()
    if (
        len(census) != 213
        or census["candidate_id"].duplicated().any()
        or census["sampling_frame_index"].tolist() != list(range(1, 214))
        or observed_counts != expected_counts
        or summary.get("formal_admission_status_counts") != expected_counts
        or summary.get("n_source_frame") != 213
        or summary.get("forbidden_execution", {}).get("proteinmpnn_forward_executions") != 0
        or summary.get("forbidden_execution", {}).get("scale1b_v1_modified") is not False
    ):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "scale1a2_contract_mismatch",
            "Frozen Scale-1A2 frame, order, or admission partition differs",
        )

    _require_columns(
        common_masks,
        {"candidate_id", "canonical_position", "common_mask"},
        "Scale-1A2 common masks",
    )
    if (
        common_masks[["candidate_id", "canonical_position"]].duplicated().any()
        or set(common_masks["candidate_id"]) != set(census["candidate_id"])
    ):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "common_mask_identity_mismatch",
            "Scale-1A2 common-mask identity/key set differs",
        )
    canonical_axes = common_masks.groupby("candidate_id", sort=False)[
        "canonical_position"
    ].agg(["min", "max", "count"])
    if not (
        canonical_axes["min"].eq(1)
        & canonical_axes["max"].eq(canonical_axes["count"])
    ).all():
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "canonical_position_axis_mismatch",
            "Scale-1A2 canonical position axes must be contiguous 1..length",
        )
    admitted_ids = set(
        census.loc[
            census["formal_admission_status"].eq("FORMALLY_ADMITTED"),
            "candidate_id",
        ]
    )
    observed_common = (
        common_masks.loc[common_masks["candidate_id"].isin(admitted_ids)]
        .assign(
            common_mask=lambda table: table["common_mask"]
            .astype("boolean")
            .fillna(False)
            .astype(int)
        )
        .groupby("candidate_id", sort=False)["common_mask"]
        .sum()
    )
    expected_common = census.set_index("candidate_id").loc[
        sorted(admitted_ids), "common_mask_count"
    ]
    if not observed_common.reindex(expected_common.index).astype(int).eq(
        expected_common.astype(int)
    ).all():
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "common_mask_count_mismatch",
            "Scale-1A2 admitted common-mask counts do not reconcile",
        )

    if (
        len(scale1b_panel) != 52
        or len(scale1b_primary) != 46
        or scale1b_panel["pair_id"].duplicated().any()
        or not set(scale1b_panel["pair_id"]).issubset(admitted_ids)
        or scale1b_manifest.get("freeze_status")
        != "SCALE1B_COHORT_FREEZE_COMPLETE"
        or scale1b_manifest.get("SCALE1_OUTCOMES_NOT_OBSERVED") is not True
        or protocol_manifest.get("protocol_status")
        != "SCALE1B_PROTOCOL_FREEZE_COMPLETE"
        or protocol_manifest.get("SCALE1B_SCORING_NOT_STARTED") is not True
    ):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "scale1b_v1_contract_mismatch",
            "Frozen Scale-1B-v1 cohort or protocol contract differs",
        )

    metadata = _build_metadata(census, inventory, discovery, scale1b_panel)
    metadata["sequence_length"] = metadata["pair_id"].map(
        canonical_axes["count"]
    ).astype("Int64")
    admitted = metadata.loc[
        metadata["formal_admission_status"].eq("FORMALLY_ADMITTED")
    ].copy()
    pending = metadata.loc[
        metadata["formal_admission_status"].eq("PENDING_HUMAN_VARIANT_REVIEW")
    ].copy()
    if (
        len(admitted) != 135
        or len(pending) != 51
        or not admitted["redundancy_metadata_resolved"].all()
        or admitted["sequence_length"].isna().any()
    ):
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_BINDING,
            "admitted_redundancy_binding_incomplete",
            "All 135 admitted proteins require authoritative cluster bindings",
        )
    implementation_path = Path(__file__).resolve()
    artifacts.append(
        {
            "path": _logical(paths, implementation_path),
            "sha256": sha256_file(implementation_path),
            "label": "effective Scale-1A3 scientific implementation",
        }
    )
    return RedundancyDiversityInputs(
        census=census.reset_index(drop=True),
        common_masks=common_masks.reset_index(drop=True),
        admitted_metadata=admitted.reset_index(drop=True),
        pending_metadata=pending.reset_index(drop=True),
        scale1b_panel=scale1b_panel.reset_index(drop=True),
        input_artifacts=tuple(artifacts),
    )


def _normalize_cluster(value: Any, *, allow_unresolved: bool) -> str | None:
    if value is None or pd.isna(value):
        if allow_unresolved:
            return None
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_BINDING,
            "unresolved_admitted_cluster",
            "An admitted protein lacks an authoritative 30% cluster binding",
        )
    cluster = str(value)
    if not _CLUSTER_PATTERN.fullmatch(cluster):
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_BINDING,
            "invalid_cluster_identifier",
            f"Invalid authoritative cluster identifier: {cluster}",
        )
    return cluster


def _decision_from_counts(
    *,
    n_nr_admitted: int,
    n_nr_upper: int,
    pending_upper_bound_exact: bool,
) -> str:
    if n_nr_admitted >= 100:
        return CORE_SCALE_REACHED
    if not pending_upper_bound_exact:
        return BLOCKED_REDUNDANCY_BINDING
    if n_nr_upper >= 100:
        return PENDING_VARIANT_REVIEW_PRIORITY
    return ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED


def compute_capacity_snapshot(
    admitted_clusters: Iterable[Any], pending_bindings: Iterable[Any]
) -> CapacitySnapshot:
    """Compute cluster-set capacity without accepting diversity metadata."""
    admitted = [
        _normalize_cluster(value, allow_unresolved=False)
        for value in admitted_clusters
    ]
    admitted_set = {value for value in admitted if value is not None}
    if not admitted_set:
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_BINDING,
            "empty_admitted_cluster_set",
            "The admitted cluster set cannot be empty",
        )
    pending_values = list(pending_bindings)
    resolved_pending: list[str] = []
    unresolved = 0
    for value in pending_values:
        normalized = _normalize_cluster(value, allow_unresolved=True)
        if normalized is None:
            unresolved += 1
        else:
            resolved_pending.append(normalized)
    pending_set = set(resolved_pending)
    overlap = pending_set & admitted_set
    new_clusters = pending_set - admitted_set
    exact = unresolved == 0
    n_nr_upper = len(admitted_set) + len(new_clusters)
    decision = _decision_from_counts(
        n_nr_admitted=len(admitted_set),
        n_nr_upper=n_nr_upper,
        pending_upper_bound_exact=exact,
    )
    return CapacitySnapshot(
        admitted_clusters=tuple(sorted(admitted_set)),
        pending_resolved_clusters=tuple(sorted(pending_set)),
        n_nr_admitted=len(admitted_set),
        n_pending_proteins=len(pending_values),
        n_pending_cluster_resolved=len(resolved_pending),
        n_pending_cluster_unresolved=unresolved,
        n_unique_pending_resolved_clusters=len(pending_set),
        n_pending_clusters_already_represented=len(overlap),
        n_pending_new_clusters=len(new_clusters),
        n_nr_upper=n_nr_upper,
        pending_upper_bound_exact=exact,
        capacity_decision=decision,
    )


def validate_decision_invariance(pre_decision: str, post_decision: str) -> None:
    """Reject any diversity-associated capacity decision drift."""
    if pre_decision != post_decision:
        raise RedundancyDiversityError(
            BLOCKED_DECISION_INVARIANT,
            "decision_invariance_failure",
            "Optional diversity metadata changed the frozen capacity decision",
            capacity_decision_pre_diversity=pre_decision,
            capacity_decision_post_diversity_validation=post_decision,
        )


def validate_diversity_independence(
    snapshot: CapacitySnapshot, diversity_status: str
) -> dict[str, Any]:
    """Revalidate the pure capacity result after descriptive enrichment."""
    if diversity_status not in {
        CATH_METADATA_COMPLETE,
        CATH_METADATA_PARTIAL,
        CATH_METADATA_UNAVAILABLE,
    }:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_diversity_metadata_status",
            f"Unsupported diversity metadata status: {diversity_status}",
        )
    pre_decision = snapshot.capacity_decision
    post_decision = _decision_from_counts(
        n_nr_admitted=snapshot.n_nr_admitted,
        n_nr_upper=snapshot.n_nr_upper,
        pending_upper_bound_exact=snapshot.pending_upper_bound_exact,
    )
    validate_decision_invariance(pre_decision, post_decision)
    return {
        "capacity_decision_pre_diversity": pre_decision,
        "capacity_decision_post_diversity_validation": post_decision,
        "decision_independent_of_diversity_metadata": True,
        "diversity_metadata_status": diversity_status,
    }


def assign_redundancy_roles(metadata: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic pre-outcome roles for a cohort union."""
    if metadata["sequence_cluster"].isna().any():
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_DEFINITION,
            "missing_frozen_redundancy_assignment",
            "At least one admitted protein lacks a frozen sequence cluster",
        )
    if not metadata["sequence_cluster"].astype(str).str.match(r"^30:.+").all():
        raise RedundancyDiversityError(
            BLOCKED_REDUNDANCY_DEFINITION,
            "invalid_frozen_redundancy_assignment",
            "Frozen sequence clusters must use the canonical 30:<cluster_id> convention",
        )
    required = {
        "pair_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "sequence_cluster",
        "redundancy_metadata_resolved",
        "common_mask_count",
        "common_mask_fraction",
        "sampling_frame_index",
    }
    if missing := sorted(required - set(metadata.columns)):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "representative_metadata_missing",
            f"Representative metadata fields are missing: {missing}",
        )
    output = metadata.copy()
    output["redundancy_cluster_id"] = output["sequence_cluster"].astype(str)
    ranked = output.sort_values(
        [
            "redundancy_cluster_id",
            "redundancy_metadata_resolved",
            "common_mask_count",
            "common_mask_fraction",
            "sampling_frame_index",
            "canonical_accession",
            "pdb_id",
            "pdb_chain",
            "pair_id",
        ],
        ascending=[True, False, False, False, True, True, True, True, True],
        kind="stable",
    )
    representative_ids = set(
        ranked.groupby("redundancy_cluster_id", sort=False).head(1)["pair_id"]
    )
    output["is_primary_representative"] = output["pair_id"].isin(
        representative_ids
    )
    output["cohort_role"] = output["is_primary_representative"].map(
        {True: "PRIMARY_CORE", False: "SECONDARY_RELATED_REPLICATION"}
    )
    output["analysis_origin"] = output["stage0_overlap"].map(
        {True: "STAGE0_OVERLAP", False: "SCALE1_NEW_PROTEIN"}
    )
    ranked["representative_rank_within_cluster"] = (
        ranked.groupby("redundancy_cluster_id", sort=False).cumcount() + 1
    )
    rank_by_pair = ranked.set_index("pair_id")["representative_rank_within_cluster"]
    output["representative_rank_within_cluster"] = output["pair_id"].map(
        rank_by_pair
    )
    output["redundancy_cluster_size"] = output.groupby(
        "redundancy_cluster_id"
    )["pair_id"].transform("size")
    return output.sort_values("sampling_frame_index", kind="stable").reset_index(
        drop=True
    )


def assign_admitted_representatives(metadata: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact frozen Scale-1B-v1 representative ordering."""
    required = {
        "pair_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "sequence_cluster",
        "redundancy_metadata_resolved",
        "common_mask_count",
        "common_mask_fraction",
        "sampling_frame_index",
    }
    _require_columns(metadata, required, "admitted representative metadata")
    prepared = metadata.copy()
    if "stage0_overlap" not in prepared:
        prepared["stage0_overlap"] = False
    assigned = assign_redundancy_roles(prepared)
    assigned = assigned.rename(
        columns={
            "is_primary_representative": "is_cluster_representative",
            "cohort_role": "legacy_cohort_role",
        }
    )
    return assigned.sort_values("sampling_frame_index", kind="stable").reset_index(
        drop=True
    )


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        return {key: None for key in ("min", "q25", "median", "q75", "max")}
    quantiles = np.quantile(numeric, [0, 0.25, 0.5, 0.75, 1], method="linear")
    result: dict[str, float | int | None] = {}
    for key, value in zip(("min", "q25", "median", "q75", "max"), quantiles):
        result[key] = int(value) if float(value).is_integer() else float(value)
    return result


def _counts(values: pd.Series) -> dict[str, int]:
    counts = values.fillna("UNAVAILABLE").astype(str).value_counts()
    return {
        str(key): int(value)
        for key, value in sorted(
            counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )
    }


def _next_task(decision: str) -> str | None:
    return {
        CORE_SCALE_REACHED: "SCALE1B_V2_PRECONFIRMATORY_COHORT_FREEZE",
        PENDING_VARIANT_REVIEW_PRIORITY: "SCALE1A4_PENDING_VARIANT_HUMAN_REVIEW",
        ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED: (
            "SCALE1_SOURCE_FRAME_EXPANSION_DESIGN"
        ),
    }.get(decision)


def _capacity_statistics(snapshot: CapacitySnapshot) -> dict[str, int | bool]:
    return {
        "n_nr_admitted": snapshot.n_nr_admitted,
        "n_pending_proteins": snapshot.n_pending_proteins,
        "n_pending_cluster_resolved": snapshot.n_pending_cluster_resolved,
        "n_pending_cluster_unresolved": snapshot.n_pending_cluster_unresolved,
        "n_unique_pending_resolved_clusters": (
            snapshot.n_unique_pending_resolved_clusters
        ),
        "n_pending_clusters_already_represented": (
            snapshot.n_pending_clusters_already_represented
        ),
        "n_pending_new_clusters": snapshot.n_pending_new_clusters,
        "n_nr_upper": snapshot.n_nr_upper,
        "pending_upper_bound_exact": snapshot.pending_upper_bound_exact,
    }


def build_redundancy_diversity_census(inputs: RedundancyDiversityInputs) -> RedundancyDiversityResult:
    """Build the census from frozen cluster assignments and local metadata."""
    assigned = assign_admitted_representatives(inputs.admitted_metadata)
    pending = inputs.pending_metadata.sort_values(
        "sampling_frame_index", kind="stable"
    ).reset_index(drop=True)
    snapshot = compute_capacity_snapshot(
        assigned["sequence_cluster"],
        pending["sequence_cluster"].where(
            pending["redundancy_metadata_resolved"], None
        ),
    )

    admitted_columns = [
        "sampling_frame_index",
        "candidate_id",
        "pair_id",
        "polymer_entity_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "formal_admission_status",
        "sequence_cluster",
        "redundancy_cluster_id",
        "redundancy_metadata_resolved",
        "redundancy_cluster_size",
        "representative_rank_within_cluster",
        "is_cluster_representative",
        "common_mask_count",
        "common_mask_fraction",
        "sequence_length",
        "discovery_organism",
        "discovery_taxonomy_id",
        "discovery_experimental_method",
        "discovery_resolution",
        "scale1b_v1_member",
    ]
    admitted_census = assigned[admitted_columns].copy()
    admitted_census["common_mask_count"] = admitted_census[
        "common_mask_count"
    ].astype(int)
    admitted_census["sequence_length"] = admitted_census["sequence_length"].astype(
        int
    )

    representatives = admitted_census.loc[
        admitted_census["is_cluster_representative"]
    ].copy()
    capacity_columns = [
        "sequence_cluster",
        "redundancy_cluster_size",
        "pair_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "sampling_frame_index",
        "common_mask_count",
        "common_mask_fraction",
        "sequence_length",
        "discovery_organism",
        "discovery_taxonomy_id",
        "discovery_experimental_method",
        "discovery_resolution",
    ]
    nonredundant_capacity = representatives[capacity_columns].rename(
        columns={
            "sequence_cluster": "redundancy_cluster_id",
            "pair_id": "representative_pair_id",
            "canonical_accession": "representative_canonical_accession",
            "pdb_id": "representative_pdb_id",
            "pdb_chain": "representative_pdb_chain",
            "sampling_frame_index": "representative_sampling_frame_index",
            "common_mask_count": "representative_common_mask_count",
            "common_mask_fraction": "representative_common_mask_fraction",
            "sequence_length": "representative_sequence_length",
            "discovery_organism": "representative_organism",
            "discovery_taxonomy_id": "representative_taxonomy_id",
            "discovery_experimental_method": "representative_experimental_method",
            "discovery_resolution": "representative_resolution",
        }
    )
    nonredundant_capacity = nonredundant_capacity.sort_values(
        "redundancy_cluster_id", kind="stable"
    ).reset_index(drop=True)

    admitted_clusters = set(snapshot.admitted_clusters)
    pending_capacity = pending[
        [
            "sampling_frame_index",
            "candidate_id",
            "pair_id",
            "polymer_entity_id",
            "canonical_accession",
            "pdb_id",
            "pdb_chain",
            "formal_admission_status",
            "sequence_cluster",
            "redundancy_metadata_resolved",
            "sequence_length",
            "discovery_organism",
            "discovery_taxonomy_id",
        ]
    ].copy()
    pending_capacity = pending_capacity.rename(
        columns={"sequence_cluster": "redundancy_cluster_id"}
    )
    pending_capacity["cluster_already_represented"] = pending_capacity[
        "redundancy_cluster_id"
    ].isin(admitted_clusters) & pending_capacity["redundancy_metadata_resolved"]
    pending_capacity["pending_new_cluster"] = (
        pending_capacity["redundancy_metadata_resolved"]
        & ~pending_capacity["cluster_already_represented"]
    )
    pending_capacity["contributes_unique_capacity_unit"] = False
    first_new = (
        pending_capacity.loc[pending_capacity["pending_new_cluster"]]
        .sort_values("sampling_frame_index", kind="stable")
        .groupby("redundancy_cluster_id", sort=False)
        .head(1)
        .index
    )
    pending_capacity.loc[first_new, "contributes_unique_capacity_unit"] = True
    pending_capacity["sequence_length"] = pending_capacity["sequence_length"].astype(
        "Int64"
    )

    domain_annotations = nonredundant_capacity[
        [
            "redundancy_cluster_id",
            "representative_pair_id",
            "representative_canonical_accession",
            "representative_sequence_length",
            "representative_organism",
            "representative_taxonomy_id",
            "representative_experimental_method",
            "representative_resolution",
        ]
    ].copy()
    domain_annotations["cath_annotation_status"] = CATH_METADATA_UNAVAILABLE
    domain_annotations["cath_domain_ids"] = pd.Series(
        [pd.NA] * len(domain_annotations), dtype="string"
    )
    domain_annotations["domain_count"] = pd.Series(
        [pd.NA] * len(domain_annotations), dtype="Int64"
    )

    decision_audit = validate_diversity_independence(
        snapshot, CATH_METADATA_UNAVAILABLE
    )
    cluster_sizes = admitted_census.groupby("redundancy_cluster_id").size()
    cluster_size_distribution = cluster_sizes.value_counts().sort_index()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scale1a3_status": snapshot.capacity_decision,
        "n_source_frame": 213,
        "n_formally_admitted": len(admitted_census),
        "n_pending_human_variant_review": len(pending_capacity),
        "n_scientific_failures": 27,
        **_capacity_statistics(snapshot),
        **decision_audit,
        "cluster_statistics": {
            "singleton_clusters": int(cluster_sizes.eq(1).sum()),
            "multi_member_clusters": int(cluster_sizes.gt(1).sum()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_size_distribution": {
                str(size): int(count)
                for size, count in cluster_size_distribution.items()
            },
        },
        "cath_metadata_status": CATH_METADATA_UNAVAILABLE,
        "diversity_metadata_coverage": {
            "representative_count": len(domain_annotations),
            "canonical_length_available": int(
                domain_annotations["representative_sequence_length"].notna().sum()
            ),
            "organism_available": int(
                domain_annotations["representative_organism"].notna().sum()
            ),
            "taxonomy_available": int(
                domain_annotations["representative_taxonomy_id"].notna().sum()
            ),
            "cath_available": 0,
            "domain_count_available": 0,
        },
        "diversity_summaries": {
            "representative_sequence_length": _distribution(
                domain_annotations["representative_sequence_length"]
            ),
            "representative_organism_counts": _counts(
                domain_annotations["representative_organism"]
            ),
            "representative_taxonomy_counts": _counts(
                domain_annotations["representative_taxonomy_id"]
            ),
            "representative_experimental_method_counts": _counts(
                domain_annotations["representative_experimental_method"]
            ),
        },
        "next_task": _next_task(snapshot.capacity_decision),
        "next_task_started": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scale1a3_status": snapshot.capacity_decision,
        "upstream_artifacts": list(inputs.input_artifacts),
        "implementation_provenance": next(
            record
            for record in inputs.input_artifacts
            if record["label"] == "effective Scale-1A3 scientific implementation"
        ),
        "redundancy_convention": {
            "provider": "RCSB rcsb_cluster_membership",
            "definition": "sequence_identity_cluster",
            "identity_threshold_percent": 30,
            "cluster_id_serialization": "30:<RCSB cluster_id>",
            "assignment_source": (
                "artifacts/dataset/reports/census/candidate_inventory_v1.tsv:"
                "sequence_cluster"
            ),
            "assignments_recomputed": False,
        },
        "representative_rule": {
            "source": "frozen Scale-1B-v1 representative rule",
            "priority": [
                "resolved redundancy metadata",
                "larger common_mask_count",
                "larger common_mask_fraction",
                "earlier sampling_frame_index",
                "canonical_accession/pdb_id/pdb_chain/pair_id",
            ],
        },
        "capacity_statistics": _capacity_statistics(snapshot),
        **decision_audit,
        "capacity_dependency_graph": {
            "inputs": [
                "frozen formal admission status",
                "authoritative frozen 30% cluster binding",
            ],
            "optional_diversity_inputs": [],
            "reverse_edge_from_diversity": False,
        },
        "diversity_metadata": {
            "status": CATH_METADATA_UNAVAILABLE,
            "network_retrieval_performed": False,
            "capacity_decision_dependency": False,
            "domain_annotations_are_set_valued": True,
        },
        "outputs": {},
        "ADMISSION_CONTRACT_UNCHANGED": True,
        "SEQUENCE_REDUNDANCY_THRESHOLD": 0.30,
        "SCALE1B_V1_IMMUTABLE": True,
        "SCALE1_SCORING_NOT_STARTED": True,
        "NO_OUTCOME_DEPENDENT_SELECTION": True,
        "PROTEINMPNN_EXECUTED": False,
        "SCALE1B_V2_NOT_STARTED": True,
        "scope_confirmations": {
            "network_access_used": False,
            "downloads_performed": False,
            "cluster_assignments_recomputed": False,
            "admission_changed": False,
            "scale1_scoring_performed": False,
        },
        "manifest_self_hash_policy": (
            "reported_by_cli_after_write_to_avoid_recursive_self_hash"
        ),
    }
    result = RedundancyDiversityResult(
        status=snapshot.capacity_decision,
        admitted_census=admitted_census.reset_index(drop=True),
        nonredundant_capacity=nonredundant_capacity,
        pending_capacity=pending_capacity.reset_index(drop=True),
        domain_annotations=domain_annotations,
        capacity_snapshot=snapshot,
        summary=summary,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )
    _validate_result(result)
    return result


def _validate_result(result: RedundancyDiversityResult) -> None:
    if (
        len(result.admitted_census) != 135
        or len(result.pending_capacity) != 51
        or len(result.nonredundant_capacity) != result.capacity_snapshot.n_nr_admitted
        or len(result.domain_annotations) != result.capacity_snapshot.n_nr_admitted
        or result.admitted_census["pair_id"].duplicated().any()
        or result.pending_capacity["pair_id"].duplicated().any()
        or result.nonredundant_capacity["redundancy_cluster_id"].duplicated().any()
        or int(result.admitted_census["is_cluster_representative"].sum())
        != result.capacity_snapshot.n_nr_admitted
        or int(result.pending_capacity["contributes_unique_capacity_unit"].sum())
        != result.capacity_snapshot.n_pending_new_clusters
    ):
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "result_cardinality_mismatch",
            "Scale-1A3 structured result cardinalities do not reconcile",
        )
    expected_clusters = set(result.capacity_snapshot.admitted_clusters)
    if (
        set(result.admitted_census["redundancy_cluster_id"]) != expected_clusters
        or set(result.nonredundant_capacity["redundancy_cluster_id"])
        != expected_clusters
        or result.summary["scale1a3_status"] != result.status
        or result.summary["capacity_decision_pre_diversity"] != result.status
        or result.summary["capacity_decision_post_diversity_validation"]
        != result.status
        or result.manifest["capacity_decision_pre_diversity"] != result.status
        or result.manifest["capacity_decision_post_diversity_validation"]
        != result.status
    ):
        raise RedundancyDiversityError(
            BLOCKED_DECISION_INVARIANT,
            "result_decision_or_cluster_mismatch",
            "Scale-1A3 result keys or decisions do not reconcile",
        )
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
        result.admitted_census,
        result.nonredundant_capacity,
        result.pending_capacity,
        result.domain_annotations,
    ):
        if forbidden.intersection(column.lower() for column in frame.columns):
            raise RedundancyDiversityError(
                BLOCKED_INPUT_INTEGRITY,
                "outcome_dependent_field_present",
                "A Scale-1 outcome field entered the pre-outcome census",
            )


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_output_path",
            f"Output is outside project roots: {path.name}",
        ) from exc


def _render_json(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_scale1a3_json",
            "Scale-1A3 JSON cannot be serialized",
        ) from exc


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise RedundancyDiversityError(
            BLOCKED_INPUT_INTEGRITY,
            "immutable_scale1a3_artifact_conflict",
            f"Immutable output differs: {path.name}",
        )
    atomic_write_new_bytes(path, payload)
    return "created"


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
            if path.stat().st_size == temporary.stat().st_size and sha256_file(
                path
            ) == sha256_file(temporary):
                return "reused_identical"
            raise RedundancyDiversityError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_scale1a3_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RedundancyDiversityError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_scale1a3_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(result: RedundancyDiversityResult, paths: ProjectPaths) -> None:
    for record in result.input_artifacts:
        path = _resolve(paths, str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RedundancyDiversityError(
                BLOCKED_INPUT_INTEGRITY,
                "upstream_input_changed",
                f"Frozen input changed: {record['label']}",
            )


def materialize_redundancy_diversity_census(
    result: RedundancyDiversityResult,
    paths: ProjectPaths,
    *,
    config: RedundancyDiversityConfig | None = None,
) -> dict[str, Any]:
    """Materialize all tables and summary, then write the manifest last."""
    config = config or RedundancyDiversityConfig()
    _validate_result(result)
    _rehash_inputs(result, paths)
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "admitted_redundancy_census": (
            output_root / "scale1a3_admitted_redundancy_census.parquet"
        ),
        "nonredundant_capacity": (
            output_root / "scale1a3_nonredundant_capacity.parquet"
        ),
        "pending_cluster_capacity": (
            output_root / "scale1a3_pending_cluster_capacity.parquet"
        ),
        "domain_annotations": (
            output_root / "scale1a3_domain_annotations.parquet"
        ),
        "diversity_summary": output_root / "scale1a3_diversity_summary.json",
        "manifest": output_root / "scale1a3_manifest.json",
    }
    frames = {
        "admitted_redundancy_census": result.admitted_census,
        "nonredundant_capacity": result.nonredundant_capacity,
        "pending_cluster_capacity": result.pending_capacity,
        "domain_annotations": result.domain_annotations,
    }
    write_status = {
        name: _write_immutable_parquet(targets[name], frame)
        for name, frame in frames.items()
    }
    write_status["diversity_summary"] = _write_immutable_bytes(
        targets["diversity_summary"], _render_json(result.summary)
    )
    outputs: dict[str, dict[str, Any]] = {
        name: {
            "path": _logical(paths, targets[name]),
            "rows": len(frame),
            "sha256": sha256_file(targets[name]),
        }
        for name, frame in frames.items()
    }
    outputs["diversity_summary"] = {
        "path": _logical(paths, targets["diversity_summary"]),
        "sha256": sha256_file(targets["diversity_summary"]),
    }
    manifest = {**result.manifest, "outputs": outputs}
    write_status["manifest"] = _write_immutable_bytes(
        targets["manifest"], _render_json(manifest)
    )
    outputs["manifest"] = {
        "path": _logical(paths, targets["manifest"]),
        "sha256": sha256_file(targets["manifest"]),
    }
    return {
        "scale1a3_status": result.status,
        "write_status": write_status,
        "outputs": outputs,
        "next_task": _next_task(result.status),
        "next_task_started": False,
    }


def run_redundancy_diversity_census(
    paths: ProjectPaths, *, config: RedundancyDiversityConfig | None = None
) -> dict[str, Any]:
    """Validate, build, and immutably materialize Scale-1A3."""
    config = config or RedundancyDiversityConfig()
    inputs = validate_redundancy_diversity_inputs(paths, config)
    result = build_redundancy_diversity_census(inputs)
    return materialize_redundancy_diversity_census(result, paths, config=config)
