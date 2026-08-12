"""Offline pre-outcome Scale-1B scoring-panel and redundancy-core freeze."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths

from .models import DerivationError
from .policies.identity import extract_canonical_sequence

SCHEMA_VERSION = "dual-uq.scale1b-cohort-freeze.v1"
DEFAULT_OUTPUT_ROOT = "experiments/p2_design_baseline/scale1/scale1b_freeze"

FREEZE_COMPLETE = "SCALE1B_COHORT_FREEZE_COMPLETE"
BLOCKED_DIVERSITY_CAPACITY = "SCALE1B_BLOCKED_DIVERSITY_CAPACITY"
BLOCKED_REDUNDANCY_DEFINITION = "SCALE1B_BLOCKED_REDUNDANCY_DEFINITION"
BLOCKED_INPUT_INTEGRITY = "SCALE1B_BLOCKED_INPUT_INTEGRITY"


class Scale1BCohortFreezeError(RuntimeError):
    """A structured cohort-freeze blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class Scale1BCohortFreezeConfig:
    admission_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_census.parquet"
    )
    common_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/scale1_common_masks.parquet"
    )
    admission_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_summary.json"
    )
    admission_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_manifest.json"
    )
    planning_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/planning/"
        "scale1_power_planning_manifest.json"
    )
    planning_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/planning/"
        "scale1_power_planning_summary.json"
    )
    planning_feasibility_ref: str = (
        "experiments/p2_design_baseline/scale1/planning/"
        "scale1_feasibility_table.parquet"
    )
    planning_effects_ref: str = (
        "experiments/p2_design_baseline/scale1/planning/"
        "scale1_stage0_planning_effects.parquet"
    )
    planning_simulation_ref: str = (
        "experiments/p2_design_baseline/scale1/planning/"
        "scale1_power_simulation.parquet"
    )
    sampling_frame_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_audit.parquet"
    )
    sampling_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_manifest.json"
    )
    inventory_tsv_ref: str = (
        "artifacts/dataset/reports/census/candidate_inventory_v1.tsv"
    )
    inventory_json_ref: str = (
        "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    )
    discovery_ref: str = "data/processed/discovery/discovered_candidates.parquet"
    cluster_implementation_ref: str = "src/dual_uq/rcsb_discovery.py"
    output_root_ref: str = DEFAULT_OUTPUT_ROOT

    expected_admission_census_sha256: str = (
        "ca01e17f55637d0f4584aa7bbb8940e50ea4aaf25b9c7ee7f741bb51a4095a7f"
    )
    expected_common_masks_sha256: str = (
        "face3d6f211585d452f474b090adb618d121aada849a4cbd492e92e9b60a03b9"
    )
    expected_admission_summary_sha256: str = (
        "50afd84f6ac4a3104fc817015c89b9e1127b16bb8b5c8f688fb385cee02ecc0f"
    )
    expected_admission_manifest_sha256: str = (
        "54227ee11e67fdb6da12cbf9001b603875b1fd3c28d603608d384f0c50988223"
    )
    expected_planning_manifest_sha256: str = (
        "e59f3633e8e5ee6c58e399464ffeb8103d484dc04431a7e4cf0b8d5301a56c4e"
    )
    expected_planning_summary_sha256: str = (
        "37b99e417b9fbd4c43c6fca8bfeeb98e6aba6011c011456228aedc0b3fc1d511"
    )
    expected_planning_feasibility_sha256: str = (
        "4a66a9afb8a55f96ef8ec83cdae6be12643965c857054c80f42d40a90914b404"
    )
    expected_planning_effects_sha256: str = (
        "44c68fa3df16e2a2e16ab4ba90d31be56d8010782cab09208ef7b9a3273f8669"
    )
    expected_planning_simulation_sha256: str = (
        "684bb9c9d3ed9063236bb278e6374e7c5c7acac57cd29e4e24257e9299aede05"
    )
    expected_sampling_frame_sha256: str = (
        "695f3587dae24b5945f975a7611c677fca9a5e89f728d2f3802f77b5a31a2d4d"
    )
    expected_sampling_manifest_sha256: str = (
        "de4baae228b15adda0ad41bba674d6a45470a728be92b374f20aa30b72ef4fca"
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


@dataclass(frozen=True)
class Scale1BCohortFreezeInputs:
    admitted: pd.DataFrame
    metadata: pd.DataFrame
    planning_summary: dict[str, Any]
    planning_manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Scale1BCohortFreezeResult:
    status: str
    scoring_panel: pd.DataFrame
    primary_core: pd.DataFrame
    redundancy_clusters: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_input_path",
            f"Invalid logical path: {logical_ref}",
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            expected_sha256=expected,
            observed_sha256=observed,
        )
    return path, {"path": logical_ref, "sha256": observed, "label": label}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not isinstance(value, dict):
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must be a JSON object",
        )
    return value


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def _read_tsv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def _extract_sequence(
    paths: ProjectPaths, row: dict[str, Any]
) -> dict[str, Any]:
    logical_ref = str(row["canonical_sequence_source_relative"])
    source_path = _resolve(paths, logical_ref)
    expected_file_sha = str(row["canonical_sequence_source_file_sha256"])
    if not source_path.is_file() or sha256_file(source_path) != expected_file_sha:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "canonical_sequence_source_drift",
            f"Canonical sequence source changed for {row['pair_id']}",
        )
    try:
        canonical = extract_canonical_sequence(
            source_path.read_bytes(), str(row["canonical_accession"])
        )
    except (OSError, DerivationError) as exc:
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "canonical_sequence_unresolvable",
            f"Canonical sequence cannot be resolved for {row['pair_id']}",
        ) from exc
    if (
        canonical["sequence_sha256"] != str(row["canonical_sequence_sha256"])
        or canonical["sequence_length"] != int(row["canonical_sequence_length"])
    ):
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "canonical_sequence_identity_drift",
            f"Canonical sequence identity changed for {row['pair_id']}",
        )
    return {
        "canonical_sequence": canonical["sequence"],
        "canonical_sequence_sha256": canonical["sequence_sha256"],
        "canonical_sequence_source_relative": logical_ref,
        "canonical_sequence_source_file_sha256": expected_file_sha,
    }


def validate_scale1b_inputs(
    paths: ProjectPaths, config: Scale1BCohortFreezeConfig
) -> Scale1BCohortFreezeInputs:
    """Validate all frozen admission, planning, and redundancy trust roots."""
    bindings = (
        (config.admission_census_ref, config.expected_admission_census_sha256, "Scale-1A1 admission census"),
        (config.common_masks_ref, config.expected_common_masks_sha256, "Scale-1A1 common masks"),
        (config.admission_summary_ref, config.expected_admission_summary_sha256, "Scale-1A1 summary"),
        (config.admission_manifest_ref, config.expected_admission_manifest_sha256, "Scale-1A1 manifest"),
        (config.planning_manifest_ref, config.expected_planning_manifest_sha256, "Scale-1 planning manifest"),
        (config.planning_summary_ref, config.expected_planning_summary_sha256, "Scale-1 planning summary"),
        (config.planning_feasibility_ref, config.expected_planning_feasibility_sha256, "Scale-1 planning feasibility"),
        (config.planning_effects_ref, config.expected_planning_effects_sha256, "Scale-1 planning effects"),
        (config.planning_simulation_ref, config.expected_planning_simulation_sha256, "Scale-1 planning simulation"),
        (config.sampling_frame_ref, config.expected_sampling_frame_sha256, "Scale-1A0 sampling frame"),
        (config.sampling_manifest_ref, config.expected_sampling_manifest_sha256, "Scale-1A0 manifest"),
        (config.inventory_tsv_ref, config.expected_inventory_tsv_sha256, "candidate inventory TSV"),
        (config.inventory_json_ref, config.expected_inventory_json_sha256, "candidate inventory JSON"),
        (config.discovery_ref, config.expected_discovery_sha256, "canonical discovery source"),
        (config.cluster_implementation_ref, config.expected_cluster_implementation_sha256, "RCSB cluster implementation"),
    )
    resolved: dict[str, Path] = {}
    artifacts: list[dict[str, Any]] = []
    for logical_ref, expected, label in bindings:
        path, record = _require_hash(paths, logical_ref, expected, label)
        resolved[logical_ref] = path
        artifacts.append(record)

    census = _read_parquet(resolved[config.admission_census_ref], "admission census")
    admission_summary = _read_json(
        resolved[config.admission_summary_ref], "admission summary"
    )
    admission_manifest = _read_json(
        resolved[config.admission_manifest_ref], "admission manifest"
    )
    planning_manifest = _read_json(
        resolved[config.planning_manifest_ref], "planning manifest"
    )
    planning_summary = _read_json(
        resolved[config.planning_summary_ref], "planning summary"
    )
    sampling_frame = _read_parquet(
        resolved[config.sampling_frame_ref], "sampling frame"
    )
    sampling_manifest = _read_json(
        resolved[config.sampling_manifest_ref], "sampling manifest"
    )
    inventory = _read_tsv(resolved[config.inventory_tsv_ref], "candidate inventory")
    discovery = _read_parquet(resolved[config.discovery_ref], "canonical discovery")

    admitted = census.loc[census["admission_status"].eq("FORMALLY_ADMITTED")].copy()
    if len(census) != 72 or len(admitted) != 52 or admitted["pair_id"].duplicated().any():
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "admission_pool_mismatch",
            "Scale-1A1 formally admitted pool is not exactly 52 unique proteins",
        )
    if (
        admission_summary.get("terminal_status_counts", {}).get("FORMALLY_ADMITTED")
        != 52
        or admission_manifest.get("scale1a1_status")
        != "SCALE1A1_FORMAL_CENSUS_COMPLETE"
        or admission_manifest.get("SCALE1_SCORING_NOT_STARTED") is not True
    ):
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "admission_contract_mismatch",
            "Scale-1A1 admission contract differs",
        )
    expected_planning_outputs = {
        "feasibility": (
            config.planning_feasibility_ref,
            config.expected_planning_feasibility_sha256,
            52,
        ),
        "stage0_planning_effects": (
            config.planning_effects_ref,
            config.expected_planning_effects_sha256,
            8,
        ),
        "power_simulation": (
            config.planning_simulation_ref,
            config.expected_planning_simulation_sha256,
            360,
        ),
        "summary": (
            config.planning_summary_ref,
            config.expected_planning_summary_sha256,
            None,
        ),
    }
    for name, (expected_path, expected_sha, expected_rows) in expected_planning_outputs.items():
        record = planning_manifest.get("outputs", {}).get(name, {})
        if (
            record.get("path") != expected_path
            or record.get("sha256") != expected_sha
            or (expected_rows is not None and record.get("rows") != expected_rows)
        ):
            raise Scale1BCohortFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "planning_manifest_binding_mismatch",
                f"Planning manifest does not bind {name}",
            )
    if (
        planning_manifest.get("planning_status")
        != "SCALE1_PROTEIN_POWER_PLANNING_COMPLETE"
        or planning_manifest.get("SCALE1_SCORING_NOT_STARTED") is not True
        or planning_manifest.get("SCALE1B_COHORT_NOT_SELECTED") is not True
        or planning_summary.get("admitted_count") != 52
        or planning_summary.get("additional_admitted_needed") != 0
        or planning_summary.get("recommended_mechanical_rule")
        != "NO_ADDITIONAL_COMMON_MASK_FEASIBILITY_FILTER_REQUIRED"
        or planning_summary.get("diversity_capacity")
        != "DIVERSITY_CAPACITY_NOT_YET_ASSESSED"
    ):
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "planning_contract_mismatch",
            "Frozen Scale-1 planning contract differs",
        )
    if (
        sampling_manifest.get("canonical_discovery_source", {}).get("sha256")
        != config.expected_discovery_sha256
        or sampling_manifest.get("source_candidate_artifact", {}).get("sha256")
        != config.expected_inventory_json_sha256
    ):
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "metadata_provenance_mismatch",
            "Scale-1A0 metadata provenance differs",
        )

    inventory_columns = [
        "pair_id",
        "candidate_index",
        "canonical_source_row",
        "polymer_entity_id",
        "PDB",
        "chain",
        "UniProt",
        "sequence_cluster",
        "protein_family_if_available",
        "sampling_stratum_prior",
        "round1_member",
    ]
    metadata = admitted.merge(
        inventory[inventory_columns],
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_inventory"),
    )
    metadata = metadata.merge(
        sampling_frame[
            ["pair_id", "stage0_declared_member", "stage0_admitted_member"]
        ],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    discovery_rows = discovery.reset_index().rename(
        columns={"index": "canonical_source_row_discovery"}
    )
    metadata = metadata.merge(
        discovery_rows[
            [
                "canonical_source_row_discovery",
                "polymer_entity_id",
                "pdb_id",
                "chain_id",
                "uniprot_id",
                "sequence_cluster",
                "length",
                "experimental_method",
                "resolution",
                "organism",
                "taxonomy_id",
            ]
        ],
        left_on="canonical_source_row",
        right_on="canonical_source_row_discovery",
        how="left",
        validate="many_to_one",
        suffixes=("", "_discovery"),
    )
    identity_ok = (
        metadata["candidate_index"].eq(metadata["sampling_frame_index"])
        & metadata["canonical_source_row_inventory"].eq(
            metadata["canonical_source_row"]
        )
        & metadata["polymer_entity_id_inventory"].eq(metadata["polymer_entity_id"])
        & metadata["PDB"].str.lower().eq(metadata["pdb_id"].str.lower())
        & metadata["chain"].eq(metadata["pdb_chain"])
        & metadata["UniProt"].eq(metadata["canonical_accession"])
        & metadata["polymer_entity_id_discovery"].eq(metadata["polymer_entity_id"])
        & metadata["pdb_id_discovery"].str.lower().eq(metadata["pdb_id"].str.lower())
        & metadata["chain_id"].eq(metadata["pdb_chain"])
        & metadata["uniprot_id"].eq(metadata["canonical_accession"])
        & metadata["sequence_cluster_discovery"].eq(metadata["sequence_cluster"])
    )
    if len(metadata) != 52 or not identity_ok.all():
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "metadata_identity_mismatch",
            "Canonical inventory/discovery identities do not match Scale-1A1",
        )
    metadata["redundancy_metadata_resolved"] = (
        metadata["sequence_cluster"].notna()
        & metadata["sequence_cluster"].astype(str).str.match(r"^30:.+")
    )
    metadata["stage0_overlap"] = metadata["stage0_admitted_member"].astype(bool)
    metadata["common_mask_fraction"] = metadata[
        "common_mask_fraction_of_mapped"
    ].astype(float)
    sequences = [_extract_sequence(paths, row) for row in metadata.to_dict("records")]
    for key in sequences[0]:
        metadata[key] = [record[key] for record in sequences]
    seen_artifacts = {(record["path"], record["sha256"]) for record in artifacts}
    for row in metadata.to_dict("records"):
        source_record = (
            str(row["canonical_sequence_source_relative"]),
            str(row["canonical_sequence_source_file_sha256"]),
        )
        if source_record not in seen_artifacts:
            artifacts.append(
                {
                    "path": source_record[0],
                    "sha256": source_record[1],
                    "label": "frozen canonical sequence source",
                }
            )
            seen_artifacts.add(source_record)
    return Scale1BCohortFreezeInputs(
        admitted=admitted.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        planning_summary=planning_summary,
        planning_manifest=planning_manifest,
        input_artifacts=tuple(artifacts),
    )


def assign_redundancy_roles(metadata: pd.DataFrame) -> pd.DataFrame:
    """Assign one deterministic pre-outcome representative per frozen cluster."""
    if metadata["sequence_cluster"].isna().any():
        raise Scale1BCohortFreezeError(
            BLOCKED_REDUNDANCY_DEFINITION,
            "missing_frozen_redundancy_assignment",
            "At least one admitted protein lacks a frozen sequence cluster",
        )
    if not metadata["sequence_cluster"].astype(str).str.match(r"^30:.+").all():
        raise Scale1BCohortFreezeError(
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
        raise Scale1BCohortFreezeError(
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
    ranked = ranked.copy()
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


def _count_mapping(values: pd.Series) -> dict[str, int]:
    counts = values.fillna("UNAVAILABLE").astype(str).value_counts()
    return {
        key: int(value)
        for key, value in sorted(
            counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )
    }


def _distribution(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    quantiles = np.quantile(
        array, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1], method="linear"
    )
    return {
        "min": int(quantiles[0]) if float(quantiles[0]).is_integer() else float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "max": int(quantiles[6]) if float(quantiles[6]).is_integer() else float(quantiles[6]),
    }


def _scoring_panel_columns() -> list[str]:
    return [
        "candidate_id",
        "pair_id",
        "polymer_entity_id",
        "pdb_id",
        "pdb_chain",
        "canonical_accession",
        "sampling_frame_index",
        "canonical_source_row",
        "canonical_sequence",
        "canonical_sequence_sha256",
        "canonical_sequence_length",
        "canonical_sequence_source_relative",
        "canonical_sequence_source_file_sha256",
        "mapped_residue_count",
        "common_mask_count",
        "common_mask_fraction",
        "sequence_cluster",
        "redundancy_cluster_id",
        "redundancy_metadata_resolved",
        "redundancy_cluster_size",
        "representative_rank_within_cluster",
        "is_primary_representative",
        "cohort_role",
        "stage0_overlap",
        "analysis_origin",
        "in_full_scoring_panel",
        "in_primary_nonredundant_core",
        "in_new_protein_primary_core",
        "protein_family_if_available",
        "structural_family_if_available",
        "frozen_domain_annotation_if_available",
        "discovery_stratum_if_available",
        "frozen_sampling_stratum_prior",
        "discovery_sequence_length",
        "organism",
        "taxonomy_id",
        "experimental_method",
        "resolution",
    ]


def build_scale1b_cohort_freeze(
    inputs: Scale1BCohortFreezeInputs,
) -> Scale1BCohortFreezeResult:
    """Build the canonical all-52 panel and one-per-cluster primary core."""
    assigned = assign_redundancy_roles(inputs.metadata)
    if len(assigned) != 52 or assigned["pair_id"].duplicated().any():
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "scoring_panel_identity_mismatch",
            "The scoring panel must contain exactly 52 unique admitted proteins",
        )
    representative_counts = assigned.groupby("redundancy_cluster_id")[
        "is_primary_representative"
    ].sum()
    if not representative_counts.eq(1).all():
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "representative_cardinality_mismatch",
            "Each redundancy cluster must have exactly one representative",
        )
    assigned["in_full_scoring_panel"] = True
    assigned["in_primary_nonredundant_core"] = assigned[
        "is_primary_representative"
    ]
    assigned["in_new_protein_primary_core"] = (
        assigned["is_primary_representative"] & ~assigned["stage0_overlap"]
    )
    assigned["structural_family_if_available"] = pd.NA
    assigned["frozen_domain_annotation_if_available"] = pd.NA
    assigned["discovery_stratum_if_available"] = pd.NA
    assigned["frozen_sampling_stratum_prior"] = assigned[
        "sampling_stratum_prior"
    ]
    assigned["discovery_sequence_length"] = assigned["length"].astype(int)
    scoring_panel = assigned[_scoring_panel_columns()].copy()
    primary_core = scoring_panel.loc[
        scoring_panel["is_primary_representative"]
    ].reset_index(drop=True)
    cluster_columns = [
        "pair_id",
        "canonical_accession",
        "sampling_frame_index",
        "redundancy_cluster_id",
        "redundancy_metadata_resolved",
        "redundancy_cluster_size",
        "common_mask_count",
        "common_mask_fraction",
        "representative_rank_within_cluster",
        "is_primary_representative",
        "cohort_role",
        "stage0_overlap",
        "analysis_origin",
    ]
    redundancy_clusters = scoring_panel[cluster_columns].copy()
    cluster_sizes = (
        scoring_panel.groupby("redundancy_cluster_id", sort=True)
        .size()
        .astype(int)
    )
    cluster_distribution = cluster_sizes.value_counts().sort_index()
    primary_n = len(primary_core)
    status = FREEZE_COMPLETE if primary_n >= 16 else BLOCKED_DIVERSITY_CAPACITY
    capacity_status = (
        "PRIMARY_CORE_CAPACITY_ADEQUATE"
        if primary_n >= 16
        else "PRIMARY_CORE_CAPACITY_INSUFFICIENT"
    )
    full_stage0 = int(scoring_panel["stage0_overlap"].sum())
    primary_stage0 = int(primary_core["stage0_overlap"].sum())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "freeze_status": status,
        "scoring_panel_n": len(scoring_panel),
        "primary_core_n": primary_n,
        "secondary_replication_n": len(scoring_panel) - primary_n,
        "new_protein_primary_core_n": int(
            scoring_panel["in_new_protein_primary_core"].sum()
        ),
        "cluster_statistics": {
            "total_clusters": len(cluster_sizes),
            "singleton_clusters": int(cluster_sizes.eq(1).sum()),
            "multi_member_clusters": int(cluster_sizes.gt(1).sum()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_size_distribution": {
                str(size): int(count)
                for size, count in cluster_distribution.items()
            },
        },
        "stage0_overlap_counts": {
            "full_scoring_panel": full_stage0,
            "primary_core": primary_stage0,
            "secondary_replication": full_stage0 - primary_stage0,
            "new_protein_primary_core": int(
                scoring_panel["in_new_protein_primary_core"].sum()
            ),
        },
        "capacity_status": capacity_status,
        "capacity_checks": {
            "ge_16": primary_n >= 16,
            "ge_24": primary_n >= 24,
            "ge_30": primary_n >= 30,
            "ge_40": primary_n >= 40,
        },
        "capacity_interpretation": (
            "N=16 is a planning sanity floor, not the desired cohort size."
        ),
        "diversity_metadata_status": "DIVERSITY_METADATA_LIMITED",
        "diversity_metadata_availability": {
            "canonical_sequence": "AVAILABLE_FOR_ALL_52",
            "sequence_identity_cluster": "AVAILABLE_FOR_ALL_52",
            "organism_and_taxonomy": "AVAILABLE_FOR_ALL_52",
            "frozen_sampling_stratum_prior": "AVAILABLE_FOR_ALL_52",
            "family_or_superfamily": "UNAVAILABLE",
            "structural_class": "UNAVAILABLE",
            "discovery_stratum": "UNAVAILABLE",
            "frozen_domain_annotation": "UNAVAILABLE",
        },
        "diversity_summaries": {
            "full_panel_length": _distribution(
                scoring_panel["canonical_sequence_length"]
            ),
            "primary_core_length": _distribution(
                primary_core["canonical_sequence_length"]
            ),
            "full_panel_sampling_prior_counts": _count_mapping(
                scoring_panel["frozen_sampling_stratum_prior"]
            ),
            "primary_core_sampling_prior_counts": _count_mapping(
                primary_core["frozen_sampling_stratum_prior"]
            ),
            "full_panel_organism_counts": _count_mapping(
                scoring_panel["organism"]
            ),
            "primary_core_organism_counts": _count_mapping(
                primary_core["organism"]
            ),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_status": status,
        "upstream_artifacts": list(inputs.input_artifacts),
        "redundancy_convention": {
            "assignment_source": (
                "data/processed/discovery/discovered_candidates.parquet:"
                "sequence_cluster"
            ),
            "canonical_inventory_binding": (
                "artifacts/dataset/reports/census/candidate_inventory_v1.tsv:"
                "sequence_cluster"
            ),
            "implementation": "src/dual_uq/rcsb_discovery.py:_sequence_cluster",
            "implementation_sha256": (
                "db70dfcecf21588be1a8ea84da36cac1e06ea4de1985598a6e9221ef073960e8"
            ),
            "provider": "RCSB rcsb_cluster_membership",
            "definition": "sequence_identity_cluster",
            "identity_threshold_percent": 30,
            "cluster_id_serialization": "30:<RCSB cluster_id>",
            "assignments_recomputed": False,
        },
        "representative_rule": {
            "priority": [
                "resolved redundancy metadata over unresolved metadata",
                "larger common_mask_count",
                "larger common_mask_fraction",
                "earlier frozen sampling_frame_index",
                "stable canonical identity ordering",
            ],
            "stable_canonical_identity_order": [
                "canonical_accession",
                "pdb_id",
                "pdb_chain",
                "pair_id",
            ],
            "stage0_membership_used": False,
            "scale1_outcomes_used": False,
        },
        "membership_definitions": {
            "FULL_SCORING_PANEL": "all 52 FORMALLY_ADMITTED proteins",
            "PRIMARY_NONREDUNDANT_CORE": (
                "one deterministic representative per frozen redundancy cluster"
            ),
            "SECONDARY_RELATED_REPLICATION": (
                "all nonrepresentatives from multi-member redundancy clusters"
            ),
            "NEW_PROTEIN_PRIMARY_CORE": (
                "PRIMARY_NONREDUNDANT_CORE excluding frozen Stage-0 admitted members"
            ),
            "STAGE0_OVERLAP": (
                "scale1_sampling_frame_audit.parquet:stage0_admitted_member"
            ),
        },
        "capacity_floor": 16,
        "capacity_floor_interpretation": (
            "planning sanity floor, not the desired cohort size"
        ),
        "outputs": {},
        "manifest_self_hash_policy": (
            "reported_by_cli_after_write_to_avoid_recursive_self_hash"
        ),
        "SCALE1B_SCORING_PANEL_FROZEN": True,
        "SCALE1B_PRIMARY_CORE_FROZEN": True,
        "SCALE1_OUTCOMES_NOT_OBSERVED": True,
        "NO_OUTCOME_DEPENDENT_SELECTION": True,
        "STAGE0_LOCAL_ANALYSIS_STOP": True,
        "scope_confirmations": {
            "network_access_used": False,
            "downloads_performed": False,
            "proteinmpnn_executed": False,
            "fixed_probes_constructed": False,
            "scale1_scoring_performed": False,
        },
    }
    return Scale1BCohortFreezeResult(
        status=status,
        scoring_panel=scoring_panel,
        primary_core=primary_core,
        redundancy_clusters=redundancy_clusters,
        summary=summary,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise Scale1BCohortFreezeError(
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
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_cohort_freeze_output",
            "Cohort-freeze JSON cannot be serialized",
        ) from exc


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise Scale1BCohortFreezeError(
            BLOCKED_INPUT_INTEGRITY,
            "immutable_cohort_freeze_artifact_conflict",
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
            raise Scale1BCohortFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_cohort_freeze_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Scale1BCohortFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_cohort_freeze_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(result: Scale1BCohortFreezeResult, paths: ProjectPaths) -> None:
    for record in result.input_artifacts:
        path = _resolve(paths, str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise Scale1BCohortFreezeError(
                BLOCKED_INPUT_INTEGRITY,
                "upstream_input_changed",
                f"Frozen input changed: {record['label']}",
            )


def materialize_scale1b_cohort_freeze(
    result: Scale1BCohortFreezeResult,
    paths: ProjectPaths,
    *,
    config: Scale1BCohortFreezeConfig | None = None,
) -> dict[str, Any]:
    """Write three tables and summary, then the immutable manifest last."""
    config = config or Scale1BCohortFreezeConfig()
    _rehash_inputs(result, paths)
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "scoring_panel": output_root / "scale1b_scoring_panel.parquet",
        "primary_core": output_root / "scale1b_primary_core.parquet",
        "redundancy_clusters": output_root / "scale1b_redundancy_clusters.parquet",
        "summary": output_root / "scale1b_cohort_summary.json",
        "manifest": output_root / "scale1b_cohort_freeze_manifest.json",
    }
    frames = {
        "scoring_panel": result.scoring_panel,
        "primary_core": result.primary_core,
        "redundancy_clusters": result.redundancy_clusters,
    }
    write_status = {
        name: _write_immutable_parquet(targets[name], frame)
        for name, frame in frames.items()
    }
    write_status["summary"] = _write_immutable_bytes(
        targets["summary"], _render_json(result.summary)
    )
    outputs: dict[str, dict[str, Any]] = {
        name: {
            "path": _logical(paths, targets[name]),
            "rows": len(frame),
            "sha256": sha256_file(targets[name]),
        }
        for name, frame in frames.items()
    }
    outputs["summary"] = {
        "path": _logical(paths, targets["summary"]),
        "sha256": sha256_file(targets["summary"]),
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
        "freeze_status": result.status,
        "write_status": write_status,
        "outputs": outputs,
    }


def run_scale1b_cohort_freeze(
    paths: ProjectPaths, *, config: Scale1BCohortFreezeConfig | None = None
) -> dict[str, Any]:
    """Validate frozen inputs, build the pre-outcome cohort, and materialize it."""
    config = config or Scale1BCohortFreezeConfig()
    inputs = validate_scale1b_inputs(paths, config)
    result = build_scale1b_cohort_freeze(inputs)
    return materialize_scale1b_cohort_freeze(result, paths, config=config)
