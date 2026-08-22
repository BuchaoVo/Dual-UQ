"""StructCal v1 model-independent release construction and validation.

This module converts already-frozen scientific cohorts into one public Core.
It owns no discovery, admission, mapping, annotation, model, or scoring logic.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.benchmark.ids import structure_id as canonical_structure_id
from dual_uq.benchmark.schema_registry import SchemaRegistry
from dual_uq.benchmark.validation import (
    validate_core_relationships,
    validate_frame_contract,
)
from dual_uq.dataset.controlled_relational_geometry_sources import ApoHoloParentSource
from dual_uq.dataset.identity_split import build_connected_identity_assignments


class StructCalReleaseError(ValueError):
    """Raised when a frozen source cannot satisfy the StructCal v1 contract."""


ARMS = (
    "representation_variation",
    "ligand_state",
    "functional_state",
    "controlled_perturbation",
)
SPLITS = ("TRAIN", "VALIDATION", "LOCKED_TEST")
SPLIT_TARGETS = {"TRAIN": 0.70, "VALIDATION": 0.15, "LOCKED_TEST": 0.15}
SPLIT_PROTOCOL_VERSION = "structcal_global_cluster_split_v1"
RELEASE_ID = "structcal_v1"
SCHEMA_VERSION = "2.0.0"

_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_FORBIDDEN_PUBLIC_EXACT = {
    "model_score",
    "model_response",
    "r_local",
    "d_excess",
    "j_full",
}
_FORBIDDEN_PUBLIC_PATTERN = re.compile(
    r"(^proteinmpnn_|^esm_if1_|_sha256$|_checksum$)", re.IGNORECASE
)


def _required(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise StructCalReleaseError(f"{label} missing required columns: {missing}")


def build_global_protein_universe(
    arm_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Collapse frozen cross-Arm records to one canonical biological protein row."""

    if not arm_tables:
        raise StructCalReleaseError("global protein universe has no Arm inputs")
    records: list[pd.DataFrame] = []
    required = {"protein_id", "uniprot_id", "canonical_sequence"}
    for arm, frame in arm_tables.items():
        if frame.empty:
            continue
        _required(frame, required, f"{arm} proteins")
        rows = frame[list(required)].copy()
        rows["protein_id"] = rows["protein_id"].astype(str).str.strip().str.upper()
        rows["uniprot_id"] = rows["uniprot_id"].astype(str).str.strip().str.upper()
        rows["canonical_sequence"] = (
            rows["canonical_sequence"].astype(str).str.strip().str.upper()
        )
        if rows[["protein_id", "uniprot_id", "canonical_sequence"]].eq("").any().any():
            raise StructCalReleaseError(f"{arm} contains empty canonical protein fields")
        if not rows["protein_id"].eq(rows["uniprot_id"]).all():
            raise StructCalReleaseError(
                f"{arm} protein_id must equal its canonical UniProt identity"
            )
        records.append(rows)
    if not records:
        raise StructCalReleaseError("global protein universe is empty")
    combined = pd.concat(records, ignore_index=True)
    sequence_counts = combined.groupby("protein_id")["canonical_sequence"].nunique()
    if (sequence_counts != 1).any():
        conflicts = sequence_counts.loc[sequence_counts.ne(1)].index.tolist()
        raise StructCalReleaseError(
            f"cross-Arm protein identity has conflicting canonical sequences: {conflicts[:5]}"
        )
    accession_counts = combined.groupby("protein_id")["uniprot_id"].nunique()
    if (accession_counts != 1).any():
        raise StructCalReleaseError("one protein_id maps to multiple canonical accessions")
    universe = combined.drop_duplicates("protein_id", keep="first").sort_values(
        "protein_id", kind="mergesort"
    )
    universe["canonical_length"] = universe["canonical_sequence"].str.len().astype(int)
    universe["canonical_sequence_status"] = universe["canonical_sequence"].map(
        lambda sequence: (
            "STANDARD_20AA" if set(str(sequence)) <= _STANDARD_AA else "NONSTANDARD"
        )
    )
    return universe[
        [
            "protein_id",
            "uniprot_id",
            "canonical_sequence",
            "canonical_length",
            "canonical_sequence_status",
        ]
    ].reset_index(drop=True)


def _cluster_balance_table(
    proteins: pd.DataFrame, clusters: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    protein_cluster = proteins[["protein_id"]].merge(
        clusters[["protein_id", "identity_cluster_id"]],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
    if protein_cluster["identity_cluster_id"].isna().any():
        raise StructCalReleaseError("global clustering does not cover every protein")
    rows = protein_cluster.groupby("identity_cluster_id", sort=True).size().rename(
        "protein_count"
    ).to_frame()
    pair_cluster = pairs.merge(
        protein_cluster, on="protein_id", how="left", validate="many_to_one"
    )
    if pair_cluster["identity_cluster_id"].isna().any():
        raise StructCalReleaseError("a formal pair references a protein outside clustering")
    rows["pair_count"] = pair_cluster.groupby("identity_cluster_id").size()
    rows["pair_count"] = rows["pair_count"].fillna(0).astype(int)
    for field in ("arm", "state_family"):
        if field not in pair_cluster:
            continue
        values = pair_cluster[field].dropna().astype(str).unique().tolist()
        for value in sorted(values):
            name = f"{field}:{value}"
            rows[name] = (
                pair_cluster.loc[pair_cluster[field].astype(str).eq(value)]
                .groupby("identity_cluster_id")
                .size()
            )
            rows[name] = rows[name].fillna(0).astype(int)
    return rows.reset_index()


def assign_global_cluster_split(
    proteins: pd.DataFrame,
    clusters: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Assign whole global identity clusters with one predeclared seeded procedure."""

    _required(proteins, {"protein_id", "canonical_length"}, "proteins")
    _required(clusters, {"protein_id", "identity_cluster_id"}, "clusters")
    _required(pairs, {"pair_id", "protein_id", "arm"}, "pairs")
    if proteins.empty or clusters.empty:
        raise StructCalReleaseError("global split requires non-empty proteins and clusters")
    if proteins["protein_id"].duplicated().any() or clusters["protein_id"].duplicated().any():
        raise StructCalReleaseError("global split inputs contain duplicate protein_id")
    if set(proteins["protein_id"]) != set(clusters["protein_id"]):
        raise StructCalReleaseError("clustering input differs from global protein universe")

    balance = _cluster_balance_table(proteins, clusters, pairs)
    metric_columns = [column for column in balance.columns if column != "identity_cluster_id"]
    totals = {column: float(balance[column].sum()) for column in metric_columns}
    randomizer = random.Random(int(seed))
    tie_order = balance["identity_cluster_id"].astype(str).tolist()
    randomizer.shuffle(tie_order)
    tie_rank = {cluster_id: index for index, cluster_id in enumerate(tie_order)}

    def priority(row: dict[str, object]) -> tuple[float, int]:
        contributions = [
            float(row[column]) / total
            for column, total in totals.items()
            if total > 0
        ]
        return (-max(contributions, default=0.0), tie_rank[str(row["identity_cluster_id"])])

    ordered = sorted(balance.to_dict(orient="records"), key=priority)
    current = {
        split: {column: 0.0 for column in metric_columns} for split in SPLITS
    }
    cluster_split: dict[str, str] = {}

    def loss(candidate_split: str, cluster: dict[str, object]) -> float:
        value = 0.0
        for split in SPLITS:
            for column in metric_columns:
                total = totals[column]
                if total <= 0:
                    continue
                observed = current[split][column]
                if split == candidate_split:
                    observed += float(cluster[column])
                target = total * SPLIT_TARGETS[split]
                value += ((observed - target) / max(target, 1.0)) ** 2
        return value

    for cluster in ordered:
        selected = min(SPLITS, key=lambda split: (loss(split, cluster), SPLITS.index(split)))
        cluster_id = str(cluster["identity_cluster_id"])
        cluster_split[cluster_id] = selected
        for column in metric_columns:
            current[selected][column] += float(cluster[column])

    assignments = clusters[["protein_id", "identity_cluster_id"]].copy()
    assignments["split"] = assignments["identity_cluster_id"].astype(str).map(cluster_split)
    if assignments["split"].isna().any():
        raise StructCalReleaseError("split assignment did not cover every identity cluster")
    assignments["identity_threshold"] = 0.30
    assignments["coverage_threshold"] = 0.80
    assignments["coverage_mode"] = 0
    assignments["clustering_method"] = "connected_components"
    assignments["clustering_tool"] = "MMseqs2"
    assignments["clustering_version"] = "18.8cc5c"
    assignments["split_seed"] = int(seed)
    assignments["split_protocol_version"] = SPLIT_PROTOCOL_VERSION
    assignments = assignments.sort_values("protein_id", kind="mergesort").reset_index(
        drop=True
    )
    if assignments.groupby("identity_cluster_id")["split"].nunique().max() != 1:
        raise StructCalReleaseError("identity cluster leakage across splits")
    if len(balance) >= len(SPLITS) and set(assignments["split"]) != set(SPLITS):
        raise StructCalReleaseError("global split did not populate every release partition")
    return assignments


def audit_public_core_schema(tables: Mapping[str, pd.DataFrame]) -> None:
    """Reject hashes, model capabilities, and outcomes from public Core tables."""

    for table, frame in tables.items():
        for field in frame.columns:
            normalized = str(field).lower()
            if (
                normalized in _FORBIDDEN_PUBLIC_EXACT
                or _FORBIDDEN_PUBLIC_PATTERN.search(normalized)
            ):
                raise StructCalReleaseError(
                    f"forbidden public Core field in {table}: {field}"
                )


def normalize_structcal_core_dtypes(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Restore schema-level nullable integer semantics after cross-Arm concatenation."""

    normalized = {name: frame.copy() for name, frame in tables.items()}
    integer_columns = {
        "proteins": ("canonical_length",),
        "condition_pairs": (
            "common_mapped_count",
            "common_coordinate_visible_count",
        ),
        "residue_mappings": ("canonical_position",),
        "splits": ("coverage_mode", "split_seed"),
    }
    for table, columns in integer_columns.items():
        if table not in normalized:
            continue
        for column in columns:
            if column in normalized[table]:
                normalized[table][column] = pd.to_numeric(
                    normalized[table][column], errors="raise"
                ).astype("Int64")
    return normalized


def safe_fraction(numerator: object, denominator: object) -> float | None:
    """Return a bounded fraction or ``None`` for unavailable source facts."""

    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0:
        return None
    return max(0.0, min(1.0, top / bottom))


@dataclass(frozen=True, slots=True)
class StructCalSourcePaths:
    """Frozen model-independent source identities for StructCal v1."""

    representation_pairs: str = (
        "experiments/p2_design_baseline/scale1/scale1b_v2/"
        "scale1b_v2_primary_cohort.parquet"
    )
    representation_mappings: str = (
        "experiments/p2_design_baseline/scale1/scale1b_v2/"
        "scale1b_v2_primary_common_masks.parquet"
    )
    representation_validity: str = (
        "experiments/p2_design_baseline/scale1b-v2/pair_validity/"
        "pair_validity.parquet"
    )
    apo_holo_root: str = (
        "experiments/interventions/biological_states/"
        "apo_holo_selective_admission_release"
    )
    functional_root: str = (
        "experiments/interventions/functional_states/construction/"
        "functional_state_admission"
    )
    controlled_root: str = (
        "experiments/interventions/controlled_perturbations/"
        "relational_geometry_confirmatory"
    )
    functional_core_root: str = "benchmark/core"
    functional_annotation_root: str = "benchmark/annotations"
    apo_sequence_table: str = (
        "experiments/comparisons/method_design/identity_split/"
        "cluster_assignments.parquet"
    )

    def read_ledger(self) -> tuple[str, ...]:
        return (
            self.representation_pairs,
            self.representation_mappings,
            self.representation_validity,
            f"{self.apo_holo_root}/primary_pairs.parquet",
            f"{self.apo_holo_root}/candidate_structures.parquet",
            f"{self.apo_holo_root}/residue_mappings.parquet",
            f"{self.apo_holo_root}/pair_structural_descriptors.parquet",
            f"{self.apo_holo_root}/residue_structural_descriptors.parquet",
            f"{self.apo_holo_root}/ligand_sites.parquet",
            f"{self.functional_root}/primary_pairs.parquet",
            f"{self.functional_core_root}/proteins.parquet",
            f"{self.functional_core_root}/structures.parquet",
            f"{self.functional_core_root}/condition_pairs.parquet",
            f"{self.functional_core_root}/residue_mappings.parquet",
            f"{self.functional_core_root}/benchmark_instances.parquet",
            f"{self.functional_annotation_root}/pair_structural_descriptors.parquet",
            f"{self.functional_annotation_root}/residue_structural_descriptors.parquet",
            f"{self.controlled_root}/parent_cohort.parquet",
            f"{self.controlled_root}/geometry/selected_instances.parquet",
            f"{self.controlled_root}/geometry/structural_descriptors.parquet",
            self.apo_sequence_table,
        )


@dataclass(frozen=True, slots=True)
class StructCalReleaseBundle:
    core: dict[str, pd.DataFrame]
    annotations: dict[str, pd.DataFrame]
    tracks: dict[str, pd.DataFrame]
    metadata: dict[str, dict[str, Any]]


def _text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    cleaned = str(value).strip()
    return cleaned if cleaned and cleaned.lower() not in {"nan", "none", "<na>"} else None


def _portable_join(base: str, value: object) -> str:
    text = _text(value)
    if text is None:
        raise StructCalReleaseError(f"missing required source path below {base}")
    path = Path(text)
    if path.is_absolute() or text.startswith(("data/", "experiments/")):
        return path.as_posix()
    return (Path(base) / path).as_posix()


def _condition_structure_id(
    *, arm: str, source_structure_id: str, label: str, protein_id: str
) -> str:
    return canonical_structure_id(
        source_structure_id=source_structure_id,
        semantics=arm,
        condition_label=label,
        protein=protein_id,
    )


def _mapping_status(mapped_1: bool, mapped_2: bool, visible_1: bool, visible_2: bool) -> str:
    if visible_1 and visible_2:
        return "COMMON_VISIBLE"
    if mapped_1 and mapped_2:
        return "COMMON_MAPPED_NOT_VISIBLE"
    if mapped_1 or mapped_2:
        return "PARTIAL_MAPPING"
    return "UNMAPPED"


def _missing_reason(mapped: bool, visible: bool) -> str | None:
    if not mapped:
        return "UNMAPPED_IN_FROZEN_SOURCE"
    if not visible:
        return "COORDINATE_NOT_VISIBLE_IN_FROZEN_SOURCE"
    return None


def _residue_id(chain: object, number: object, insertion: object = None) -> str | None:
    chain_text = _text(chain)
    number_text = _text(number)
    if chain_text is None or number_text is None:
        return None
    try:
        numeric = float(number_text)
        if numeric.is_integer():
            number_text = str(int(numeric))
    except ValueError:
        pass
    insertion_text = _text(insertion)
    suffix = "" if insertion_text in {None, ".", "?"} else insertion_text
    return f"{chain_text}:{number_text}{suffix}"


def _read_sources(root: Path, paths: StructCalSourcePaths) -> dict[str, pd.DataFrame]:
    missing = [relative for relative in paths.read_ledger() if not (root / relative).is_file()]
    if missing:
        raise StructCalReleaseError(f"frozen source artifacts are missing: {missing[:5]}")
    tables = {
        "rv_pairs": pd.read_parquet(root / paths.representation_pairs),
        "rv_mappings": pd.read_parquet(root / paths.representation_mappings),
        "rv_validity": pd.read_parquet(root / paths.representation_validity),
        "apo_pairs": pd.read_parquet(root / paths.apo_holo_root / "primary_pairs.parquet"),
        "apo_structures": pd.read_parquet(root / paths.apo_holo_root / "candidate_structures.parquet"),
        "apo_mappings": pd.read_parquet(root / paths.apo_holo_root / "residue_mappings.parquet"),
        "apo_pair_descriptors": pd.read_parquet(root / paths.apo_holo_root / "pair_structural_descriptors.parquet"),
        "apo_residue_descriptors": pd.read_parquet(root / paths.apo_holo_root / "residue_structural_descriptors.parquet"),
        "apo_ligands": pd.read_parquet(root / paths.apo_holo_root / "ligand_sites.parquet"),
        "functional_primary": pd.read_parquet(root / paths.functional_root / "primary_pairs.parquet"),
        "functional_proteins": pd.read_parquet(root / paths.functional_core_root / "proteins.parquet"),
        "functional_structures": pd.read_parquet(root / paths.functional_core_root / "structures.parquet"),
        "functional_pairs": pd.read_parquet(root / paths.functional_core_root / "condition_pairs.parquet"),
        "functional_mappings": pd.read_parquet(root / paths.functional_core_root / "residue_mappings.parquet"),
        "functional_instances": pd.read_parquet(root / paths.functional_core_root / "benchmark_instances.parquet"),
        "functional_pair_descriptors": pd.read_parquet(root / paths.functional_annotation_root / "pair_structural_descriptors.parquet"),
        "functional_residue_descriptors": pd.read_parquet(root / paths.functional_annotation_root / "residue_structural_descriptors.parquet"),
        "controlled_parents": pd.read_parquet(root / paths.controlled_root / "parent_cohort.parquet"),
        "controlled_selected": pd.read_parquet(root / paths.controlled_root / "geometry/selected_instances.parquet"),
        "controlled_descriptors": pd.read_parquet(root / paths.controlled_root / "geometry/structural_descriptors.parquet"),
        "apo_sequences": pd.read_parquet(root / paths.apo_sequence_table),
    }
    forbidden_sources = re.compile(
        r"proteinmpnn|esm_if1|model_response|local_response|sequence_score",
        re.IGNORECASE,
    )
    outcome_paths = [relative for relative in paths.read_ledger() if forbidden_sources.search(relative)]
    if outcome_paths:
        raise StructCalReleaseError(f"outcome-dependent source path in release input: {outcome_paths}")
    return tables


def _source_protein_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rv = tables["rv_pairs"].rename(
        columns={
            "canonical_accession": "uniprot_id",
            "canonical_sequence_length": "canonical_length",
        }
    )
    rv = rv.assign(protein_id=rv["uniprot_id"].astype(str).str.upper())
    apo_ids = set(tables["apo_pairs"]["protein_id"].astype(str))
    apo = tables["apo_sequences"].loc[
        tables["apo_sequences"]["protein_id"].astype(str).isin(apo_ids),
        ["protein_id", "sequence"],
    ].drop_duplicates("protein_id")
    apo = apo.rename(columns={"sequence": "canonical_sequence"}).assign(
        uniprot_id=lambda frame: frame["protein_id"]
    )
    functional_ids = set(tables["functional_primary"]["uniprot_id"].astype(str))
    functional = tables["functional_proteins"].loc[
        tables["functional_proteins"]["protein_id"].astype(str).isin(functional_ids),
        ["protein_id", "uniprot_id", "canonical_sequence"],
    ]
    controlled_ids = set(tables["controlled_selected"]["protein_id"].astype(str))
    controlled = apo.loc[apo["protein_id"].astype(str).isin(controlled_ids)].copy()
    if set(apo["protein_id"].astype(str)) != apo_ids:
        raise StructCalReleaseError("Apo/Holo canonical sequence table does not cover PRIMARY proteins")
    if set(functional["protein_id"].astype(str)) != functional_ids:
        raise StructCalReleaseError("functional Core sequence table does not cover PRIMARY proteins")
    if set(controlled["protein_id"].astype(str)) != controlled_ids:
        raise StructCalReleaseError("controlled parent proteins lack frozen canonical sequences")
    return {
        "representation_variation": rv[["protein_id", "uniprot_id", "canonical_sequence"]],
        "ligand_state": apo[["protein_id", "uniprot_id", "canonical_sequence"]],
        "functional_state": functional,
        "controlled_perturbation": controlled,
    }


def _build_representation_tables(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    pairs = tables["rv_pairs"].copy()
    validity = tables["rv_validity"].copy()
    if set(pairs["pair_id"].astype(str)) != set(validity["pair_id"].astype(str)):
        raise StructCalReleaseError("PDB/AFDB primary and pair-validity identities disagree")
    validity = validity.set_index("pair_id", drop=False)
    structures: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    clean_pairs: set[str] = set()
    for source in pairs.to_dict(orient="records"):
        pair_id = str(source["pair_id"])
        protein_id = str(source["canonical_accession"]).upper()
        facts = validity.loc[pair_id]
        pdb_source_id = f"{str(source['pdb_id']).lower()}:{source['pdb_chain']}"
        afdb_source_id = str(source["afdb_model_id"])
        pdb_structure_id = _condition_structure_id(
            arm="representation_variation",
            source_structure_id=pdb_source_id,
            label="PDB",
            protein_id=protein_id,
        )
        afdb_structure_id = _condition_structure_id(
            arm="representation_variation",
            source_structure_id=afdb_source_id,
            label="AFDB",
            protein_id=protein_id,
        )
        structures.extend(
            [
                {
                    "structure_id": pdb_structure_id,
                    "source_structure_id": pdb_source_id,
                    "protein_id": protein_id,
                    "arm": "representation_variation",
                    "condition_type": "experimental_representation",
                    "condition_label": "PDB",
                    "state_family": None,
                    "source_type": "PDB",
                    "source_accession": str(source["pdb_id"]).lower(),
                    "chain_id": _text(source["pdb_chain"]),
                    "entity_id": _text(source["polymer_entity_id"]),
                    "assembly_id": None,
                    "experimental_method": None,
                    "source_file_ref": str(source["pdb_structure_ref"]),
                    "state_evidence_tier": None,
                    "state_evidence_source": "frozen_representation_primary",
                    "parent_structure_id": None,
                },
                {
                    "structure_id": afdb_structure_id,
                    "source_structure_id": afdb_source_id,
                    "protein_id": protein_id,
                    "arm": "representation_variation",
                    "condition_type": "predicted_representation",
                    "condition_label": "AFDB",
                    "state_family": None,
                    "source_type": "AFDB",
                    "source_accession": afdb_source_id,
                    "chain_id": "A",
                    "entity_id": None,
                    "assembly_id": None,
                    "experimental_method": None,
                    "source_file_ref": str(source["afdb_structure_ref"]),
                    "state_evidence_tier": None,
                    "state_evidence_source": "frozen_representation_primary",
                    "parent_structure_id": None,
                },
            ]
        )
        high_comparability = bool(facts["high_comparability_eligible"])
        if high_comparability:
            clean_pairs.add(pair_id)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "representation_variation",
                "condition_1_structure_id": pdb_structure_id,
                "condition_2_structure_id": afdb_structure_id,
                "condition_1_label": "PDB",
                "condition_2_label": "AFDB",
                "pair_role": "PRIMARY",
                "admission_status": "ADMITTED",
                "sequence_comparable": str(facts["identity_status"]) == "exact",
                "construct_comparable": not bool(facts.get("known_construct_difference", False)),
                "assembly_comparable": _text(facts.get("context_comparability_status")) is not None,
                "mapping_comparable": str(facts["mapping_status"]) == "formally_admitted",
                "pair_provenance": "scale1b_v2_primary_plus_frozen_pair_validity",
                "state_family": None,
                "admission_reason": None,
                "common_mapped_count": int(facts["mapped_residue_count"]) if pd.notna(facts["mapped_residue_count"]) else None,
                "common_mapped_fraction": float(facts["mapped_fraction_of_canonical_observed"]) if pd.notna(facts["mapped_fraction_of_canonical_observed"]) else None,
                "common_coordinate_visible_count": int(facts["common_mask_count_observed"]),
                "common_coordinate_visible_fraction": float(facts["common_mask_fraction_of_canonical_observed"]),
                "ligand_context_class": _text(facts.get("context_comparability_status")),
                "perturbation_family": None,
                "perturbation_dose": None,
                "perturbation_dose_unit": None,
            }
        )
        instance_rows.append(
            {
                "instance_id": pair_id,
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "representation_variation",
                "local_sensitivity_eligible": high_comparability,
                "generation_eligible": high_comparability,
                "compatibility_eligible": high_comparability,
                "multistate_eligible": False,
                "geometry_evaluable": False,
                "ligand_evaluable": False,
                "eligibility_reason": None if high_comparability else "NOT_IN_FROZEN_CLEAN_SUBSET",
            }
        )

    mapping_rows: list[dict[str, object]] = []
    for row in tables["rv_mappings"].to_dict(orient="records"):
        mapped_1 = bool(row["pdb_mapped"])
        mapped_2 = bool(row["afdb_mapped"])
        visible_1 = bool(row["pdb_backbone_complete"])
        visible_2 = bool(row["afdb_backbone_complete"])
        residue_1 = _residue_id(row["auth_asym_id"], row["auth_seq_id"], row["insertion_code"])
        residue_2 = _residue_id("A", row["canonical_position"])
        if mapped_1 and residue_1 is None:
            raise StructCalReleaseError("frozen PDB mapping has mapped residue without identity")
        mapping_rows.append(
            {
                "pair_id": str(row["candidate_id"]),
                "protein_id": str(row["canonical_accession"]).upper(),
                "canonical_position": int(row["canonical_position"]),
                "canonical_aa": str(row["canonical_aa"]),
                "condition_1_residue_id": residue_1 if mapped_1 else None,
                "condition_2_residue_id": residue_2 if mapped_2 else None,
                "condition_1_aa": _text(row["pdb_observed_aa"]) if mapped_1 else None,
                "condition_2_aa": _text(row["afdb_observed_aa"]) if mapped_2 else None,
                "condition_1_mapped": mapped_1,
                "condition_2_mapped": mapped_2,
                "condition_1_coordinate_visible": visible_1,
                "condition_2_coordinate_visible": visible_2,
                "common_mapped": mapped_1 and mapped_2,
                "common_coordinate_visible": visible_1 and visible_2,
                "condition_1_missing_reason": _missing_reason(mapped_1, visible_1),
                "condition_2_missing_reason": _missing_reason(mapped_2, visible_2),
                "mapping_status": _mapping_status(mapped_1, mapped_2, visible_1, visible_2),
            }
        )
    return (
        {
            "structures": pd.DataFrame(structures).drop_duplicates("structure_id"),
            "condition_pairs": pd.DataFrame(pair_rows),
            "residue_mappings": pd.DataFrame(mapping_rows),
            "benchmark_instances": pd.DataFrame(instance_rows),
        },
        clean_pairs,
    )


def _apo_structure_metadata(tables: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    rows = tables["apo_structures"].drop_duplicates("polymer_entity_id", keep="first")
    return rows.set_index("polymer_entity_id").to_dict(orient="index")


def _build_apo_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    metadata = _apo_structure_metadata(tables)
    structures: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    geometry_pairs = set(
        tables["apo_pair_descriptors"].loc[
            tables["apo_pair_descriptors"]["geometry_status"].astype(str).str.lower().eq("available"),
            "pair_id",
        ].astype(str)
    )
    ligand_pairs = set(tables["apo_ligands"]["pair_id"].astype(str))
    for source in tables["apo_pairs"].to_dict(orient="records"):
        pair_id = str(source["pair_id"])
        protein_id = str(source["protein_id"]).upper()
        ids: dict[str, str] = {}
        for side, label in (("apo", "APO"), ("holo", "HOLO")):
            entity_key = str(source[f"{side}_polymer_entity_id"])
            facts = metadata.get(entity_key, {})
            source_structure_id = f"{str(source[f'{side}_pdb_id']).lower()}:{entity_key}"
            structure_id = _condition_structure_id(
                arm="ligand_state",
                source_structure_id=source_structure_id,
                label=label,
                protein_id=protein_id,
            )
            ids[side] = structure_id
            structures.append(
                {
                    "structure_id": structure_id,
                    "source_structure_id": source_structure_id,
                    "protein_id": protein_id,
                    "arm": "ligand_state",
                    "condition_type": "experimental_ligand_state",
                    "condition_label": label,
                    "state_family": None,
                    "source_type": "PDB",
                    "source_accession": str(source[f"{side}_pdb_id"]).lower(),
                    "chain_id": _text(source[f"{side}_chain_id"]),
                    "entity_id": _text(facts.get("entity_id")) or entity_key,
                    "assembly_id": _text(facts.get("assembly_ids")),
                    "experimental_method": _text(facts.get("experimental_method")),
                    "source_file_ref": str(source[f"{side}_mmcif_relative_path"]),
                    "state_evidence_tier": "PRIMARY",
                    "state_evidence_source": "frozen_apo_holo_admission",
                    "parent_structure_id": None,
                }
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "ligand_state",
                "condition_1_structure_id": ids["apo"],
                "condition_2_structure_id": ids["holo"],
                "condition_1_label": "APO",
                "condition_2_label": "HOLO",
                "pair_role": "PRIMARY",
                "admission_status": "ADMITTED",
                "sequence_comparable": float(source["sequence_identity"]) == 1.0,
                "construct_comparable": int(source["construct_mismatch_count"]) == 0,
                "assembly_comparable": str(source["assembly_comparability"]) == "comparable",
                "mapping_comparable": int(source["common_mapped_count"]) > 0,
                "pair_provenance": "apo_holo_selective_admission_release",
                "state_family": None,
                "admission_reason": None,
                "common_mapped_count": int(source["common_mapped_count"]),
                "common_mapped_fraction": float(source["common_fraction"]),
                "common_coordinate_visible_count": int(source["common_mapped_count"]),
                "common_coordinate_visible_fraction": float(source["common_fraction"]),
                "ligand_context_class": "APO_HOLO",
                "perturbation_family": None,
                "perturbation_dose": None,
                "perturbation_dose_unit": None,
            }
        )
        instance_rows.append(
            {
                "instance_id": pair_id,
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "ligand_state",
                "local_sensitivity_eligible": True,
                "generation_eligible": True,
                "compatibility_eligible": True,
                "multistate_eligible": True,
                "geometry_evaluable": pair_id in geometry_pairs,
                "ligand_evaluable": pair_id in ligand_pairs,
                "eligibility_reason": None,
            }
        )

    mapping_rows: list[dict[str, object]] = []
    for row in tables["apo_mappings"].to_dict(orient="records"):
        canonical_aa = _text(row.get("uniprot_residue_name_apo")) or _text(
            row.get("uniprot_residue_name_holo")
        )
        if canonical_aa is None:
            raise StructCalReleaseError("Apo/Holo mapping lacks canonical amino-acid identity")
        mapping_rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "protein_id": str(row["protein_id"]).upper(),
                "canonical_position": int(row["canonical_position"]),
                "canonical_aa": canonical_aa,
                "condition_1_residue_id": _residue_id(row["pdb_chain_id_apo"], row["pdb_residue_number_apo"]),
                "condition_2_residue_id": _residue_id(row["pdb_chain_id_holo"], row["pdb_residue_number_holo"]),
                "condition_1_aa": _text(row["uniprot_residue_name_apo"]),
                "condition_2_aa": _text(row["uniprot_residue_name_holo"]),
                "condition_1_mapped": True,
                "condition_2_mapped": True,
                "condition_1_coordinate_visible": True,
                "condition_2_coordinate_visible": True,
                "common_mapped": True,
                "common_coordinate_visible": True,
                "condition_1_missing_reason": None,
                "condition_2_missing_reason": None,
                "mapping_status": "COMMON_VISIBLE",
            }
        )
    return {
        "structures": pd.DataFrame(structures).drop_duplicates("structure_id"),
        "condition_pairs": pd.DataFrame(pair_rows),
        "residue_mappings": pd.DataFrame(mapping_rows),
        "benchmark_instances": pd.DataFrame(instance_rows),
    }


def _build_functional_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    primary_ids = set(tables["functional_primary"]["pair_id"].astype(str))
    pairs = tables["functional_pairs"].loc[
        tables["functional_pairs"]["pair_id"].astype(str).isin(primary_ids)
    ].copy()
    if set(pairs["pair_id"].astype(str)) != primary_ids or len(pairs) != len(primary_ids):
        raise StructCalReleaseError("functional canonical Core does not preserve PRIMARY pair identity")
    pairs = pairs.rename(columns={"structural_condition_semantics": "arm"})
    structure_ids = set(pairs["condition_1_structure_id"]) | set(
        pairs["condition_2_structure_id"]
    )
    structures = tables["functional_structures"].loc[
        tables["functional_structures"]["structure_id"].isin(structure_ids)
    ].copy()
    structures = structures.rename(columns={"structural_condition_semantics": "arm"})
    structures["source_accession"] = structures["pdb_id"].where(
        structures["pdb_id"].notna(), structures["source_structure_id"]
    )
    structures = structures[
        [
            "structure_id", "source_structure_id", "protein_id", "arm", "condition_type",
            "condition_label", "state_family", "source_type", "source_accession", "chain_id",
            "entity_id", "assembly_id", "experimental_method", "source_file_ref",
            "state_evidence_tier", "state_evidence_source", "parent_structure_id",
        ]
    ]
    mappings = tables["functional_mappings"].loc[
        tables["functional_mappings"]["pair_id"].astype(str).isin(primary_ids)
    ].copy()
    instances = tables["functional_instances"].loc[
        tables["functional_instances"]["pair_id"].astype(str).isin(primary_ids)
    ].copy()
    instances = instances.rename(
        columns={
            "structural_condition_semantics": "arm",
            "generative_propagation_eligible": "generation_eligible",
            "sequence_scoring_eligible": "compatibility_eligible",
            "multistate_generation_eligible": "multistate_eligible",
        }
    )
    return {
        "structures": structures,
        "condition_pairs": pairs,
        "residue_mappings": mappings,
        "benchmark_instances": instances,
    }


def _build_controlled_tables(
    root: Path, paths: StructCalSourcePaths, tables: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    selected = tables["controlled_selected"].copy()
    selected_parents = set(selected["parent_structure_id"].astype(str))
    parents = tables["controlled_parents"].loc[
        tables["controlled_parents"]["parent_structure_id"].astype(str).isin(selected_parents)
    ].copy()
    if set(parents["parent_structure_id"].astype(str)) != selected_parents:
        raise StructCalReleaseError("selected controlled instances lack frozen parent facts")
    parent_lookup = parents.set_index("parent_structure_id")
    source = ApoHoloParentSource(root, parent_panel=parents)
    inherited_by_parent: dict[str, pd.DataFrame] = {}
    structures: list[dict[str, object]] = []
    for parent_id, parent in parent_lookup.iterrows():
        record = source.resolve_parent(
            parent,
            inherited_pair_id=f"structcal_inheritance:{parent_id}",
        )
        inherited_by_parent[str(parent_id)] = record.mapping
        structures.append(
            {
                "structure_id": str(parent_id),
                "source_structure_id": str(parent["source_structure_id"]),
                "protein_id": str(parent["protein_id"]).upper(),
                "arm": "controlled_perturbation",
                "condition_type": "controlled_reference",
                "condition_label": "REFERENCE",
                "state_family": None,
                "source_type": "PDB",
                "source_accession": str(parent["source_structure_id"]),
                "chain_id": _text(parent["chain_id"]),
                "entity_id": None,
                "assembly_id": None,
                "experimental_method": _text(parent["experimental_method"]),
                "source_file_ref": str(parent["source_file_ref"]),
                "state_evidence_tier": None,
                "state_evidence_source": "controlled_confirmatory_parent_selection",
                "parent_structure_id": None,
            }
        )

    pair_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    mapping_frames: list[pd.DataFrame] = []
    geometry_base = f"{paths.controlled_root}/geometry"
    for row in selected.to_dict(orient="records"):
        pair_id = str(row["pair_id"])
        protein_id = str(row["protein_id"]).upper()
        parent_id = str(row["parent_structure_id"])
        parent = parent_lookup.loc[parent_id]
        perturbed_structure_id = str(row["candidate_id"])
        structures.append(
            {
                "structure_id": perturbed_structure_id,
                "source_structure_id": str(row["candidate_id"]),
                "protein_id": protein_id,
                "arm": "controlled_perturbation",
                "condition_type": "controlled_perturbation",
                "condition_label": "PERTURBED",
                "state_family": None,
                "source_type": "CONTROLLED_PERTURBATION",
                "source_accession": str(row["candidate_id"]),
                "chain_id": _text(parent["chain_id"]),
                "entity_id": None,
                "assembly_id": None,
                "experimental_method": None,
                "source_file_ref": _portable_join(geometry_base, row["perturbed_source_file_ref"]),
                "state_evidence_tier": None,
                "state_evidence_source": "controlled_confirmatory_selected_instance",
                "parent_structure_id": parent_id,
            }
        )
        inherited = inherited_by_parent[parent_id].copy()
        inherited["pair_id"] = pair_id
        inherited["protein_id"] = protein_id
        mapping_frames.append(inherited)
        common_count = int(inherited["common_mapped"].sum())
        visible_count = int(inherited["common_coordinate_visible"].sum())
        canonical_count = len(inherited)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "controlled_perturbation",
                "condition_1_structure_id": parent_id,
                "condition_2_structure_id": perturbed_structure_id,
                "condition_1_label": "REFERENCE",
                "condition_2_label": "PERTURBED",
                "pair_role": "PRIMARY",
                "admission_status": "ADMITTED",
                "sequence_comparable": True,
                "construct_comparable": True,
                "assembly_comparable": True,
                "mapping_comparable": common_count == canonical_count,
                "pair_provenance": "controlled_relational_geometry_confirmatory_selected",
                "state_family": None,
                "admission_reason": None,
                "common_mapped_count": common_count,
                "common_mapped_fraction": common_count / canonical_count,
                "common_coordinate_visible_count": visible_count,
                "common_coordinate_visible_fraction": visible_count / canonical_count,
                "ligand_context_class": None,
                "perturbation_family": str(row["perturbation_family"]),
                "perturbation_dose": float(row["requested_dose"]),
                "perturbation_dose_unit": str(row["requested_dose_unit"]),
            }
        )
        instance_rows.append(
            {
                "instance_id": str(row["pilot_instance_id"]),
                "pair_id": pair_id,
                "protein_id": protein_id,
                "arm": "controlled_perturbation",
                "local_sensitivity_eligible": True,
                "generation_eligible": True,
                "compatibility_eligible": True,
                "multistate_eligible": False,
                "geometry_evaluable": True,
                "ligand_evaluable": False,
                "eligibility_reason": None,
            }
        )
    return {
        "structures": pd.DataFrame(structures).drop_duplicates("structure_id"),
        "condition_pairs": pd.DataFrame(pair_rows),
        "residue_mappings": pd.concat(mapping_frames, ignore_index=True),
        "benchmark_instances": pd.DataFrame(instance_rows),
    }


def _normalize_pair_descriptors(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    apo = tables["apo_pair_descriptors"].copy()
    apo_rows = pd.DataFrame(
        {
            "pair_id": apo["pair_id"].astype(str),
            "aligned_ca_rmsd": apo["aligned_ca_rmsd"],
            "median_residue_displacement": apo["median_residue_displacement"],
            "upper_tail_residue_displacement": apo["p90_residue_displacement"],
            "global_contact_change": None,
            "descriptor_method": "frozen_apo_holo_geometry",
            "descriptor_version": "apo_holo_release_v1",
        }
    )
    functional_ids = set(tables["functional_primary"]["pair_id"].astype(str))
    controlled = tables["controlled_selected"].copy()
    controlled_rows = pd.DataFrame(
        {
            "pair_id": controlled["pair_id"].astype(str),
            "aligned_ca_rmsd": controlled["realized_ca_rmsd"],
            "median_residue_displacement": None,
            "upper_tail_residue_displacement": None,
            "global_contact_change": controlled["realized_contact_change"],
            "descriptor_method": controlled["perturbation_method"].astype(str),
            "descriptor_version": controlled["perturbation_version"].astype(str),
        }
    )
    return pd.concat(
        [
            apo_rows,
            tables["functional_pair_descriptors"].loc[
                tables["functional_pair_descriptors"]["pair_id"].astype(str).isin(functional_ids)
            ],
            controlled_rows,
        ],
        ignore_index=True,
    )


def _normalize_residue_descriptors(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    apo = tables["apo_residue_descriptors"]
    apo_rows = pd.DataFrame(
        {
            "pair_id": apo["pair_id"].astype(str),
            "canonical_position": apo["canonical_position"].astype(int),
            "ca_displacement": apo["aligned_ca_displacement"],
            "local_pairwise_distance_change": apo["local_pairwise_distance_change"],
            "neighborhood_geometry_change": None,
            "contact_gain": None,
            "contact_loss": None,
            "descriptor_method": "frozen_apo_holo_geometry",
            "descriptor_version": "apo_holo_release_v1",
        }
    )
    functional_ids = set(tables["functional_primary"]["pair_id"].astype(str))
    functional = tables["functional_residue_descriptors"].loc[
        tables["functional_residue_descriptors"]["pair_id"].astype(str).isin(functional_ids)
    ]
    return pd.concat([apo_rows, functional], ignore_index=True)


def _normalize_ligands(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    source = tables["apo_ligands"]
    return pd.DataFrame(
        {
            "pair_id": source["pair_id"].astype(str),
            "canonical_position": source["canonical_position"].astype(int),
            "ligand_id": "ANY_HOLO_LIGAND",
            "ligand_class": None,
            "ligand_contact": source["ligand_proximal"].astype(bool),
            "ligand_distance": None,
            "pocket_membership": source["ligand_proximal"].astype(bool),
            "annotation_method": "frozen_apo_holo_ligand_proximity",
            "annotation_version": "apo_holo_release_v1",
        }
    )


def _normalize_perturbations(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    source = tables["controlled_selected"]
    return pd.DataFrame(
        {
            "pair_id": source["pair_id"].astype(str),
            "perturbation_family": source["perturbation_family"].astype(str),
            "requested_dose": source["requested_dose"].astype(float),
            "requested_dose_unit": source["requested_dose_unit"].astype(str),
            "realized_ca_rmsd": source["realized_ca_rmsd"].astype(float),
            "realized_pairwise_distance_change": source["realized_pairwise_distance_change"].astype(float),
            "realized_contact_change": source["realized_contact_change"].astype(float),
            "random_seed": source["random_seed"].map(int),
            "perturbation_method": source["perturbation_method"].astype(str),
            "perturbation_version": source["perturbation_version"].astype(str),
        }
    )


def run_global_mmseqs_clustering(
    proteins: pd.DataFrame, *, mmseqs_binary: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen all-vs-all MMseqs2 protocol and form connected components."""

    binary = Path(mmseqs_binary).expanduser().resolve()
    if not binary.is_file():
        raise StructCalReleaseError(f"MMseqs2 binary is unavailable: {binary}")
    version = subprocess.run(
        [str(binary), "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != "18.8cc5c":
        raise StructCalReleaseError(f"MMseqs2 version {version} != frozen 18.8cc5c")
    with tempfile.TemporaryDirectory(prefix="structcal-mmseqs-") as temporary:
        work = Path(temporary)
        fasta = work / "structcal_global_proteins.fasta"
        hits_path = work / "identity_hits.tsv"
        temp_path = work / "tmp"
        lines: list[str] = []
        for row in proteins.sort_values("protein_id", kind="mergesort").itertuples(index=False):
            lines.extend([f">{row.protein_id}", str(row.canonical_sequence)])
        fasta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        command = [
            str(binary), "easy-search", str(fasta), str(fasta), str(hits_path), str(temp_path),
            "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
            "--max-seqs", "100000", "--add-self-matches", "1", "--threads", "16",
            "--format-output", "query,target,pident,alnlen,qcov,tcov",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        hits = pd.read_csv(
            hits_path,
            sep="\t",
            header=None,
            names=["query", "target", "pident", "alnlen", "qcov", "tcov"],
        )
    sequence_table = proteins.rename(columns={"canonical_sequence": "sequence"})[
        ["protein_id", "sequence"]
    ]
    components = build_connected_identity_assignments(hits, sequence_table).rename(
        columns={"sequence_cluster_id": "identity_cluster_id"}
    )
    components["identity_cluster_id"] = "STRUCTCAL30:" + components[
        "identity_cluster_id"
    ].astype(str)
    protocol = {
        "tool": "MMseqs2",
        "version": version,
        "minimum_sequence_identity": 0.30,
        "coverage_threshold": 0.80,
        "coverage_mode": 0,
        "cluster_interpretation": "connected_components",
        "max_sequences_per_query": 100000,
        "self_matches": True,
        "input_unit": "one canonical sequence per StructCal protein_id",
        "command_semantics": "easy-search all-vs-all followed by connected components",
        "stderr_tail": completed.stderr[-2000:],
    }
    return components, protocol


def _canonical_core(
    proteins: pd.DataFrame,
    arms: list[dict[str, pd.DataFrame]],
    splits: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    names = ("structures", "condition_pairs", "residue_mappings", "benchmark_instances")
    core = {name: pd.concat([arm[name] for arm in arms], ignore_index=True) for name in names}
    core["proteins"] = proteins
    core["splits"] = splits
    ordered = {
        "proteins": core["proteins"].sort_values("protein_id", kind="mergesort").reset_index(drop=True),
        "structures": core["structures"].sort_values("structure_id", kind="mergesort").reset_index(drop=True),
        "condition_pairs": core["condition_pairs"].sort_values("pair_id", kind="mergesort").reset_index(drop=True),
        "residue_mappings": core["residue_mappings"].sort_values(["pair_id", "canonical_position"], kind="mergesort").reset_index(drop=True),
        "benchmark_instances": core["benchmark_instances"].sort_values("instance_id", kind="mergesort").reset_index(drop=True),
        "splits": core["splits"].sort_values("protein_id", kind="mergesort").reset_index(drop=True),
    }
    return normalize_structcal_core_dtypes(ordered)


def _track_views(
    pairs: pd.DataFrame, splits: pd.DataFrame, clean_representation_pairs: set[str]
) -> dict[str, pd.DataFrame]:
    membership = pairs[["pair_id", "protein_id", "arm", "state_family"]].merge(
        splits[["protein_id", "identity_cluster_id", "split"]],
        on="protein_id",
        how="left",
        validate="many_to_one",
    )
    track_i = membership.loc[
        membership["pair_id"].astype(str).isin(clean_representation_pairs)
    ].copy()
    track_i["benchmark_role"] = "INVARIANCE_NATURALISTIC"
    track_ii = membership.loc[
        membership["arm"].isin(["ligand_state", "functional_state"])
    ].copy()
    track_ii["benchmark_role"] = track_ii["arm"].map(
        {
            "ligand_state": "SENSITIVITY_APO_HOLO",
            "functional_state": "SENSITIVITY_FUNCTIONAL_STATE",
        }
    )
    columns = [
        "pair_id", "protein_id", "identity_cluster_id", "split", "benchmark_role",
        "arm", "state_family",
    ]
    return {
        "track_i_invariance": track_i[columns].sort_values("pair_id", kind="mergesort").reset_index(drop=True),
        "track_ii_sensitivity": track_ii[columns].sort_values("pair_id", kind="mergesort").reset_index(drop=True),
    }


def validate_structcal_bundle(
    bundle: StructCalReleaseBundle, *, schema_root: Path
) -> dict[str, Any]:
    """Execute release-level schema, relational, leakage, and Track checks."""

    registry = SchemaRegistry(schema_root)
    audit_public_core_schema(bundle.core)
    validate_core_relationships(bundle.core, registry)
    for name, frame in bundle.annotations.items():
        validate_frame_contract(frame, name, registry)
    proteins = bundle.core["proteins"].set_index("protein_id")
    pairs = bundle.core["condition_pairs"]
    structures = bundle.core["structures"].set_index("structure_id")
    mappings = bundle.core["residue_mappings"]
    splits = bundle.core["splits"]
    if not proteins["canonical_length"].eq(proteins["canonical_sequence"].str.len()).all():
        raise StructCalReleaseError("canonical protein length mismatch")
    for row in pairs.itertuples(index=False):
        left = structures.loc[row.condition_1_structure_id]
        right = structures.loc[row.condition_2_structure_id]
        if left.protein_id != row.protein_id or right.protein_id != row.protein_id:
            raise StructCalReleaseError("pair protein identity disagrees with condition structures")
        if left.arm != row.arm or right.arm != row.arm:
            raise StructCalReleaseError("pair Arm disagrees with condition structures")
    protein_lengths = proteins["canonical_length"].to_dict()
    protein_sequences = proteins["canonical_sequence"].to_dict()
    for row in mappings.itertuples(index=False):
        if row.canonical_position > protein_lengths[row.protein_id]:
            raise StructCalReleaseError("mapping position exceeds canonical sequence length")
        if protein_sequences[row.protein_id][row.canonical_position - 1] != row.canonical_aa:
            raise StructCalReleaseError("mapping canonical amino acid disagrees with protein sequence")
        if bool(row.common_mapped) != bool(row.condition_1_mapped and row.condition_2_mapped):
            raise StructCalReleaseError("common_mapped flag is internally inconsistent")
        if bool(row.common_coordinate_visible) != bool(
            row.condition_1_coordinate_visible and row.condition_2_coordinate_visible
        ):
            raise StructCalReleaseError("common coordinate visibility is internally inconsistent")
    cluster_leakage = int(
        (splits.groupby("identity_cluster_id")["split"].nunique() > 1).sum()
    )
    if cluster_leakage:
        raise StructCalReleaseError("cross-split identity-cluster leakage")
    pair_ids = set(pairs["pair_id"].astype(str))
    split_lookup = splits.set_index("protein_id")
    for track_name, track in bundle.tracks.items():
        if not set(track["pair_id"].astype(str)).issubset(pair_ids):
            raise StructCalReleaseError(f"{track_name} references a pair outside Core")
        resolved = track["protein_id"].map(split_lookup["split"])
        if not resolved.eq(track["split"]).all():
            raise StructCalReleaseError(f"{track_name} split inheritance mismatch")
    if "track_iii_calibration" in bundle.tracks:
        raise StructCalReleaseError("Track-III must not materialize a duplicate cohort")
    for annotation, frame in bundle.annotations.items():
        if not set(frame["pair_id"].astype(str)).issubset(pair_ids):
            raise StructCalReleaseError(f"{annotation} references a pair outside Core")
    return {
        "protein_key_unique": not bundle.core["proteins"]["protein_id"].duplicated().any(),
        "pair_key_unique": not pairs["pair_id"].duplicated().any(),
        "mapping_key_unique": not mappings.duplicated(["pair_id", "canonical_position"]).any(),
        "foreign_keys_valid": True,
        "pair_structure_identity_valid": True,
        "mapping_semantics_valid": True,
        "track_integrity_valid": True,
        "identity_cluster_leakage": cluster_leakage,
        "outcome_leakage": 0,
        "track_iii_duplicate_cohort": False,
    }


def _distribution_summary(
    core: Mapping[str, pd.DataFrame], tracks: Mapping[str, pd.DataFrame], arm_proteins: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    proteins = core["proteins"]
    splits = core["splits"]
    pairs = core["condition_pairs"].merge(
        splits[["protein_id", "identity_cluster_id", "split"]], on="protein_id", validate="many_to_one"
    )
    cluster_sizes = splits.groupby("identity_cluster_id").size()
    arm_sets = {arm: set(frame["protein_id"].astype(str)) for arm, frame in arm_proteins.items()}
    overlaps: dict[str, int] = {}
    arms = list(arm_sets)
    for index, left in enumerate(arms):
        for right in arms[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(arm_sets[left] & arm_sets[right])
    length_bins = pd.cut(
        proteins.set_index("protein_id")["canonical_length"],
        bins=[0, 199, 399, 799, float("inf")],
        labels=["LT_200", "200_399", "400_799", "GE_800"],
    )
    length_by_split = (
        splits.assign(length_bin=splits["protein_id"].map(length_bins))
        .groupby(["split", "length_bin"], observed=False)
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    return {
        "global_protein_universe": {
            "unique_proteins": len(proteins),
            "arm_unique_proteins": {arm: len(values) for arm, values in arm_sets.items()},
            "cross_arm_overlap": overlaps,
        },
        "global_clustering": {
            "proteins_clustered": len(splits),
            "identity_clusters": int(splits["identity_cluster_id"].nunique()),
            "singleton_clusters": int((cluster_sizes == 1).sum()),
            "multi_protein_clusters": int((cluster_sizes > 1).sum()),
            "largest_cluster_size": int(cluster_sizes.max()),
        },
        "global_split": {
            "clusters_per_split": splits.groupby("split")["identity_cluster_id"].nunique().astype(int).to_dict(),
            "proteins_per_split": splits["split"].value_counts().astype(int).to_dict(),
            "pairs_per_split": pairs["split"].value_counts().astype(int).to_dict(),
            "arm_pairs_per_split": pairs.groupby(["split", "arm"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "state_family_pairs_per_split": pairs.dropna(subset=["state_family"]).groupby(["split", "state_family"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "length_bins_per_split": length_by_split,
            "cluster_leakage": 0,
        },
        "tracks": {
            name: {
                "pairs": len(frame),
                "proteins": int(frame["protein_id"].nunique()),
                "clusters": int(frame["identity_cluster_id"].nunique()),
                "roles": frame["benchmark_role"].value_counts().astype(int).to_dict(),
                "state_families": frame["state_family"].dropna().value_counts().astype(int).to_dict(),
            }
            for name, frame in tracks.items()
        },
    }


def build_structcal_v1_bundle(
    project_root: Path,
    *,
    mmseqs_binary: Path,
    source_paths: StructCalSourcePaths | None = None,
) -> StructCalReleaseBundle:
    """Build the complete in-memory StructCal v1 release from frozen sources."""

    root = Path(project_root).expanduser().resolve()
    paths = source_paths or StructCalSourcePaths()
    tables = _read_sources(root, paths)
    arm_proteins = _source_protein_tables(tables)
    proteins = build_global_protein_universe(arm_proteins)
    representation, clean_pairs = _build_representation_tables(tables)
    apo = _build_apo_tables(tables)
    functional = _build_functional_tables(tables)
    controlled = _build_controlled_tables(root, paths, tables)
    pre_split_pairs = pd.concat(
        [
            representation["condition_pairs"], apo["condition_pairs"],
            functional["condition_pairs"], controlled["condition_pairs"],
        ],
        ignore_index=True,
    )
    clusters, clustering_protocol = run_global_mmseqs_clustering(
        proteins, mmseqs_binary=mmseqs_binary
    )
    splits = assign_global_cluster_split(
        proteins, clusters, pre_split_pairs, seed=20260822
    )
    core = _canonical_core(proteins, [representation, apo, functional, controlled], splits)
    annotations = {
        "pair_structural_descriptors": _normalize_pair_descriptors(tables).sort_values("pair_id", kind="mergesort").reset_index(drop=True),
        "residue_structural_descriptors": _normalize_residue_descriptors(tables).sort_values(["pair_id", "canonical_position"], kind="mergesort").reset_index(drop=True),
        "ligand_annotations": _normalize_ligands(tables).sort_values(["pair_id", "canonical_position"], kind="mergesort").reset_index(drop=True),
        "perturbation_descriptors": _normalize_perturbations(tables).sort_values("pair_id", kind="mergesort").reset_index(drop=True),
    }
    tracks = _track_views(core["condition_pairs"], splits, clean_pairs)
    summary = _distribution_summary(core, tracks, arm_proteins)
    metadata = _protocol_metadata(
        clustering_protocol=clustering_protocol,
        source_paths=paths,
        summary=summary,
    )
    bundle = StructCalReleaseBundle(core=core, annotations=annotations, tracks=tracks, metadata=metadata)
    validation = validate_structcal_bundle(bundle, schema_root=root / "schemas")
    bundle.metadata["release_manifest"]["validation"] = validation
    return bundle


def _protocol_metadata(
    *,
    clustering_protocol: dict[str, Any],
    source_paths: StructCalSourcePaths,
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    benchmark = {
        "benchmark_name": "StructCal",
        "benchmark_version": "1.0.0",
        "release_id": RELEASE_ID,
        "status": "FROZEN",
        "scientific_scope": "invariance-sensitivity calibration across structural conditions",
        "arms": list(ARMS),
        "tracks": {
            "TRACK_I_STRUCTURAL_INVARIANCE": {
                "question": "stability under nuisance or representation-level structural variation",
                "materialized_view": "tracks/track_i_invariance.parquet",
                "primary_role": "INVARIANCE_NATURALISTIC",
                "membership": "frozen clean PDB/AFDB representation-variation subset",
            },
            "TRACK_II_FUNCTIONAL_SENSITIVITY": {
                "question": "selective response to biologically meaningful structural-state variation",
                "materialized_view": "tracks/track_ii_sensitivity.parquet",
                "membership": "Apo/Holo PRIMARY plus functional-state PRIMARY",
            },
            "TRACK_III_INVARIANCE_SENSITIVITY_CALIBRATION": {
                "question": "joint nuisance invariance and functional sensitivity retention",
                "materialized_view": None,
                "contract": "joint Track-I x Track-II Pareto/multidimensional evaluation",
                "scalar_weighted_score": False,
            },
        },
        "controlled_confirmatory_role": (
            "Core structural calibration resource; intentionally excluded from Track-I because "
            "its frozen tiers manipulate relational geometry rather than establish nuisance invariance"
        ),
        "statistical_units": {
            "measurement": "PAIR",
            "biological_aggregation": "PROTEIN",
            "inferential_independence": "IDENTITY_CLUSTER_30",
            "aggregation_order": ["residue", "pair", "protein", "identity_cluster_30", "cohort"],
        },
    }
    split = {
        "split_values": list(SPLITS),
        "target_fractions": SPLIT_TARGETS,
        "target_fraction_scope": "each_normalized_balance_metric",
        "optimization_metric_families": [
            "protein_count",
            "pair_count",
            "arm:<value>",
            "state_family:<value>",
        ],
        "identity_cluster_count_targeted": False,
        "assignment_unit": "identity_cluster_30",
        "split_root_seed": 20260822,
        "protocol_version": SPLIT_PROTOCOL_VERSION,
        "procedure": (
            "single seeded deterministic greedy cluster assignment minimizing the "
            "summed squared target-relative deviation across all normalized "
            "pre-outcome balance metrics"
        ),
        "balance_attributes": ["protein_count", "formal_pair_count", "arm", "state_family"],
        "outcome_attributes_used": [],
        "seed_search": False,
    }
    statistical = {
        "pair_measurement_unit": "PAIR",
        "primary_biological_aggregation_unit": "PROTEIN",
        "inferential_resampling_unit": "identity_cluster_30",
        "bootstrap_replicates": 10000,
        "confidence_level": 0.95,
        "interval": "percentile_bootstrap",
        "cluster_weighting": "equal_cluster_contribution",
        "system_comparison": "paired_cluster_bootstrap",
    }
    generation = {
        "sequences_per_condition": 64,
        "native_standard_temperature": 0.1,
        "temperature_when_not_native": "NOT_APPLICABLE",
        "seed_schedule": list(range(64)),
        "paired_condition_seed_matching": True,
        "preserve_native_decoding_semantics": True,
        "independent_position_sampling_may_replace_autoregressive_generation": False,
    }
    evaluator = {
        "reference_panel": ["ProteinMPNN", "ESM-IF1"],
        "roles": ["SELF", "CROSS_PROTEINMPNN", "CROSS_ESM_IF1"],
        "generator_equals_evaluator_required": False,
        "raw_scores_may_be_averaged_across_evaluators": False,
        "interpretation": "model-based sequence-structure compatibility",
        "not_direct_measurements_of": ["physical_stability", "experimental_fitness", "biological_function"],
        "canonical_scoring_concepts": [
            "sequence_id", "generator_model_id", "evaluator_model_id", "pair_id",
            "target_condition_structure_id", "raw_score", "score_direction",
            "reference_score", "baseline_adjusted_score",
        ],
    }
    return {
        "benchmark_definition": benchmark,
        "clustering_protocol": clustering_protocol,
        "split_protocol": split,
        "statistical_protocol": statistical,
        "generation_protocol": generation,
        "evaluator_protocol": evaluator,
        "release_manifest": {
            "release_id": RELEASE_ID,
            "benchmark_name": "StructCal",
            "benchmark_version": "1.0.0",
            "schema_version": SCHEMA_VERSION,
            "protocol_versions": {
                "clustering": "structcal_global_identity_30_v1",
                "split": SPLIT_PROTOCOL_VERSION,
                "statistical": "structcal_statistical_v1",
                "generation": "structcal_generation_v1",
                "evaluator": "structcal_evaluator_v1",
            },
            "artifact_relative_paths": [],
            "row_counts": {},
            "source_release_identities": list(source_paths.read_ledger()),
            "code_revision": "UNRESOLVED_UNTIL_MATERIALIZATION",
            "creation_metadata": {},
            "release_status": "VALIDATED",
            "outcome_leakage_audit": {
                "model_dependent_inputs_read": [],
                "outcome_leakage": 0,
            },
            "summary": summary,
        },
    }


def materialize_structcal_v1(
    bundle: StructCalReleaseBundle,
    *,
    project_root: Path,
    release_root: Path,
) -> Path:
    """Publish a fully validated bundle without overwriting an existing release."""

    root = Path(project_root).resolve()
    destination = Path(release_root)
    destination = destination if destination.is_absolute() else root / destination
    if destination.exists():
        raise StructCalReleaseError(f"release destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        artifacts: list[str] = []
        row_counts: dict[str, int] = {}
        for layer, frames in (
            ("core", bundle.core),
            ("annotations", bundle.annotations),
            ("tracks", bundle.tracks),
        ):
            for name, frame in frames.items():
                relative = f"{layer}/{name}.parquet"
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                artifacts.append(relative)
                row_counts[relative] = len(frame)
        metadata = {name: dict(value) for name, value in bundle.metadata.items()}
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        worktree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        metadata["release_manifest"] = dict(metadata["release_manifest"])
        metadata["release_manifest"].update(
            {
                "artifact_relative_paths": sorted(
                    artifacts + [f"metadata/{name}.json" for name in metadata]
                ),
                "row_counts": row_counts,
                "code_revision": revision,
                "creation_metadata": {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "materializer": "dual_uq.dataset.structcal_release.materialize_structcal_v1",
                    "working_tree_state": (
                        "DIRTY_UNCOMMITTED_AUTHORIZED_TASK_CONTEXT"
                        if worktree_dirty
                        else "CLEAN"
                    ),
                },
                "release_status": "RELEASED",
            }
        )
        for name, value in metadata.items():
            relative = f"metadata/{name}.json"
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, destination)
    except Exception:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
        raise
    return destination


def validate_materialized_structcal_v1(
    release_root: Path, *, schema_root: Path
) -> dict[str, Any]:
    """Reload a published StructCal v1 release and execute all definition hard gates."""

    root = Path(release_root).expanduser().resolve()
    if not root.is_dir():
        raise StructCalReleaseError(f"StructCal release root is missing: {root}")
    core_names = (
        "proteins", "structures", "condition_pairs", "residue_mappings",
        "benchmark_instances", "splits",
    )
    annotation_names = (
        "pair_structural_descriptors", "residue_structural_descriptors",
        "ligand_annotations", "perturbation_descriptors",
    )
    track_names = ("track_i_invariance", "track_ii_sensitivity")
    metadata_names = (
        "benchmark_definition", "clustering_protocol", "split_protocol",
        "statistical_protocol", "generation_protocol", "evaluator_protocol",
        "release_manifest",
    )

    def load_parquet(layer: str, name: str) -> pd.DataFrame:
        path = root / layer / f"{name}.parquet"
        if not path.is_file():
            raise StructCalReleaseError(f"release artifact is missing: {path.relative_to(root)}")
        return pd.read_parquet(path)

    def load_json(name: str) -> dict[str, Any]:
        path = root / "metadata" / f"{name}.json"
        if not path.is_file():
            raise StructCalReleaseError(f"release metadata is missing: {path.relative_to(root)}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise StructCalReleaseError(f"release metadata is unreadable: {name}") from exc
        if not isinstance(value, dict):
            raise StructCalReleaseError(f"release metadata must be an object: {name}")
        return value

    core = {name: load_parquet("core", name) for name in core_names}
    annotations = {name: load_parquet("annotations", name) for name in annotation_names}
    tracks = {name: load_parquet("tracks", name) for name in track_names}
    metadata = {name: load_json(name) for name in metadata_names}
    bundle = StructCalReleaseBundle(core=core, annotations=annotations, tracks=tracks, metadata=metadata)
    validation = validate_structcal_bundle(bundle, schema_root=Path(schema_root))

    manifest = metadata["release_manifest"]
    manifest_schema = SchemaRegistry(schema_root).get("release_manifest")
    missing_manifest = sorted(set(manifest_schema.required).difference(manifest))
    extra_manifest = sorted(set(manifest).difference(manifest_schema.properties))
    if missing_manifest or extra_manifest:
        raise StructCalReleaseError(
            f"release manifest schema mismatch: missing={missing_manifest} extra={extra_manifest}"
        )
    if (
        manifest.get("release_id") != RELEASE_ID
        or manifest.get("benchmark_name") != "StructCal"
        or manifest.get("release_status") != "RELEASED"
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise StructCalReleaseError("release manifest identity/status is not frozen StructCal v1")
    actual_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    declared_paths = sorted(str(path) for path in manifest["artifact_relative_paths"])
    if actual_paths != declared_paths:
        raise StructCalReleaseError("release manifest artifact paths do not match published files")
    frames = {**core, **annotations, **tracks}
    for relative, expected in manifest["row_counts"].items():
        path = Path(relative)
        name = path.stem
        if name not in frames or len(frames[name]) != int(expected):
            raise StructCalReleaseError(f"release manifest row count mismatch: {relative}")

    benchmark = metadata["benchmark_definition"]
    clustering = metadata["clustering_protocol"]
    split = metadata["split_protocol"]
    statistical = metadata["statistical_protocol"]
    generation = metadata["generation_protocol"]
    evaluator = metadata["evaluator_protocol"]
    protocol_checks = (
        benchmark.get("status") == "FROZEN",
        set(benchmark.get("tracks", {}))
        == {
            "TRACK_I_STRUCTURAL_INVARIANCE",
            "TRACK_II_FUNCTIONAL_SENSITIVITY",
            "TRACK_III_INVARIANCE_SENSITIVITY_CALIBRATION",
        },
        clustering.get("tool") == "MMseqs2",
        clustering.get("version") == "18.8cc5c",
        clustering.get("minimum_sequence_identity") == 0.30,
        clustering.get("coverage_threshold") == 0.80,
        clustering.get("coverage_mode") == 0,
        clustering.get("cluster_interpretation") == "connected_components",
        split.get("split_values") == list(SPLITS),
        split.get("split_root_seed") == 20260822,
        split.get("target_fraction_scope") == "each_normalized_balance_metric",
        split.get("optimization_metric_families")
        == ["protein_count", "pair_count", "arm:<value>", "state_family:<value>"],
        split.get("identity_cluster_count_targeted") is False,
        split.get("outcome_attributes_used") == [],
        split.get("seed_search") is False,
        statistical.get("inferential_resampling_unit") == "identity_cluster_30",
        statistical.get("bootstrap_replicates") == 10000,
        statistical.get("confidence_level") == 0.95,
        statistical.get("interval") == "percentile_bootstrap",
        statistical.get("system_comparison") == "paired_cluster_bootstrap",
        generation.get("sequences_per_condition") == 64,
        generation.get("native_standard_temperature") == 0.1,
        generation.get("seed_schedule") == list(range(64)),
        evaluator.get("reference_panel") == ["ProteinMPNN", "ESM-IF1"],
        evaluator.get("raw_scores_may_be_averaged_across_evaluators") is False,
        manifest.get("outcome_leakage_audit", {}).get("model_dependent_inputs_read") == [],
        manifest.get("outcome_leakage_audit", {}).get("outcome_leakage") == 0,
    )
    if not all(protocol_checks):
        raise StructCalReleaseError("StructCal v1 protocol metadata is incomplete")
    if (root / "tracks/track_iii_calibration.parquet").exists():
        raise StructCalReleaseError("Track-III duplicate cohort was materialized")
    return {
        "release_id": RELEASE_ID,
        "protein_count": len(core["proteins"]),
        "pair_count": len(core["condition_pairs"]),
        "identity_cluster_count": int(core["splits"]["identity_cluster_id"].nunique()),
        "identity_cluster_leakage": validation["identity_cluster_leakage"],
        "outcome_leakage": validation["outcome_leakage"],
        "protocol_complete": True,
        "artifact_paths_complete": True,
        "row_counts_complete": True,
    }


__all__ = [
    "ARMS",
    "SPLITS",
    "SPLIT_PROTOCOL_VERSION",
    "StructCalReleaseBundle",
    "StructCalReleaseError",
    "StructCalSourcePaths",
    "assign_global_cluster_split",
    "audit_public_core_schema",
    "build_global_protein_universe",
    "build_structcal_v1_bundle",
    "materialize_structcal_v1",
    "normalize_structcal_core_dtypes",
    "run_global_mmseqs_clustering",
    "safe_fraction",
    "validate_materialized_structcal_v1",
    "validate_structcal_bundle",
]
