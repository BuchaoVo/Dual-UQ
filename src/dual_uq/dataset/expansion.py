"""Prospective, outcome-blind Scale-1 source-frame expansion planning."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from dual_uq.afdb import AFDB_PREDICTION_API, get_afdb_prediction_records
from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.rcsb_discovery import (
    RCSB_DATA_API,
    fetch_candidate_metadata,
)

SCALE1_EXPANSION_DESIGN_READY = "SCALE1_EXPANSION_DESIGN_READY"
SCALE1_EXPANSION_UNIVERSE_CAPACITY_INSUFFICIENT = (
    "SCALE1_EXPANSION_UNIVERSE_CAPACITY_INSUFFICIENT"
)
SCALE1_EXPANSION_SOURCE_QUERY_CHANGE_REQUIRED = (
    "SCALE1_EXPANSION_SOURCE_QUERY_CHANGE_REQUIRED"
)
BLOCKED_INPUT_INTEGRITY = "BLOCKED_INPUT_INTEGRITY"
BLOCKED_REDUNDANCY_BINDING = "BLOCKED_REDUNDANCY_BINDING"

_HISTORICAL_A3_IMPLEMENTATION_LABEL = (
    "effective Scale-1A3 scientific implementation"
)
_HISTORICAL_A3_IMPLEMENTATION_PATH = (
    "src/dual_uq/dataset/scale1a3_redundancy_diversity.py"
)

NEW_EXTERNAL_CLUSTER = "NEW_EXTERNAL_CLUSTER"
PENDING_ONLY_RESCUE = "PENDING_ONLY_RESCUE"
FAILURE_ONLY_RESCUE = "FAILURE_ONLY_RESCUE"
MIXED_UNADMITTED_RESCUE = "MIXED_UNADMITTED_RESCUE"
ADMITTED_COVERED_CLUSTER = "ADMITTED_COVERED_CLUSTER"

_CLUSTER_PATTERN = re.compile(r"^30:.+$")
_ADMITTED = "FORMALLY_ADMITTED"
_PENDING = "PENDING_HUMAN_VARIANT_REVIEW"
_FORBIDDEN_SELECTION_COLUMNS = {
    "common_mask_count",
    "common_mask_fraction",
    "paired_identity",
    "mismatch_count",
    "formal_admission_status",
    "terminal_scientific_reason",
    "pdb_afdb_rmsd",
    "tm_score",
    "p",
    "m",
    "sdfi",
    "proteinmpnn_score",
    "top1_disagreement",
    "regret",
    "structural_excess",
}


@dataclass(frozen=True)
class ExpansionConfig:
    """Portable frozen inputs and planning values for Scale-1E0."""

    scale1a3_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/scale1a3_manifest.json"
    )
    expected_scale1a3_manifest_sha256: str = (
        "eac4e19e6645d3aa6935e17bf8f25d1116d32a67dd8d3a4c78e68bc93d680d56"
    )
    scale1a2_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_census.parquet"
    )
    expected_scale1a2_census_sha256: str = (
        "fc69e13d3d94def1a751b5528230fa5f15dd1ae741d8361b4187236167120793"
    )
    discovery_ref: str = "data/processed/discovery/discovered_candidates.parquet"
    expected_discovery_sha256: str = (
        "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436"
    )
    query_ref: str = "data/processed/discovery/rcsb_screening_query.json"
    expected_query_sha256: str = (
        "88243fc800e1710575b350522b86f8016a4a2f32e2c6d7e81821491b953ce6c6"
    )
    identifier_list_ref: str = (
        "data/processed/discovery/rcsb_polymer_entities.txt"
    )
    expected_identifier_list_sha256: str = (
        "011276ca7236cd586c8992c4ad5b85d2961a788ec4c54e0be171da0db516e848"
    )
    scale1b_cohort_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_freeze_manifest.json"
    )
    expected_scale1b_cohort_manifest_sha256: str = (
        "275f7373f4f5de9b0a1f12c2b5585bc7ef91e88d11dc2b61b387326798e6e730"
    )
    scale1b_protocol_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_protocol_freeze_manifest.json"
    )
    expected_scale1b_protocol_manifest_sha256: str = (
        "485a22a89d0fc532cdee72c09ed7faa36c5fdc54079aa2a971479d41c53af918"
    )
    output_root_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_design"
    )
    original_probed_count: int = 220
    original_eligible_count: int = 213
    query_identifier_count: int = 500
    minimum_target: int = 100
    primary_target: int = 120
    stretch_target: int = 150


@dataclass(frozen=True)
class ExpansionInputs:
    """Validated frozen state plus the unprobed same-query identifier suffix."""

    original_candidates: pd.DataFrame
    original_outcomes: pd.DataFrame
    cluster_state: pd.DataFrame
    query_identifiers: tuple[str, ...]
    remaining_query_identifiers: tuple[str, ...]
    input_artifacts: tuple[dict[str, Any], ...]
    query_payload: dict[str, Any]
    config: ExpansionConfig


@dataclass(frozen=True)
class ExpansionResult:
    """Single canonical source for every Scale-1E0 output."""

    status: str
    cluster_state: pd.DataFrame
    expansion_universe: pd.DataFrame
    target_clusters: pd.DataFrame
    planning_scenarios: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


class ExpansionError(RuntimeError):
    """Structured Scale-1E0 blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


def _is_historical_a3_implementation_artifact(
    record: Mapping[str, Any],
) -> bool:
    """Recognize frozen A3 self-provenance after its source owner moved.

    The historical A3 manifest is immutable release evidence. Its own
    implementation record is retained as metadata, but it is not a live
    scientific input that can be rehashed after the semantic source move.
    """

    return (
        str(record.get("label")) == _HISTORICAL_A3_IMPLEMENTATION_LABEL
        and str(record.get("path")) == _HISTORICAL_A3_IMPLEMENTATION_PATH
    )


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "nonportable_input_path",
            f"Invalid logical path: {logical_ref}",
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_upstream_input",
            f"Missing {label}: {logical_ref}",
        )
    observed = sha256_file(path)
    if observed != expected:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            path=logical_ref,
            expected_sha256=expected,
            observed_sha256=observed,
        )
    return path, {"path": logical_ref, "sha256": observed, "label": label}


def _original_prospective_metadata(
    discovery: pd.DataFrame, census: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = discovery.loc[discovery["discovery_status"].eq("eligible")].copy()
    eligible["candidate_id"] = (
        eligible["pdb_id"].astype(str).str.lower()
        + "_"
        + eligible["chain_id"].astype(str)
        + "__"
        + eligible["uniprot_id"].astype(str).str.upper()
    )
    if len(eligible) != 213 or eligible["candidate_id"].duplicated().any():
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "original_eligible_identity_mismatch",
            "Frozen discovery must contain exactly 213 unique eligible candidates",
        )
    outcomes = census[["candidate_id", "formal_admission_status"]].rename(
        columns={"candidate_id": "pair_id"}
    )
    if set(eligible["candidate_id"]) != set(outcomes["pair_id"]):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "original_frame_identity_mismatch",
            "Frozen discovery and Scale-1A2 identities differ",
        )
    output = pd.DataFrame(
        {
            "candidate_id": eligible["candidate_id"],
            "pair_id": eligible["candidate_id"],
            "polymer_entity_id": eligible["polymer_entity_id"].astype(str),
            "canonical_accession": eligible["uniprot_id"].astype(str).str.upper(),
            "pdb_id": eligible["pdb_id"].astype(str).str.lower(),
            "pdb_chain": eligible["chain_id"].astype(str),
            "sequence_cluster": eligible["sequence_cluster"].astype(str),
            "identity_resolved": True,
            "identifier_unambiguous": True,
            "afdb_exact_record_resolved": eligible["afdb_model_entity_id"].notna(),
            "canonical_sequence_available": eligible["afdb_available"].fillna(False),
            "supported_length": eligible["length"].between(100, 500, inclusive="both"),
            "auth_chain_count": eligible["auth_asym_ids"].fillna("").astype(str).map(
                lambda value: len([item for item in value.split(";") if item])
            ),
            "length": pd.to_numeric(eligible["length"], errors="coerce").astype("Int64"),
            "experimental_method": eligible["experimental_method"],
            "resolution": pd.to_numeric(eligible["resolution"], errors="coerce"),
            "organism": eligible["organism"],
            "taxonomy_id": pd.to_numeric(eligible["taxonomy_id"], errors="coerce").astype("Int64"),
            "afdb_model_entity_id": eligible["afdb_model_entity_id"],
            "candidate_origin": "ORIGINAL_213_FRAME",
            "current_frame_overlap": True,
        }
    )
    return output.reset_index(drop=True), outcomes.reset_index(drop=True)


def load_expansion_inputs(
    paths: ProjectPaths,
    config: ExpansionConfig | None = None,
) -> ExpansionInputs:
    """Load and SHA-gate frozen A3/A2/v1 state before any discovery."""

    config = config or ExpansionConfig()
    a3_path, a3_artifact = _require_hash(
        paths,
        config.scale1a3_manifest_ref,
        config.expected_scale1a3_manifest_sha256,
        "Scale-1A3 manifest",
    )
    census_path, census_artifact = _require_hash(
        paths,
        config.scale1a2_census_ref,
        config.expected_scale1a2_census_sha256,
        "Scale-1A2 census",
    )
    discovery_path, discovery_artifact = _require_hash(
        paths,
        config.discovery_ref,
        config.expected_discovery_sha256,
        "canonical discovery source",
    )
    query_path, query_artifact = _require_hash(
        paths, config.query_ref, config.expected_query_sha256, "frozen RCSB query"
    )
    identifiers_path, identifiers_artifact = _require_hash(
        paths,
        config.identifier_list_ref,
        config.expected_identifier_list_sha256,
        "frozen RCSB identifier list",
    )
    _, cohort_manifest_artifact = _require_hash(
        paths,
        config.scale1b_cohort_manifest_ref,
        config.expected_scale1b_cohort_manifest_sha256,
        "Scale-1B-v1 cohort manifest",
    )
    _, protocol_manifest_artifact = _require_hash(
        paths,
        config.scale1b_protocol_manifest_ref,
        config.expected_scale1b_protocol_manifest_sha256,
        "Scale-1B-v1 protocol manifest",
    )
    try:
        a3_manifest = json.loads(a3_path.read_text(encoding="utf-8"))
        query_payload = json.loads(query_path.read_text(encoding="utf-8"))
        census = pd.read_parquet(census_path)
        discovery = pd.read_parquet(discovery_path)
    except Exception as exc:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_parse_failure",
            "A frozen Scale-1E0 input could not be parsed",
        ) from exc
    statistics = a3_manifest.get("capacity_statistics") or {}
    expected_statistics = {
        "n_nr_admitted": 63,
        "n_nr_upper": 78,
        "n_pending_proteins": 51,
        "n_pending_new_clusters": 15,
    }
    if (
        any(int(statistics.get(key, -1)) != value for key, value in expected_statistics.items())
        or a3_manifest.get("scale1a3_status")
        != "ADDITIONAL_SOURCE_FRAME_EXPANSION_REQUIRED"
        or len(census) != 213
        or len(discovery) != config.original_probed_count
    ):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_state_reconstruction_mismatch",
            "Frozen Scale-1A3/A2/discovery cardinalities do not reconstruct",
        )
    original_candidates, outcomes = _original_prospective_metadata(discovery, census)
    frame_for_state = original_candidates[["pair_id", "sequence_cluster"]].merge(
        outcomes, on="pair_id", how="left", validate="one_to_one"
    )
    cluster_state = reconstruct_cluster_state(frame_for_state)
    if int(cluster_state["contains_admitted"].sum()) != 63:
        raise ExpansionError(
            BLOCKED_REDUNDANCY_BINDING,
            "admitted_cluster_count_mismatch",
            "Scale-1A3 must reconstruct exactly 63 admitted clusters",
        )
    identifiers = tuple(
        line.strip()
        for line in identifiers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(identifiers) != config.query_identifier_count or len(set(identifiers)) != len(identifiers):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "query_identifier_cardinality_mismatch",
            "Frozen RCSB identifier list must contain 500 unique records",
        )
    attempted = set(discovery["polymer_entity_id"].astype(str))
    remaining = tuple(identifier for identifier in identifiers if identifier not in attempted)
    if len(remaining) != config.query_identifier_count - config.original_probed_count:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "remaining_query_identity_mismatch",
            "Unprobed identifier suffix does not reconcile with frozen discovery",
        )
    artifacts = [
        a3_artifact,
        census_artifact,
        discovery_artifact,
        query_artifact,
        identifiers_artifact,
        cohort_manifest_artifact,
        protocol_manifest_artifact,
    ]
    seen = {(item["path"], item["sha256"]) for item in artifacts}
    for item in [*(a3_manifest.get("upstream_artifacts") or []), *(a3_manifest.get("outputs") or {}).values()]:
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            continue
        key = (str(item["path"]), str(item["sha256"]))
        if key not in seen:
            if _is_historical_a3_implementation_artifact(item):
                artifacts.append(
                    {
                        "path": key[0],
                        "sha256": key[1],
                        "label": item.get("label", _HISTORICAL_A3_IMPLEMENTATION_LABEL),
                    }
                )
                seen.add(key)
                continue
            path = _resolve(paths, key[0])
            if not path.is_file() or sha256_file(path) != key[1]:
                raise ExpansionError(
                    BLOCKED_INPUT_INTEGRITY,
                    "a3_bound_artifact_mismatch",
                    f"Scale-1A3-bound artifact changed: {key[0]}",
                )
            artifacts.append(
                {"path": key[0], "sha256": key[1], "label": item.get("label", "Scale-1A3-bound artifact")}
            )
            seen.add(key)
    return ExpansionInputs(
        original_candidates=original_candidates,
        original_outcomes=outcomes,
        cluster_state=cluster_state,
        query_identifiers=identifiers,
        remaining_query_identifiers=remaining,
        input_artifacts=tuple(artifacts),
        query_payload=query_payload,
        config=config,
    )


def discover_remaining_query_candidates(
    identifiers: Iterable[str],
    provider: Callable[[str], dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Record-isolated metadata discovery for a frozen identifier collection."""

    identifier_values = tuple(str(value) for value in identifiers)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for identifier in identifier_values:
        try:
            record = provider(str(identifier))
            required = {
                "polymer_entity_id",
                "pdb_id",
                "chain_id",
                "auth_asym_ids",
                "uniprot_id",
                "all_uniprot_ids",
                "length",
                "experimental_method",
                "resolution",
                "sequence_cluster",
                "afdb_model_entity_id",
                "afdb_returned_accession",
                "afdb_exact_record_count",
                "afdb_sequence_present",
            }
            if missing := sorted(required - set(record)):
                raise ValueError(f"metadata fields missing: {missing}")
            accession = str(record["uniprot_id"]).upper()
            if (
                str(record["afdb_returned_accession"]).upper() != accession
                or int(record["afdb_exact_record_count"]) < 1
            ):
                failures.append(
                    {
                        "polymer_entity_id": str(identifier),
                        "failure_code": "exact_afdb_identity_unresolved",
                        "message": "AFDB metadata lacks one exact candidate-accession record",
                    }
                )
                continue
            cluster = str(record["sequence_cluster"])
            if not _CLUSTER_PATTERN.match(cluster):
                failures.append(
                    {
                        "polymer_entity_id": str(identifier),
                        "failure_code": "redundancy_binding_unresolved",
                        "message": "Authoritative 30% RCSB cluster is unavailable",
                    }
                )
                continue
            pdb_id = str(record["pdb_id"]).lower()
            chain = str(record["chain_id"])
            pair_id = f"{pdb_id}_{chain}__{accession}"
            auth_asym_ids = str(record.get("auth_asym_ids") or "")
            rows.append(
                {
                    "candidate_id": pair_id,
                    "pair_id": pair_id,
                    "polymer_entity_id": str(record["polymer_entity_id"]),
                    "canonical_accession": accession,
                    "pdb_id": pdb_id,
                    "pdb_chain": chain,
                    "sequence_cluster": cluster,
                    "identity_resolved": True,
                    "identifier_unambiguous": bool(chain and accession),
                    "afdb_exact_record_resolved": True,
                    "canonical_sequence_available": bool(record["afdb_sequence_present"]),
                    "supported_length": 100 <= int(record["length"]) <= 500,
                    "auth_chain_count": len([item for item in auth_asym_ids.split(";") if item]),
                    "length": int(record["length"]),
                    "experimental_method": str(record["experimental_method"]),
                    "resolution": float(record["resolution"]) if pd.notna(record["resolution"]) else float("nan"),
                    "organism": record.get("organism"),
                    "taxonomy_id": record.get("taxonomy_id"),
                    "afdb_model_entity_id": record["afdb_model_entity_id"],
                    "afdb_exact_record_count": int(
                        record["afdb_exact_record_count"]
                    ),
                    "afdb_exact_model_ids": record.get("afdb_exact_model_ids"),
                    "afdb_fragment_resolution_deferred": bool(
                        record.get("afdb_fragment_resolution_deferred", False)
                    ),
                    "afdb_record_count": int(record.get("afdb_record_count", 1)),
                    "rcsb_entity_source_url": record.get("rcsb_entity_source_url"),
                    "rcsb_entry_source_url": record.get("rcsb_entry_source_url"),
                    "afdb_metadata_source_url": record.get("afdb_metadata_source_url"),
                    "candidate_origin": "EXPANSION_SAME_QUERY_UNPROBED",
                    "current_frame_overlap": False,
                }
            )
        except Exception as exc:  # noqa: BLE001 - discovery is record-isolated
            failures.append(
                {
                    "polymer_entity_id": str(identifier),
                    "failure_code": "metadata_provider_failure",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    eligible = pd.DataFrame(rows)
    if not eligible.empty:
        eligible = eligible.drop_duplicates("polymer_entity_id", keep="first")
    counts = Counter(item["failure_code"] for item in failures)
    return eligible.reset_index(drop=True), {
        "attempted_count": len(identifier_values),
        "eligible_count": len(eligible),
        "failed_count": len(failures),
        "failure_code_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def fetch_prospective_candidate_metadata(identifier: str) -> dict[str, Any]:
    """Fetch metadata and partition exact-accession AFDB records without selection."""

    row = fetch_candidate_metadata(identifier, require_single_uniprot=True)
    if row is None:
        raise LookupError("RCSB candidate identity is not uniquely resolvable")
    accession = str(row["uniprot_id"]).upper()
    records = get_afdb_prediction_records(accession)
    exact = [
        record
        for record in records
        if str(record.get("uniprotAccession") or "").upper() == accession
        and not bool(record.get("isComplex", False))
    ]
    if not exact:
        raise LookupError(
            f"Expected at least one exact AFDB record for {accession}; observed 0"
        )
    model_ids = sorted(
        {
            str(record.get("modelEntityId") or record.get("entryId") or "")
            for record in exact
            if str(record.get("modelEntityId") or record.get("entryId") or "")
        }
    )
    if not model_ids:
        raise LookupError(f"Exact AFDB records for {accession} lack model identities")
    sequence_present = any(
        bool(
            str(record.get("uniprotSequence") or record.get("sequence") or "")
            .replace("\n", "")
            .replace(" ", "")
        )
        for record in exact
    )
    entry_id, entity_id = str(identifier).split("_", 1)
    return {
        **row,
        "afdb_returned_accession": accession,
        "afdb_exact_record_count": len(exact),
        "afdb_record_count": len(records),
        "afdb_model_entity_id": model_ids[0] if len(exact) == 1 else None,
        "afdb_exact_model_ids": ";".join(model_ids),
        "afdb_sequence_present": sequence_present,
        "afdb_fragment_resolution_deferred": len(exact) > 1,
        "rcsb_entity_source_url": (
            f"{RCSB_DATA_API}/polymer_entity/{entry_id}/{entity_id}"
        ),
        "rcsb_entry_source_url": f"{RCSB_DATA_API}/entry/{entry_id}",
        "afdb_metadata_source_url": AFDB_PREDICTION_API.format(
            accession=accession
        ),
    }


def _global_primary_order(primaries: pd.DataFrame) -> pd.DataFrame:
    output = primaries.copy()
    output["_resolution"] = pd.to_numeric(output["resolution"], errors="coerce").fillna(float("inf"))
    output["_chains"] = pd.to_numeric(output["auth_chain_count"], errors="coerce").fillna(float("inf"))
    output = output.sort_values(
        [
            "provenance_complete",
            "identity_resolved",
            "identifier_unambiguous",
            "afdb_exact_record_resolved",
            "canonical_sequence_available",
            "supported_length",
            "_chains",
            "_resolution",
            "sequence_cluster",
            "polymer_entity_id",
            "canonical_accession",
            "pdb_chain",
            "pair_id",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    output["prospective_global_primary_rank"] = range(1, len(output) + 1)
    return output.drop(columns=["_resolution", "_chains"])


def build_expansion_design(
    inputs: ExpansionInputs,
    external_candidates: pd.DataFrame,
    discovery_audit: dict[str, Any],
) -> ExpansionResult:
    """Build the complete prospective expansion design from frozen metadata."""

    original = inputs.original_candidates.copy()
    external = external_candidates.copy()
    if not external.empty:
        if external["polymer_entity_id"].duplicated().any() or external["pair_id"].duplicated().any():
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "external_candidate_identity_duplicate",
                "Prospective external candidates must have unique identities",
            )
        if set(external["polymer_entity_id"]).intersection(original["polymer_entity_id"]):
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "external_original_identity_overlap",
                "Expansion candidates overlap the immutable original frame",
            )
    universe = pd.concat([original, external], ignore_index=True, sort=False)
    universe["target_class"] = universe["sequence_cluster"].map(
        lambda value: classify_target_cluster(str(value), inputs.cluster_state)
    )
    selected = select_primary_and_reserves(universe)
    selected["admitted_covered_flag"] = selected["target_class"].eq(
        ADMITTED_COVERED_CLUSTER
    )
    selected["reserve_eligible"] = selected["selection_role"].eq(
        "RESERVE_CANDIDATE"
    )
    selected["selection_reason"] = selected["selection_role"].map(
        {
            "PRIMARY_CANDIDATE": "one_cluster_first_outcome_blind_rank_1",
            "RESERVE_CANDIDATE": "same_cluster_outcome_blind_reserve",
            "INELIGIBLE_ADMITTED_COVERED": "cluster_already_admitted_covered",
        }
    )
    primaries = _global_primary_order(
        selected.loc[selected["selection_role"].eq("PRIMARY_CANDIDATE")]
    )
    rank_by_pair = primaries.set_index("pair_id")["prospective_global_primary_rank"]
    selected["prospective_global_primary_rank"] = selected["pair_id"].map(rank_by_pair).astype("Int64")

    reserve_counts = (
        selected.loc[selected["selection_role"].eq("RESERVE_CANDIDATE")]
        .groupby("sequence_cluster")
        .size()
    )
    target_columns = [
        "sequence_cluster",
        "target_class",
        "pair_id",
        "candidate_id",
        "polymer_entity_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "candidate_origin",
        "prospective_global_primary_rank",
        "selection_reason",
    ]
    targets = primaries[target_columns].rename(
        columns={
            "pair_id": "primary_pair_id",
            "candidate_id": "primary_candidate_id",
            "polymer_entity_id": "primary_polymer_entity_id",
            "canonical_accession": "primary_canonical_accession",
            "pdb_id": "primary_pdb_id",
            "pdb_chain": "primary_pdb_chain",
            "candidate_origin": "primary_candidate_origin",
            "selection_reason": "selection_provenance",
        }
    )
    targets["reserve_candidate_count"] = (
        targets["sequence_cluster"].map(reserve_counts).fillna(0).astype(int)
    )
    targets["target_source_category"] = targets["target_class"].map(
        lambda value: "NEW_EXTERNAL" if value == NEW_EXTERNAL_CLUSTER else "RESCUE_EXISTING"
    )
    targets = targets.sort_values("prospective_global_primary_rank", kind="stable").reset_index(drop=True)

    evidence = estimate_cluster_yield(
        inputs.original_candidates,
        inputs.original_outcomes,
    )
    actionable_count = len(targets)
    scenarios = plan_capacity_scenarios(
        starting_capacity=63,
        targets=(
            inputs.config.minimum_target,
            inputs.config.primary_target,
            inputs.config.stretch_target,
        ),
        evidence=evidence,
        actionable_cluster_count=actionable_count,
    )
    wave = plan_expansion_waves(
        scenarios,
        primary_target=inputs.config.primary_target,
        actionable_cluster_count=actionable_count,
    )
    wave1_ids = set(
        targets.head(wave["recommended_wave1_primary_n"])["primary_pair_id"]
    )
    targets["wave_assignment"] = targets["primary_pair_id"].map(
        lambda pair_id: "WAVE_1" if pair_id in wave1_ids else "WAVE_2_OR_RESERVE_TRIGGER"
    )
    selected["wave_assignment"] = selected["pair_id"].map(
        targets.set_index("primary_pair_id")["wave_assignment"]
    )
    selected.loc[selected["selection_role"].eq("RESERVE_CANDIDATE"), "wave_assignment"] = "SAME_CLUSTER_RESERVE"
    selected.loc[selected["admitted_covered_flag"], "wave_assignment"] = "NOT_IN_EXPANSION"

    original_ranked = rank_prospective_candidates(inputs.original_candidates)
    representatives = original_ranked.loc[
        original_ranked["prospective_rank_within_cluster"].eq(1),
        ["sequence_cluster", "pair_id", "polymer_entity_id", "canonical_accession", "pdb_id", "pdb_chain"],
    ].rename(
        columns={
            "pair_id": "representative_pair_id",
            "polymer_entity_id": "representative_polymer_entity_id",
            "canonical_accession": "representative_canonical_accession",
            "pdb_id": "representative_pdb_id",
            "pdb_chain": "representative_pdb_chain",
        }
    )
    cluster_state = inputs.cluster_state.merge(
        representatives, on="sequence_cluster", how="left", validate="one_to_one"
    )

    target_counts = targets["target_class"].value_counts()
    primary_gap = inputs.config.primary_target - 63
    status = (
        SCALE1_EXPANSION_DESIGN_READY
        if actionable_count >= primary_gap
        else SCALE1_EXPANSION_UNIVERSE_CAPACITY_INSUFFICIENT
    )
    summary = {
        "schema_version": "dual-uq.scale1-expansion-design.v1",
        "scale1_expansion_design_status": status,
        "current_capacity": {
            "formally_admitted_proteins": 135,
            "admitted_30pct_clusters": 63,
            "pending_proteins": 51,
            "pending_new_clusters": 15,
            "n_nr_upper": 78,
        },
        "original_redundancy_diagnosis": {
            "original_frame_candidates": 213,
            "represented_30pct_clusters": len(cluster_state),
            "admitted_clusters": int(cluster_state["cluster_state"].eq("ADMITTED_COVERED").sum()),
            "clusters_containing_pending_no_admitted": int(cluster_state["contains_pending_no_admitted"].sum()),
            "pure_pending_only_clusters": int(cluster_state["cluster_state"].eq("PENDING_ONLY").sum()),
            "failure_only_clusters": int(cluster_state["cluster_state"].eq("FAILURE_ONLY").sum()),
            "mixed_unadmitted_clusters": int(cluster_state["cluster_state"].eq("MIXED_UNADMITTED").sum()),
        },
        "expansion_universe": {
            "n_candidates_total": len(selected),
            "n_clusters_total": int(selected["sequence_cluster"].nunique()),
            "n_admitted_covered_clusters": int(selected.loc[selected["admitted_covered_flag"], "sequence_cluster"].nunique()),
            "n_unadmitted_existing_clusters": int(targets["target_class"].ne(NEW_EXTERNAL_CLUSTER).sum()),
            "n_new_external_clusters": int(target_counts.get(NEW_EXTERNAL_CLUSTER, 0)),
            "n_discovered_target_clusters": actionable_count,
            "n_eligible_target_clusters": actionable_count,
            "n_actionable_target_clusters": actionable_count,
            "n_primary_candidates": actionable_count,
            "n_clusters_with_reserve_capacity": int((targets["reserve_candidate_count"] > 0).sum()),
            "n_total_reserve_candidates": int(targets["reserve_candidate_count"].sum()),
            "discovery_audit": discovery_audit,
        },
        "historical_planning_evidence": evidence,
        "capacity_targets": {
            "minimum": inputs.config.minimum_target,
            "primary": inputs.config.primary_target,
            "stretch": inputs.config.stretch_target,
            "required_additional_clusters": {
                str(inputs.config.minimum_target): inputs.config.minimum_target - 63,
                str(inputs.config.primary_target): inputs.config.primary_target - 63,
                str(inputs.config.stretch_target): inputs.config.stretch_target - 63,
            },
        },
        "wave_design": wave,
        "optional_diversity_metadata": {
            "status": "DESCRIPTIVE_LOCAL_METADATA_ONLY",
            "fields": ["organism", "taxonomy_id", "experimental_method", "resolution", "length"],
            "blocks_primary_design": False,
        },
        "interpretation": {
            "original_213_frame_role": "SOURCE_NEUTRAL_REFERENCE_CENSUS",
            "expansion_frame_role": "PROSPECTIVE_CLUSTER_COVERAGE_ENRICHMENT",
            "combined_prevalence_estimation_forbidden": True,
            "combined_frames_capacity_use_only": True,
        },
        "limitations": [
            "Planning yields are retrospective point estimates, not guarantees.",
            "Expansion candidates are coverage-enriched and cannot estimate source-population prevalence.",
            "No coordinate, mapping, common-mask, or admission outcome has been observed for external candidates.",
        ],
    }
    manifest = {
        "schema_version": "dual-uq.scale1-expansion-design.v1",
        "scale1_expansion_design_status": status,
        "upstream_artifacts": list(inputs.input_artifacts),
        "implementation_provenance": {
            "path": "src/dual_uq/dataset/expansion.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "redundancy_convention": {
            "provider": "RCSB rcsb_cluster_membership",
            "identity_threshold_percent": 30,
            "cluster_id_serialization": "30:<cluster_id>",
            "reclustered": False,
        },
        "original_discovery": {
            "query_path": inputs.config.query_ref,
            "query_sha256": inputs.config.expected_query_sha256,
            "identifier_list_path": inputs.config.identifier_list_ref,
            "identifier_list_sha256": inputs.config.expected_identifier_list_sha256,
            "query_identifier_count": len(inputs.query_identifiers),
            "original_probed_count": 220,
            "original_eligible_count": 213,
            "same_query_unprobed_attempted": discovery_audit["attempted_count"],
            "scientific_eligibility_rule_changed": False,
            "source_query_changes": [],
        },
        "target_cluster_definitions": {
            NEW_EXTERNAL_CLUSTER: "cluster absent from original 213-frame",
            PENDING_ONLY_RESCUE: "pending-only original cluster without admitted member",
            FAILURE_ONLY_RESCUE: "failure-only original cluster without admitted member",
            MIXED_UNADMITTED_RESCUE: "mixed pending/failure original cluster without admitted member",
            ADMITTED_COVERED_CLUSTER: "original cluster with at least one admitted member",
        },
        "candidate_ranking_rule": [
            "complete provenance and identity",
            "unambiguous identifiers",
            "exact AFDB accession record",
            "canonical sequence field availability",
            "supported query length",
            "fewer author chains",
            "better experimental resolution",
            "stable canonical identity tie-break",
        ],
        "planning_scenario_definitions": {
            "CONSERVATIVE_REALIZED_CLUSTER_YIELD": "admitted original clusters / all original clusters",
            "CLUSTER_AWARE_PRIMARY_YIELD": "admitted retrospective outcome-blind primaries / all retrospective primaries",
            "OPTIMISTIC_CANDIDATE_YIELD": "candidate admission yield with unique-target-cluster assumption; non-guaranteed",
        },
        "capacity_targets": [100, 120, 150],
        "network_operations": {
            "metadata_only": bool(discovery_audit["attempted_count"]),
            "same_query_unprobed_identifier_count": discovery_audit["attempted_count"],
            "coordinate_downloads": 0,
        },
        "ORIGINAL_213_FRAME_IMMUTABLE": True,
        "EXPANSION_IS_CLUSTER_COVERAGE_ENRICHED": True,
        "EXPANSION_IS_OUTCOME_BLIND": True,
        "COMBINED_PREVALENCE_ESTIMATION_FORBIDDEN": True,
        "ADMISSION_CONTRACT_UNCHANGED": True,
        "SCALE1B_V1_IMMUTABLE": True,
        "SCALE1B_V1_SCORING_HOLD": True,
        "NO_COORDINATE_ACQUISITION": True,
        "NO_FORMAL_ADMISSION": True,
        "NO_PROTEINMPNN": True,
        "NO_SCALE1B_V2_SELECTION": True,
    }
    result = ExpansionResult(
        status=status,
        cluster_state=cluster_state.reset_index(drop=True),
        expansion_universe=selected.sort_values(
            ["current_frame_overlap", "sequence_cluster", "prospective_rank_within_cluster"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True),
        target_clusters=targets,
        planning_scenarios=scenarios,
        summary=summary,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )
    validate_expansion_result(result)
    return result


def validate_expansion_result(result: ExpansionResult) -> None:
    """Reconcile the canonical result before any output is written."""

    if (
        len(result.cluster_state) != 87
        or int(result.cluster_state["contains_admitted"].sum()) != 63
        or result.cluster_state["sequence_cluster"].duplicated().any()
        or result.expansion_universe["pair_id"].duplicated().any()
        or result.target_clusters["sequence_cluster"].duplicated().any()
        or len(result.planning_scenarios) != 9
    ):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "result_cardinality_mismatch",
            "Scale-1E0 result cardinalities do not reconcile",
        )
    primaries = result.expansion_universe.loc[
        result.expansion_universe["selection_role"].eq("PRIMARY_CANDIDATE")
    ]
    if (
        len(primaries) != len(result.target_clusters)
        or primaries["sequence_cluster"].duplicated().any()
        or primaries["admitted_covered_flag"].any()
        or set(primaries["sequence_cluster"]) != set(result.target_clusters["sequence_cluster"])
    ):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "one_cluster_first_violation",
            "Primary and target cluster identities do not reconcile",
        )
    forbidden = _FORBIDDEN_SELECTION_COLUMNS.intersection(
        column.lower() for column in result.expansion_universe.columns
    )
    if forbidden:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "outcome_dependent_field_present",
            f"Expansion universe contains forbidden outcomes: {sorted(forbidden)}",
        )
    if set(result.planning_scenarios["desired_capacity"]) != {100, 120, 150}:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "capacity_target_mismatch",
            "Planning scenarios must preserve 100/120/150 targets",
        )


def _render_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "immutable_expansion_artifact_conflict",
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
            if path.stat().st_size == temporary.stat().st_size and sha256_file(path) == sha256_file(temporary):
                return "reused_identical"
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "immutable_expansion_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        os.link(temporary, path)
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(result: ExpansionResult, paths: ProjectPaths) -> None:
    for record in result.input_artifacts:
        if _is_historical_a3_implementation_artifact(record):
            continue
        path = _resolve(paths, str(record["path"]))
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "upstream_input_changed",
                f"Frozen input changed: {record.get('label', record['path'])}",
            )
    implementation = result.manifest["implementation_provenance"]
    implementation_path = _resolve(paths, str(implementation["path"]))
    if sha256_file(implementation_path) != implementation["sha256"]:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "implementation_changed_before_materialization",
            "Scale-1E0 implementation changed after result construction",
        )


def materialize_expansion_design(
    result: ExpansionResult,
    paths: ProjectPaths,
    *,
    config: ExpansionConfig | None = None,
) -> dict[str, Any]:
    """Write four tables and summary, then the immutable manifest last."""

    config = config or ExpansionConfig()
    validate_expansion_result(result)
    _rehash_inputs(result, paths)
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "cluster_state": output_root / "scale1_expansion_cluster_state.parquet",
        "expansion_universe": output_root / "scale1_expansion_universe.parquet",
        "cluster_targets": output_root / "scale1_expansion_cluster_targets.parquet",
        "planning_scenarios": output_root / "scale1_expansion_planning_scenarios.parquet",
        "design_summary": output_root / "scale1_expansion_design_summary.json",
        "manifest": output_root / "scale1_expansion_design_manifest.json",
    }
    frames = {
        "cluster_state": result.cluster_state,
        "expansion_universe": result.expansion_universe,
        "cluster_targets": result.target_clusters,
        "planning_scenarios": result.planning_scenarios,
    }
    write_status = {
        name: _write_immutable_parquet(targets[name], frame)
        for name, frame in frames.items()
    }
    write_status["design_summary"] = _write_immutable_bytes(
        targets["design_summary"], _render_json(result.summary)
    )
    outputs = {
        name: {
            "path": paths.logical_ref(targets[name]),
            "rows": len(frame),
            "sha256": sha256_file(targets[name]),
        }
        for name, frame in frames.items()
    }
    outputs["design_summary"] = {
        "path": paths.logical_ref(targets["design_summary"]),
        "sha256": sha256_file(targets["design_summary"]),
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
        "scale1_expansion_design_status": result.status,
        "write_status": write_status,
        "outputs": outputs,
        "next_task": (
            "SCALE1_EXPANSION_WAVE1_ACQUISITION_AND_ADMISSION_FREEZE"
            if result.status == SCALE1_EXPANSION_DESIGN_READY
            else None
        ),
        "next_task_started": False,
    }


def _reuse_existing_release(
    paths: ProjectPaths,
    inputs: ExpansionInputs,
    config: ExpansionConfig,
) -> dict[str, Any] | None:
    output_root = _resolve(paths, config.output_root_ref)
    targets = {
        "cluster_state": output_root / "scale1_expansion_cluster_state.parquet",
        "expansion_universe": output_root / "scale1_expansion_universe.parquet",
        "cluster_targets": output_root / "scale1_expansion_cluster_targets.parquet",
        "planning_scenarios": output_root / "scale1_expansion_planning_scenarios.parquet",
        "design_summary": output_root / "scale1_expansion_design_summary.json",
        "manifest": output_root / "scale1_expansion_design_manifest.json",
    }
    existing = {name for name, path in targets.items() if path.exists()}
    if not existing:
        return None
    if existing != set(targets):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "partial_existing_expansion_release",
            "A partial immutable Scale-1E0 release already exists",
            existing=sorted(existing),
            missing=sorted(set(targets) - existing),
        )
    try:
        manifest = json.loads(targets["manifest"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "existing_manifest_unreadable",
            "Existing Scale-1E0 manifest cannot be parsed",
        ) from exc
    for name, record in (manifest.get("outputs") or {}).items():
        if name not in targets or not isinstance(record, dict):
            continue
        if sha256_file(targets[name]) != str(record.get("sha256")):
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "existing_output_sha256_mismatch",
                f"Existing Scale-1E0 output changed: {name}",
            )
        if "rows" in record and len(pd.read_parquet(targets[name])) != int(
            record["rows"]
        ):
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "existing_output_row_count_mismatch",
                f"Existing Scale-1E0 output row count changed: {name}",
            )
    expected_upstream = {
        (str(item["path"]), str(item["sha256"]))
        for item in inputs.input_artifacts
    }
    observed_upstream = {
        (str(item["path"]), str(item["sha256"]))
        for item in (manifest.get("upstream_artifacts") or [])
    }
    if expected_upstream != observed_upstream:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "existing_release_input_binding_mismatch",
            "Existing Scale-1E0 release binds a different frozen input set",
        )
    implementation = manifest.get("implementation_provenance") or {}
    implementation_path = _resolve(paths, str(implementation.get("path", "")))
    if (
        not implementation_path.is_file()
        or sha256_file(implementation_path) != implementation.get("sha256")
    ):
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "existing_release_implementation_mismatch",
            "Existing Scale-1E0 release binds a different implementation",
        )
    status = str(manifest.get("scale1_expansion_design_status"))
    allowed = {
        SCALE1_EXPANSION_DESIGN_READY,
        SCALE1_EXPANSION_UNIVERSE_CAPACITY_INSUFFICIENT,
        SCALE1_EXPANSION_SOURCE_QUERY_CHANGE_REQUIRED,
    }
    if status not in allowed:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "existing_release_status_invalid",
            "Existing Scale-1E0 release has an invalid terminal status",
        )
    return {
        "scale1_expansion_design_status": status,
        "write_status": {name: "reused_identical" for name in targets},
        "outputs": {
            **(manifest.get("outputs") or {}),
            "manifest": {
                "path": paths.logical_ref(targets["manifest"]),
                "sha256": sha256_file(targets["manifest"]),
            },
        },
        "network_access_used": False,
        "next_task": (
            "SCALE1_EXPANSION_WAVE1_ACQUISITION_AND_ADMISSION_FREEZE"
            if status == SCALE1_EXPANSION_DESIGN_READY
            else None
        ),
        "next_task_started": False,
    }


def run_expansion_design(
    paths: ProjectPaths,
    *,
    config: ExpansionConfig | None = None,
    provider: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate inputs, discover the fixed suffix, plan, and materialize."""

    config = config or ExpansionConfig()
    inputs = load_expansion_inputs(paths, config)
    if existing := _reuse_existing_release(paths, inputs, config):
        return existing
    eligible, discovery_audit = discover_remaining_query_candidates(
        inputs.remaining_query_identifiers,
        provider or fetch_prospective_candidate_metadata,
    )
    result = build_expansion_design(inputs, eligible, discovery_audit)
    materialized = materialize_expansion_design(
        result, paths, config=config
    )
    return {**materialized, "network_access_used": True}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_required_columns",
            f"{label} is missing required columns: {missing}",
            missing=missing,
        )


def reconstruct_cluster_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct disjoint original-frame cluster states."""

    _require_columns(
        frame,
        {"pair_id", "sequence_cluster", "formal_admission_status"},
        "original frame",
    )
    if frame["sequence_cluster"].isna().any() or not frame[
        "sequence_cluster"
    ].astype(str).str.match(_CLUSTER_PATTERN).all():
        raise ExpansionError(
            BLOCKED_REDUNDANCY_BINDING,
            "invalid_cluster_binding",
            "Every original-frame protein must have an authoritative 30% cluster",
        )

    records: list[dict[str, Any]] = []
    for cluster_id, group in frame.groupby("sequence_cluster", sort=True):
        statuses = set(group["formal_admission_status"].astype(str))
        has_admitted = _ADMITTED in statuses
        has_pending = _PENDING in statuses
        has_failure = any(value not in {_ADMITTED, _PENDING} for value in statuses)
        if has_admitted:
            state = "ADMITTED_COVERED"
        elif has_pending and has_failure:
            state = "MIXED_UNADMITTED"
        elif has_pending:
            state = "PENDING_ONLY"
        elif has_failure:
            state = "FAILURE_ONLY"
        else:
            raise ExpansionError(
                BLOCKED_INPUT_INTEGRITY,
                "unrecognized_cluster_state",
                f"Cluster {cluster_id} has no recognized admission state",
            )
        records.append(
            {
                "sequence_cluster": str(cluster_id),
                "cluster_state": state,
                "protein_count": len(group),
                "admitted_protein_count": int(
                    group["formal_admission_status"].eq(_ADMITTED).sum()
                ),
                "pending_protein_count": int(
                    group["formal_admission_status"].eq(_PENDING).sum()
                ),
                "failure_protein_count": int(
                    (~group["formal_admission_status"].isin([_ADMITTED, _PENDING])).sum()
                ),
                "contains_admitted": has_admitted,
                "contains_pending_no_admitted": has_pending and not has_admitted,
            }
        )
    return pd.DataFrame(records)


def classify_target_cluster(cluster_id: str, cluster_state: pd.DataFrame) -> str:
    """Classify a cluster relative to the frozen original frame."""

    matches = cluster_state.loc[
        cluster_state["sequence_cluster"].astype(str).eq(str(cluster_id))
    ]
    if matches.empty:
        return NEW_EXTERNAL_CLUSTER
    state = str(matches.iloc[0]["cluster_state"])
    return {
        "ADMITTED_COVERED": ADMITTED_COVERED_CLUSTER,
        "PENDING_ONLY": PENDING_ONLY_RESCUE,
        "FAILURE_ONLY": FAILURE_ONLY_RESCUE,
        "MIXED_UNADMITTED": MIXED_UNADMITTED_RESCUE,
    }[state]


def _reject_outcome_columns(frame: pd.DataFrame) -> None:
    present = sorted(
        column
        for column in frame.columns
        if column.lower() in _FORBIDDEN_SELECTION_COLUMNS
    )
    if present:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "outcome_dependent_selection_input",
            f"Prospective ranking received forbidden outcome columns: {present}",
            columns=present,
        )


def rank_prospective_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen, metadata-only deterministic ranking tuple."""

    _reject_outcome_columns(candidates)
    required = {
        "candidate_id",
        "pair_id",
        "polymer_entity_id",
        "canonical_accession",
        "pdb_id",
        "pdb_chain",
        "sequence_cluster",
        "identity_resolved",
        "identifier_unambiguous",
        "afdb_exact_record_resolved",
        "canonical_sequence_available",
        "supported_length",
        "auth_chain_count",
        "resolution",
    }
    _require_columns(candidates, required, "prospective candidates")
    output = candidates.copy()
    output["provenance_complete"] = output[
        [
            "identity_resolved",
            "identifier_unambiguous",
            "afdb_exact_record_resolved",
            "canonical_sequence_available",
            "supported_length",
        ]
    ].fillna(False).all(axis=1)
    output["_resolution_rank"] = pd.to_numeric(
        output["resolution"], errors="coerce"
    ).fillna(float("inf"))
    output["_chain_count_rank"] = pd.to_numeric(
        output["auth_chain_count"], errors="coerce"
    ).fillna(float("inf"))
    output = output.sort_values(
        [
            "sequence_cluster",
            "provenance_complete",
            "identity_resolved",
            "identifier_unambiguous",
            "afdb_exact_record_resolved",
            "canonical_sequence_available",
            "supported_length",
            "_chain_count_rank",
            "_resolution_rank",
            "polymer_entity_id",
            "canonical_accession",
            "pdb_chain",
            "pair_id",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)
    output["prospective_rank_within_cluster"] = (
        output.groupby("sequence_cluster", sort=False).cumcount() + 1
    )
    return output.drop(columns=["_resolution_rank", "_chain_count_rank"])


def select_primary_and_reserves(candidates: pd.DataFrame) -> pd.DataFrame:
    """Allocate at most one capacity-bearing primary per target cluster."""

    _require_columns(candidates, {"target_class"}, "target candidates")
    ranked = rank_prospective_candidates(candidates)
    ranked["primary_eligible"] = (
        ranked["target_class"].ne(ADMITTED_COVERED_CLUSTER)
        & ranked["provenance_complete"]
    )
    eligible_rank = (
        ranked.loc[ranked["primary_eligible"]]
        .groupby("sequence_cluster", sort=False)
        .cumcount()
        .add(1)
    )
    ranked["eligible_rank_within_cluster"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked.loc[ranked["primary_eligible"], "eligible_rank_within_cluster"] = eligible_rank.astype("Int64")
    ranked["selection_role"] = "INELIGIBLE_ADMITTED_COVERED"
    ranked.loc[
        ranked["primary_eligible"] & ranked["eligible_rank_within_cluster"].eq(1),
        "selection_role",
    ] = "PRIMARY_CANDIDATE"
    ranked.loc[
        ranked["primary_eligible"] & ranked["eligible_rank_within_cluster"].gt(1),
        "selection_role",
    ] = "RESERVE_CANDIDATE"
    return ranked


def estimate_cluster_yield(
    prospective_metadata: pd.DataFrame,
    admission_outcomes: pd.DataFrame,
) -> dict[str, float | int]:
    """Estimate historical yields after outcome-blind primary selection."""

    _require_columns(
        admission_outcomes,
        {"pair_id", "formal_admission_status"},
        "historical outcomes",
    )
    ranked = rank_prospective_candidates(prospective_metadata)
    primaries = ranked.loc[
        ranked["prospective_rank_within_cluster"].eq(1),
        ["pair_id", "sequence_cluster"],
    ]
    joined = primaries.merge(
        admission_outcomes[["pair_id", "formal_admission_status"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if joined["formal_admission_status"].isna().any():
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "retrospective_outcome_missing",
            "A retrospectively selected primary lacks a frozen admission outcome",
        )
    all_joined = prospective_metadata[["pair_id", "sequence_cluster"]].merge(
        admission_outcomes[["pair_id", "formal_admission_status"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    cluster_success = all_joined.groupby("sequence_cluster")[
        "formal_admission_status"
    ].apply(lambda values: bool(values.eq(_ADMITTED).any()))
    n_clusters = len(cluster_success)
    primary_admitted = int(joined["formal_admission_status"].eq(_ADMITTED).sum())
    candidate_admitted = int(all_joined["formal_admission_status"].eq(_ADMITTED).sum())
    return {
        "n_unique_clusters_in_frame": n_clusters,
        "n_clusters_with_at_least_one_admitted": int(cluster_success.sum()),
        "n_clusters_with_no_admitted": int((~cluster_success).sum()),
        "cluster_level_admission_success_fraction": float(cluster_success.mean()),
        "retrospective_primary_count": len(joined),
        "retrospective_primary_admitted_count": primary_admitted,
        "primary_candidate_admission_fraction": float(primary_admitted / len(joined)),
        "candidate_count": len(all_joined),
        "candidate_admitted_count": candidate_admitted,
        "candidate_level_admission_fraction": float(candidate_admitted / len(all_joined)),
    }


def plan_capacity_scenarios(
    *,
    starting_capacity: int,
    targets: tuple[int, ...],
    evidence: dict[str, float | int],
    actionable_cluster_count: int,
) -> pd.DataFrame:
    """Create transparent point-planning scenarios for each capacity target."""

    definitions = (
        (
            "CONSERVATIVE_REALIZED_CLUSTER_YIELD",
            "cluster_level_admission_success_fraction",
            False,
        ),
        (
            "CLUSTER_AWARE_PRIMARY_YIELD",
            "primary_candidate_admission_fraction",
            False,
        ),
        (
            "OPTIMISTIC_CANDIDATE_YIELD",
            "candidate_level_admission_fraction",
            True,
        ),
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        gap = max(0, int(target) - int(starting_capacity))
        for scenario, key, optimistic in definitions:
            yield_value = float(evidence[key])
            requirement = math.ceil(gap / yield_value) if gap and yield_value > 0 else gap
            rows.append(
                {
                    "starting_capacity": int(starting_capacity),
                    "desired_capacity": int(target),
                    "required_new_admitted_clusters": int(gap),
                    "planning_scenario": scenario,
                    "yield_basis": key,
                    "yield_value": yield_value,
                    "estimated_primary_candidate_requirement": int(requirement),
                    "actionable_cluster_count": int(actionable_cluster_count),
                    "actionable_universe_feasible": bool(
                        actionable_cluster_count >= requirement
                    ),
                    "optimistic_non_guaranteed": bool(optimistic),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["desired_capacity", "planning_scenario"], kind="stable"
    ).reset_index(drop=True)


def plan_expansion_waves(
    scenarios: pd.DataFrame,
    *,
    primary_target: int,
    actionable_cluster_count: int,
) -> dict[str, Any]:
    """Recommend Wave 1 from non-optimistic primary-target scenarios."""

    subset = scenarios.loc[
        scenarios["desired_capacity"].eq(primary_target)
        & ~scenarios["optimistic_non_guaranteed"].astype(bool)
    ]
    if len(subset) != 2:
        raise ExpansionError(
            BLOCKED_INPUT_INTEGRITY,
            "planning_scenario_incomplete",
            "Primary target requires two non-optimistic planning scenarios",
        )
    required = int(subset["estimated_primary_candidate_requirement"].max())
    return {
        "primary_target": int(primary_target),
        "recommended_wave1_primary_n": min(required, int(actionable_cluster_count)),
        "non_optimistic_requirement": required,
        "actionable_capacity_sufficient": bool(actionable_cluster_count >= required),
        "reserve_strategy": "same_cluster_reserves_only_after_distinct_cluster_primaries",
        "wave2_trigger_inputs": [
            "remaining_admitted_cluster_gap",
            "structured_acquisition_attrition",
            "structured_formal_admission_attrition",
            "reserve_availability",
            "remaining_actionable_target_clusters",
        ],
        "wave2_outcome_independent": True,
    }
