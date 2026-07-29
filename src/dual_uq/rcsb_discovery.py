from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .afdb import get_afdb_prediction_metadata
from .net import post_json, request_json


RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_API = "https://data.rcsb.org/rest/v1/core"


def build_polymer_entity_query(
    *,
    rows: int,
    methods: list[str],
    resolution_max: float,
    length_min: int,
    length_max: int,
    sequence_identity_grouping: int | None = 30,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "in",
                        "value": methods,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_max,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": length_min,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less_or_equal",
                        "value": length_max,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_name"
                        ),
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }

    if sequence_identity_grouping is not None:
        query["request_options"]["group_by"] = {
            "aggregation_method": "sequence_identity",
            "similarity_cutoff": sequence_identity_grouping,
        }
        query["request_options"]["group_by_return_type"] = "representatives"

    return query


def search_polymer_entities(query: dict[str, Any]) -> list[str]:
    try:
        response = post_json(RCSB_SEARCH_API, query)
    except RuntimeError:
        # Some server versions reject group-by for certain combinations.
        fallback = dict(query)
        fallback["request_options"] = dict(query["request_options"])
        fallback["request_options"].pop("group_by", None)
        fallback["request_options"].pop("group_by_return_type", None)
        response = post_json(RCSB_SEARCH_API, fallback)

    result_set = response.get("result_set", [])
    identifiers = []
    for item in result_set:
        identifier = str(item.get("identifier", "")).strip()
        if identifier:
            identifiers.append(identifier)
    return identifiers


def _reference_uniprots(entity: dict[str, Any]) -> list[str]:
    container = entity.get("rcsb_polymer_entity_container_identifiers", {})
    identifiers = container.get("reference_sequence_identifiers") or []
    accessions = []
    for item in identifiers:
        if str(item.get("database_name", "")).upper() != "UNIPROT":
            continue
        accession = str(item.get("database_accession", "")).strip().upper()
        if accession:
            accessions.append(accession)
    return sorted(set(accessions))


def _sequence_cluster(entity: dict[str, Any]) -> str | None:
    memberships = entity.get("rcsb_cluster_membership") or []
    preferred = {30: 0, 40: 1, 50: 2, 70: 3, 90: 4, 95: 5, 100: 6}
    candidates = []
    for item in memberships:
        identity = item.get("identity")
        cluster_id = item.get("cluster_id")
        if identity is None or cluster_id is None:
            continue
        try:
            identity_int = int(identity)
        except (TypeError, ValueError):
            continue
        candidates.append((preferred.get(identity_int, 99), identity_int, str(cluster_id)))
    if not candidates:
        return None
    _, identity, cluster_id = sorted(candidates)[0]
    return f"{identity}:{cluster_id}"


def _first_organism(entity: dict[str, Any]) -> tuple[str | None, int | None]:
    organisms = entity.get("rcsb_entity_source_organism") or []
    if not organisms:
        return None, None
    organism = organisms[0]
    name = organism.get("ncbi_scientific_name")
    taxonomy = organism.get("ncbi_taxonomy_id")
    try:
        taxonomy_int = int(taxonomy) if taxonomy is not None else None
    except (TypeError, ValueError):
        taxonomy_int = None
    return name, taxonomy_int


def _global_plddt(metadata: dict[str, Any]) -> float | None:
    for key in ("globalMetricValue", "globalMetric", "confidenceScore"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def fetch_candidate_metadata(
    polymer_entity_id: str,
    *,
    require_single_uniprot: bool = True,
) -> dict[str, Any] | None:
    if "_" not in polymer_entity_id:
        return None
    entry_id, entity_id = polymer_entity_id.split("_", 1)
    if len(entry_id) != 4:
        return None

    entity = request_json(
        f"{RCSB_DATA_API}/polymer_entity/{entry_id}/{entity_id}"
    )
    entry = request_json(f"{RCSB_DATA_API}/entry/{entry_id}")

    accessions = _reference_uniprots(entity)
    if require_single_uniprot and len(accessions) != 1:
        return None
    if not accessions:
        return None

    container = entity.get("rcsb_polymer_entity_container_identifiers", {})
    auth_asym_ids = container.get("auth_asym_ids") or []
    if not auth_asym_ids:
        return None

    entity_poly = entity.get("entity_poly") or {}
    length = entity_poly.get("rcsb_sample_sequence_length")
    if length is None:
        sequence = entity_poly.get("pdbx_seq_one_letter_code_can") or ""
        length = len("".join(str(sequence).split()))
    try:
        length_int = int(length)
    except (TypeError, ValueError):
        return None

    entry_info = entry.get("rcsb_entry_info") or {}
    resolution_values = entry_info.get("resolution_combined") or []
    resolution = None
    if resolution_values:
        try:
            resolution = float(min(resolution_values))
        except (TypeError, ValueError):
            resolution = None

    methods = [
        str(item.get("method"))
        for item in (entry.get("exptl") or [])
        if item.get("method")
    ]
    organism, taxonomy = _first_organism(entity)

    protein_chain_count = entry_info.get("deposited_polymer_entity_instance_count")
    protein_entity_count = entry_info.get("polymer_entity_count_protein")

    return {
        "polymer_entity_id": polymer_entity_id,
        "pdb_id": entry_id.lower(),
        "entity_id": entity_id,
        "chain_id": str(auth_asym_ids[0]),
        "auth_asym_ids": ";".join(str(x) for x in auth_asym_ids),
        "uniprot_id": accessions[0],
        "all_uniprot_ids": ";".join(accessions),
        "length": length_int,
        "experimental_method": ";".join(methods),
        "resolution": resolution,
        "organism": organism,
        "taxonomy_id": taxonomy,
        "sequence_cluster": _sequence_cluster(entity),
        "protein_chain_count": protein_chain_count,
        "protein_entity_count": protein_entity_count,
        "initial_release_date": (
            (entry.get("rcsb_accession_info") or {}).get("initial_release_date")
        ),
    }


def discover_candidates(
    identifiers: Iterable[str],
    *,
    require_single_uniprot: bool,
    afdb_probe_limit: int,
    request_pause_seconds: float,
) -> pd.DataFrame:
    rows = []
    for index, identifier in enumerate(identifiers):
        if index >= afdb_probe_limit:
            break
        try:
            row = fetch_candidate_metadata(
                identifier,
                require_single_uniprot=require_single_uniprot,
            )
            if row is None:
                continue
            metadata = get_afdb_prediction_metadata(row["uniprot_id"])
            row["afdb_available"] = True
            row["afdb_global_plddt"] = _global_plddt(metadata)
            row["afdb_model_entity_id"] = metadata.get("modelEntityId")
            row["afdb_version"] = metadata.get("latestVersion")
            row["discovery_status"] = "eligible"
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "polymer_entity_id": identifier,
                    "discovery_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if request_pause_seconds > 0:
            time.sleep(request_pause_seconds)

    return pd.DataFrame(rows)


def _length_bin(length: int, bins: list[list[int]]) -> str:
    for lower, upper in bins:
        if int(lower) <= int(length) <= int(upper):
            return f"{lower}-{upper}"
    return "outside"


def assign_provisional_strata(
    table: pd.DataFrame,
    definitions: dict[str, Any],
) -> pd.DataFrame:
    output = table.copy()
    output["provisional_stratum"] = "balanced_background"

    high = definitions["high_global_confidence"]
    high_mask = (
        output["afdb_global_plddt"].fillna(-np.inf) >= high["global_plddt_min"]
    ) & (output["length"] <= high["length_max"])
    output.loc[high_mask, "provisional_stratum"] = "high_global_confidence"

    low = definitions["lower_global_confidence"]
    low_mask = output["afdb_global_plddt"].fillna(np.inf) <= low["global_plddt_max"]
    output.loc[low_mask, "provisional_stratum"] = "lower_global_confidence"

    long_cfg = definitions["long_backbone_proxy"]
    long_mask = output["length"] >= long_cfg["length_min"]
    # Long backbones are retained as a dedicated PAE-enrichment stream unless
    # they are already clearly low-confidence.
    output.loc[
        long_mask & ~low_mask,
        "provisional_stratum",
    ] = "long_backbone_proxy"
    return output


def select_screening_pool(
    eligible: pd.DataFrame,
    *,
    quotas: dict[str, int],
    definitions: dict[str, Any],
    length_bins: list[list[int]],
    unique_sequence_cluster: bool,
    max_per_organism: int,
    seed: int,
    prefer_single_protein_chain: bool,
) -> pd.DataFrame:
    table = assign_provisional_strata(eligible, definitions)
    table["length_bin"] = table["length"].map(lambda x: _length_bin(int(x), length_bins))
    table["single_chain_proxy"] = (
        pd.to_numeric(table["protein_chain_count"], errors="coerce").fillna(999) == 1
    )

    rng = np.random.default_rng(seed)
    table["_random"] = rng.random(len(table))
    table = table.sort_values(
        [
            "single_chain_proxy",
            "resolution",
            "afdb_global_plddt",
            "_random",
        ],
        ascending=[
            False if prefer_single_protein_chain else True,
            True,
            False,
            True,
        ],
        na_position="last",
    )

    selected_rows = []
    used_uniprots: set[str] = set()
    used_clusters: set[str] = set()
    organism_counts: Counter[str] = Counter()

    def can_take(row: pd.Series) -> bool:
        uniprot = str(row["uniprot_id"])
        cluster = str(row.get("sequence_cluster") or "")
        organism = str(row.get("organism") or "UNKNOWN")
        if uniprot in used_uniprots:
            return False
        if unique_sequence_cluster and cluster and cluster != "nan" and cluster in used_clusters:
            return False
        if organism_counts[organism] >= max_per_organism:
            return False
        return True

    def take(row: pd.Series, selection_reason: str) -> None:
        record = row.to_dict()
        record["selection_reason"] = selection_reason
        record["selected_for_screening"] = True
        selected_rows.append(record)
        used_uniprots.add(str(row["uniprot_id"]))
        cluster = str(row.get("sequence_cluster") or "")
        if cluster and cluster != "nan":
            used_clusters.add(cluster)
        organism_counts[str(row.get("organism") or "UNKNOWN")] += 1

    # Round-robin across length bins inside each quota to avoid a length-skewed pool.
    for stratum, quota in quotas.items():
        subset = table[table["provisional_stratum"] == stratum]
        taken = 0
        while taken < int(quota):
            progress = False
            for length_bin in [f"{a}-{b}" for a, b in length_bins]:
                candidates = subset[subset["length_bin"] == length_bin]
                for _, row in candidates.iterrows():
                    if can_take(row):
                        take(row, f"quota:{stratum}")
                        taken += 1
                        progress = True
                        break
                if taken >= int(quota):
                    break
            if not progress:
                break

    target_total = int(sum(quotas.values()))
    if len(selected_rows) < target_total:
        for _, row in table.iterrows():
            if can_take(row):
                take(row, "quota_fill")
            if len(selected_rows) >= target_total:
                break

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected

    selected = selected.drop(columns=["_random"], errors="ignore")
    selected.insert(0, "screening_index", range(1, len(selected) + 1))
    return selected
