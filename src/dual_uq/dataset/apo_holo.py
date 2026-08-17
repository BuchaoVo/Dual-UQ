"""Provenance-aware experimental apo/holo cohort construction.

The first layer of this module contains deterministic, evidence-preserving
contracts used by retrieval and pair-validity workflows.  It intentionally does
not depend on inverse-folding models or on downstream structural-response
measurements.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_json, atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.geometry import kabsch_align, pairwise_distances, rmsd
from dual_uq.net import download_file, post_json, request_json
from dual_uq.pdb_archive import RCSB_MMCIF_URL, fetch_pdb_mmcif
from dual_uq.rcsb_discovery import RCSB_DATA_API, RCSB_SEARCH_API, fetch_candidate_metadata
from dual_uq.sifts import SIFTS_XML_URL, fetch_sifts_xml

RCSB_GRAPHQL_API = "https://data.rcsb.org/graphql"


def _build_experimental_protein_query(rows: int) -> dict[str, Any]:
    if rows < 1:
        raise ValueError("rows must be positive")
    return {
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
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": int(rows)},
            "results_content_type": ["experimental"],
        },
    }


def build_holo_anchor_query(rows: int) -> dict[str, Any]:
    """Build a compact RCSB search query for experimental protein entities with ligands."""

    query = _build_experimental_protein_query(rows)
    query["query"]["nodes"].insert(
        1,
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater",
                "value": 0,
            },
        },
    )
    return query


def build_uniprot_entity_query(accession: str, rows: int) -> dict[str, Any]:
    """Build an exact UniProt polymer-entity search query."""

    normalized = str(accession).strip().upper()
    if not normalized:
        raise ValueError("accession must be non-empty")
    if rows < 1:
        raise ValueError("rows must be positive")
    # Counterpart search must include apo candidates with no non-polymer
    # components; only the anchor query is ligand-bearing.
    query = _build_experimental_protein_query(rows)
    query["query"]["nodes"].append(
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": normalized,
            },
        }
    )
    return query


def discover_holo_anchor_ids(search_response: dict[str, Any]) -> list[str]:
    """Extract unique polymer-entity identifiers from a Search response."""

    identifiers: list[str] = []
    seen: set[str] = set()
    for item in search_response.get("result_set") or []:
        identifier = str(item.get("identifier") or "").strip()
        if identifier and identifier not in seen:
            seen.add(identifier)
            identifiers.append(identifier)
    return identifiers


def _cache_shard_dir(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.stem}.{suffix}")


def _migrate_json_cache_shards(cache_path: Path, shard_dir: Path, key_name: str) -> None:
    """Split an older monolithic cache once, without rewriting it per request."""

    marker = shard_dir / ".migration_complete"
    if marker.is_file() or not cache_path.is_file():
        return
    try:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return
    entries = loaded.get(key_name) if isinstance(loaded, dict) else None
    if not isinstance(entries, dict):
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    for key, value in entries.items():
        shard_path = shard_dir / f"{key}.json"
        if not shard_path.exists():
            atomic_write_json(shard_path, value)
    atomic_write_new_bytes(marker, b"migrated\n")


def _empty_discovery_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pdb_id",
            "polymer_entity_id",
            "entity_id",
            "chain_id",
            "auth_asym_ids",
            "uniprot_id",
            "all_uniprot_ids",
            "length",
            "sequence",
            "experimental_method",
            "resolution",
            "nonpolymer_entity_ids",
            "nonpolymer_entity_count",
            "assembly_ids",
            "metadata_status",
            "metadata_error",
            "census_status",
            "census_failure_code",
            "census_failure_message",
            "discovery_stage",
            "holo_anchor_candidate",
        ]
    )


def _search_identifiers(
    query: dict[str, Any],
    *,
    cache_path: Path | None = None,
    cache_key: str | None = None,
) -> tuple[list[str], int | None]:
    key = cache_key or "default"
    shard_path: Path | None = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        shard_dir = _cache_shard_dir(cache_path, "pages")
        _migrate_json_cache_shards(cache_path, shard_dir, "pages")
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
    cached: Any = None
    if shard_path is not None and shard_path.is_file():
        try:
            cached = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            cached = None
    elif cache_path is not None and cache_path.is_file():
        # Small legacy fallback for caches written before sharding.
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = (loaded.get("pages") or {}).get(key) if isinstance(loaded, dict) else None
        except (OSError, TypeError, ValueError):
            cached = None
    if isinstance(cached, dict) and cached.get("status") == "complete":
        response = cached.get("response") or {}
    else:
        try:
            response = post_json(RCSB_SEARCH_API, query)
        except Exception as exc:
            if shard_path is not None:
                atomic_write_json(shard_path, {
                    "status": "network_unresolved",
                    "error": str(exc),
                })
            raise
        if shard_path is not None:
            atomic_write_json(shard_path, {"status": "complete", "response": response})
    return discover_holo_anchor_ids(response), (
        int(response["total_count"]) if response.get("total_count") is not None else None
    )


def _entry_id_from_identifier(identifier: str) -> str:
    return str(identifier).split("_", 1)[0].strip().lower()


def _graphql_entry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    entries = data.get("entries") or data.get("entry") or []
    if isinstance(entries, dict):
        entries = [entries]
    return [item for item in entries if isinstance(item, dict)]


def normalize_graphql_entities(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize batched GraphQL entry/entity metadata without inferring state."""

    columns = [
        "pdb_id",
        "polymer_entity_id",
        "entity_id",
        "chain_id",
        "auth_asym_ids",
        "uniprot_id",
        "all_uniprot_ids",
        "length",
        "sequence",
        "experimental_method",
        "resolution",
        "nonpolymer_entity_ids",
        "nonpolymer_entity_count",
        "assembly_ids",
        "metadata_status",
        "metadata_error",
    ]
    rows: list[dict[str, Any]] = []
    for entry in _graphql_entry_rows(payload):
        entry_id = str(entry.get("rcsb_id") or entry.get("id") or "").lower()
        entry_info = entry.get("rcsb_entry_info") or {}
        identifiers = entry.get("rcsb_entry_container_identifiers") or {}
        polymer_entities = entry.get("polymer_entities") or entry.get("polymer_entity") or []
        if isinstance(polymer_entities, dict):
            polymer_entities = [polymer_entities]
        if not polymer_entities:
            rows.append(
                {
                    "pdb_id": entry_id or None,
                    "polymer_entity_id": None,
                    "metadata_status": "partial",
                }
            )
            continue
        for entity in polymer_entities:
            entity_id = str(entity.get("rcsb_id") or entity.get("id") or "").lower()
            entity_poly = entity.get("entity_poly") or {}
            container = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            references = container.get("reference_sequence_identifiers") or []
            uniprots = [
                str(item.get("database_accession") or "").upper()
                for item in references
                if str(item.get("database_name") or "").upper() == "UNIPROT"
                and item.get("database_accession")
            ]
            asym_ids = container.get("auth_asym_ids") or container.get("asym_ids") or []
            nonpoly_ids = identifiers.get("non_polymer_entity_ids") or []
            row = {
                "pdb_id": entry_id or None,
                "polymer_entity_id": entity_id or None,
                "entity_id": container.get("entity_id") or entity.get("entity_id"),
                "chain_id": str(asym_ids[0]) if asym_ids else None,
                "auth_asym_ids": ";".join(str(value) for value in asym_ids),
                "uniprot_id": uniprots[0] if uniprots else None,
                "all_uniprot_ids": ";".join(sorted(set(uniprots))) or None,
                "length": entity_poly.get("rcsb_sample_sequence_length")
                or entity_poly.get("pdbx_seq_one_letter_code_can_length"),
                "sequence": entity_poly.get("pdbx_seq_one_letter_code_can"),
                "experimental_method": (entry_info.get("experimental_method") or [None])[0]
                if isinstance(entry_info.get("experimental_method"), list)
                else entry_info.get("experimental_method"),
                "resolution": (entry_info.get("resolution_combined") or [None])[0]
                if isinstance(entry_info.get("resolution_combined"), list)
                else entry_info.get("resolution_combined"),
                "nonpolymer_entity_ids": ";".join(str(value) for value in nonpoly_ids),
                "nonpolymer_entity_count": len(nonpoly_ids),
                "assembly_ids": ";".join(
                    str(value) for value in identifiers.get("assembly_ids") or []
                ),
                "metadata_status": "complete" if entity_id and uniprots else "partial",
            }
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def _graphql_query() -> str:
    return """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_info { experimental_method resolution_combined }
        rcsb_entry_container_identifiers { non_polymer_entity_ids assembly_ids }
        polymer_entities {
          rcsb_id
          entity_poly { pdbx_seq_one_letter_code_can rcsb_sample_sequence_length }
          rcsb_polymer_entity_container_identifiers {
            entity_id asym_ids auth_asym_ids
            reference_sequence_identifiers { database_name database_accession }
          }
        }
      }
    }
    """


def fetch_graphql_metadata_batch(ids: list[str], cache_path: Path) -> pd.DataFrame:
    """Fetch one deterministic GraphQL metadata chunk with resumable cache."""

    normalized_ids = sorted({_entry_id_from_identifier(value) for value in ids if value})
    if not normalized_ids:
        return pd.DataFrame()
    cache_path = Path(cache_path)
    cache_identity = {"ids": normalized_ids, "query": _graphql_query()}
    key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    shard_dir = _cache_shard_dir(cache_path, "chunks")
    _migrate_json_cache_shards(cache_path, shard_dir, "chunks")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{key}.json"
    cached: Any = None
    if shard_path.is_file():
        try:
            cached = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            cached = None
    elif cache_path.is_file():
        # Small legacy fallback for a cache whose migration marker was not yet written.
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = (loaded.get("chunks") or {}).get(key) if isinstance(loaded, dict) else None
        except (OSError, ValueError, TypeError):
            cached = None
    if isinstance(cached, dict) and cached.get("status") == "complete":
        return normalize_graphql_entities(cached.get("payload") or {})
    try:
        payload = post_json(
            RCSB_GRAPHQL_API,
            {"query": _graphql_query(), "variables": {"ids": normalized_ids}},
        )
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(shard_path, {
            "ids": normalized_ids,
            "status": "network_unresolved",
            "error": str(exc),
        })
        return pd.DataFrame(
            [
                {
                    "pdb_id": value,
                    "metadata_status": "network_unresolved",
                    "metadata_error": str(exc),
                }
                for value in normalized_ids
            ]
        )
    if payload.get("errors"):
        error_text = "; ".join(
            str(item.get("message") or item)
            for item in payload.get("errors", [])
            if isinstance(item, dict)
        ) or "GraphQL response contained errors"
        atomic_write_json(shard_path, {
            "ids": normalized_ids,
            "status": "network_unresolved",
            "error": error_text,
            "payload": payload,
        })
        return pd.DataFrame(
            [
                {
                    "pdb_id": value,
                    "metadata_status": "network_unresolved",
                    "metadata_error": error_text,
                }
                for value in normalized_ids
            ]
        )
    atomic_write_json(shard_path, {"ids": normalized_ids, "status": "complete", "payload": payload})
    return normalize_graphql_entities(payload)


def _decorate_discovery_rows(
    rows: pd.DataFrame,
    *,
    stage: str,
    anchor: bool,
) -> pd.DataFrame:
    if rows.empty:
        return _empty_discovery_frame()
    result = rows.copy()
    result["census_status"] = result["metadata_status"].map(
        lambda value: "metadata_available"
        if value in {"complete", "partial"}
        else "metadata_failed"
    )
    result["discovery_stage"] = stage
    result["holo_anchor_candidate"] = bool(anchor)
    return result


def _bind_metadata_to_search_entities(
    metadata: pd.DataFrame,
    identifiers: list[str],
) -> pd.DataFrame:
    """Bind entry-level GraphQL results back to requested entity IDs.

    Missing GraphQL entity rows remain explicit partial/unresolved records;
    they are not silently treated as absent from the RCSB corpus.
    """

    requested = [str(identifier).strip().lower() for identifier in identifiers if identifier]
    if not requested:
        return _empty_discovery_frame()
    result = metadata.copy() if not metadata.empty else _empty_discovery_frame()
    if "polymer_entity_id" in result:
        result["polymer_entity_id"] = result["polymer_entity_id"].astype("string").str.lower()
        result = result.loc[result["polymer_entity_id"].isin(set(requested))].copy()
    observed = set(result.get("polymer_entity_id", pd.Series(dtype=str)).dropna().astype(str))
    missing = [identifier for identifier in requested if identifier not in observed]
    if missing:
        errors = result.get("metadata_error", pd.Series(dtype=str))
        error_text = str(errors.dropna().iloc[0]) if len(errors.dropna()) else None
        extra = _empty_discovery_frame()
        extra = extra.reindex(range(len(missing)))
        extra["pdb_id"] = [identifier.split("_", 1)[0] for identifier in missing]
        extra["polymer_entity_id"] = missing
        extra["metadata_status"] = (
            "network_unresolved" if error_text else "partial"
        )
        extra["metadata_error"] = error_text
        result = pd.concat([result, extra], ignore_index=True, sort=False)
    return result


def discover_holo_anchor_accessions(
    project_root: Path,
    rows: int,
    cache_path: Path | None = None,
    *,
    start: int = 0,
) -> pd.DataFrame:
    """Discover ligand-bearing experimental entities and their UniProt anchors.

    Search results are only identifiers.  GraphQL metadata is fetched in a
    resumable batch, then restricted back to the exact polymer entities
    returned by Search; no entry-level sibling is inferred to be an anchor.
    """

    query = build_holo_anchor_query(rows)
    query["request_options"]["paginate"]["start"] = int(start)
    search_cache = cache_path or (
        Path(project_root) / "data/raw/apo_holo/metadata_cache/holo_anchor_search.json"
    )
    try:
        identifiers, total_count = _search_identifiers(
            query,
            cache_path=search_cache,
            cache_key=f"start={start};rows={rows}",
        )
    except Exception as exc:  # noqa: BLE001
        result = _empty_discovery_frame()
        result.attrs.update(
            total_count=None,
            retrieved_count=0,
            census_truncated=True,
            discovery_failure=str(exc),
        )
        return result
    metadata_cache = Path(project_root) / "data/raw/apo_holo/metadata_cache/holo_anchor.json"
    if cache_path is not None and cache_path.name != search_cache.name:
        metadata_cache = cache_path
    metadata = fetch_graphql_metadata_batch(identifiers, metadata_cache)
    metadata = _bind_metadata_to_search_entities(metadata, identifiers)
    result = _decorate_discovery_rows(metadata, stage="holo_anchor", anchor=True)
    result.attrs.update(
        total_count=total_count,
        retrieved_count=len(identifiers),
        census_truncated=bool(total_count is not None and len(identifiers) < total_count),
        discovery_query="RCSB Search experimental protein entities with non-polymer components",
    )
    return result


def _build_uniprot_group_query(accessions: list[str], rows: int) -> dict[str, Any]:
    if len(accessions) == 1:
        return build_uniprot_entity_query(accessions[0], rows)
    return {
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
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": (
                                    "rcsb_polymer_entity_container_identifiers."
                                    "reference_sequence_identifiers.database_accession"
                                ),
                                "operator": "exact_match",
                                "value": accession,
                            },
                        }
                        for accession in accessions
                    ],
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": int(rows)},
            "results_content_type": ["experimental"],
        },
    }


def discover_uniprot_counterparts(
    accessions: Iterable[str],
    project_root: Path,
    cache_path: Path | None = None,
    rows: int = 1_000,
    max_results: int | None = None,
) -> pd.DataFrame:
    """Find experimental polymer entities for the exact anchor accessions."""

    normalized = sorted({str(value).strip().upper() for value in accessions if str(value).strip()})
    if not normalized:
        result = _empty_discovery_frame()
        result.attrs.update(total_count=0, retrieved_count=0, census_truncated=False)
        return result
    metadata_cache = cache_path or (
        Path(project_root) / "data/raw/apo_holo/metadata_cache/uniprot_counterparts.json"
    )
    search_cache = Path(project_root) / "data/raw/apo_holo/metadata_cache/uniprot_search.json"
    all_frames: list[pd.DataFrame] = []
    total_count: int | None = 0
    total_identifiers = 0
    truncated = False
    for chunk_index in range(0, len(normalized), 100):
        chunk = normalized[chunk_index : chunk_index + 100]
        start = 0
        while True:
            page_rows = rows
            if max_results is not None:
                remaining = max_results - total_identifiers
                if remaining <= 0:
                    break
                page_rows = min(page_rows, remaining)
            query = _build_uniprot_group_query(chunk, page_rows)
            query["request_options"]["paginate"]["start"] = int(start)
            try:
                identifiers, page_total = _search_identifiers(
                    query,
                    cache_path=search_cache,
                    cache_key=f"chunk={chunk_index};start={start};rows={page_rows}",
                )
            except Exception as exc:  # noqa: BLE001
                unresolved = _empty_discovery_frame()
                unresolved = unresolved.reindex(range(len(chunk)))
                unresolved["polymer_entity_id"] = [
                    f"network_unresolved_{value}" for value in chunk
                ]
                unresolved["metadata_status"] = "network_unresolved"
                unresolved["metadata_error"] = str(exc)
                all_frames.append(
                    _decorate_discovery_rows(
                        unresolved, stage="uniprot_counterpart", anchor=False
                    )
                )
                truncated = True
                break
            if page_total is not None:
                total_count = (total_count or 0) + page_total if start == 0 else total_count
            if not identifiers:
                break
            metadata = fetch_graphql_metadata_batch(identifiers, metadata_cache)
            if not metadata.empty and "uniprot_id" in metadata:
                metadata = metadata.loc[
                    metadata["uniprot_id"].isna()
                    | metadata["uniprot_id"].isin(normalized)
                ].copy()
            metadata = _bind_metadata_to_search_entities(metadata, identifiers)
            all_frames.append(
                _decorate_discovery_rows(
                    metadata, stage="uniprot_counterpart", anchor=False
                )
            )
            total_identifiers += len(identifiers)
            if max_results is not None and total_identifiers >= max_results:
                truncated = True
                break
            if (
                page_total is None
                or len(identifiers) < page_rows
                or start + len(identifiers) >= page_total
            ):
                break
            start += len(identifiers)
        if max_results is not None and total_identifiers >= max_results:
            break
    result = (
        pd.concat(all_frames, ignore_index=True, sort=False)
        if all_frames
        else _empty_discovery_frame()
    )
    if not result.empty and "polymer_entity_id" in result:
        result = result.drop_duplicates("polymer_entity_id", keep="first")
    result.attrs.update(
        total_count=total_count,
        retrieved_count=total_identifiers,
        # ``total_count`` is the sum of per-accession-group Search counts.
        # The same polymer entity can reference more than one accession and is
        # deliberately de-duplicated below, so comparing it to the unique row
        # count would falsely label a complete census as truncated.  Pagination
        # and network state are the authoritative completeness signals here.
        census_truncated=bool(truncated),
        discovery_query="RCSB Search exact UniProt accessions across experimental entities",
    )
    return result


def run_expanded_discovery(config: ApoHoloConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run holo-anchored discovery without fetching raw structure assets."""

    rows = config.max_polymer_entities or config.census_page_size
    anchor_pages: list[pd.DataFrame] = []
    anchor_start = 0
    anchor_total: int | None = None
    while True:
        page = discover_holo_anchor_accessions(
            config.project_root,
            rows,
            start=anchor_start,
        )
        anchor_pages.append(page)
        page_total = page.attrs.get("total_count")
        if page_total is not None:
            anchor_total = int(page_total)
        page_retrieved = int(page.attrs.get("retrieved_count", 0))
        if config.max_polymer_entities is not None:
            break
        if page_retrieved == 0 or anchor_start + page_retrieved >= (anchor_total or 0):
            break
        anchor_start += page_retrieved
    anchors = (
        pd.concat(anchor_pages, ignore_index=True, sort=False)
        if anchor_pages
        else _empty_discovery_frame()
    )
    if not anchors.empty and "polymer_entity_id" in anchors:
        anchors = anchors.drop_duplicates("polymer_entity_id", keep="first")
    anchors.attrs.update(
        total_count=anchor_total,
        retrieved_count=sum(int(page.attrs.get("retrieved_count", 0)) for page in anchor_pages),
        # Intermediate Search pages are expected to report ``retrieved <
        # total_count``.  Only the aggregate pagination result determines
        # completeness; otherwise every multi-page census is falsely marked
        # truncated.
        census_truncated=bool(
            config.max_polymer_entities is not None
            or (
                anchor_total is not None
                and sum(int(page.attrs.get("retrieved_count", 0)) for page in anchor_pages)
                < anchor_total
            )
            or any(bool(page.attrs.get("discovery_failure")) for page in anchor_pages)
        ),
    )
    accessions = anchors.get("uniprot_id", pd.Series(dtype=str)).dropna().astype(str).unique()
    counterparts = discover_uniprot_counterparts(
        accessions,
        config.project_root,
        rows=rows,
        max_results=config.max_polymer_entities,
    )
    frames = [frame for frame in (anchors, counterparts) if not frame.empty]
    if not frames:
        census = _empty_discovery_frame()
    else:
        census = pd.concat(frames, ignore_index=True, sort=False)
        if "polymer_entity_id" in census:
            census = census.drop_duplicates("polymer_entity_id", keep="first")
        census = census.sort_values(
            [column for column in ("uniprot_id", "polymer_entity_id") if column in census],
            kind="mergesort",
        ).reset_index(drop=True)
    anchor_accession_count = int(anchors.get("uniprot_id", pd.Series(dtype=str)).dropna().nunique())
    counterpart_accession_count = int(
        counterparts.get("uniprot_id", pd.Series(dtype=str)).dropna().nunique()
    )
    anchor_truncated = bool(anchors.attrs.get("census_truncated", False))
    counterpart_truncated = bool(counterparts.attrs.get("census_truncated", False))
    census.attrs.update(
        total_count=counterparts.attrs.get("total_count")
        if counterparts.attrs.get("total_count") is not None
        else anchors.attrs.get("total_count"),
        retrieved_count=len(census),
        census_truncated=bool(
            config.max_polymer_entities is not None
            or anchor_truncated
            or counterpart_truncated
        ),
        census_query="holo-anchored RCSB Search + batched GraphQL metadata",
        discovery_mode="holo_anchored_graphql",
        anchor_structure_count=len(anchors),
        anchor_protein_count=anchor_accession_count,
        counterpart_structure_count=len(counterparts),
        counterpart_protein_count=counterpart_accession_count,
    )
    anchor_summary = (
        anchors.groupby("uniprot_id", dropna=True, sort=True)
        .size()
        .rename("anchor_structure_count")
        .reset_index()
        if not anchors.empty and "uniprot_id" in anchors
        else pd.DataFrame(columns=["uniprot_id", "anchor_structure_count"])
    )
    anchor_summary.attrs.update(anchors.attrs)
    return census, anchor_summary


class ApoHoloError(ValueError):
    """Raised when apo/holo evidence violates a declared contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class LigandClass(str, Enum):
    BIOLOGICAL_SMALL_MOLECULE = "biological_small_molecule"
    COFACTOR = "cofactor"
    ION = "ion"
    ADDITIVE_OR_SOLVENT = "additive_or_solvent"
    AMBIGUOUS = "ambiguous"


class StateLabel(str, Enum):
    HOLO = "holo"
    APO = "apo"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ApoHoloConfig:
    """Configuration for census, admission, and geometry boundaries."""

    project_root: Path
    census_page_size: int = 1_000
    census_workers: int = 8
    max_polymer_entities: int | None = None
    min_common_fraction: float = 0.90
    min_sequence_identity: float = 0.95
    max_construct_mismatch: int = 0
    contact_cutoff_angstrom: float = 8.0

    def __post_init__(self) -> None:
        root = Path(self.project_root).expanduser().resolve()
        if self.census_page_size < 1:
            raise ValueError("census_page_size must be positive")
        if self.census_workers < 1:
            raise ValueError("census_workers must be positive")
        if self.max_polymer_entities is not None and self.max_polymer_entities < 1:
            raise ValueError("max_polymer_entities must be positive when supplied")
        if not 0.0 <= self.min_common_fraction <= 1.0:
            raise ValueError("min_common_fraction must be in [0, 1]")
        if not 0.0 <= self.min_sequence_identity <= 1.0:
            raise ValueError("min_sequence_identity must be in [0, 1]")
        if self.max_construct_mismatch < 0:
            raise ValueError("max_construct_mismatch must be non-negative")
        if self.contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be positive")
        object.__setattr__(self, "project_root", root)


_KNOWN_COFACTORS = {
    "FAD",
    "FMN",
    "HEM",
    "NAD",
    "NAP",
    "SAM",
    "SAH",
    "COA",
}
_KNOWN_BIOLOGICAL_LIGANDS = {
    "ADP",
    "AMP",
    "ATP",
    "GDP",
    "GTP",
    "CMP",
    "DCT",
    "DTP",
    "UMP",
    "UDP",
    "FPP",
    "IPP",
}
_KNOWN_IONS = {
    "CA",
    "CL",
    "FE",
    "K",
    "MG",
    "MN",
    "NA",
    "ZN",
}
_KNOWN_ADDITIVES = {
    "ACT",
    "DMS",
    "EDO",
    "GOL",
    "PEG",
    "SO4",
    "PO4",
    "TRS",
    "WAT",
}
_BIOLOGICAL_NAME_MARKERS = (
    "substrate",
    "product",
    "inhibitor",
    "nucleotide",
    "drug",
)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", ".", "?", "nan", "None"} else text


def classify_ligand_component(
    ccd_id: str | None,
    chemical_name: str | None,
    heavy_atom_count: int | None,
) -> LigandClass:
    """Classify one component conservatively from explicit CCD evidence.

    Unknown components remain ambiguous; the function never promotes an
    arbitrary non-polymer component to a biological ligand.
    """

    ccd = (_clean_text(ccd_id) or "").upper()
    name = (_clean_text(chemical_name) or "").lower()
    if ccd in _KNOWN_COFACTORS or "cofactor" in name:
        return LigandClass.COFACTOR
    if ccd in _KNOWN_ADDITIVES or any(token in name for token in ("solvent", "buffer", "sulfate", "glycerol", "water")):
        return LigandClass.ADDITIVE_OR_SOLVENT
    if ccd in _KNOWN_IONS or name.endswith(" ion") or "metal ion" in name:
        return LigandClass.ION
    if (
        ccd in _KNOWN_BIOLOGICAL_LIGANDS or any(token in name for token in _BIOLOGICAL_NAME_MARKERS)
    ) and (heavy_atom_count is None or int(heavy_atom_count) >= 4):
        return LigandClass.BIOLOGICAL_SMALL_MOLECULE
    return LigandClass.AMBIGUOUS


def classify_state(
    ligand_annotations: Iterable[dict[str, Any]],
    *,
    annotation_complete: bool = False,
) -> StateLabel:
    """Return holo only with explicit biological-ligand evidence.

    A complete authoritative non-polymer inventory with no biological/cofactor
    ligand supports an apo label only when no ambiguous organic component is
    present.  Missing or incomplete CCD evidence remains unresolved; absence
    of one CCD record is never treated as apo.
    """

    classes = []
    for annotation in ligand_annotations:
        value = annotation.get("ligand_class")
        try:
            classes.append(value if isinstance(value, LigandClass) else LigandClass(str(value)))
        except ValueError:
            classes.append(LigandClass.AMBIGUOUS)
    if any(
        ligand_class in classes
        for ligand_class in (
            LigandClass.BIOLOGICAL_SMALL_MOLECULE,
            LigandClass.COFACTOR,
        )
    ):
        return StateLabel.HOLO
    if annotation_complete and all(
        ligand_class in {LigandClass.ION, LigandClass.ADDITIVE_OR_SOLVENT}
        for ligand_class in classes
    ):
        return StateLabel.APO
    return StateLabel.UNRESOLVED


def sequence_identity(left: str | None, right: str | None) -> float | None:
    """Return exact identity on an explicitly aligned, equal-length overlap."""

    first = _clean_text(left)
    second = _clean_text(right)
    if first is None or second is None or len(first) != len(second) or not first:
        return None
    return float(sum(a == b for a, b in zip(first.upper(), second.upper())) / len(first))


def construct_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> float | None:
    """Return interval Jaccard overlap, preserving unknown bounds."""

    values = (left_start, left_end, right_start, right_end)
    if any(value is None for value in values):
        return None
    if left_start > left_end or right_start > right_end:
        return None
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return float(intersection / union) if union else None


def assembly_comparability(left: str | None, right: str | None) -> str:
    """Compare explicit assembly labels without inferring missing context."""

    first = (_clean_text(left) or "").casefold()
    second = (_clean_text(right) or "").casefold()
    if not first or not second:
        return "unknown"
    return "comparable" if first == second else "not_comparable"


def _resolution_key(row: pd.Series) -> float:
    values = [row.get("apo_resolution"), row.get("holo_resolution")]
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return max(finite) if finite else float("inf")


def select_primary_pair(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select one deterministic pair per protein without using outcome columns."""

    if pairs.empty:
        return pairs.copy()
    required = {"protein_id", "common_mapped_count", "sequence_identity"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ApoHoloError("schema_mismatch", f"candidate pairs missing columns: {missing}")
    work = pairs.copy()
    work["_common"] = pd.to_numeric(work["common_mapped_count"], errors="coerce").fillna(-1)
    work["_identity"] = pd.to_numeric(work["sequence_identity"], errors="coerce").fillna(-1.0)
    work["_construct"] = pd.to_numeric(work.get("construct_mismatch_count", 0), errors="coerce").fillna(10**9)
    work["_method"] = work.get("experimental_method_comparable", False).map(bool).astype(int)
    work["_resolution"] = work.apply(_resolution_key, axis=1)
    for column in ("apo_pdb_id", "holo_pdb_id"):
        if column not in work:
            work[column] = ""
    ordered = work.sort_values(
        ["protein_id", "_common", "_identity", "_construct", "_method", "_resolution", "apo_pdb_id", "holo_pdb_id"],
        ascending=[True, False, False, True, False, True, True, True],
        kind="mergesort",
    )
    result = ordered.groupby("protein_id", sort=False, as_index=False).head(1)
    return result.drop(columns=["_common", "_identity", "_construct", "_method", "_resolution"], errors="ignore").reset_index(drop=True)


def _census_query(page_size: int, start: int) -> dict[str, Any]:
    return {
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
            "paginate": {"start": int(start), "rows": int(page_size)},
            "results_content_type": ["experimental"],
        },
    }


def _entry_metadata(pdb_id: str) -> dict[str, Any]:
    entry = request_json(f"{RCSB_DATA_API}/entry/{pdb_id}")
    identifiers = entry.get("rcsb_entry_container_identifiers") or {}
    info = entry.get("rcsb_entry_info") or {}
    return {
        "nonpolymer_entity_ids": ";".join(
            str(value) for value in identifiers.get("non_polymer_entity_ids") or []
        ),
        "nonpolymer_entity_count": int(info.get("nonpolymer_entity_count") or 0),
        "assembly_ids": ";".join(str(value) for value in identifiers.get("assembly_ids") or []),
        "assembly_id_primary": str((identifiers.get("assembly_ids") or [None])[0]) if identifiers.get("assembly_ids") else None,
        "entry_url": f"{RCSB_DATA_API}/entry/{pdb_id}",
    }


def _census_record(identifier: str) -> dict[str, Any] | None:
    """Fetch one metadata record while preserving a structured failure row."""

    try:
        metadata = fetch_candidate_metadata(identifier, require_single_uniprot=True)
        if metadata is None:
            return None
        metadata.update(_entry_metadata(str(metadata["pdb_id"])))
        metadata["census_status"] = "metadata_available"
        metadata["census_source_url"] = (
            f"{RCSB_DATA_API}/polymer_entity/"
            f"{metadata['pdb_id']}/{metadata['entity_id']}"
        )
        return metadata
    except Exception as exc:  # noqa: BLE001
        return {
            "polymer_entity_id": identifier,
            "census_status": "metadata_failed",
            "census_failure_code": type(exc).__name__,
            "census_failure_message": str(exc),
        }


def run_metadata_census(config: ApoHoloConfig) -> pd.DataFrame:
    """Retrieve RCSB polymer-entity metadata without downloading raw structures."""

    rows: list[dict[str, Any]] = []
    total_count: int | None = None
    start = 0
    seen: set[str] = set()
    while True:
        if config.max_polymer_entities is not None and start >= config.max_polymer_entities:
            break
        page_limit = config.census_page_size
        if config.max_polymer_entities is not None:
            page_limit = min(page_limit, config.max_polymer_entities - start)
        response = post_json(RCSB_SEARCH_API, _census_query(page_limit, start))
        total_count = int(response.get("total_count") or 0)
        result_set = response.get("result_set") or []
        if not result_set:
            break
        identifiers = []
        for item in result_set:
            identifier = str(item.get("identifier") or "").strip()
            if identifier and identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)
        with ThreadPoolExecutor(max_workers=config.census_workers) as executor:
            for metadata in executor.map(_census_record, identifiers):
                if metadata is not None:
                    rows.append(metadata)
        start += len(result_set)
        if len(result_set) < page_limit:
            break

    census = pd.DataFrame(rows)
    if not census.empty and "polymer_entity_id" in census.columns:
        census = census.sort_values("polymer_entity_id", kind="mergesort").reset_index(drop=True)
    truncated = bool(total_count is not None and start < total_count)
    census.attrs["total_count"] = total_count
    census.attrs["retrieved_count"] = int(start)
    census.attrs["census_truncated"] = truncated
    census.attrs["census_query"] = "protein-only experimental polymer entities with UniProt reference"
    return census


def identify_candidate_groups(census: pd.DataFrame) -> pd.DataFrame:
    """Retain all structures in UniProt groups with at least two entities."""

    required = {"polymer_entity_id", "uniprot_id"}
    missing = sorted(required - set(census.columns))
    if missing:
        raise ApoHoloError("schema_mismatch", f"census missing columns: {missing}")
    work = census.copy()
    available = work["census_status"].eq("metadata_available") if "census_status" in work else pd.Series(True, index=work.index)
    counts = work.loc[available].groupby("uniprot_id")["polymer_entity_id"].transform("nunique")
    selected = work.loc[available & counts.ge(2)].copy()
    selected["candidate_group_status"] = "potential_apo_holo_group"
    return selected.sort_values(["uniprot_id", "polymer_entity_id"], kind="mergesort").reset_index(drop=True)


def _assembly_context_record(record: dict[str, Any]) -> dict[str, Any]:
    pdb_id = str(record["pdb_id"]).lower()
    assembly_id = record.get("assembly_id_primary") or "1"
    try:
        payload = request_json(f"{RCSB_DATA_API}/assembly/{pdb_id}/{assembly_id}")
        assembly = payload.get("pdbx_struct_assembly") or {}
        return {
            "polymer_entity_id": record["polymer_entity_id"],
            "assembly_oligomeric_details": assembly.get("oligomeric_details"),
            "assembly_oligomeric_count": assembly.get("oligomeric_count"),
            "assembly_details": assembly.get("details"),
            "assembly_status": "available",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "polymer_entity_id": record["polymer_entity_id"],
            "assembly_oligomeric_details": None,
            "assembly_oligomeric_count": None,
            "assembly_details": None,
            "assembly_status": "unavailable",
            "assembly_error": str(exc),
        }


def load_assembly_context(
    candidate_entities: pd.DataFrame,
    *,
    workers: int = 8,
) -> pd.DataFrame:
    """Fetch assembly context only after a UniProt candidate group is found."""

    if workers < 1:
        raise ValueError("workers must be positive")
    records = candidate_entities.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_assembly_context_record, records))
    return pd.DataFrame(rows)


def _asset_row(
    *,
    root: Path,
    asset_type: str,
    source_url: str,
    path: Path,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "asset_type": asset_type,
        "source_url": source_url,
        "relative_path": path.relative_to(root).as_posix(),
        "retrieval_status": status,
        "error": error,
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file():
        row["size_bytes"] = int(path.stat().st_size)
        row["sha256"] = sha256_file(path)
    return row


def acquire_candidate_assets(
    config: ApoHoloConfig,
    candidate_entities: pd.DataFrame,
) -> pd.DataFrame:
    """Acquire raw assets only for entities in the candidate table."""

    required = {"polymer_entity_id", "pdb_id", "uniprot_id"}
    missing = sorted(required - set(candidate_entities.columns))
    if missing:
        raise ApoHoloError("schema_mismatch", f"candidate entities missing columns: {missing}")
    root = config.project_root
    rows: list[dict[str, Any]] = []
    for record in candidate_entities.to_dict(orient="records"):
        pdb_id = str(record["pdb_id"]).lower()
        accession = str(record["uniprot_id"]).upper()
        entity_id = str(record["polymer_entity_id"])
        prefix = {"polymer_entity_id": entity_id, "pdb_id": pdb_id, "uniprot_id": accession}
        assets = [
            (
                "pdb_mmcif",
                RCSB_MMCIF_URL.format(pdb_id=pdb_id),
                root / "data/raw/apo_holo/pdb" / f"{pdb_id}.cif",
                lambda pdb_id=pdb_id, root=root: fetch_pdb_mmcif(pdb_id, root / "data/raw/apo_holo/pdb"),
            ),
            (
                "sifts_xml",
                SIFTS_XML_URL.format(pdb_id=pdb_id),
                root / "data/raw/apo_holo/sifts" / f"{pdb_id}.xml.gz",
                lambda pdb_id=pdb_id, root=root: fetch_sifts_xml(pdb_id, root / "data/raw/apo_holo/sifts"),
            ),
            (
                "uniprot_canonical_json",
                f"https://rest.uniprot.org/uniprotkb/{accession}.json",
                root / "data/raw/apo_holo/uniprot" / f"{accession}.json",
                lambda accession=accession, root=root: download_file(
                    f"https://rest.uniprot.org/uniprotkb/{accession}.json",
                    root / "data/raw/apo_holo/uniprot" / f"{accession}.json",
                ),
            ),
        ]
        for asset_type, source_url, expected_path, fetcher in assets:
            try:
                path = fetcher()
                row = _asset_row(root=root, asset_type=asset_type, source_url=source_url, path=Path(path), status="VALID")
            except Exception as exc:  # noqa: BLE001
                row = _asset_row(root=root, asset_type=asset_type, source_url=source_url, path=expected_path, status="FAILED", error=str(exc))
            row.update(prefix)
            rows.append(row)
        for entity_id in [
            value for value in str(record.get("nonpolymer_entity_ids") or "").split(";") if value
        ]:
            try:
                nonpoly = request_json(f"{RCSB_DATA_API}/nonpolymer_entity/{pdb_id}/{entity_id}")
                component = (nonpoly.get("pdbx_entity_nonpoly") or {}).get("comp_id")
                if not component:
                    continue
                source_url = f"{RCSB_DATA_API}/chemcomp/{component}"
                path = root / "data/raw/apo_holo/ccd" / f"{str(component).upper()}.json"
                path = Path(download_file(source_url, path))
                row = _asset_row(root=root, asset_type="ccd_json", source_url=source_url, path=path, status="VALID")
                row.update(prefix)
                row["nonpolymer_entity_id"] = entity_id
                row["ccd_id"] = str(component).upper()
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                row = _asset_row(
                    root=root,
                    asset_type="ccd_json",
                    source_url=f"{RCSB_DATA_API}/chemcomp/unknown",
                    path=root / "data/raw/apo_holo/ccd" / "unknown.json",
                    status="FAILED",
                    error=str(exc),
                )
                row.update(prefix)
                row["nonpolymer_entity_id"] = entity_id
                rows.append(row)
    return pd.DataFrame(rows)


def _mapping_count(mapping: Any) -> int | None:
    if mapping is None:
        return None
    if isinstance(mapping, pd.DataFrame) and "uniprot_residue_number" in mapping:
        return int(mapping["uniprot_residue_number"].dropna().nunique())
    return None


def _structure_state(record: dict[str, Any], ligand_annotations: dict[str, Any]) -> StateLabel:
    explicit = record.get("state_label")
    if explicit is not None:
        try:
            return explicit if isinstance(explicit, StateLabel) else StateLabel(str(explicit))
        except ValueError:
            return StateLabel.UNRESOLVED
    annotations = ligand_annotations.get(str(record.get("polymer_entity_id")), [])
    return classify_state(
        annotations,
        annotation_complete=bool(record.get("ligand_annotation_complete", False)),
    )


def build_candidate_pairs(
    structures: pd.DataFrame,
    mappings: dict[str, pd.DataFrame],
    ligand_annotations: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Construct every possible apo/holo pair and preserve exclusion reasons."""

    required = {"pdb_id", "polymer_entity_id", "uniprot_id"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise ApoHoloError("schema_mismatch", f"structures missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for accession, group in structures.groupby("uniprot_id", sort=True):
        records = group.to_dict(orient="records")
        states = {
            str(record["polymer_entity_id"]): _structure_state(record, ligand_annotations)
            for record in records
        }
        holo = [record for record in records if states[str(record["polymer_entity_id"])] is StateLabel.HOLO]
        apo = [record for record in records if states[str(record["polymer_entity_id"])] is StateLabel.APO]
        unresolved = [record for record in records if states[str(record["polymer_entity_id"])] is StateLabel.UNRESOLVED]
        if holo:
            if not apo:
                apo = unresolved
            pair_records = [(apo_record, holo_record) for apo_record in apo for holo_record in holo]
        else:
            # Preserve candidate pairs even when ligand annotations are
            # unresolved; this is evidence for exclusion, never an apo claim.
            pair_records = list(combinations(records, 2))
        for apo_record, holo_record in pair_records:
                apo_id = str(apo_record["polymer_entity_id"])
                holo_id = str(holo_record["polymer_entity_id"])
                apo_mapping = mappings.get(apo_id)
                holo_mapping = mappings.get(holo_id)
                common_count = None
                if isinstance(apo_mapping, pd.DataFrame) and isinstance(holo_mapping, pd.DataFrame):
                    common_count = len(
                        set(pd.to_numeric(apo_mapping.get("uniprot_residue_number"), errors="coerce").dropna().astype(int))
                        & set(pd.to_numeric(holo_mapping.get("uniprot_residue_number"), errors="coerce").dropna().astype(int))
                    )
                common_count = common_count if common_count is not None else int(min(_mapping_count(apo_mapping) or 0, _mapping_count(holo_mapping) or 0))
                canonical_length = pd.to_numeric(
                    pd.Series([apo_record.get("canonical_sequence_length", apo_record.get("length"))]), errors="coerce"
                ).iloc[0]
                sequence_id = apo_record.get("sequence_identity")
                if sequence_id is None:
                    sequence_id = holo_record.get("sequence_identity")
                identity = float(sequence_id) if sequence_id is not None and np.isfinite(float(sequence_id)) else None
                common_fraction = float(common_count / canonical_length) if canonical_length and np.isfinite(canonical_length) else None
                assembly_status = assembly_comparability(
                    apo_record.get("assembly_label", apo_record.get("oligomeric_details")),
                    holo_record.get("assembly_label", holo_record.get("oligomeric_details")),
                )
                method_status = (
                    str(apo_record.get("experimental_method")) == str(holo_record.get("experimental_method"))
                    if apo_record.get("experimental_method") is not None and holo_record.get("experimental_method") is not None
                    else None
                )
                reasons: list[str] = []
                state_pair_status = "resolved" if states[apo_id] is StateLabel.APO and states[holo_id] is StateLabel.HOLO else "unresolved"
                if state_pair_status != "resolved":
                    reasons.append("unresolved_ligand_state")
                if identity is None:
                    reasons.append("missing_sequence_identity")
                if common_fraction is None:
                    reasons.append("missing_common_coverage")
                if assembly_status == "unknown":
                    reasons.append("unknown_assembly_context")
                elif assembly_status != "comparable":
                    reasons.append("incomparable_assembly_context")
                admission = "ADMITTED" if not reasons else "EXCLUDED"
                rows.append(
                    {
                        "protein_id": str(accession),
                        "pair_id": f"{apo_record['pdb_id']}_{holo_record['pdb_id']}__{accession}",
                        "apo_pdb_id": str(apo_record["pdb_id"]),
                        "holo_pdb_id": str(holo_record["pdb_id"]),
                        "apo_polymer_entity_id": apo_id,
                        "holo_polymer_entity_id": holo_id,
                        "uniprot_id": str(accession),
                        "sequence_cluster": apo_record.get("sequence_cluster") or holo_record.get("sequence_cluster"),
                        "apo_state": states[apo_id].value,
                        "holo_state": states[holo_id].value,
                        "apo_chain_id": apo_record.get("chain_id"),
                        "holo_chain_id": holo_record.get("chain_id"),
                        "apo_mmcif_relative_path": apo_record.get("pdb_mmcif_relative_path"),
                        "holo_mmcif_relative_path": holo_record.get("pdb_mmcif_relative_path"),
                        "apo_sifts_relative_path": apo_record.get("sifts_relative_path"),
                        "holo_sifts_relative_path": holo_record.get("sifts_relative_path"),
                        "state_pair_status": state_pair_status,
                        "common_mapped_count": common_count,
                        "canonical_sequence_length": int(canonical_length) if canonical_length and np.isfinite(canonical_length) else None,
                        "common_fraction": common_fraction,
                        "sequence_identity": identity,
                        "construct_mismatch_count": int(apo_record.get("construct_mismatch_count", 0) or 0),
                        "assembly_comparability": assembly_status,
                        "experimental_method_comparable": method_status,
                        "apo_resolution": apo_record.get("resolution"),
                        "holo_resolution": holo_record.get("resolution"),
                        "admission_status": admission,
                        "exclusion_reasons": ";".join(reasons) if reasons else None,
                    }
                )
    return pd.DataFrame(rows)


def load_ligand_annotations(candidate_entities: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Load explicit RCSB non-polymer and CCD evidence for candidate entities."""

    def load_record(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        entity_key = str(record["polymer_entity_id"])
        pdb_id = str(record["pdb_id"]).lower()
        entries: list[dict[str, Any]] = []
        for nonpoly_id in [
            value
            for value in str(record.get("nonpolymer_entity_ids") or "").split(";")
            if value
        ]:
            try:
                nonpoly = request_json(
                    f"{RCSB_DATA_API}/nonpolymer_entity/{pdb_id}/{nonpoly_id}"
                )
                entity = nonpoly.get("pdbx_entity_nonpoly") or {}
                component = str(entity.get("comp_id") or "").upper()
                if not component:
                    entries.append(
                        {
                            "nonpolymer_entity_id": nonpoly_id,
                            "ligand_class": LigandClass.AMBIGUOUS.value,
                        }
                    )
                    continue
                ccd = request_json(f"{RCSB_DATA_API}/chemcomp/{component}")
                info = ccd.get("rcsb_chem_comp_info") or {}
                chem = ccd.get("chem_comp") or {}
                ligand_class = classify_ligand_component(
                    component,
                    chem.get("name"),
                    info.get("atom_count_heavy"),
                )
                entries.append(
                    {
                        "nonpolymer_entity_id": nonpoly_id,
                        "ccd_id": component,
                        "chemical_name": chem.get("name"),
                        "heavy_atom_count": info.get("atom_count_heavy"),
                        "ligand_class": ligand_class.value,
                        "auth_asym_ids": ";".join(
                            str(value)
                            for value in (
                                (
                                    nonpoly.get(
                                        "rcsb_nonpolymer_entity_container_identifiers"
                                    )
                                    or {}
                                ).get("auth_asym_ids")
                                or []
                            )
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                entries.append(
                    {
                        "nonpolymer_entity_id": nonpoly_id,
                        "ligand_class": LigandClass.AMBIGUOUS.value,
                        "annotation_status": "FAILED",
                        "annotation_error": str(exc),
                    }
                )
        return entity_key, entries

    records = candidate_entities.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(records)))) as executor:
        return dict(executor.map(load_record, records))


def build_candidate_structure_table(
    census: pd.DataFrame,
    assets: pd.DataFrame,
    ligand_annotations: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Combine census, asset provenance, and explicit ligand-state evidence."""

    work = census.copy()
    rows: list[dict[str, Any]] = []
    asset_paths_by_entity: dict[str, dict[str, str]] = {}
    if not assets.empty and {
        "polymer_entity_id",
        "asset_type",
        "relative_path",
        "retrieval_status",
    }.issubset(assets.columns):
        for entity_id, group in assets.groupby(
            assets["polymer_entity_id"].astype(str), sort=False
        ):
            paths: dict[str, str] = {}
            for asset_type, asset_group in group.groupby("asset_type", sort=False):
                valid = asset_group.loc[
                    asset_group["retrieval_status"].astype(str).str.startswith("VALID")
                ]
                if not valid.empty:
                    paths[str(asset_type)] = str(valid.iloc[0]["relative_path"])
            asset_paths_by_entity[str(entity_id)] = paths
    for record in work.to_dict(orient="records"):
        entity_id = str(record["polymer_entity_id"])
        annotations = ligand_annotations.get(entity_id, [])
        state = classify_state(
            annotations,
            annotation_complete=record.get("nonpolymer_entity_count") is not None,
        )
        asset_paths = asset_paths_by_entity.get(entity_id, {})
        rows.append(
            {
                **record,
                "protein_id": str(record.get("uniprot_id")),
                "state_label": state.value,
                "ligand_annotation_complete": record.get("nonpolymer_entity_count") is not None,
                "ligand_annotations": annotations,
                "ligand_class_summary": ";".join(sorted({str(item.get("ligand_class")) for item in annotations})) or None,
                "pdb_mmcif_relative_path": asset_paths.get("pdb_mmcif"),
                "sifts_relative_path": asset_paths.get("sifts_xml"),
                "uniprot_relative_path": asset_paths.get("uniprot_canonical_json"),
                "canonical_sequence_length": record.get("length"),
                "assembly_label": record.get("assembly_oligomeric_details"),
                "assembly_oligomeric_count": record.get("assembly_oligomeric_count"),
            }
        )
    return pd.DataFrame(rows)


def _parse_candidate_mapping_record(
    record: dict[str, Any], project_root: Path
) -> tuple[str, pd.DataFrame | None]:
    from dual_uq.dataset.services.mapping import parse_sifts_mapping_with_explicit_labels

    path = record.get("sifts_relative_path")
    structure_path = record.get("pdb_mmcif_relative_path")
    entity_id = str(record["polymer_entity_id"])
    if not path or not structure_path:
        return entity_id, None
    try:
        mapping = parse_sifts_mapping_with_explicit_labels(
            project_root / str(path),
            project_root / str(structure_path),
            chain_id=str(record.get("chain_id")),
            uniprot_id=str(record.get("uniprot_id")),
        )
    except (OSError, LookupError, ValueError):
        mapping = None
    return entity_id, mapping


def load_candidate_mappings(
    structures: pd.DataFrame,
    *,
    project_root: Path,
    workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Parse only the exact candidate SIFTS bindings; failures remain absent."""

    if workers < 1:
        raise ValueError("workers must be positive")

    records = structures.to_dict(orient="records")
    mappings: dict[str, pd.DataFrame] = {}
    if workers == 1:
        parsed = (
            _parse_candidate_mapping_record(record, project_root)
            for record in records
        )
        for entity_id, mapping in parsed:
            if mapping is not None:
                mappings[entity_id] = mapping
        return mappings

    # Bound the number of large pandas results in flight.  Submitting the
    # entire cohort at once can fill the process-pool result queue and make
    # progress appear stalled even though individual parses are healthy.
    batch_size = 32
    with ProcessPoolExecutor(max_workers=min(workers, 16)) as executor:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            parsed = executor.map(
                _parse_candidate_mapping_record,
                batch,
                [project_root] * len(batch),
            )
            for entity_id, mapping in parsed:
                if mapping is not None:
                    mappings[entity_id] = mapping
    return mappings


def annotate_sequence_identity(
    structures: pd.DataFrame,
    mappings: dict[str, pd.DataFrame],
    *,
    project_root: Path,
) -> pd.DataFrame:
    """Compute exact mapped-overlap identity against the canonical UniProt record."""

    from Bio.Data.PDBData import protein_letters_3to1_extended

    work = structures.copy()
    identities: list[float | None] = []
    mapped_counts: list[int | None] = []
    starts: list[int | None] = []
    ends: list[int | None] = []
    for record in work.to_dict(orient="records"):
        mapping = mappings.get(str(record["polymer_entity_id"]))
        sequence_path = record.get("uniprot_relative_path")
        canonical = None
        if sequence_path:
            try:
                payload = json.loads((project_root / str(sequence_path)).read_text(encoding="utf-8"))
                canonical = str((payload.get("sequence") or {}).get("value") or "").strip().upper()
            except (OSError, ValueError, TypeError):
                canonical = None
        if mapping is None or not canonical:
            identities.append(None)
            mapped_counts.append(_mapping_count(mapping))
            starts.append(None)
            ends.append(None)
            continue
        mapped = mapping.copy()
        positions = pd.to_numeric(mapped.get("uniprot_residue_number"), errors="coerce").astype("Int64")
        pdb_names = mapped.get("pdb_residue_name", pd.Series(index=mapped.index, dtype=object)).astype(str).str.upper()
        observed = pdb_names.map(lambda value: protein_letters_3to1_extended.get(value))
        comparable = positions.notna() & observed.notna() & positions.ge(1) & positions.le(len(canonical))
        if not comparable.any():
            identities.append(None)
        else:
            canonical_values = positions[comparable].astype(int).map(
                lambda index, canonical=canonical: canonical[index - 1]
            )
            identities.append(float((observed[comparable].to_numpy() == canonical_values.to_numpy()).mean()))
        mapped_positions = positions.dropna().astype(int)
        mapped_counts.append(int(mapped_positions.nunique()))
        starts.append(int(mapped_positions.min()) if len(mapped_positions) else None)
        ends.append(int(mapped_positions.max()) if len(mapped_positions) else None)
    work["sequence_identity"] = identities
    work["mapped_residue_count"] = mapped_counts
    work["mapped_uniprot_start"] = starts
    work["mapped_uniprot_end"] = ends
    return work


def _pair_common_coordinates(
    apo_by_position: dict[int, np.ndarray],
    holo_by_position: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Pair finite C-alpha coordinates by canonical position, not row order."""

    positions = sorted(set(apo_by_position).intersection(holo_by_position))
    if len(positions) < 3:
        raise ApoHoloError(
            "insufficient_common_coordinates",
            "fewer than three paired C-alpha coordinates are available",
        )
    apo = np.asarray([apo_by_position[position] for position in positions], dtype=float)
    holo = np.asarray([holo_by_position[position] for position in positions], dtype=float)
    if apo.shape != holo.shape or apo.ndim != 2 or apo.shape[1] != 3:
        raise ApoHoloError(
            "insufficient_common_coordinates",
            "paired C-alpha coordinate arrays are not shape-compatible",
        )
    if not np.isfinite(apo).all() or not np.isfinite(holo).all():
        raise ApoHoloError(
            "nonfinite_coordinates",
            "paired C-alpha coordinates contain non-finite values",
        )
    return positions, apo, holo


def _normalize_insertion_code(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _load_ca_coordinates_by_canonical(
    mmcif_path: Path,
    *,
    chain_id: str,
    mapping: pd.DataFrame,
    prefix: str,
) -> dict[int, np.ndarray]:
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(mmcif_path.stem, str(mmcif_path))
    model = next(structure.get_models())
    chain = next((item for item in model.get_chains() if str(item.id) == str(chain_id)), None)
    if chain is None:
        raise ApoHoloError("missing_chain", f"chain {chain_id!r} not found in {mmcif_path}")
    if "canonical_position" not in mapping:
        raise ApoHoloError("mapping_schema_mismatch", "mapping lacks canonical_position")
    seq_column = f"{prefix}_auth_seq_id"
    ins_column = f"insertion_code_{prefix}"
    if seq_column not in mapping:
        raise ApoHoloError("mapping_schema_mismatch", f"mapping lacks {seq_column}")
    positions_by_key: dict[tuple[int, str], list[int]] = {}
    for position, seq, insertion in zip(
        pd.to_numeric(mapping["canonical_position"], errors="coerce"),
        pd.to_numeric(mapping[seq_column], errors="coerce"),
        mapping.get(ins_column, pd.Series("", index=mapping.index)),
    ):
        if pd.isna(position) or pd.isna(seq):
            continue
        key = (int(seq), _normalize_insertion_code(insertion))
        positions_by_key.setdefault(key, []).append(int(position))
    # A single PDB residue identifier cannot represent two canonical residues
    # without an explicit, lossless correspondence.  Leave such positions out
    # of the geometry-visible set instead of choosing by row order.
    unique_keys = {
        key: values[0]
        for key, values in positions_by_key.items()
        if len(values) == 1
    }
    coords_by_key: dict[tuple[int, str], np.ndarray] = {}
    for residue in chain.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        key = (int(residue.id[1]), _normalize_insertion_code(residue.id[2]))
        if key in unique_keys:
            coordinate = np.asarray(residue["CA"].coord, dtype=float)
            if np.isfinite(coordinate).all():
                coords_by_key[key] = coordinate
    return {position: coords_by_key[key] for key, position in unique_keys.items() if key in coords_by_key}


def _load_ca_coordinates(
    mmcif_path: Path,
    *,
    chain_id: str,
    mapping: pd.DataFrame,
    prefix: str,
) -> np.ndarray:
    """Load coordinates in canonical order for the legacy private helper."""

    by_position = _load_ca_coordinates_by_canonical(
        mmcif_path, chain_id=chain_id, mapping=mapping, prefix=prefix
    )
    positions = sorted(by_position)
    if len(positions) < 3:
        raise ApoHoloError(
            "insufficient_common_coordinates",
            "fewer than three C-alpha coordinates are available",
        )
    return np.asarray([by_position[position] for position in positions], dtype=float)


def _ligand_proximity_labels(
    mmcif_path: Path,
    *,
    chain_id: str,
    mapping: pd.DataFrame,
    ligand_annotations: list[dict[str, Any]],
    cutoff: float,
    canonical_positions: Iterable[int] | None = None,
) -> list[bool]:
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(mmcif_path.stem, str(mmcif_path))
    model = next(structure.get_models())
    ligand_ids = {str(item.get("ccd_id", "")).upper() for item in ligand_annotations if item.get("ccd_id")}
    ligand_atoms = [
        np.asarray(atom.coord, dtype=float)
        for chain in model.get_chains()
        for residue in chain.get_residues()
        if str(residue.resname).strip().upper() in ligand_ids or str(residue.id[0]).startswith("H_")
        for atom in residue.get_atoms()
    ]
    label_count = len(list(canonical_positions)) if canonical_positions is not None else len(mapping)
    if not ligand_atoms:
        return [False] * label_count
    ligand_coordinates = np.asarray(ligand_atoms, dtype=float)
    labels: list[bool] = []
    seq_column = "holo_auth_seq_id"
    ins_column = "insertion_code_holo"
    chain = next((item for item in model.get_chains() if str(item.id) == str(chain_id)), None)
    if chain is None:
        return [False] * label_count
    coords_by_key = {
        (int(residue.id[1]), str(residue.id[2]).strip()): np.asarray(residue["CA"].coord, dtype=float)
        for residue in chain.get_residues()
        if residue.id[0] == " " and "CA" in residue
    }
    work = mapping.copy()
    if canonical_positions is not None:
        requested = {int(value) for value in canonical_positions}
        work["_canonical_position"] = pd.to_numeric(
            work["canonical_position"], errors="coerce"
        )
        work = work.loc[work["_canonical_position"].isin(requested)].sort_values(
            "_canonical_position", kind="mergesort"
        )
    for seq, insertion in zip(
        pd.to_numeric(work[seq_column], errors="coerce"),
        work.get(ins_column, pd.Series("", index=work.index)),
    ):
        if pd.isna(seq):
            labels.append(False)
            continue
        key = (int(seq), _normalize_insertion_code(insertion))
        coordinates = coords_by_key.get(key)
        labels.append(bool(coordinates is not None and np.min(np.linalg.norm(ligand_coordinates - coordinates, axis=1)) <= cutoff))
    return labels


def characterize_pair_assets(
    pair: pd.Series | dict[str, Any],
    *,
    mappings: dict[str, pd.DataFrame],
    project_root: Path,
    ligand_annotations: dict[str, list[dict[str, Any]]],
    config: ApoHoloConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Characterize one admitted pair using exact mapped coordinates."""

    values = pair if isinstance(pair, dict) else pair.to_dict()
    apo_mapping = mappings.get(str(values["apo_polymer_entity_id"]))
    holo_mapping = mappings.get(str(values["holo_polymer_entity_id"]))
    if apo_mapping is None or holo_mapping is None:
        raise ApoHoloError("missing_mapping_asset", "both pair mappings are required")
    common = build_common_residue_mapping(apo_mapping, holo_mapping)
    apo_path = project_root / str(values["apo_mmcif_relative_path"])
    holo_path = project_root / str(values["holo_mmcif_relative_path"])
    apo_by_position = _load_ca_coordinates_by_canonical(
        apo_path, chain_id=str(values["apo_chain_id"]), mapping=common, prefix="apo"
    )
    holo_by_position = _load_ca_coordinates_by_canonical(
        holo_path, chain_id=str(values["holo_chain_id"]), mapping=common, prefix="holo"
    )
    canonical_positions, apo_coordinates, holo_coordinates = _pair_common_coordinates(
        apo_by_position, holo_by_position
    )
    proximal = _ligand_proximity_labels(
        holo_path,
        chain_id=str(values["holo_chain_id"]),
        mapping=common,
        ligand_annotations=ligand_annotations.get(str(values["holo_polymer_entity_id"]), []),
        cutoff=config.contact_cutoff_angstrom,
        canonical_positions=canonical_positions,
    )
    result = characterize_coordinates(
        apo_coordinates,
        holo_coordinates,
        canonical_positions=canonical_positions,
        ligand_proximal=proximal,
        contact_cutoff_angstrom=config.contact_cutoff_angstrom,
    )
    pair_summary = dict(result.pair_summary)
    pair_summary.update(
        {
            "pair_id": values.get("pair_id"),
            "protein_id": values.get("protein_id"),
            "apo_pdb_id": values.get("apo_pdb_id"),
            "holo_pdb_id": values.get("holo_pdb_id"),
            "geometry_status": "available",
        }
    )
    residue = result.residue_table.copy()
    residue.insert(0, "pair_id", values.get("pair_id"))
    residue.insert(1, "protein_id", values.get("protein_id"))
    return residue, pair_summary


def summarize_apo_holo(
    census: pd.DataFrame,
    pairs: pd.DataFrame,
    primary_pairs: pd.DataFrame,
    alternative_pairs: pd.DataFrame,
    pair_descriptors: pd.DataFrame,
    residue_descriptors: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive Q1–Q6 fields at the protein/cluster unit."""

    reason_counts: dict[str, int] = {}
    if not pairs.empty and "exclusion_reasons" in pairs:
        for reason_string in pairs["exclusion_reasons"].dropna().astype(str):
            for reason in (value for value in reason_string.split(";") if value):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    state_counts = pairs["state_pair_status"].value_counts().to_dict() if not pairs.empty and "state_pair_status" in pairs else {}
    clusters = primary_pairs.get("sequence_cluster", pd.Series(dtype=str)).dropna().astype(str).nunique() if not primary_pairs.empty else 0
    finite_rmsd = pd.to_numeric(pair_descriptors.get("aligned_ca_rmsd", pd.Series(dtype=float)), errors="coerce").dropna()
    finite_displacement = pd.to_numeric(residue_descriptors.get("aligned_ca_displacement", pd.Series(dtype=float)), errors="coerce").dropna()
    geometry_status = pair_descriptors.get("geometry_status", pd.Series(dtype=str))
    primary_count = len(primary_pairs)
    truncated = bool(census.attrs.get("census_truncated", False))
    if truncated:
        decision = "LIMITED"
    elif primary_count == 0:
        decision = "BLOCKED"
    else:
        decision = "PASS"
    return {
        "decision": decision,
        "census_truncated": truncated,
        "candidate_structure_count": len(census),
        "candidate_protein_count": int(census.get("uniprot_id", pd.Series(dtype=str)).dropna().nunique()),
        "candidate_pair_count": len(pairs),
        "excluded_pair_count": int((pairs.get("admission_status", pd.Series(dtype=str)) == "EXCLUDED").sum()) if not pairs.empty else 0,
        "primary_pair_count": primary_count,
        "alternative_pair_count": len(alternative_pairs),
        "independent_cluster_count": int(clusters),
        "state_pair_status_counts": {str(key): int(value) for key, value in state_counts.items()},
        "exclusion_reason_counts": reason_counts,
        "geometry_available_pair_count": int(geometry_status.eq("available").sum()) if len(geometry_status) else 0,
        "global_rmsd_summary": {
            "median": float(finite_rmsd.median()) if len(finite_rmsd) else None,
            "q90": float(finite_rmsd.quantile(0.90)) if len(finite_rmsd) else None,
        },
        "local_change_summary": {
            "median_residue_displacement": float(finite_displacement.median()) if len(finite_displacement) else None,
            "q90_residue_displacement": float(finite_displacement.quantile(0.90)) if len(finite_displacement) else None,
        },
        "ligand_proximal_vs_distal": "descriptive only; no causal ligand claim",
        "q1_candidate_proteins": int(census.get("uniprot_id", pd.Series(dtype=str)).dropna().nunique()),
        "q2_admitted_proteins": primary_count,
        "q3_major_exclusions": reason_counts,
        "q4_structural_change_range": {
            "rmsd_median": float(finite_rmsd.median()) if len(finite_rmsd) else None,
            "displacement_q90": float(finite_displacement.quantile(0.90)) if len(finite_displacement) else None,
        },
        "q5_ligand_proximity_pattern": "descriptive only; assess from proximal/distal summaries",
        "q6_next_experiment_ready": bool(primary_count > 0 and not truncated),
    }


_ATTRITION_REASON_PRIORITY = (
    "metadata_network_unresolved",
    "unresolved_ligand_state",
    "sequence_mismatch",
    "construct_domain_mismatch",
    "insufficient_common_mapping",
    "assembly_context_mismatch",
    "experimental_entity_mismatch",
    "other",
)


def diagnose_apo_holo_attrition(
    pairs: pd.DataFrame,
    *,
    min_common_fraction: float = 0.90,
    min_sequence_identity: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an outcome-blind pair-level attrition diagnosis.

    The diagnosis preserves every input pair, records one deterministic primary
    reason, and retains all observed secondary reasons.  It only uses admission
    fields already present in the pair table; missing evidence is not converted
    into a scientific failure reason.
    """

    required = {
        "pair_id",
        "state_pair_status",
        "sequence_identity",
        "construct_mismatch_count",
        "common_fraction",
        "assembly_comparability",
        "admitted",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ApoHoloError("schema_mismatch", f"attrition diagnosis missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for values in pairs.to_dict(orient="records"):
        observed = {
            str(item)
            for item in str(values.get("exclusion_reasons") or "").split(";")
            if item
        }
        categories: set[str] = set()
        metadata_status = str(values.get("metadata_status") or "")
        census_status = str(values.get("census_status") or "")
        metadata_error = values.get("metadata_error")
        if metadata_error or metadata_status in {"network_unresolved", "partial", "missing"} or census_status in {
            "network_unresolved",
            "unresolved",
        }:
            categories.add("metadata_network_unresolved")
        if str(values.get("state_pair_status") or "") != "resolved":
            categories.add("unresolved_ligand_state")
        identity = pd.to_numeric(values.get("sequence_identity"), errors="coerce")
        if pd.isna(identity) or float(identity) < min_sequence_identity:
            categories.add("sequence_mismatch")
        mismatch = pd.to_numeric(values.get("construct_mismatch_count"), errors="coerce")
        if not pd.isna(mismatch) and int(mismatch) > 0:
            categories.add("construct_domain_mismatch")
        common = pd.to_numeric(values.get("common_fraction"), errors="coerce")
        if pd.isna(common) or float(common) < min_common_fraction:
            categories.add("insufficient_common_mapping")
        if str(values.get("assembly_comparability") or "") != "comparable":
            categories.add("assembly_context_mismatch")
        experimental_comparable = values.get("experimental_method_comparable")
        if experimental_comparable is not None and not pd.isna(experimental_comparable) and not bool(experimental_comparable):
            categories.add("experimental_entity_mismatch")

        # Preserve existing reason strings that have no dedicated category.
        known_existing = {
            "unresolved_ligand_state",
            "sequence_identity_below_threshold",
            "common_coverage_below_threshold",
            "construct_mismatch_exceeds_threshold",
            "assembly_context_not_comparable",
            "incomparable_assembly_context",
        }
        if observed.difference(known_existing):
            categories.add("other")
        if not categories and not bool(values.get("admitted")):
            categories.add("other")
        ordered = [reason for reason in _ATTRITION_REASON_PRIORITY if reason in categories]
        primary = ordered[0] if ordered else None
        secondary = [reason for reason in ordered[1:]]
        rows.append(
            {
                **values,
                "primary_exclusion_reason": primary,
                "secondary_exclusion_reasons": ";".join(secondary) or None,
                "observed_exclusion_reasons": ";".join(sorted(observed)) or None,
            }
        )

    diagnosis = pd.DataFrame(rows, columns=[*pairs.columns, "primary_exclusion_reason", "secondary_exclusion_reasons", "observed_exclusion_reasons"])
    total = len(diagnosis)
    primary_counts = diagnosis["primary_exclusion_reason"].value_counts(dropna=False).to_dict() if total else {}
    secondary_counts: dict[str, int] = {}
    for value in diagnosis["secondary_exclusion_reasons"].dropna().astype(str):
        for reason in value.split(";"):
            secondary_counts[reason] = secondary_counts.get(reason, 0) + 1

    def _count(mask: pd.Series) -> int:
        return int(mask.fillna(False).sum())

    sequence_identity = pd.to_numeric(diagnosis["sequence_identity"], errors="coerce")
    construct_mismatch = pd.to_numeric(diagnosis["construct_mismatch_count"], errors="coerce")
    common_fraction = pd.to_numeric(diagnosis["common_fraction"], errors="coerce")
    funnel = {
        "candidate_pair": total,
        "ligand_state_resolved": _count(diagnosis["state_pair_status"].eq("resolved")),
        "same_uniprot_entity_bound": _count(
            diagnosis.get("uniprot_id", pd.Series(index=diagnosis.index)).notna()
            & diagnosis.get("apo_polymer_entity_id", pd.Series(index=diagnosis.index)).notna()
            & diagnosis.get("holo_polymer_entity_id", pd.Series(index=diagnosis.index)).notna()
        ),
        "sequence_compatible": _count(sequence_identity.ge(min_sequence_identity)),
        "construct_compatible": _count(construct_mismatch.fillna(np.inf).le(0)),
        "mapping_compatible": _count(common_fraction.ge(min_common_fraction)),
        "assembly_compatible": _count(diagnosis["assembly_comparability"].eq("comparable")),
        "final_admitted": _count(diagnosis["admitted"].eq(True)),
    }
    categories = {
        "unresolved_apo_state": _count(diagnosis.get("apo_state", pd.Series(index=diagnosis.index)).eq("unresolved")),
        "unresolved_holo_ligand_relevance": _count(diagnosis.get("holo_state", pd.Series(index=diagnosis.index)).eq("unresolved")),
        "no_true_apo_counterpart": 0,
        "sequence_mismatch": _count(sequence_identity.lt(min_sequence_identity) | sequence_identity.isna()),
        "construct_domain_mismatch": _count(construct_mismatch.gt(0)),
        "insufficient_common_mapping": _count(common_fraction.lt(min_common_fraction) | common_fraction.isna()),
        "missing_residue_incompatibility": 0,
        "assembly_context_mismatch": _count(~diagnosis["assembly_comparability"].eq("comparable")),
        "experimental_entity_mismatch": _count(diagnosis.get("experimental_method_comparable", pd.Series(index=diagnosis.index)).eq(False)),
        "metadata_network_unresolved": _count(diagnosis["primary_exclusion_reason"].eq("metadata_network_unresolved")),
        "other": _count(diagnosis["primary_exclusion_reason"].eq("other")),
    }
    return diagnosis, {
        "pair_count": total,
        "primary_reason_counts": {str(key): int(value) for key, value in primary_counts.items() if pd.notna(key)},
        "secondary_reason_counts": secondary_counts,
        "category_counts": categories,
        "category_fractions": {key: (value / total if total else None) for key, value in categories.items()},
        "funnel": funnel,
        "thresholds": {
            "min_common_fraction": min_common_fraction,
            "min_sequence_identity": min_sequence_identity,
        },
        "missing_residue_incompatibility_note": "0 means no explicit admission reason or retained evidence encoded this category; no inference was made from partial mapping alone.",
    }


def admit_pair(row: pd.Series | dict[str, Any], config: ApoHoloConfig) -> dict[str, Any]:
    """Apply strict pair-validity criteria without using structural outcomes."""

    values = row if isinstance(row, dict) else row.to_dict()
    existing_reasons = values.get("exclusion_reasons")
    if existing_reasons is None or (
        isinstance(existing_reasons, (float, np.floating)) and np.isnan(existing_reasons)
    ):
        reasons: list[str] = []
    else:
        reasons = [str(value) for value in str(existing_reasons).split(";") if value]
    if values.get("state_pair_status") != "resolved":
        reasons.append("unresolved_ligand_state")
    identity = values.get("sequence_identity")
    if identity is None or float(identity) < config.min_sequence_identity:
        reasons.append("sequence_identity_below_threshold")
    fraction = values.get("common_fraction")
    if fraction is None or float(fraction) < config.min_common_fraction:
        reasons.append("common_coverage_below_threshold")
    if int(values.get("construct_mismatch_count") or 0) > config.max_construct_mismatch:
        reasons.append("construct_mismatch_exceeds_threshold")
    if values.get("assembly_comparability") != "comparable":
        reasons.append("assembly_context_not_comparable")
    return {
        "admitted": not reasons,
        "admission_status": "ADMITTED" if not reasons else "EXCLUDED",
        "exclusion_reasons": ";".join(dict.fromkeys(reasons)) if reasons else None,
    }


def select_primary_and_alternatives(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one admitted pair per protein and preserve other admitted pairs."""

    admitted = pairs.loc[pairs.get("admission_status", pd.Series(index=pairs.index)).eq("ADMITTED")].copy()
    if admitted.empty:
        return admitted.copy(), admitted.copy()
    primary = select_primary_pair(admitted)
    key_columns = ["protein_id", "apo_pdb_id", "holo_pdb_id"]
    marker = primary[key_columns].astype(str).agg("|".join, axis=1)
    admitted_marker = admitted[key_columns].astype(str).agg("|".join, axis=1)
    alternatives = admitted.loc[~admitted_marker.isin(set(marker))].reset_index(drop=True)
    return primary.reset_index(drop=True), alternatives


def build_common_residue_mapping(apo_mapping: pd.DataFrame, holo_mapping: pd.DataFrame) -> pd.DataFrame:
    """Join mapping rows by exact UniProt position only."""

    for label, frame in (("apo", apo_mapping), ("holo", holo_mapping)):
        if "uniprot_residue_number" not in frame.columns:
            raise ApoHoloError("schema_mismatch", f"{label} mapping lacks uniprot_residue_number")
    left = apo_mapping.copy()
    right = holo_mapping.copy()
    left["canonical_position"] = pd.to_numeric(left["uniprot_residue_number"], errors="coerce").astype("Int64")
    right["canonical_position"] = pd.to_numeric(right["uniprot_residue_number"], errors="coerce").astype("Int64")
    left = left.dropna(subset=["canonical_position"]).drop_duplicates("canonical_position")
    right = right.dropna(subset=["canonical_position"]).drop_duplicates("canonical_position")
    merged = left.merge(right, on="canonical_position", how="inner", suffixes=("_apo", "_holo"), validate="one_to_one")
    for prefix in ("apo", "holo"):
        source = f"{prefix}_auth_seq_id"
        if source not in merged:
            raw = merged.get(f"pdb_residue_number_{prefix}")
            merged[source] = pd.to_numeric(raw.astype(str).str.extract(r"([+-]?\d+)")[0], errors="coerce").astype("Int64")
    return merged.sort_values("canonical_position", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class GeometryResult:
    residue_table: pd.DataFrame
    pair_summary: dict[str, Any]


def _contact_neighbors(coordinates: np.ndarray, cutoff: float) -> list[set[int]]:
    distances = pairwise_distances(coordinates)
    neighbours: list[set[int]] = []
    for index in range(len(coordinates)):
        neighbours.append(set(np.flatnonzero((distances[index] <= cutoff) & (distances[index] > 0)).tolist()))
    return neighbours


def characterize_coordinates(
    apo_coordinates: np.ndarray,
    holo_coordinates: np.ndarray,
    *,
    canonical_positions: Iterable[int],
    ligand_proximal: Iterable[bool],
    contact_cutoff_angstrom: float,
) -> GeometryResult:
    """Characterize apo/holo state variation on one exact common residue set."""

    apo = np.asarray(apo_coordinates, dtype=float)
    holo = np.asarray(holo_coordinates, dtype=float)
    if apo.shape != holo.shape or apo.ndim != 2 or apo.shape[1] != 3 or len(apo) < 3:
        raise ApoHoloError("insufficient_common_coordinates", "at least three paired C-alpha coordinates are required")
    positions = [int(value) for value in canonical_positions]
    proximal = [bool(value) for value in ligand_proximal]
    if len(positions) != len(apo) or len(proximal) != len(apo):
        raise ApoHoloError("geometry_key_mismatch", "positions and ligand labels must match coordinates")
    aligned_holo, _, _ = kabsch_align(holo, apo)
    displacement = np.linalg.norm(aligned_holo - apo, axis=1)
    apo_distances = pairwise_distances(apo)
    holo_distances = pairwise_distances(aligned_holo)
    apo_neighbours = _contact_neighbors(apo, contact_cutoff_angstrom)
    holo_neighbours = _contact_neighbors(aligned_holo, contact_cutoff_angstrom)
    local_change: list[float | None] = []
    turnover: list[float | None] = []
    for index in range(len(apo)):
        neighbours = sorted(apo_neighbours[index] | holo_neighbours[index])
        if not neighbours:
            local_change.append(None)
            turnover.append(None)
            continue
        local_change.append(float(np.mean(np.abs(apo_distances[index, neighbours] - holo_distances[index, neighbours]))))
        turnover.append(float(len(apo_neighbours[index] ^ holo_neighbours[index]) / len(set(neighbours))))
    residue = pd.DataFrame(
        {
            "canonical_position": positions,
            "aligned_ca_displacement": displacement,
            "local_pairwise_distance_change": local_change,
            "contact_turnover_fraction": turnover,
            "ligand_proximal": proximal,
            "geometry_status": "available",
        }
    )
    summary = {
        "common_residue_count": len(residue),
        "aligned_ca_rmsd": rmsd(aligned_holo, apo),
        "median_residue_displacement": float(np.median(displacement)),
        "p90_residue_displacement": float(np.quantile(displacement, 0.90, method="linear")),
        "ligand_proximal_count": int(sum(proximal)),
        "ligand_distal_count": int(len(proximal) - sum(proximal)),
        "ligand_proximal_median_displacement": float(residue.loc[residue["ligand_proximal"], "aligned_ca_displacement"].median()) if any(proximal) else None,
        "ligand_distal_median_displacement": float(residue.loc[~residue["ligand_proximal"], "aligned_ca_displacement"].median()) if not all(proximal) else None,
        "median_local_pairwise_distance_change": float(pd.to_numeric(residue["local_pairwise_distance_change"], errors="coerce").median()),
    }
    return GeometryResult(residue_table=residue, pair_summary=summary)


@dataclass(frozen=True)
class ApoHoloResult:
    """Canonical structured result consumed by every release renderer."""

    census: pd.DataFrame
    pairs: pd.DataFrame
    admission: pd.DataFrame
    primary_pairs: pd.DataFrame
    alternative_pairs: pd.DataFrame
    residue_mappings: pd.DataFrame
    ligand_sites: pd.DataFrame
    pair_descriptors: pd.DataFrame
    residue_descriptors: pd.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]
    asset_ledger: pd.DataFrame | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_immutable_table(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_parquet(index=False)
    if not isinstance(payload, bytes):
        raise TypeError("pandas parquet serialization did not return bytes")
    if path.exists():
        if sha256_file(path) != __import__("hashlib").sha256(payload).hexdigest():
            raise ApoHoloError("immutable_conflict", f"refusing to overwrite {path}")
        return
    atomic_write_new_bytes(path, payload)


def render_apo_holo_release(result: ApoHoloResult, output_dir: str | Path) -> None:
    """Render all cohort tables and report files from one structured result."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "candidate_structures.parquet": result.census,
        "candidate_pairs.parquet": result.pairs,
        "admission_decisions.parquet": result.admission,
        "primary_pairs.parquet": result.primary_pairs,
        "alternative_pairs.parquet": result.alternative_pairs,
        "residue_mappings.parquet": result.residue_mappings,
        "ligand_sites.parquet": result.ligand_sites,
        "pair_structural_descriptors.parquet": result.pair_descriptors,
        "residue_structural_descriptors.parquet": result.residue_descriptors,
    }
    artifact_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for filename, frame in tables.items():
        path = output / filename
        _write_immutable_table(frame, path)
        artifact_hashes[filename] = sha256_file(path)
        row_counts[filename] = len(frame)
    if result.asset_ledger is not None:
        path = output / "asset_ledger.parquet"
        _write_immutable_table(result.asset_ledger, path)
        artifact_hashes[path.name] = sha256_file(path)
        row_counts[path.name] = len(result.asset_ledger)
    summary = _json_safe(dict(result.summary))
    summary.setdefault("decision", "BLOCKED")
    summary.setdefault("candidate_structure_count", row_counts["candidate_structures.parquet"])
    summary.setdefault("candidate_pair_count", row_counts["candidate_pairs.parquet"])
    summary.setdefault("primary_pair_count", row_counts["primary_pairs.parquet"])
    summary.setdefault("alternative_pair_count", row_counts["alternative_pairs.parquet"])
    summary_path = output / "summary.json"
    atomic_write_json(summary_path, summary)
    artifact_hashes["summary.json"] = sha256_file(summary_path)
    report = "# Experimental apo/holo cohort\n\n"
    report += f"Decision: **{summary['decision']}**\n\n"
    report += "## Cohort\n\n"
    report += f"- Candidate structures: `{summary.get('candidate_structure_count')}`.\n"
    report += f"- Candidate proteins: `{summary.get('candidate_protein_count', 'unknown')}`.\n"
    report += f"- Admitted primary pairs: `{summary.get('primary_pair_count')}`.\n"
    report += f"- Independent 30%-identity clusters: `{summary.get('independent_cluster_count', 'unknown')}`.\n\n"
    report += "## Pair validity\n\n"
    report += f"- Candidate pairs: `{summary.get('candidate_pair_count')}`; exclusions: `{summary.get('excluded_pair_count', 'unknown')}`.\n"
    report += f"- Major exclusion causes: `{summary.get('exclusion_reason_counts', {})}`.\n\n"
    report += "## State characterization\n\n"
    report += f"- Global RMSD summary: `{summary.get('global_rmsd_summary', 'unknown')}`.\n"
    report += f"- Local change summary: `{summary.get('local_change_summary', 'unknown')}`.\n"
    report += f"- Ligand-proximal versus distal: `{summary.get('ligand_proximal_vs_distal', 'unknown')}`.\n\n"
    report += "## Q1–Q6\n\n"
    for key in ("q1_candidate_proteins", "q2_admitted_proteins", "q3_major_exclusions", "q4_structural_change_range", "q5_ligand_proximity_pattern", "q6_next_experiment_ready"):
        report += f"- **{key}**: `{summary.get(key, 'unknown')}`.\n"
    report += "\nInterpretation is limited to experimental structural-state variation; no ligand causality, functional transition, biological uncertainty, or inverse-folding claim is made.\n"
    report_path = output / "report.md"
    atomic_write_new_bytes(report_path, report.encode("utf-8"))
    artifact_hashes["report.md"] = sha256_file(report_path)
    manifest = {
        "analysis": "experimental_apo_holo_state_dataset",
        "decision": summary["decision"],
        "row_counts": row_counts,
        "artifacts": artifact_hashes,
        "provenance": _json_safe(result.provenance),
        "interpretation_boundary": "experimental structural-state variation only; no causal, functional, biological, or inverse-folding claim",
    }
    atomic_write_json(output / "manifest.json", manifest)
