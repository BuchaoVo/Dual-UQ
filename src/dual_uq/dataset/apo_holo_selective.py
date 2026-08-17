"""Selective, outcome-blind apo/holo state resolution and admission planning.

This module is deliberately separate from the historical bounded constructor.  It
uses the complete metadata census to resolve structure-level state before any raw
structure acquisition, then reduces the Cartesian pair space by an explicit
comparability skyline.  No structural response, inverse-folding score, or model
output is used by any function here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_json
from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.apo_holo import (
    LigandClass,
    StateLabel,
    _cache_shard_dir,
    _clean_text,
    classify_ligand_component,
)
from dual_uq.net import download_file, post_json
from dual_uq.pdb_archive import RCSB_MMCIF_URL, fetch_pdb_mmcif
from dual_uq.sifts import SIFTS_XML_URL, fetch_sifts_xml

RCSB_GRAPHQL_API = "https://data.rcsb.org/graphql"


def _nonpolymer_query() -> str:
    return """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_container_identifiers { non_polymer_entity_ids }
        nonpolymer_entities {
          rcsb_id
          pdbx_entity_nonpoly { entity_id comp_id name }
          rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
        }
      }
    }
    """


def _normalize_nonpolymer_payload(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entries = (payload.get("data") or {}).get("entries") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pdb_id = str(entry.get("rcsb_id") or "").lower()
        identifiers = entry.get("rcsb_entry_container_identifiers") or {}
        expected = {str(value) for value in identifiers.get("non_polymer_entity_ids") or []}
        observed: set[str] = set()
        for item in entry.get("nonpolymer_entities") or []:
            if not isinstance(item, dict):
                continue
            nonpoly_id = str(item.get("rcsb_id") or "").rsplit("_", 1)[-1]
            entity = item.get("pdbx_entity_nonpoly") or {}
            container = item.get("rcsb_nonpolymer_entity_container_identifiers") or {}
            component = _clean_text(entity.get("comp_id"))
            if nonpoly_id:
                observed.add(nonpoly_id)
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "nonpolymer_entity_id": nonpoly_id or None,
                    "ccd_id": component.upper() if component else None,
                    "chemical_name": _clean_text(entity.get("name")),
                    "auth_asym_ids": ";".join(str(value) for value in container.get("auth_asym_ids") or []),
                    "inventory_status": "complete" if component else "incomplete",
                }
            )
        if expected - observed:
            for nonpoly_id in sorted(expected - observed):
                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "nonpolymer_entity_id": nonpoly_id,
                        "ccd_id": None,
                        "chemical_name": None,
                        "auth_asym_ids": "",
                        "inventory_status": "incomplete",
                    }
                )
    columns = [
        "pdb_id",
        "nonpolymer_entity_id",
        "ccd_id",
        "chemical_name",
        "auth_asym_ids",
        "inventory_status",
    ]
    return pd.DataFrame(rows, columns=columns)


def fetch_nonpolymer_metadata(
    pdb_ids: Iterable[str],
    *,
    cache_root: Path,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Fetch/cache non-polymer entity metadata without downloading structures."""

    ids = sorted({str(value).strip().lower() for value in pdb_ids if str(value).strip()})
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(cache_root)
    shard_dir = _cache_shard_dir(root / "nonpolymer_graphql.json", "chunks")
    shard_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    query = _nonpolymer_query()
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        key = hashlib.sha256(
            json.dumps({"ids": batch, "query": query}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        path = shard_dir / f"{key}.json"
        payload: dict[str, Any]
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    payload = cached.get("payload") or {}
                else:
                    payload = {}
            except (OSError, TypeError, ValueError):
                payload = {}
        else:
            try:
                payload = post_json(
                    RCSB_GRAPHQL_API,
                    {"query": query, "variables": {"ids": batch}},
                )
            except Exception as exc:  # noqa: BLE001
                atomic_write_json(path, {"status": "network_unresolved", "error": str(exc), "ids": batch})
                payload = {"data": {"entries": []}}
            else:
                atomic_write_json(path, {"status": "complete", "ids": batch, "payload": payload})
        frame = _normalize_nonpolymer_payload(payload)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["pdb_id", "nonpolymer_entity_id", "ccd_id", "chemical_name", "auth_asym_ids", "inventory_status"]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["pdb_id", "nonpolymer_entity_id"], keep="last"
    ).sort_values(["pdb_id", "nonpolymer_entity_id"], kind="mergesort").reset_index(drop=True)


def build_component_reference(component_rows: pd.DataFrame) -> pd.DataFrame:
    """Create one conservative CCD reference row per component identifier."""

    if component_rows.empty:
        return pd.DataFrame(columns=["ccd_id", "chemical_name", "heavy_atom_count", "ligand_class", "reference_status"])
    if "ccd_id" not in component_rows:
        raise ValueError("component_rows must contain ccd_id")
    work = component_rows.copy()
    work["ccd_id"] = work["ccd_id"].astype("string").str.upper()
    rows: list[dict[str, Any]] = []
    for ccd_id, group in work.dropna(subset=["ccd_id"]).groupby("ccd_id", sort=True):
        names = [value for value in group.get("chemical_name", pd.Series(dtype=object)).dropna().astype(str).unique()]
        heavy_values = pd.to_numeric(group.get("heavy_atom_count", pd.Series(dtype=float)), errors="coerce").dropna()
        classes = {
            classify_ligand_component(
                str(ccd_id),
                name,
                int(heavy_values.iloc[0]) if len(heavy_values) else None,
            ).value
            for name in (names or [None])
        }
        conflict = len(classes) > 1 or len(names) > 1 or heavy_values.nunique() > 1
        rows.append(
            {
                "ccd_id": str(ccd_id),
                "chemical_name": names[0] if names else None,
                "heavy_atom_count": int(heavy_values.iloc[0]) if len(heavy_values) else None,
                "ligand_class": LigandClass.AMBIGUOUS.value if conflict else next(iter(classes)),
                "reference_status": "conflicting" if conflict else "resolved",
            }
        )
    return pd.DataFrame(rows).sort_values("ccd_id", kind="mergesort").reset_index(drop=True)


def annotate_component_rows(component_rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative component class to each structure/entity inventory row."""

    if component_rows.empty:
        return component_rows.copy()
    required = {"ccd_id", "ligand_class", "reference_status"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reference missing columns: {sorted(required - set(reference.columns))}")
    return component_rows.merge(
        reference[["ccd_id", "chemical_name", "heavy_atom_count", "ligand_class", "reference_status"]],
        on="ccd_id",
        how="left",
        suffixes=("", "_reference"),
        validate="many_to_one",
    ).assign(
        ligand_class=lambda frame: frame["ligand_class"].fillna(LigandClass.AMBIGUOUS.value),
        inventory_status=lambda frame: np.where(
            frame["inventory_status"].eq("complete") & frame["reference_status"].eq("resolved"),
            "complete",
            "incomplete",
        ),
    )


def resolve_structure_states(census: pd.DataFrame, component_rows: pd.DataFrame) -> pd.DataFrame:
    """Resolve HOLO/APO/UNRESOLVED from complete per-structure inventories."""

    required = {"polymer_entity_id", "pdb_id", "uniprot_id", "nonpolymer_entity_count", "nonpolymer_entity_ids"}
    missing = sorted(required - set(census.columns))
    if missing:
        raise ValueError(f"census missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    inventory_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not component_rows.empty:
        for key, group in component_rows.groupby(["pdb_id", "nonpolymer_entity_id"], sort=False):
            inventory_by_key[(str(key[0]).lower(), str(key[1]))] = group.to_dict(orient="records")
    for record in census.to_dict(orient="records"):
        pdb_id = str(record["pdb_id"]).lower()
        expected = {value for value in str(record.get("nonpolymer_entity_ids") or "").split(";") if value}
        count = record.get("nonpolymer_entity_count")
        complete = count is not None and np.isfinite(float(count)) and int(count) == len(expected)
        inventory: list[dict[str, Any]] = []
        if expected:
            for nonpoly_id in sorted(expected):
                records = inventory_by_key.get((pdb_id, nonpoly_id))
                if records is None:
                    complete = False
                else:
                    inventory.extend(records)
        if expected and len(inventory) != len(expected):
            complete = False
        classes = {str(item.get("ligand_class")) for item in inventory}
        if not complete:
            state = StateLabel.UNRESOLVED
            evidence = "incomplete_component_inventory"
        elif classes & {LigandClass.BIOLOGICAL_SMALL_MOLECULE.value, LigandClass.COFACTOR.value}:
            state = StateLabel.HOLO
            evidence = "resolved_biological_ligand_or_cofactor"
        elif classes.issubset({LigandClass.ION.value, LigandClass.ADDITIVE_OR_SOLVENT.value}) and all(
            str(item.get("reference_status")) == "resolved" for item in inventory
        ):
            state = StateLabel.APO
            evidence = "complete_inventory_without_biological_ligand"
        else:
            state = StateLabel.UNRESOLVED
            evidence = "ambiguous_component_relevance"
        rows.append(
            {
                **record,
                "state_label": state.value,
                "state_evidence": evidence,
                "ligand_inventory_complete": bool(complete),
                "ligand_component_count_observed": len(inventory),
                "ligand_class_summary": ";".join(sorted(classes)) or None,
            }
        )
    return pd.DataFrame(rows)


def _assembly_rank(value: Any) -> int:
    text = _clean_text(value)
    if not text:
        return 0
    return 1


def _method_rank(left: Any, right: Any) -> int:
    first = (_clean_text(left) or "").casefold()
    second = (_clean_text(right) or "").casefold()
    return 1 if first and second and first == second else 0


def build_metadata_pairs(structures: pd.DataFrame) -> pd.DataFrame:
    """Construct all state-resolved pairs with outcome-independent metadata fields."""

    required = {"polymer_entity_id", "pdb_id", "uniprot_id", "state_label"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise ValueError(f"structures missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for protein_id, group in structures.groupby("uniprot_id", dropna=True, sort=True):
        apo = group.loc[group["state_label"].eq(StateLabel.APO.value)]
        holo = group.loc[group["state_label"].eq(StateLabel.HOLO.value)]
        apo_records = apo.to_dict(orient="records")
        holo_records = holo.to_dict(orient="records")
        for left in apo_records:
            left_length = pd.to_numeric(left.get("length"), errors="coerce")
            left_sequence = _clean_text(left.get("sequence"))
            for right in holo_records:
                right_length = pd.to_numeric(right.get("length"), errors="coerce")
                if np.isfinite(left_length) and np.isfinite(right_length):
                    length_difference = abs(float(left_length) - float(right_length))
                    length_similarity = 1.0 - length_difference / max(float(left_length), float(right_length), 1.0)
                else:
                    length_difference = None
                    length_similarity = None
                seq_right = _clean_text(right.get("sequence"))
                # An aligned identity value is deliberately retained only for an
                # exact sequence match.  Non-identical raw strings have no
                # authoritative alignment at this metadata-only stage.
                identity = 1.0 if left_sequence and seq_right and left_sequence == seq_right else None
                assembly_left = str(left.get("assembly_ids") or "").split(";")[0]
                assembly_right = str(right.get("assembly_ids") or "").split(";")[0]
                assembly_known = bool(assembly_left and assembly_right)
                assembly_comparable = bool(assembly_known and assembly_left == assembly_right)
                method_comparable = _method_rank(left.get("experimental_method"), right.get("experimental_method"))
                resolution_values = [
                    float(value)
                    for value in (left.get("resolution"), right.get("resolution"))
                    if value is not None and np.isfinite(float(value))
                ]
                resolution = max(resolution_values) if resolution_values else None
                pair_id = f"{left['pdb_id']}_{right['pdb_id']}__{protein_id}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "protein_id": str(protein_id),
                        "apo_polymer_entity_id": str(left["polymer_entity_id"]),
                        "holo_polymer_entity_id": str(right["polymer_entity_id"]),
                        "apo_pdb_id": str(left["pdb_id"]).lower(),
                        "holo_pdb_id": str(right["pdb_id"]).lower(),
                        "apo_state": StateLabel.APO.value,
                        "holo_state": StateLabel.HOLO.value,
                        "sequence_identity": identity,
                        "sequence_exact": bool(left_sequence and seq_right and left_sequence == seq_right),
                        "length_difference": length_difference,
                        "length_similarity": length_similarity,
                        "assembly_comparable": assembly_comparable if assembly_known else None,
                        "experimental_method_comparable": method_comparable if method_comparable else None,
                        "metadata_complete": bool(left_sequence and seq_right and left.get("length") is not None and right.get("length") is not None),
                        "resolution_pair_max": resolution,
                        "outcome_blind": True,
                    }
                )
    return pd.DataFrame(rows)


def _dominance_values(row: pd.Series) -> dict[str, float | None]:
    def numeric(name: str) -> float | None:
        value = row.get(name)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return float(value)

    resolution = numeric("resolution_pair_max")
    return {
        "sequence_identity": numeric("sequence_identity"),
        "length_similarity": numeric("length_similarity"),
        "assembly_comparable": None if row.get("assembly_comparable") is None else float(bool(row.get("assembly_comparable"))),
        "experimental_method_comparable": None if row.get("experimental_method_comparable") is None else float(bool(row.get("experimental_method_comparable"))),
        "metadata_complete": float(bool(row.get("metadata_complete"))),
        "sequence_exact": float(bool(row.get("sequence_exact"))),
        "resolution_pair_max": None if resolution is None else -resolution,
    }


def reduce_non_dominated_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Remove strictly dominated pairs after collapsing exact comparability ties.

    Exact metadata-equivalent pairs are a redundancy class, not an outcome-based
    cap.  One deterministic representative is retained and the class size is
    carried explicitly; the complete Cartesian pair table remains available to
    audit the reduction.
    """

    if pairs.empty:
        result = pairs.copy()
        result["non_dominated"] = pd.Series(dtype=bool)
        result["provisional_best"] = pd.Series(dtype=bool)
        return result
    rows: list[pd.DataFrame] = []
    for protein_id, group in pairs.groupby("protein_id", sort=True):
        signature = [
            "sequence_identity",
            "length_similarity",
            "assembly_comparable",
            "experimental_method_comparable",
            "metadata_complete",
            "sequence_exact",
            "resolution_pair_max",
        ]
        work = group.sort_values("pair_id", kind="mergesort").reset_index(drop=True).copy()
        work["equivalence_class_size"] = work.groupby(signature, dropna=False, sort=False)["pair_id"].transform("size")
        work = work.drop_duplicates(signature, keep="first").reset_index(drop=True)
        vectors = pd.DataFrame([_dominance_values(row) for _, row in work.iterrows()])[signature].replace({None: np.nan}).to_numpy(dtype=float)
        dominated = np.zeros(len(work), dtype=bool)
        # Chunked skyline computation avoids a quadratic boolean allocation for
        # the few proteins with tens of thousands of metadata pairs.
        for start in range(0, len(work), 256):
            stop = min(start + 256, len(work))
            candidate = vectors[start:stop]
            shared = (~np.isnan(vectors)[None, :, :]) & (~np.isnan(candidate)[:, None, :])
            ge = (~shared) | (vectors[None, :, :] >= candidate[:, None, :])
            gt = shared & (vectors[None, :, :] > candidate[:, None, :])
            dominated[start:stop] = (
                ge.all(axis=2) & gt.any(axis=2) & shared.any(axis=2)
            ).any(axis=1)
        work["non_dominated"] = ~dominated
        retained = work.loc[~dominated].copy()
        retained["provisional_best"] = False
        if not retained.empty:
            best = retained.sort_values(
                ["sequence_identity", "length_similarity", "resolution_pair_max", "pair_id"],
                ascending=[False, False, True, True],
                na_position="last",
                kind="mergesort",
            ).index[0]
            retained.loc[best, "provisional_best"] = True
        rows.append(retained)
    return pd.concat(rows, ignore_index=True).sort_values(["protein_id", "pair_id"], kind="mergesort").reset_index(drop=True)


def unique_structure_assets(reduced_pairs: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Return one raw-asset request row per participating polymer entity."""

    if reduced_pairs.empty:
        return structures.iloc[0:0].copy()
    ids = set(reduced_pairs["apo_polymer_entity_id"].astype(str)) | set(reduced_pairs["holo_polymer_entity_id"].astype(str))
    result = structures.loc[structures["polymer_entity_id"].astype(str).isin(ids)].copy()
    return result.sort_values(["uniprot_id", "state_label", "polymer_entity_id"], kind="mergesort").drop_duplicates("polymer_entity_id").reset_index(drop=True)


def _ledger_row(
    root: Path,
    *,
    asset_type: str,
    source_url: str,
    path: Path,
    status: str,
    **binding: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **binding,
        "asset_type": asset_type,
        "source_url": source_url,
        "relative_path": path.relative_to(root).as_posix(),
        "retrieval_status": status,
        "error": None,
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file():
        row["size_bytes"] = int(path.stat().st_size)
        row["sha256"] = sha256_file(path)
    return row


def acquire_selective_assets(
    config: Any,
    structures: pd.DataFrame,
    component_inventory: pd.DataFrame,
    *,
    workers: int = 8,
) -> pd.DataFrame:
    """Acquire only unique structures/components surviving metadata reduction.

    The ledger is entity-bound even when a PDB, UniProt record, or CCD payload is
    shared by multiple selected entities. Existing non-empty files are reused by
    the canonical download helpers.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(config.project_root)
    records = structures.to_dict(orient="records")

    def fetch_structure(record: dict[str, Any]) -> list[dict[str, Any]]:
        pdb_id = str(record["pdb_id"]).lower()
        entity_id = str(record["polymer_entity_id"])
        rows: list[dict[str, Any]] = []
        assets = [
            (
                "pdb_mmcif",
                RCSB_MMCIF_URL.format(pdb_id=pdb_id),
                root / "data/raw/apo_holo/pdb" / f"{pdb_id}.cif",
                lambda: fetch_pdb_mmcif(pdb_id, root / "data/raw/apo_holo/pdb"),
            ),
            (
                "sifts_xml",
                SIFTS_XML_URL.format(pdb_id=pdb_id),
                root / "data/raw/apo_holo/sifts" / f"{pdb_id}.xml.gz",
                lambda: fetch_sifts_xml(pdb_id, root / "data/raw/apo_holo/sifts"),
            ),
        ]
        for asset_type, source_url, expected, fetcher in assets:
            try:
                path = Path(fetcher())
                rows.append(_ledger_row(root, asset_type=asset_type, source_url=source_url, path=path, status="VALID", polymer_entity_id=entity_id, pdb_id=pdb_id, uniprot_id=str(record["uniprot_id"]).upper()))
            except Exception as exc:  # noqa: BLE001
                row = _ledger_row(root, asset_type=asset_type, source_url=source_url, path=expected, status="FAILED", polymer_entity_id=entity_id, pdb_id=pdb_id, uniprot_id=str(record["uniprot_id"]).upper())
                row["error"] = str(exc)
                rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=workers) as executor:
        structure_rows = [row for batch in executor.map(fetch_structure, records) for row in batch]

    unique_accessions = sorted({str(value).upper() for value in structures["uniprot_id"].dropna()})
    def fetch_uniprot(accession: str) -> list[dict[str, Any]]:
        source_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        path = root / "data/raw/apo_holo/uniprot" / f"{accession}.json"
        try:
            actual = Path(download_file(source_url, path))
            row = _ledger_row(root, asset_type="uniprot_canonical_json", source_url=source_url, path=actual, status="VALID", uniprot_id=accession)
        except Exception as exc:  # noqa: BLE001
            row = _ledger_row(root, asset_type="uniprot_canonical_json", source_url=source_url, path=path, status="FAILED", uniprot_id=accession)
            row["error"] = str(exc)
        return [row]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        uniprot_rows = [row for batch in executor.map(fetch_uniprot, unique_accessions) for row in batch]

    unique_ccd = sorted({str(value).upper() for value in component_inventory.get("ccd_id", pd.Series(dtype=str)).dropna()})
    def fetch_ccd(ccd_id: str) -> list[dict[str, Any]]:
        source_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}"
        path = root / "data/raw/apo_holo/ccd" / f"{ccd_id}.json"
        try:
            actual = Path(download_file(source_url, path))
            row = _ledger_row(root, asset_type="ccd_json", source_url=source_url, path=actual, status="VALID", ccd_id=ccd_id)
        except Exception as exc:  # noqa: BLE001
            row = _ledger_row(root, asset_type="ccd_json", source_url=source_url, path=path, status="FAILED", ccd_id=ccd_id)
            row["error"] = str(exc)
        return [row]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        ccd_rows = [row for batch in executor.map(fetch_ccd, unique_ccd) for row in batch]

    ledger = pd.DataFrame(structure_rows + uniprot_rows + ccd_rows)
    return ledger.sort_values(["asset_type", "polymer_entity_id", "uniprot_id", "ccd_id"], na_position="last", kind="mergesort").reset_index(drop=True)


def bind_shared_assets_to_entities(
    ledger: pd.DataFrame,
    structures: pd.DataFrame,
    component_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Expand shared UniProt/CCD asset rows into explicit entity bindings."""

    if ledger.empty:
        return ledger.copy()
    rows = ledger.to_dict(orient="records")
    accessions = structures[["polymer_entity_id", "pdb_id", "uniprot_id"]].drop_duplicates().to_dict(orient="records")
    # Build stable lookup lists once.  The previous implementation filtered the
    # full entity/component frames for every shared asset row, turning a
    # perfectly bounded binding step into an O(n*m) operation for the selective
    # cohort.  These indexes preserve the original row order and multiplicity.
    accessions_by_uniprot: dict[str, list[dict[str, Any]]] = {}
    for binding in accessions:
        accessions_by_uniprot.setdefault(
            str(binding.get("uniprot_id") or "").upper(), []
        ).append(binding)
    for row in ledger.loc[ledger["asset_type"].eq("uniprot_canonical_json")].to_dict(orient="records"):
        accession = str(row.get("uniprot_id") or "").upper()
        for binding in accessions_by_uniprot.get(accession, []):
            bound = dict(row)
            bound.update(
                {
                    "polymer_entity_id": str(binding["polymer_entity_id"]),
                    "pdb_id": str(binding["pdb_id"]).lower(),
                    "asset_scope": "entity_bound_shared_payload",
                }
            )
            rows.append(bound)
    selected_components = component_inventory.loc[
        component_inventory["pdb_id"].isin(structures["pdb_id"].unique())
    ]
    components_by_ccd: dict[str, list[dict[str, Any]]] = {}
    for binding in selected_components.to_dict(orient="records"):
        components_by_ccd.setdefault(
            str(binding.get("ccd_id") or "").upper(), []
        ).append(binding)
    entities_by_pdb: dict[str, list[dict[str, Any]]] = {}
    for entity in structures.to_dict(orient="records"):
        entities_by_pdb.setdefault(str(entity.get("pdb_id") or "").lower(), []).append(entity)
    for row in ledger.loc[ledger["asset_type"].eq("ccd_json")].to_dict(orient="records"):
        ccd = str(row.get("ccd_id") or "").upper()
        for binding in components_by_ccd.get(ccd, []):
            for entity in entities_by_pdb.get(str(binding["pdb_id"]).lower(), []):
                bound = dict(row)
                bound.update(
                    {
                        "polymer_entity_id": str(entity["polymer_entity_id"]),
                        "pdb_id": str(binding["pdb_id"]).lower(),
                        "nonpolymer_entity_id": str(binding["nonpolymer_entity_id"]),
                        "asset_scope": "entity_bound_shared_payload",
                    }
                )
                rows.append(bound)
    result = pd.DataFrame(rows)
    return result.sort_values(["asset_type", "polymer_entity_id", "uniprot_id", "ccd_id"], na_position="last", kind="mergesort").reset_index(drop=True)


def build_ligand_annotation_map(
    structures: pd.DataFrame,
    component_inventory: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Build entity-keyed ligand annotations from the resolved component table."""

    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not component_inventory.empty:
        for key, group in component_inventory.groupby(["pdb_id", "nonpolymer_entity_id"], sort=False):
            indexed[(str(key[0]).lower(), str(key[1]))] = group.to_dict(orient="records")
    result: dict[str, list[dict[str, Any]]] = {}
    columns = ["polymer_entity_id", "pdb_id", "nonpolymer_entity_ids"]
    available = structures.reindex(columns=columns)
    for entity_id_value, pdb_id_value, nonpolymer_ids in available.itertuples(
        index=False, name=None
    ):
        entity_id = str(entity_id_value)
        pdb_id = str(pdb_id_value).lower()
        expected = [
            value for value in str(nonpolymer_ids or "").split(";") if value
        ]
        annotations: list[dict[str, Any]] = []
        for nonpoly_id in expected:
            annotations.extend(indexed.get((pdb_id, nonpoly_id), [{"nonpolymer_entity_id": nonpoly_id, "ligand_class": LigandClass.AMBIGUOUS.value, "annotation_status": "FAILED"}]))
        result[entity_id] = annotations
    return result


def assess_entity_asset_readiness(
    structures: pd.DataFrame,
    ledger: pd.DataFrame,
    component_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Classify selected structures as raw-valid or unresolved acquisition."""

    if ledger.empty:
        result = structures.copy()
        result["raw_asset_status"] = "unresolved_asset"
        result["raw_asset_missing_types"] = "all"
        return result
    rows: list[dict[str, Any]] = []
    valid = ledger.loc[ledger["retrieval_status"].astype(str).str.startswith("VALID")].copy()
    valid_types_by_entity: dict[str, set[str]] = {}
    if "polymer_entity_id" in valid:
        for entity_id, group in valid.dropna(subset=["polymer_entity_id"]).groupby("polymer_entity_id", sort=False):
            valid_types_by_entity[str(entity_id)] = set(group["asset_type"].astype(str))
    valid_uniprots = set(valid.loc[valid["asset_type"].eq("uniprot_canonical_json"), "uniprot_id"].dropna().astype(str).str.upper()) if "uniprot_id" in valid else set()
    valid_ccd = set(valid.loc[valid["asset_type"].eq("ccd_json"), "ccd_id"].dropna().astype(str).str.upper()) if "ccd_id" in valid else set()
    ccd_by_pdb: dict[str, set[str]] = {}
    if not component_inventory.empty:
        for pdb_id, group in component_inventory.groupby("pdb_id", sort=False):
            ccd_by_pdb[str(pdb_id).lower()] = set(group["ccd_id"].dropna().astype(str).str.upper())
    for record in structures.to_dict(orient="records"):
        entity_id = str(record["polymer_entity_id"])
        pdb_id = str(record["pdb_id"]).lower()
        accession = str(record["uniprot_id"]).upper()
        missing: list[str] = []
        entity_types = valid_types_by_entity.get(entity_id, set())
        if "pdb_mmcif" not in entity_types:
            missing.append("pdb_mmcif")
        if "sifts_xml" not in entity_types:
            missing.append("sifts_xml")
        if accession not in valid_uniprots:
            missing.append("uniprot_canonical_json")
        expected = [value for value in str(record.get("nonpolymer_entity_ids") or "").split(";") if value]
        for ccd_id in sorted(ccd_by_pdb.get(pdb_id, set()).intersection({str(value).upper() for value in expected})):
            if ccd_id not in valid_ccd:
                missing.append(f"ccd_json:{ccd_id}")
        row = dict(record)
        row["raw_asset_status"] = "raw_valid" if not missing else "unresolved_asset"
        row["raw_asset_missing_types"] = ";".join(sorted(set(missing))) or None
        rows.append(row)
    return pd.DataFrame(rows)


def build_local_reuse_ledger(
    project_root: Path,
    structures: pd.DataFrame,
    component_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize a ledger for already present raw files without network access."""

    root = Path(project_root)
    rows: list[dict[str, Any]] = []
    for record in structures.to_dict(orient="records"):
        pdb_id = str(record["pdb_id"]).lower()
        entity_id = str(record["polymer_entity_id"])
        accession = str(record["uniprot_id"]).upper()
        assets = [
            ("pdb_mmcif", f"https://files.rcsb.org/download/{pdb_id.upper()}.cif", root / "data/raw/apo_holo/pdb" / f"{pdb_id}.cif"),
            ("sifts_xml", SIFTS_XML_URL.format(pdb_id=pdb_id), root / "data/raw/apo_holo/sifts" / f"{pdb_id}.xml.gz"),
            ("uniprot_canonical_json", f"https://rest.uniprot.org/uniprotkb/{accession}.json", root / "data/raw/apo_holo/uniprot" / f"{accession}.json"),
        ]
        for asset_type, source_url, path in assets:
            status = "VALID_LOCAL_REUSE" if path.is_file() and path.stat().st_size > 0 else "UNRESOLVED_NETWORK"
            rows.append(_ledger_row(root, asset_type=asset_type, source_url=source_url, path=path, status=status, polymer_entity_id=entity_id, pdb_id=pdb_id, uniprot_id=accession))
    selected_components = component_inventory.loc[component_inventory["pdb_id"].isin(structures["pdb_id"].unique())]
    for ccd_id in sorted(selected_components.get("ccd_id", pd.Series(dtype=str)).dropna().astype(str).str.upper().unique()):
        path = root / "data/raw/apo_holo/ccd" / f"{ccd_id}.json"
        status = "VALID_LOCAL_REUSE" if path.is_file() and path.stat().st_size > 0 else "UNRESOLVED_NETWORK"
        rows.append(_ledger_row(root, asset_type="ccd_json", source_url=f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}", path=path, status=status, ccd_id=ccd_id))
    return pd.DataFrame(rows).sort_values(["asset_type", "polymer_entity_id", "uniprot_id", "ccd_id"], na_position="last", kind="mergesort").reset_index(drop=True)
