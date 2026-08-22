"""Semantic staging for the outcome-blind functional-state cohort.

The discovery/admission implementation predates the benchmark schemas and is
kept as an immutable upstream input.  This module is a narrow adapter: it
selects the admitted rows, reconstructs lossless canonical residue mappings
from the cached SIFTS/mmCIF assets, and writes a schema-shaped staging set.
No model result, geometry outcome, or historical roadmap identifier is used to
define membership.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.benchmark.ids import structure_id as canonical_structure_id
from dual_uq.benchmark.schema_registry import SchemaRegistry
from dual_uq.benchmark.tables import (
    canonical_columns,
    validate_frame,
    write_parquet_bundle_transactional,
)
from dual_uq.benchmark.validation import validate_frame_contract, validate_pair_orientation
from dual_uq.construction.eligibility import build_generic_eligibility
from dual_uq.construction.functional_state import close_candidate_attrition
from dual_uq.construction.proteins import normalize_protein
from dual_uq.construction.structures import normalize_structure
from dual_uq.core.atomic_io import atomic_write_json, atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.services.mapping import parse_sifts_mapping_with_explicit_labels
from dual_uq.structure_io import (
    join_residue_mapping_to_ca,
    load_chain_ca_table,
    residue_name_to_one_letter,
)

SEMANTICS = "functional_state"
_CORE_TABLES = (
    "proteins",
    "structures",
    "condition_pairs",
    "residue_mappings",
    "benchmark_instances",
)
_ANNOTATION_TABLES = ("pair_structural_descriptors", "residue_structural_descriptors")
_AUDIT_TABLES = (
    "candidate_attrition",
    "asset_ledger",
    "cluster_assignments",
    "state_family_summary",
    "state_annotations",
)
_FORBIDDEN_RESIDUE_FIELDS = {
    "ca_displacement", "aligned_rmsd", "local_pairwise_distance_change",
    "neighborhood_geometry_change", "contact_change", "ligand_distance",
    "model_score", "model_probability", "model_response", "d_excess",
    "generative_js",
}
_STATE_LABELS = {
    "OPEN_CLOSED": {"OPEN", "CLOSED"},
    "INWARD_OUTWARD": {"INWARD_FACING", "OUTWARD_FACING"},
    "ACTIVE_INACTIVE": {"ACTIVE", "INACTIVE"},
    "RESTING_ACTIVATED": {"RESTING", "ACTIVATED"},
    "PRE_POST": {"PRE_TRANSITION", "POST_TRANSITION"},
}


class FunctionalStateCanonicalError(ValueError):
    """Raised when semantic staging cannot satisfy its table contracts."""


def _repository_root() -> Path:
    """Return the repository root containing this package and frozen schemas."""

    return Path(__file__).resolve().parents[3]


def _schema_columns(schema_root: Path, table: str) -> tuple[str, ...]:
    return canonical_columns(table, SchemaRegistry(schema_root))


def _validate_schema_frame(frame: pd.DataFrame, table: str, schema_root: Path) -> None:
    """Validate a DataFrame against the single frozen schema authority."""
    try:
        validate_frame_contract(frame, table, SchemaRegistry(schema_root))
    except ValueError as exc:
        raise FunctionalStateCanonicalError(str(exc)) from exc


@dataclass(frozen=True)
class FunctionalStateCanonicalConfig:
    """Portable paths for canonical functional-state staging."""

    project_root: Path
    admission_root: Path
    discovery_root: Path
    output_root: Path
    audit_root: Path | None = None
    mapping_workers: int = 8

    def __post_init__(self) -> None:
        root = Path(self.project_root).expanduser().resolve()
        object.__setattr__(self, "project_root", root)
        for name in ("admission_root", "discovery_root", "output_root", "audit_root"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = Path(raw).expanduser()
            object.__setattr__(self, name, value if value.is_absolute() else root / value)
        if self.audit_root is None:
            object.__setattr__(self, "audit_root", root / "experiments/interventions/functional_states/construction")
        if self.mapping_workers < 1:
            raise ValueError("mapping_workers must be positive")


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value = str(value).strip()
    return value if value and value.lower() not in {"nan", "none", "<na>"} else None


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _flag_series(values: pd.Series) -> pd.Series:
    return values.map(_flag).astype(bool)


def _portable(root: Path, value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    path = Path(text)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix() if path.is_absolute() else path.as_posix()
    except ValueError:
        return path.as_posix()


def _sequence_status(sequence: str, source: Any = None) -> str:
    status = _text(source)
    if status == "UNRESOLVED":
        return status
    if not sequence:
        return "UNRESOLVED"
    return "STANDARD_20AA" if set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY") else "NONSTANDARD"


def _admitted_pairs(admission: dict[str, pd.DataFrame]) -> pd.DataFrame:
    primary = admission.get("primary_pairs", pd.DataFrame()).copy()
    alternatives = admission.get("alternative_pairs", pd.DataFrame()).copy()
    if not primary.empty or not alternatives.empty:
        result = pd.concat([primary, alternatives], ignore_index=True, sort=False).drop_duplicates("pair_id")
        return result.loc[~result.get("state_contrast_category", pd.Series(index=result.index)).eq("PURE_LIGAND_STATE_ONLY")].copy()
    pairs = admission.get("candidate_pairs", pd.DataFrame()).copy()
    result = pairs.loc[pairs.get("final_pair_state", pd.Series(index=pairs.index)).isin(["PRIMARY", "ALTERNATIVE"])]
    return result.loc[~result.get("state_contrast_category", pd.Series(index=result.index)).eq("PURE_LIGAND_STATE_ONLY")].copy()


def _structure_lookup(structures: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if structures.empty or "polymer_entity_id" not in structures:
        return {}
    rows = structures.drop_duplicates("polymer_entity_id", keep="first")
    return rows.set_index("polymer_entity_id").to_dict(orient="index")


def _ledger_lookup(ledger: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if ledger.empty:
        return {}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ledger.to_dict(orient="records"):
        key = (str(row.get("asset_type") or ""), str(row.get("pdb_id") or row.get("uniprot_id") or "").upper())
        result.setdefault(key, row)
    return result


def _structure_id(row: dict[str, Any], side: str) -> str:
    pdb_id = str(row.get("pdb_" + side) or "").lower()
    entity_id = str(row.get("entity_" + side) or "")
    source_id = f"{pdb_id}:{entity_id}"
    return canonical_structure_id(
        source_structure_id=source_id,
        semantics=SEMANTICS,
        condition_label=_text(row.get("state_" + side)) or "UNRESOLVED",
        protein=str(row.get("uniprot_id") or "").upper(),
    )


def _build_proteins(
    admitted: pd.DataFrame,
    structures: pd.DataFrame,
    clusters: pd.DataFrame,
    schema_root: Path,
) -> pd.DataFrame:
    lookup = _structure_lookup(structures)
    rows: list[dict[str, Any]] = []
    for protein, group in admitted.groupby("uniprot_id", sort=True):
        source = None
        for entity in group[["entity_a", "entity_b"]].to_numpy().ravel():
            if str(entity) in lookup and _text(lookup[str(entity)].get("canonical_sequence")):
                source = lookup[str(entity)]
                break
        sequence = _text((source or {}).get("canonical_sequence")) or ""
        if not sequence:
            raise FunctionalStateCanonicalError(f"admitted protein has no canonical sequence: {protein}")
        normalized = normalize_protein(
            str(protein).upper(), sequence,
            source_provenance=_text((source or {}).get("uniprot_relative_path")) or "unresolved",
        )
        rows.append(normalized)
    return pd.DataFrame(rows, columns=_schema_columns(schema_root, "proteins")).sort_values("protein_id", kind="mergesort").reset_index(drop=True)


def _annotation_lookup(annotations: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if annotations.empty or "polymer_entity_id" not in annotations:
        return {}
    result = {}
    for row in annotations.to_dict(orient="records"):
        result.setdefault((str(row.get("polymer_entity_id")), str(row.get("state_label"))), row)
    return result


def _build_structures(
    admitted: pd.DataFrame,
    structures: pd.DataFrame,
    annotations: pd.DataFrame,
    ledger: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    lookup = _structure_lookup(structures)
    annotations_by_key = _annotation_lookup(annotations)
    assets = _ledger_lookup(ledger)
    rows: list[dict[str, Any]] = []
    for pair in admitted.to_dict(orient="records"):
        for side in ("a", "b"):
            entity = str(pair[f"entity_{side}"])
            source = dict(lookup.get(entity, {}))
            state_label = _text(pair.get(f"state_{side}")) or "UNRESOLVED"
            pdb_id = str(pair.get(f"pdb_{side}") or source.get("pdb_id") or "").lower()
            asset = assets.get(("pdb_mmcif", pdb_id.upper()), {})
            evidence = annotations_by_key.get((entity, state_label), {})
            normalized = normalize_structure(
                source_structure_id=f"{pdb_id}:{entity}",
                protein_id=str(pair["uniprot_id"]).upper(),
                structural_condition_semantics=SEMANTICS,
                condition_label=state_label,
                condition_type="experimental",
                source_type="PDB",
                source_file_ref=_portable(root, source.get("mmcif_relative_path")) or _portable(root, asset.get("relative_path")) or "unresolved",
                state_family=_text(pair.get("state_family")),
                source_accession=pdb_id,
                chain_id=_text(pair.get(f"chain_{side}")) or _text(source.get("chain_id")),
                entity_id=entity,
                assembly_id=(_text(source.get("assembly_ids")) or "").split(";")[0] or None,
                experimental_method=_text(source.get("experimental_method")),
                state_evidence_tier=_text(pair.get(f"evidence_tier_{side}")) or _text(evidence.get("evidence_tier")),
                state_evidence_source=_text(pair.get(f"evidence_source_{side}")) or _text(evidence.get("evidence_source")),
            )
            normalized.update({
                "structure_id": _structure_id(pair, side),
                "parent_structure_id": None,
            })
            rows.append(normalized)
    columns = list(_schema_columns(root / "schemas", "structures"))
    return pd.DataFrame(rows, columns=columns).drop_duplicates("structure_id").sort_values("structure_id", kind="mergesort").reset_index(drop=True)


def _residue_id(row: dict[str, Any]) -> str | None:
    chain = _text(row.get("auth_asym_id")) or _text(row.get("chain_id"))
    raw_auth = row.get("auth_seq_id")
    if isinstance(raw_auth, float) and math.isfinite(raw_auth) and raw_auth.is_integer():
        raw_auth = int(raw_auth)
    auth = _text(raw_auth)
    insertion = _text(row.get("insertion_code")) or ""
    if chain is None or auth is None:
        return None
    return f"{chain}:{auth}{insertion}"


def _resolve_asset_path(root: Path, value: Any) -> Path | None:
    text = _text(value)
    if text is None:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def _load_one_persisted_identity_fact(
    payload: tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], Path, bool],
) -> tuple[str, pd.DataFrame]:
    row, asset_lookup, root, identity_only = payload
    entity = str(row["polymer_entity_id"])
    pdb_id = str(row.get("pdb_id") or "").lower()
    asset = asset_lookup.get(("pdb_mmcif", pdb_id.upper()), {})
    mapping_path = _resolve_asset_path(
        root, row.get("mapping_relative_path")
    ) or _resolve_asset_path(root, asset_lookup.get(("sifts_xml", pdb_id.upper()), {}).get("relative_path"))
    cif_path = _resolve_asset_path(
        root, row.get("mmcif_relative_path")
    ) or _resolve_asset_path(root, asset.get("relative_path"))
    if mapping_path is None or cif_path is None or not mapping_path.is_file() or not cif_path.is_file():
        raise FunctionalStateCanonicalError(
            f"persisted SIFTS/mmCIF identity fact unavailable for {entity}"
        )
    try:
        mapping = parse_sifts_mapping_with_explicit_labels(
            mapping_path,
            cif_path,
            chain_id=str(row["chain_id"]),
            uniprot_id=str(row["uniprot_id"]),
        )
        if identity_only:
            joined = mapping.copy()
            joined["residue_one_letter"] = joined["pdb_residue_name"].map(
                residue_name_to_one_letter
            )
        else:
            ca = load_chain_ca_table(cif_path, chain_id=str(row["chain_id"]))
            joined, _ = join_residue_mapping_to_ca(mapping, ca)
    except Exception as exc:
        raise FunctionalStateCanonicalError(
            f"persisted SIFTS/mmCIF identity fact unreadable for {entity}"
        ) from exc

    position = pd.to_numeric(joined["uniprot_residue_number"], errors="coerce")
    if position.isna().any() or position.duplicated().any():
        raise FunctionalStateCanonicalError(
            f"persisted identity facts have invalid canonical positions for {entity}"
        )
    return entity, pd.DataFrame(
        {
            "canonical_position": position.astype(int),
            "residue_id": [
                _residue_id(record)
                for record in joined.to_dict(orient="records")
            ],
            "aa": joined["residue_one_letter"].map(_text),
            "mapped": True,
        }
    )


def _load_persisted_identity_facts(
    structures: pd.DataFrame,
    ledger: pd.DataFrame,
    root: Path,
    *,
    entities: set[str] | None = None,
    identity_only: bool = False,
    workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Load identity facts from already-persisted SIFTS/mmCIF assets.

    This is a source-fact reader for canonical projection. It does not align,
    infer, or re-run admission. Every returned row is keyed by the persisted
    UniProt position and carries the persisted author residue identity and
    observed residue letter.
    """

    if structures.empty:
        return {}
    required = {"polymer_entity_id", "pdb_id", "uniprot_id", "chain_id"}
    missing = sorted(required - set(structures.columns))
    if missing:
        raise FunctionalStateCanonicalError(
            f"canonical identity recovery structures missing fields: {missing}"
        )
    asset_lookup = _ledger_lookup(ledger)
    selected = structures.drop_duplicates("polymer_entity_id", keep="first")
    if entities is not None:
        selected = selected.loc[selected["polymer_entity_id"].astype(str).isin(entities)]
    payloads = [
        (row, asset_lookup, root, identity_only)
        for row in selected.to_dict(orient="records")
    ]
    if workers < 1:
        raise ValueError("identity source workers must be positive")
    if workers == 1 or len(payloads) < 2:
        result = map(_load_one_persisted_identity_fact, payloads)
    else:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=workers)
        result = pool.map(_load_one_persisted_identity_fact, payloads)
        try:
            facts = dict(result)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return facts
    return dict(result)


def _nonempty(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).str.strip().ne("")


def _has_value(value: Any) -> bool:
    return _text(value) is not None


def _validate_mapping_semantics(
    mapping: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    """Validate identity, visibility, commonness, and cross-row semantics."""

    if mapping.empty:
        return
    for side in ("condition_1", "condition_2"):
        mapped = _flag_series(mapping[f"{side}_mapped"])
        visible = _flag_series(mapping[f"{side}_coordinate_visible"])
        if (mapped & ~_nonempty(mapping[f"{side}_residue_id"])).any():
            raise FunctionalStateCanonicalError(
                f"mapped condition side requires residue_id: {side}"
            )
        if (mapped & ~_nonempty(mapping[f"{side}_aa"])).any():
            raise FunctionalStateCanonicalError(
                f"mapped condition side requires aa: {side}"
            )
        if (visible & ~mapped).any():
            raise FunctionalStateCanonicalError(
                f"coordinate-visible condition side must be mapped: {side}"
            )
    common = _flag_series(mapping["common_mapped"])
    mapped_1 = _flag_series(mapping["condition_1_mapped"])
    mapped_2 = _flag_series(mapping["condition_2_mapped"])
    if (common & ~(mapped_1 & mapped_2)).any():
        raise FunctionalStateCanonicalError(
            "common_mapped requires both condition sides mapped"
        )

    pair_structures = pairs.loc[
        :, [
            "pair_id",
            "condition_1_structure_id",
            "condition_2_structure_id",
        ]
    ].copy()
    for side in ("condition_1", "condition_2"):
        side_mapping = mapping.loc[
            :, [
                "pair_id",
                "canonical_position",
                f"{side}_residue_id",
                f"{side}_aa",
                f"{side}_mapped",
            ]
        ].merge(pair_structures, on="pair_id", how="left", validate="many_to_one")
        side_mapping["_mapped"] = _flag_series(side_mapping[f"{side}_mapped"])
        side_mapping = side_mapping.loc[side_mapping["_mapped"]].copy()
        if side_mapping.empty:
            continue
        side_mapping["_identity"] = (
            side_mapping[f"{side}_residue_id"].astype(str)
            + "\x1f"
            + side_mapping[f"{side}_aa"].astype(str)
        )
        conflicts = (
            side_mapping.groupby(
                [f"{side}_structure_id", "canonical_position"], dropna=False
            )["_identity"]
            .nunique()
            .loc[lambda values: values.gt(1)]
        )
        if not conflicts.empty:
            raise FunctionalStateCanonicalError(
                "structure identity mapping consistency failed"
            )


def _build_residue_mappings(
    admitted: pd.DataFrame,
    persisted: pd.DataFrame,
    structures: pd.DataFrame,
    ledger: pd.DataFrame,
    schema_root: Path,
    root: Path,
    *,
    identity_only: bool = False,
    identity_workers: int = 1,
) -> pd.DataFrame:
    """Project the completed admission mapping artifact into the frozen schema.

    The admission artifact contains the common coordinate-visible rows that
    were already accepted by the scientific mapping/comparability boundary.
    Canonical staging must preserve those facts; it must not reopen assets or
    reconstruct a second mapping from coordinates.
    """

    columns = list(_schema_columns(schema_root, "residue_mappings"))
    if persisted.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "pair_id",
        "canonical_position",
        "canonical_residue",
        "coordinate_observable_a",
        "coordinate_observable_b",
    }
    missing = sorted(required - set(persisted.columns))
    if missing:
        raise FunctionalStateCanonicalError(
            f"persisted residue mapping artifact missing fields: {missing}"
        )
    pair_ids = set(admitted["pair_id"].astype(str))
    source = persisted.copy()
    source["pair_id"] = source["pair_id"].astype(str)
    unknown = sorted(set(source["pair_id"]) - pair_ids)
    if unknown:
        raise FunctionalStateCanonicalError(
            f"persisted residue mapping references non-admitted pairs: {unknown[:5]}"
        )
    positions = pd.to_numeric(source["canonical_position"], errors="coerce")
    if positions.isna().any() or positions.mod(1).ne(0).any() or positions.le(0).any():
        raise FunctionalStateCanonicalError("persisted residue mapping has invalid canonical positions")
    if source.duplicated(["pair_id", "canonical_position"]).any():
        raise FunctionalStateCanonicalError("persisted residue mapping contains duplicate keys")
    if source["canonical_residue"].isna().any():
        raise FunctionalStateCanonicalError("persisted residue mapping has missing canonical residues")

    protein_lookup = admitted.assign(
        pair_id=admitted["pair_id"].astype(str),
        protein_id=admitted["uniprot_id"].astype(str).str.upper(),
    ).set_index("pair_id")["protein_id"]
    protein_ids = source["pair_id"].map(protein_lookup)
    if protein_ids.isna().any():
        raise FunctionalStateCanonicalError("persisted residue mapping has an unresolved protein foreign key")

    identity_columns = {
        "condition_1_residue_id",
        "condition_2_residue_id",
        "condition_1_aa",
        "condition_2_aa",
    }
    direct_identity_columns = identity_columns <= set(source.columns)
    direct_identity = direct_identity_columns
    if direct_identity_columns:
        for side in ("condition_1", "condition_2"):
            mapped = (
                _flag_series(source[f"{side}_mapped"])
                if f"{side}_mapped" in source
                else pd.Series(True, index=source.index)
            )
            direct_identity &= not (
                mapped
                & (
                    ~_nonempty(source[f"{side}_residue_id"])
                    | ~_nonempty(source[f"{side}_aa"])
                )
            ).any()
    identity_facts = (
        {}
        if direct_identity
        else _load_persisted_identity_facts(
            structures,
            ledger,
            root,
            entities=set(admitted[["entity_a", "entity_b"]].astype(str).to_numpy().ravel()),
            identity_only=identity_only,
            workers=identity_workers,
        )
    )
    identity_lookup = {
        entity: facts.set_index("canonical_position").to_dict(orient="index")
        for entity, facts in identity_facts.items()
    }
    pair_entities = admitted.set_index("pair_id")[["entity_a", "entity_b"]].astype(str)
    coordinate_a = _flag_series(source["coordinate_observable_a"])
    coordinate_b = _flag_series(source["coordinate_observable_b"])
    canonical_aa = source["canonical_residue"].astype(str)
    rows: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        pair_id = str(row["pair_id"])
        pair_entity = pair_entities.loc[pair_id]
        side_values: dict[str, tuple[Any, Any, bool]] = {}
        for side, coordinate_visible in (
            ("condition_1", bool(coordinate_a.loc[index])),
            ("condition_2", bool(coordinate_b.loc[index])),
        ):
            mapped = (
                _flag(row[f"{side}_mapped"])
                if f"{side}_mapped" in source
                else True
            )
            residue_id = row[f"{side}_residue_id"] if direct_identity_columns else None
            aa = row[f"{side}_aa"] if direct_identity_columns else None
            if not direct_identity:
                entity = str(pair_entity["entity_a" if side == "condition_1" else "entity_b"])
                fact = identity_lookup.get(entity, {}).get(int(row["canonical_position"]))
                fact_residue_id = None if fact is None else fact["residue_id"]
                fact_aa = None if fact is None else fact["aa"]
                if (
                    _has_value(residue_id)
                    and _has_value(fact_residue_id)
                    and str(residue_id) != str(fact_residue_id)
                ) or (
                    _has_value(aa)
                    and _has_value(fact_aa)
                    and str(aa) != str(fact_aa)
                ):
                    raise FunctionalStateCanonicalError(
                        "persisted identity conflict: "
                        f"pair={pair_id} position={row['canonical_position']} side={side}"
                    )
                if not _has_value(residue_id):
                    residue_id = fact_residue_id
                if not _has_value(aa):
                    aa = fact_aa
            if mapped and (not _has_value(residue_id) or not _has_value(aa)):
                raise FunctionalStateCanonicalError(
                    f"mapped condition side requires persisted identity: pair={pair_id} position={row['canonical_position']} side={side}"
                )
            if coordinate_visible and not mapped:
                raise FunctionalStateCanonicalError(
                    f"coordinate-visible condition side must be mapped: pair={pair_id} position={row['canonical_position']} side={side}"
                )
            side_values[side] = (residue_id if mapped else None, aa if mapped else None, mapped)
        mapped_1 = side_values["condition_1"][2]
        mapped_2 = side_values["condition_2"][2]
        common_mapped = mapped_1 and mapped_2
        common_coordinate = bool(coordinate_a.loc[index] and coordinate_b.loc[index])
        rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_ids.loc[index],
                "canonical_position": int(positions.loc[index]),
                "canonical_aa": canonical_aa.loc[index],
                "condition_1_residue_id": side_values["condition_1"][0],
                "condition_2_residue_id": side_values["condition_2"][0],
                "condition_1_aa": side_values["condition_1"][1],
                "condition_2_aa": side_values["condition_2"][1],
                "condition_1_mapped": mapped_1,
                "condition_2_mapped": mapped_2,
                "condition_1_coordinate_visible": bool(coordinate_a.loc[index]),
                "condition_2_coordinate_visible": bool(coordinate_b.loc[index]),
                "common_mapped": common_mapped,
                "common_coordinate_visible": common_coordinate,
                "condition_1_missing_reason": None if coordinate_a.loc[index] else ("COORDINATES_UNAVAILABLE" if mapped_1 else "UNMAPPED"),
                "condition_2_missing_reason": None if coordinate_b.loc[index] else ("COORDINATES_UNAVAILABLE" if mapped_2 else "UNMAPPED"),
                "mapping_status": (
                    "COMMON_VISIBLE"
                    if common_coordinate
                    else "MAPPED_NOT_VISIBLE"
                    if common_mapped
                    else "PARTIAL_OR_UNMAPPED"
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    return result.loc[:, columns].sort_values(
        ["pair_id", "canonical_position"], kind="mergesort"
    ).reset_index(drop=True)


def repair_primary_residue_mapping_identity(
    canonical_mapping: pd.DataFrame,
    primary_pairs: pd.DataFrame,
    admission_mapping: pd.DataFrame,
    admission_structures: pd.DataFrame,
    admission_ledger: pd.DataFrame,
    *,
    schema_root: Path,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Repair only PRIMARY condition-side identity from persisted source facts.

    The admission mapping remains the source of mapping, visibility, and
    coordinate facts. This function projects those frozen rows through the
    canonical owner and replaces only the four condition-side identity fields
    in the PRIMARY slice of an existing canonical mapping table.
    """

    identity_columns = [
        "condition_1_residue_id",
        "condition_2_residue_id",
        "condition_1_aa",
        "condition_2_aa",
    ]
    key_columns = ["pair_id", "canonical_position"]
    required_pairs = {"pair_id", "entity_a", "entity_b", "uniprot_id"}
    missing_pairs = sorted(required_pairs - set(primary_pairs.columns))
    if missing_pairs:
        raise FunctionalStateCanonicalError(
            f"PRIMARY pair source missing fields: {missing_pairs}"
        )
    if canonical_mapping.duplicated(key_columns).any():
        raise FunctionalStateCanonicalError(
            "canonical residue mapping contains duplicate PRIMARY repair keys"
        )
    primary_ids = set(primary_pairs["pair_id"].astype(str))
    current = canonical_mapping.copy()
    current["pair_id"] = current["pair_id"].astype(str)
    current_primary = current.loc[current["pair_id"].isin(primary_ids)].copy()
    if current_primary.empty:
        raise FunctionalStateCanonicalError("canonical residue mapping has no PRIMARY rows")

    source = admission_mapping.copy()
    source["pair_id"] = source["pair_id"].astype(str)
    source = source.loc[source["pair_id"].isin(primary_ids)].copy()
    source_ids = set(source["pair_id"])
    if source_ids != primary_ids:
        missing_source = sorted(primary_ids - source_ids)
        raise FunctionalStateCanonicalError(
            f"PRIMARY admission mapping is incomplete: missing pairs {missing_source[:5]}"
        )

    projected = _build_residue_mappings(
        primary_pairs.copy(),
        source,
        admission_structures,
        admission_ledger,
        schema_root,
        project_root,
        identity_only=True,
        identity_workers=4,
    )
    current_primary = current_primary.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    projected = projected.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    if not current_primary[key_columns].equals(projected[key_columns]):
        raise FunctionalStateCanonicalError(
            "PRIMARY repair changed canonical residue mapping keys"
        )
    preserve_columns = [column for column in current.columns if column not in identity_columns]
    if set(preserve_columns) != set(projected.columns) - set(identity_columns):
        raise FunctionalStateCanonicalError(
            "PRIMARY repair changed the residue mapping schema"
        )
    try:
        pd.testing.assert_frame_equal(
            current_primary.loc[:, preserve_columns],
            projected.loc[:, preserve_columns],
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise FunctionalStateCanonicalError(
            "PRIMARY repair changed non-identity mapping facts"
        ) from exc
    for column in identity_columns:
        existing = current_primary[column]
        present = _nonempty(existing)
        if present.any():
            left = existing.loc[present].astype(str).reset_index(drop=True)
            right = projected.loc[present, column].astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise FunctionalStateCanonicalError(
                    f"PRIMARY repair changed existing identity values: {column}"
                )

    repaired = pd.concat(
        [current.loc[~current["pair_id"].isin(primary_ids)], projected],
        ignore_index=True,
    ).sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    _validate_schema_frame(repaired, "residue_mappings", schema_root)
    before_missing = int(
        sum((~_nonempty(current_primary[column])).sum() for column in identity_columns)
    )
    after_missing = int(
        sum((~_nonempty(projected[column])).sum() for column in identity_columns)
    )
    return repaired, {
        "primary_pairs": len(primary_ids),
        "primary_rows": len(projected),
        "identity_cells_missing_before": before_missing,
        "identity_cells_missing_after": after_missing,
    }


def _build_condition_pairs(admitted: pd.DataFrame, schema_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in admitted.to_dict(orient="records"):
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "protein_id": str(pair["uniprot_id"]).upper(),
                "arm": SEMANTICS,
                "condition_1_structure_id": _structure_id(pair, "a"),
                "condition_2_structure_id": _structure_id(pair, "b"),
                "condition_1_label": _text(pair.get("state_a")) or "UNRESOLVED",
                "condition_2_label": _text(pair.get("state_b")) or "UNRESOLVED",
                "pair_role": str(pair.get("final_pair_state") or "ALTERNATIVE"),
                "admission_status": "ADMITTED",
                "sequence_comparable": _flag(pair.get("sequence_exact", False)),
                "construct_comparable": bool(float(pair.get("construct_overlap_fraction") or 0.0) >= 0.90),
                "assembly_comparable": _flag(pair.get("assembly_comparable", False)),
                "mapping_comparable": _flag(pair.get("mapping_valid", False)),
                "pair_provenance": "discovery_candidate_and_outcome_blind_admission",
                "state_family": _text(pair.get("state_family")),
                "admission_reason": None,
                "common_mapped_count": int(pair.get("common_mapped_count") or 0),
                "common_mapped_fraction": pair.get("common_mapped_fraction"),
                "common_coordinate_visible_count": int(pair.get("common_coordinate_visible_count") or 0),
                "common_coordinate_visible_fraction": pair.get("common_coordinate_visible_fraction"),
                "ligand_context_class": _text(pair.get("state_contrast_category")) or "UNRESOLVED_LIGAND_CONTEXT",
                "perturbation_family": None,
                "perturbation_dose": None,
                "perturbation_dose_unit": None,
            }
        )
    columns = list(_schema_columns(schema_root, "condition_pairs"))
    return pd.DataFrame(rows, columns=columns).sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _adapt_descriptors(admission: dict[str, pd.DataFrame], pair_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project completed structural descriptors without recomputing geometry."""

    source_pair = admission.get("pair_structural_descriptors", pd.DataFrame())
    pair_rows = []
    for pair_id in sorted(pair_ids):
        source = source_pair.loc[source_pair.get("pair_id", pd.Series(dtype=str)).astype(str).eq(pair_id)] if not source_pair.empty else pd.DataFrame()
        row = source.iloc[0].to_dict() if not source.empty else {}
        pair_rows.append(
            {
                "pair_id": pair_id,
                "aligned_ca_rmsd": row.get("aligned_ca_rmsd"),
                "median_residue_displacement": row.get("median_residue_displacement"),
                "upper_tail_residue_displacement": row.get("p90_residue_displacement"),
                "global_contact_change": row.get("global_contact_change"),
                "descriptor_method": _text(row.get("geometry_status")) or "functional_state_admission",
                "descriptor_version": "functional_state_descriptor",
            }
        )
    pair_table = pd.DataFrame(pair_rows, columns=[
        "pair_id", "aligned_ca_rmsd", "median_residue_displacement",
        "upper_tail_residue_displacement", "global_contact_change",
        "descriptor_method", "descriptor_version",
    ])
    source_residue = admission.get("residue_structural_descriptors", pd.DataFrame())
    if source_residue.empty:
        residue_table = pd.DataFrame(columns=["pair_id", "canonical_position", "ca_displacement", "local_pairwise_distance_change", "neighborhood_geometry_change", "contact_gain", "contact_loss", "descriptor_method", "descriptor_version"])
    else:
        residue_table = pd.DataFrame(
            [
                {
                    "pair_id": row.get("pair_id"),
                    "canonical_position": row.get("canonical_position"),
                    "ca_displacement": row.get("ca_displacement"),
                    "local_pairwise_distance_change": row.get("local_pairwise_distance_change"),
                    "neighborhood_geometry_change": row.get("neighborhood_geometry_change"),
                    "contact_gain": row.get("contact_gain"),
                    "contact_loss": row.get("contact_loss"),
                    "descriptor_method": _text(row.get("descriptor_method")) or "functional_state_admission",
                    "descriptor_version": "functional_state_descriptor",
                }
                for row in source_residue.to_dict(orient="records")
                if str(row.get("pair_id")) in pair_ids
            ]
        )
    return pair_table, residue_table


def _build_benchmark_instances(pairs: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    standard = proteins.set_index("protein_id")["canonical_sequence_status"].to_dict() if not proteins.empty else {}
    rows = []
    for pair in pairs.to_dict(orient="records"):
        protein = str(pair["protein_id"])
        mapped = bool(pair["mapping_comparable"])
        visible = int(pair.get("common_coordinate_visible_count") or 0)
        sequence_ok = standard.get(protein) == "STANDARD_20AA"
        eligibility = build_generic_eligibility(
            mapping_available=mapped,
            geometry_available=visible >= 3,
            ligand_available=False,
        )
        if not sequence_ok:
            eligibility["generation_eligible"] = False
            eligibility["compatibility_eligible"] = False
            eligibility["multistate_eligible"] = False
        rows.append(
            {
                "instance_id": str(pair["pair_id"]),
                "pair_id": str(pair["pair_id"]),
                "protein_id": protein,
                "arm": SEMANTICS,
                **eligibility,
            }
        )
    columns = [
        "instance_id", "pair_id", "protein_id", "arm",
        "local_sensitivity_eligible", "generation_eligible",
        "compatibility_eligible", "multistate_eligible",
        "geometry_evaluable", "ligand_evaluable", "eligibility_reason",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("instance_id", kind="mergesort").reset_index(drop=True)


def _build_candidate_attrition(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    result = candidate_pairs.copy()
    if "final_pair_state" in result:
        result["terminal_status"] = result["final_pair_state"]
    elif "terminal_status" not in result:
        result["terminal_status"] = result.get("admission_status", "UNRESOLVED")
    if "admission_reasons" in result:
        result["terminal_reason"] = result["admission_reasons"]
    elif "terminal_reason" not in result:
        result["terminal_reason"] = None
    result = result.drop(columns=["final_pair_state", "admission_reasons"], errors="ignore")
    if "terminal_reason" not in result:
        result["terminal_reason"] = None
    result["terminal_status"] = result["terminal_status"].where(result["terminal_status"].notna(), "UNRESOLVED").astype(str)
    result.loc[result["terminal_status"].str.lower().isin(["nan", "none", "<na>", ""]), "terminal_status"] = "UNRESOLVED"
    if "terminal_reason" in result:
        result["terminal_reason"] = result["terminal_reason"].astype("string").str.replace(
            "experimental_entity_mismatch", "protein_identity_mismatch", regex=False
        )
    result["outcome_blind"] = True
    result["structural_condition_semantics"] = SEMANTICS
    result["candidate_pair_id"] = result["pair_id"].astype(str)
    result["discovery_source_identity"] = "functional_state_discovery/candidate_pairs.parquet"
    return result


def _normalize_asset_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    result = ledger.copy()
    if "retrieval_status" in result:
        result["asset_status"] = result["retrieval_status"].astype(str).map(
            lambda value: "AVAILABLE" if value.startswith("AVAILABLE") else "UNAVAILABLE" if value.startswith("UNAVAILABLE") else "MALFORMED" if value.startswith("MALFORMED") else "UNRESOLVED"
        )
    return result


def canonicalize_admission_tables(
    admission: dict[str, pd.DataFrame],
    discovery: dict[str, pd.DataFrame],
    *,
    project_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert immutable admission/discovery frames into semantic staging tables."""

    admitted = _admitted_pairs(admission).copy()
    structures = admission.get("structures", discovery.get("structures", pd.DataFrame())).copy()
    annotations = admission.get("annotations", discovery.get("annotations", pd.DataFrame())).copy()
    ledger = admission.get("asset_ledger", pd.DataFrame()).copy()
    clusters = admission.get("cluster_assignments", pd.DataFrame()).copy()
    root = Path(project_root).expanduser().resolve() if project_root is not None else _repository_root()
    schema_root = root / "schemas"
    proteins = _build_proteins(admitted, structures, clusters, schema_root)
    canonical_structures = _build_structures(admitted, structures, annotations, ledger, root)
    pairs = _build_condition_pairs(admitted, schema_root)
    mappings = _build_residue_mappings(
        admitted,
        admission.get("residue_mappings", pd.DataFrame()),
        structures,
        ledger,
        schema_root,
        root,
    )
    pair_descriptors, residue_descriptors = _adapt_descriptors(admission, set(pairs["pair_id"]))
    benchmarks = _build_benchmark_instances(pairs, proteins)
    candidate = admission.get("candidate_pairs", discovery.get("candidate_pairs", pd.DataFrame()))
    attrition = _build_candidate_attrition(candidate)
    admitted_ids = set(pairs["pair_id"].astype(str))
    if not attrition.empty:
        attrition["terminal_status"] = attrition["terminal_status"].astype(str)
        duplicate_mask = attrition["terminal_status"].eq("ADMITTED") & ~attrition["pair_id"].astype(str).isin(admitted_ids)
        attrition.loc[duplicate_mask, "terminal_status"] = "EXCLUDED"
        attrition.loc[duplicate_mask, "terminal_reason"] = attrition.loc[duplicate_mask, "terminal_reason"].fillna("duplicate_semantic_pair")
        pure_mask = attrition.get("state_contrast_category", pd.Series(index=attrition.index)).eq("PURE_LIGAND_STATE_ONLY")
        attrition.loc[pure_mask & attrition["terminal_status"].isin(["PRIMARY", "ALTERNATIVE", "ADMITTED"]), "terminal_status"] = "EXCLUDED"
        attrition.loc[pure_mask, "terminal_reason"] = attrition.loc[pure_mask, "terminal_reason"].fillna("pure_ligand_state_only")
        needs_reason = attrition["terminal_status"].isin(["EXCLUDED", "UNRESOLVED"])
        attrition.loc[needs_reason, "terminal_reason"] = attrition.loc[needs_reason, "terminal_reason"].fillna("other_explicit_reason")
        attrition = close_candidate_attrition(
            attrition.rename(columns={"terminal_status": "final_pair_state", "terminal_reason": "admission_reasons"})
        )
    family_rows = []
    for family in ("OPEN_CLOSED", "INWARD_OUTWARD", "ACTIVE_INACTIVE", "RESTING_ACTIVATED", "PRE_POST", "OTHER_EXPLICIT"):
        candidate_family = attrition.loc[attrition.get("state_family", pd.Series(index=attrition.index)).eq(family)]
        for terminal in ("PRIMARY", "ALTERNATIVE", "EXCLUDED", "UNRESOLVED"):
            family_rows.append({"state_family": family, "terminal_status": terminal, "candidate_count": int(candidate_family.get("terminal_status", pd.Series(dtype=str)).eq(terminal).sum())})
    family_summary = pd.DataFrame(family_rows)
    cluster_output = clusters.copy()
    if "cluster_status" in cluster_output:
        cluster_output["cluster_status"] = cluster_output["cluster_status"].astype(str).str.replace("FROZEN_ARM_B_REUSE", "FROZEN_REUSE", regex=False)
    tables = {
        "proteins": proteins,
        "structures": canonical_structures,
        "condition_pairs": pairs,
        "residue_mappings": mappings,
        "benchmark_instances": benchmarks,
        "pair_structural_descriptors": pair_descriptors,
        "residue_structural_descriptors": residue_descriptors,
        "candidate_attrition": attrition,
        "asset_ledger": _normalize_asset_ledger(ledger),
        "cluster_assignments": cluster_output,
        "state_family_summary": family_summary,
        "state_annotations": annotations,
    }
    validate_canonical_tables(tables, schema_root=schema_root)
    return tables


def validate_canonical_tables(tables: dict[str, pd.DataFrame], *, schema_root: Path | None = None) -> None:
    """Validate keys, references, semantic ownership, and missingness fields."""

    schema_root = Path(schema_root or (_repository_root() / "schemas")).resolve()
    for name in _CORE_TABLES:
        frame = tables.get(name)
        if frame is None:
            raise FunctionalStateCanonicalError(f"missing canonical table: {name}")
        _validate_schema_frame(frame, name, schema_root)
    for name in _ANNOTATION_TABLES:
        frame = tables.get(name)
        if frame is None:
            raise FunctionalStateCanonicalError(f"missing canonical annotation table: {name}")
        _validate_schema_frame(frame, name, schema_root)
    proteins = tables["proteins"]
    if proteins["protein_id"].duplicated().any() or proteins["canonical_length"].le(0).any() or proteins["canonical_length"].ne(proteins["canonical_sequence"].str.len()).any():
        raise FunctionalStateCanonicalError("protein identity or canonical length invariant failed")
    pairs = tables["condition_pairs"]
    if pairs["pair_id"].duplicated().any() or not pairs["pair_role"].isin(["PRIMARY", "ALTERNATIVE"]).all():
        raise FunctionalStateCanonicalError("condition pair identity/role invariant failed")
    def _coherent_state(row: pd.Series) -> bool:
        semantics = str(row["arm"])
        family = row["state_family"]
        labels = {row["condition_1_label"], row["condition_2_label"]}
        if semantics != SEMANTICS:
            return len(labels) == 2 and all(isinstance(label, str) and label for label in labels)
        if family == "OTHER_EXPLICIT":
            return len(labels) == 2 and all(isinstance(label, str) and label for label in labels)
        return family in _STATE_LABELS and labels <= _STATE_LABELS[family] and len(labels) == 2

    if not pairs.apply(_coherent_state, axis=1).all():
        raise FunctionalStateCanonicalError("state family/label coherence invariant failed")
    try:
        for row in pairs.to_dict(orient="records"):
            validate_pair_orientation(row)
    except ValueError as exc:
        raise FunctionalStateCanonicalError(str(exc)) from exc
    if not pairs["protein_id"].isin(set(proteins["protein_id"])).all():
        raise FunctionalStateCanonicalError("condition pair protein foreign key failed")
    structures = tables["structures"]
    if structures["structure_id"].duplicated().any() or not structures["protein_id"].isin(set(proteins["protein_id"])).all():
        raise FunctionalStateCanonicalError("structure identity/foreign key invariant failed")
    if structures["structure_id"].eq(structures["source_structure_id"]).any():
        raise FunctionalStateCanonicalError("structure/source identity must remain distinct")
    if structures.loc[
        structures["arm"].eq(SEMANTICS), "source_type"
    ].ne("PDB").any():
        raise FunctionalStateCanonicalError("functional-state structures must be PDB sources")
    structure_ids = set(structures["structure_id"])
    if not pairs["condition_1_structure_id"].isin(structure_ids).all() or not pairs["condition_2_structure_id"].isin(structure_ids).all():
        raise FunctionalStateCanonicalError("condition pair structure foreign key failed")
    structure_lookup = structures.set_index("structure_id")["condition_label"].to_dict()
    for row in pairs.to_dict(orient="records"):
        if structure_lookup.get(row["condition_1_structure_id"]) != row["condition_1_label"] or structure_lookup.get(row["condition_2_structure_id"]) != row["condition_2_label"]:
            raise FunctionalStateCanonicalError("condition pair labels do not match actual structures")
    mapping = tables["residue_mappings"]
    if mapping.duplicated(["pair_id", "canonical_position"]).any() or not mapping["pair_id"].isin(set(pairs["pair_id"])).all():
        raise FunctionalStateCanonicalError("residue mapping key/foreign key invariant failed")
    if not mapping["protein_id"].isin(set(proteins["protein_id"])).all():
        raise FunctionalStateCanonicalError("residue mapping protein foreign key failed")
    if not set(mapping.columns).intersection(_FORBIDDEN_RESIDUE_FIELDS) == set():
        raise FunctionalStateCanonicalError("residue mapping contains descriptor/model fields")
    _validate_mapping_semantics(mapping, pairs)
    benchmark = tables["benchmark_instances"]
    if benchmark["instance_id"].duplicated().any() or not benchmark["pair_id"].isin(set(pairs["pair_id"])).all():
        raise FunctionalStateCanonicalError("benchmark instance key/foreign key invariant failed")
    if not benchmark["protein_id"].isin(set(proteins["protein_id"])).all():
        raise FunctionalStateCanonicalError("benchmark instance protein foreign key failed")
    attrition = tables.get("candidate_attrition")
    if attrition is not None and "terminal_status" in attrition and not attrition["terminal_status"].isin(["PRIMARY", "ALTERNATIVE", "EXCLUDED", "UNRESOLVED"]).all():
        raise FunctionalStateCanonicalError("candidate attrition has non-terminal status")
    if attrition is not None and "terminal_reason" in attrition:
        required_reason = attrition["terminal_status"].isin(["EXCLUDED", "UNRESOLVED"])
        if attrition.loc[required_reason, "terminal_reason"].isna().any():
            raise FunctionalStateCanonicalError("excluded/unresolved candidate lacks attrition reason")


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    payload_frame = frame.copy(deep=False)
    payload_frame.attrs = {}
    payload = payload_frame.to_parquet(index=False)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() and sha256_file(path) != digest:
        raise FunctionalStateCanonicalError(f"immutable conflict: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_new_bytes(path, payload)
    return sha256_file(path)


def _write_bytes_immutable(path: Path, encoded: bytes) -> None:
    if path.exists():
        if path.read_bytes() != encoded:
            raise FunctionalStateCanonicalError(f"immutable conflict: {path}")
        return
    atomic_write_new_bytes(path, encoded)


def _write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_bytes_immutable(path, encoded)


def _merge_existing_canonical_frames(
    frames: dict[str, pd.DataFrame],
    destinations: dict[str, Path],
    registry: SchemaRegistry,
) -> dict[str, pd.DataFrame]:
    """Merge new semantic rows with existing categories by frozen primary key."""

    merged: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        ordered = validate_frame(frame, name, registry)
        destination = destinations[name]
        if destination.is_file():
            existing = validate_frame(pd.read_parquet(destination), name, registry)
            ordered = pd.concat([existing, ordered], ignore_index=True)
        primary_key = list(registry.get(name).primary_key)
        if not primary_key:
            raise FunctionalStateCanonicalError(f"canonical table lacks a primary key: {name}")
        duplicate_mask = ordered.duplicated(primary_key, keep=False)
        for _, group in ordered.loc[duplicate_mask].groupby(primary_key, dropna=False, sort=False):
            if len(group.drop_duplicates()) > 1:
                raise FunctionalStateCanonicalError(
                    f"conflicting canonical rows for {name} primary key {tuple(group.iloc[0][primary_key])}"
                )
        ordered = ordered.drop_duplicates(primary_key, keep="first")
        if not ordered.empty:
            ordered = ordered.sort_values(primary_key, kind="mergesort")
        merged[name] = validate_frame(ordered.reset_index(drop=True), name, registry)
    return merged


def _load_admission(root: Path) -> dict[str, pd.DataFrame]:
    names = [
        "candidate_pairs", "primary_pairs", "alternative_pairs", "structures", "state_annotations",
        "asset_ledger", "residue_mappings", "pair_structural_descriptors",
        "residue_structural_descriptors", "cluster_assignments",
    ]
    result: dict[str, pd.DataFrame] = {}
    for name in names:
        path = root / f"{name}.parquet"
        if path.is_file():
            result["annotations" if name == "state_annotations" else name] = pd.read_parquet(path)
    required = {
        "candidate_pairs",
        "structures",
        "asset_ledger",
        "residue_mappings",
        "pair_structural_descriptors",
        "residue_structural_descriptors",
    }
    missing = sorted(required - set(result))
    if missing:
        raise FunctionalStateCanonicalError(f"admission output missing: {missing}")
    return result


def build_canonical_from_admission(config: FunctionalStateCanonicalConfig) -> dict[str, pd.DataFrame]:
    """Load completed admission artifacts and project them into canonical tables."""

    if any(config.admission_root.glob("*.part")):
        raise FunctionalStateCanonicalError("admission output still contains partial files")
    completion_path = config.audit_root / "construction_completion.json"
    if not completion_path.is_file():
        raise FunctionalStateCanonicalError(
            "construction completion record is missing; wait for admission/materialization completion"
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise FunctionalStateCanonicalError("construction completion record is unreadable") from exc
    if completion.get("completion_status") != "SUCCESS":
        raise FunctionalStateCanonicalError("construction completion does not establish success")
    if completion.get("validation_status") != "PASSED":
        raise FunctionalStateCanonicalError("construction completion validation did not pass")
    counts = completion.get("candidate_counts", {})
    if counts.get("candidate_pairs") != sum(
        counts.get(key, 0)
        for key in ("primary_pairs", "alternative_pairs", "excluded_pairs", "unresolved_pairs")
    ):
        raise FunctionalStateCanonicalError("admission attrition closure is incomplete")
    admission = _load_admission(config.admission_root)
    discovery = {
        name: pd.read_parquet(config.discovery_root / f"{name}.parquet")
        for name in ("candidate_pairs", "structures", "state_annotations")
        if (config.discovery_root / f"{name}.parquet").is_file()
    }
    return canonicalize_admission_tables(
        admission,
        discovery,
        project_root=config.project_root,
    )


def input_artifact_hashes(config: FunctionalStateCanonicalConfig) -> dict[str, str]:
    """Return portable source paths and hashes used by canonical staging."""

    paths = sorted(
        [*config.discovery_root.glob("*.parquet"), config.discovery_root / "summary.json", config.discovery_root / "manifest.json"]
        + list(config.admission_root.glob("*.parquet"))
        + [config.admission_root / "summary.json", config.admission_root / "manifest.json", config.admission_root / "report.md"],
        key=lambda path: path.as_posix(),
    )
    result: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            result[_portable(config.project_root, path) or path.name] = sha256_file(path)
    return result


def materialize_canonical_tables(
    tables: dict[str, pd.DataFrame],
    config: FunctionalStateCanonicalConfig,
    *,
    input_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write schema-validated Core/annotation fragments and audit evidence.

    ``config.output_root`` is the repository ``benchmark`` root.  Only the
    model-independent Core and structural annotation fragments are written
    beneath it.  Attrition, asset, and construction evidence remain in the
    task-local audit workspace and do not imply a complete benchmark release.
    """

    validate_canonical_tables(tables, schema_root=config.project_root / "schemas")
    output = config.output_root
    audit = config.audit_root or (config.project_root / "experiments/interventions/functional_states/construction")
    output_rel = _portable(config.project_root, output) or output.name
    forbidden_path_tokens = ("v1", "v2", "stage", "phase", "final", "latest", "new", "test", "arm")
    if any(token in output_rel.lower().split("/") for token in forbidden_path_tokens):
        raise FunctionalStateCanonicalError(f"non-semantic canonical output path: {output_rel}")
    audit_rel = _portable(config.project_root, audit) or audit.name
    if any(token in audit_rel.lower().split("/") for token in forbidden_path_tokens):
        raise FunctionalStateCanonicalError(f"non-semantic audit path: {audit_rel}")
    core_root = output / "core"
    annotation_root = output / "annotations"
    output.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    table_paths = {
        **{name: core_root / f"{name}.parquet" for name in _CORE_TABLES},
        **{name: annotation_root / f"{name}.parquet" for name in _ANNOTATION_TABLES},
    }
    registry = SchemaRegistry(config.project_root / "schemas")
    core_frames = {name: tables[name] for name in table_paths if name in _CORE_TABLES and name in tables}
    core_destinations = {name: table_paths[name] for name in core_frames}
    annotation_frames = {name: tables[name] for name in table_paths if name in _ANNOTATION_TABLES and name in tables}
    annotation_destinations = {name: table_paths[name] for name in annotation_frames}
    core_frames = _merge_existing_canonical_frames(core_frames, core_destinations, registry)
    annotation_frames = _merge_existing_canonical_frames(
        annotation_frames, annotation_destinations, registry
    )
    merged_tables = dict(tables)
    merged_tables.update(core_frames)
    merged_tables.update(annotation_frames)
    validate_canonical_tables(merged_tables, schema_root=config.project_root / "schemas")
    for bundle, destinations in ((core_frames, core_destinations), (annotation_frames, annotation_destinations)):
        if not bundle:
            continue
        digests = write_parquet_bundle_transactional(
            bundle,
            destinations,
            registry,
            replace_existing=True,
        )
        for name, digest in digests.items():
            logical_path = _portable(config.project_root, destinations[name]) or destinations[name].name
            hashes[logical_path] = digest
            row_counts[logical_path] = len(bundle[name])
    for name in _AUDIT_TABLES:
        if name not in tables:
            continue
        path = audit / f"{name}.parquet"
        logical_path = _portable(config.project_root, path) or path.name
        hashes[logical_path] = _write_table(tables[name], path)
        row_counts[logical_path] = len(tables[name])
    discovery_summary: dict[str, Any] = {}
    summary_path_upstream = config.admission_root / "summary.json"
    if summary_path_upstream.is_file():
        try:
            discovery_summary = json.loads(summary_path_upstream.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            discovery_summary = {}
    attrition = tables["candidate_attrition"]
    unresolved_items = (
        discovery_summary.get("discovery_query_status", {}).get("failures")
        or discovery_summary.get("query_failures")
        or []
    )
    attrition_reasons = attrition.get("terminal_reason", pd.Series(dtype=str)).fillna("unknown").astype(str)
    descriptor = tables["pair_structural_descriptors"]
    cluster = tables["cluster_assignments"]
    cluster_status = cluster.get("cluster_status", pd.Series(dtype=str)).astype(str) if not cluster.empty else pd.Series(dtype=str)
    family_breakdown = {}
    for family in ("OPEN_CLOSED", "INWARD_OUTWARD", "ACTIVE_INACTIVE", "RESTING_ACTIVATED", "PRE_POST"):
        candidate_family = attrition.loc[attrition.get("state_family", pd.Series(index=attrition.index)).eq(family)]
        pair_family = tables["condition_pairs"].loc[tables["condition_pairs"]["state_family"].eq(family)]
        family_breakdown[family] = {
            "candidates": len(candidate_family),
            "primary": int(pair_family["pair_role"].eq("PRIMARY").sum()),
            "alternative": int(pair_family["pair_role"].eq("ALTERNATIVE").sum()),
            "excluded": int(candidate_family.get("terminal_status", pd.Series(dtype=str)).eq("EXCLUDED").sum()),
            "unresolved": int(candidate_family.get("terminal_status", pd.Series(dtype=str)).eq("UNRESOLVED").sum()),
            "unique_primary_proteins": int(pair_family.loc[pair_family["pair_role"].eq("PRIMARY"), "protein_id"].nunique()),
        }
    summary = {
        "verdict": "LIMITED" if unresolved_items else "READY",
        "discovery_completion": "UPSTREAM_DISCOVERY_PRESERVED",
        "pre_post_retrieval_status": discovery_summary.get("discovery_query_status", {}).get(
            "pre_post_retry", {}
        ).get("status", "UNRESOLVED" if unresolved_items else "COMPLETE"),
        "candidate_pairs": len(attrition),
        "admitted_pairs": len(tables["condition_pairs"]),
        "primary_pairs": int(tables["condition_pairs"]["pair_role"].eq("PRIMARY").sum()),
        "alternative_pairs": int(tables["condition_pairs"]["pair_role"].eq("ALTERNATIVE").sum()),
        "represented_proteins": len(tables["proteins"]),
        "pure_ligand_only_candidates": int(attrition.get("state_contrast_category", pd.Series(dtype=str)).eq("PURE_LIGAND_STATE_ONLY").sum()),
        "dominant_attrition_reasons": attrition_reasons.value_counts().head(10).to_dict(),
        "state_family_breakdown": family_breakdown,
        "ligand_context_counts": attrition.get("state_contrast_category", pd.Series(dtype=str)).value_counts().to_dict(),
        "state_ambiguity_count": int(tables.get("state_annotations", pd.DataFrame()).get("ambiguity_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "state_conflict_count": int(attrition_reasons.str.contains("conflicting_state_evidence", na=False).sum()),
        "evidence_tier_counts": pd.concat(
            [attrition.get("evidence_tier_a", pd.Series(dtype=str)), attrition.get("evidence_tier_b", pd.Series(dtype=str))],
            ignore_index=True,
        ).dropna().astype(str).value_counts().to_dict(),
        "asset_status_counts": tables["asset_ledger"].get("asset_status", pd.Series(dtype=str)).value_counts().to_dict(),
        "structural_range": {
            field: (float(descriptor[field].dropna().median()) if field in descriptor and descriptor[field].notna().any() else None)
            for field in ("aligned_ca_rmsd", "median_residue_displacement", "upper_tail_residue_displacement", "global_contact_change")
        },
        "structural_range_quantiles": {
            field: {
                "q10": float(descriptor[field].dropna().quantile(0.10)),
                "median": float(descriptor[field].dropna().median()),
                "q90": float(descriptor[field].dropna().quantile(0.90)),
            }
            for field in ("aligned_ca_rmsd", "median_residue_displacement", "upper_tail_residue_displacement", "global_contact_change")
            if field in descriptor and descriptor[field].notna().any()
        },
        "task_eligibility": {
            field: int(tables["benchmark_instances"][field].sum())
            for field in ("local_sensitivity_eligible", "generation_eligible", "compatibility_eligible", "multistate_eligible")
        },
        "identity_clusters": int(cluster["identity_cluster_id"].nunique()) if "identity_cluster_id" in cluster else 0,
        "identity_singleton_fraction": float(cluster_status.eq("UNRESOLVED_NO_MMSEQS_BINARY").mean()) if len(cluster_status) else 0.0,
        "largest_identity_cluster": int(cluster["identity_cluster_id"].value_counts().max()) if "identity_cluster_id" in cluster and not cluster.empty else 0,
        "unresolved_items": unresolved_items,
        "model_execution": "none",
        "structural_condition_semantics": SEMANTICS,
        "outcome_blind_admission": True,
        "unresolved_preserved": True,
    }
    summary_path = audit / "summary.json"
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing_summary = {}
        if existing_summary != summary:
            raise FunctionalStateCanonicalError(f"immutable conflict: {summary_path}")
    else:
        atomic_write_json(summary_path, summary)
    hashes[_portable(config.project_root, summary_path) or summary_path.name] = sha256_file(summary_path)
    report = render_canonical_report(summary, tables)
    report_path = audit / "report.md"
    _write_bytes_immutable(report_path, report.encode("utf-8"))
    hashes[_portable(config.project_root, report_path) or report_path.name] = sha256_file(report_path)
    completion = {
        "completion_status": "SUCCESS",
        "validation_status": "PASSED",
        "construction_status": "FUNCTIONAL_STATE_CANONICAL_MATERIALIZATION_COMPLETE",
        "structural_condition_semantics": SEMANTICS,
        "canonical_roots": {
            "core": _portable(config.project_root, core_root),
            "annotations": _portable(config.project_root, annotation_root),
            "audit": _portable(config.project_root, audit),
        },
        "row_counts": row_counts,
        "candidate_counts": {
            key: int(summary[key])
            for key in (
                "candidate_pairs",
                "primary_pairs",
                "alternative_pairs",
                "excluded_pairs",
                "unresolved_pairs",
            )
            if key in summary
        },
        "state_family_breakdown": family_breakdown,
        "admission_policy": "outcome_blind_identity_mapping_comparability_state_evidence",
        "implementation_identity": {
            "admission": "dual_uq.construction.admission.evaluate_pair",
            "mapping": "dual_uq.construction.mapping.map_condition_pair",
            "comparability": "dual_uq.construction.comparability.compare_mapped_pair",
            "state_policy": "dual_uq.construction.functional_state",
            "descriptors": "dual_uq.construction.annotations",
            "canonical_staging": "dual_uq.dataset.functional_state_canonical",
        },
        "outcome_blind": True,
        "model_execution": "none",
        "frozen_upstream_untouched": True,
    }
    completion_path = audit / "construction_completion.json"
    if completion_path.is_file():
        try:
            existing_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise FunctionalStateCanonicalError(
                "construction completion record is unreadable"
            ) from exc
        if existing_completion.get("completion_status") != "SUCCESS" or existing_completion.get(
            "validation_status"
        ) != "PASSED":
            raise FunctionalStateCanonicalError(
                "construction completion record does not establish successful validation"
            )
        return existing_completion
    _write_json_immutable(completion_path, completion)
    return completion


def render_canonical_report(summary: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    pairs = tables["condition_pairs"]
    attrition = tables["candidate_attrition"]
    families = ["OPEN_CLOSED", "INWARD_OUTWARD", "ACTIVE_INACTIVE", "RESTING_ACTIVATED", "PRE_POST"]
    family_rows = []
    for family in families:
        candidate = attrition.loc[attrition.get("state_family", pd.Series(index=attrition.index)).eq(family)]
        admitted = pairs.loc[pairs["state_family"].eq(family)]
        asset_a = candidate.get("asset_status_a", pd.Series(index=candidate.index)).astype(str).eq("AVAILABLE")
        asset_b = candidate.get("asset_status_b", pd.Series(index=candidate.index)).astype(str).eq("AVAILABLE")
        family_rows.append(
            f"- {family}: candidates={len(candidate)}, asset_resolved={int((asset_a & asset_b).sum())}, mapping_valid={int(_flag_series(candidate.get('mapping_valid', pd.Series(index=candidate.index))).sum()) if not candidate.empty else 0}, primary={int(admitted['pair_role'].eq('PRIMARY').sum())}, alternatives={int(admitted['pair_role'].eq('ALTERNATIVE').sum())}, excluded={int(candidate.get('terminal_status', pd.Series(dtype=str)).eq('EXCLUDED').sum())}, unresolved={int(candidate.get('terminal_status', pd.Series(dtype=str)).eq('UNRESOLVED').sum())}."
        )
    state_categories = attrition.get("state_contrast_category", pd.Series(dtype=str)).value_counts().to_dict()
    descriptor = summary.get("structural_range", {})
    eligibility = summary.get("task_eligibility", {})
    open_closed = pairs.loc[pairs["state_family"].eq("OPEN_CLOSED")]
    pure_count = int(summary.get("pure_ligand_only_candidates", 0))
    pure_fraction = pure_count / len(attrition) if len(attrition) else 0.0
    return "\n".join(
        [
            "# Functional-state canonical staging",
            "",
            f"Verdict: **{summary['verdict']}**",
            "",
            "This staging set preserves discovery and outcome-blind admission; no model output or structural response determines membership.",
            "",
            "## Discovery completion",
            f"- PRE_POST retrieval status: {summary.get('pre_post_retrieval_status')}; candidate universe: {len(attrition)}; represented proteins: {len(tables['proteins'])}.",
            "- Unresolved query metadata is preserved; no successful discovery evidence was rewritten.",
            "",
            "## State provenance",
            "- Supported state families remain separate; evidence comes from discovery state annotations and candidate provenance.",
            f"- Ambiguous evidence rows={summary.get('state_ambiguity_count', 0)}; explicit conflict attrition rows={summary.get('state_conflict_count', 0)}.",
            f"- Evidence-tier counts across both conditions: {json.dumps(summary.get('evidence_tier_counts', {}), sort_keys=True)}.",
            f"- Ligand-context categories: {json.dumps({str(k): int(v) for k, v in state_categories.items()}, sort_keys=True)}.",
            "- Admission is sequence/mapping/construct/assembly/state-evidence based and outcome-blind.",
            "",
            "## Asset resolution and comparability",
            f"- Asset ledger rows: {len(tables['asset_ledger'])}; statuses={json.dumps(summary.get('asset_status_counts', {}), sort_keys=True)}; unresolved candidates remain in candidate_attrition.parquet.",
            f"- Canonical residue mapping rows: {len(tables['residue_mappings'])}; mapped gaps and coordinate non-observability remain explicit.",
            f"- Admitted pairs: {len(pairs)}; primary={int(pairs['pair_role'].eq('PRIMARY').sum())}; alternative={int(pairs['pair_role'].eq('ALTERNATIVE').sum())}.",
            "",
            "## Admission and state-family breakdown",
            *family_rows,
            f"- OPEN_CLOSED standalone check: {int(open_closed.loc[open_closed['pair_role'].eq('PRIMARY'), 'protein_id'].nunique())} primary proteins; diversity and dynamic range remain descriptive rather than outcome-selected.",
            f"- Dominant attrition reasons: {json.dumps(summary.get('dominant_attrition_reasons', {}), sort_keys=True)}.",
            "",
            "## Ligand-state overlap",
            f"- PURE_LIGAND_STATE_ONLY candidates: {pure_count} ({pure_fraction:.4f} of candidates); these are not primary canonical instances.",
            "",
            "## Structural range",
            f"- Median admitted descriptors: aligned C-alpha RMSD={descriptor.get('aligned_ca_rmsd')}, residue displacement={descriptor.get('median_residue_displacement')}, upper-tail displacement={descriptor.get('upper_tail_residue_displacement')}, contact change={descriptor.get('global_contact_change')}.",
            f"- Descriptor q10/median/q90: {json.dumps(summary.get('structural_range_quantiles', {}), sort_keys=True)}.",
            "- Descriptors are annotations computed after admission and never feed back into selection.",
            "",
            "## Redundancy",
            f"- Identity clusters={summary.get('identity_clusters', 0)}, singleton fraction={summary.get('identity_singleton_fraction', 0.0)}, largest cluster={summary.get('largest_identity_cluster', 0)}; unresolved extensions remain benchmark-only.",
            "",
            "## Canonical outputs",
            f"- proteins={len(tables['proteins'])}, structures={len(tables['structures'])}, condition_pairs={len(pairs)}, residue_mappings={len(tables['residue_mappings'])}, benchmark_instances={len(tables['benchmark_instances'])}.",
            f"- pair annotations={len(tables['pair_structural_descriptors'])}, residue annotations={len(tables['residue_structural_descriptors'])}.",
            "",
            "## Task eligibility",
            f"- local_sensitivity={eligibility.get('local_sensitivity_eligible', 0)}, generation={eligibility.get('generation_eligible', 0)}, compatibility={eligibility.get('compatibility_eligible', 0)}, multistate={eligibility.get('multistate_eligible', 0)}.",
            "- No model was run; eligibility is model-independent data readiness.",
            "",
            "## Schema validation",
            "- Primary keys, foreign keys, mapping-only fields, semantic category, naming, and provenance checks are enforced before materialization.",
            "- Every discovery candidate has an explicit terminal status in candidate_attrition.parquet.",
            "",
            "## Scientific interpretation",
            "- Functional_state is defensible as an experimental structural-condition cohort only for the admitted subset and extends beyond pure ligand-state variation where independent state evidence is present.",
            "- No causal, thermodynamic, fitness, function, or model-response interpretation is made.",
            "",
            "## Unresolved items",
            f"- {len(summary.get('unresolved_items', []))} upstream discovery query issue(s) remain explicit; this is why the staging verdict is {summary['verdict']}.",
            "",
            "## Next",
            "- CONTROLLED_PERTURBATION_CONSTRUCTION is not started by this task.",
            "",
            "- This is a canonical data-construction staging result, not evidence of causality, allostery, fitness, stability, or function.",
        ]
    )


__all__ = [
    "FunctionalStateCanonicalConfig",
    "FunctionalStateCanonicalError",
    "build_canonical_from_admission",
    "canonicalize_admission_tables",
    "input_artifact_hashes",
    "materialize_canonical_tables",
    "validate_canonical_tables",
]
