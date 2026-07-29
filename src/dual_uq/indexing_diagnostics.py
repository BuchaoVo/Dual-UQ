from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_INTEGER_PATTERN = re.compile(r"^-?\d+")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_mmcif_atom_site(path: str | Path) -> pd.DataFrame:
    """Read the auth/label atom-site identifiers without conflating namespaces."""
    mmcif = MMCIF2Dict(str(path))
    fields = {
        "group_pdb": "_atom_site.group_PDB",
        "atom_name": "_atom_site.label_atom_id",
        "auth_asym_id": "_atom_site.auth_asym_id",
        "label_asym_id": "_atom_site.label_asym_id",
        "auth_seq_id": "_atom_site.auth_seq_id",
        "label_seq_id": "_atom_site.label_seq_id",
        "insertion_code": "_atom_site.pdbx_PDB_ins_code",
    }
    columns = {name: _as_list(mmcif.get(key)) for name, key in fields.items()}
    lengths = {len(values) for values in columns.values()}
    if not lengths or 0 in lengths or len(lengths) != 1:
        detail = {name: len(values) for name, values in columns.items()}
        raise ValueError(f"Incomplete or inconsistent _atom_site identifiers in {path}: {detail}")
    return pd.DataFrame(columns)


def _clean_identifier(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return None
    if re.fullmatch(r"-?\d+\.0", text):
        return str(int(float(text)))
    return text


def _residue_key(sequence_id: Any, insertion_code: Any = None) -> str | None:
    sequence = _clean_identifier(sequence_id)
    if sequence is None:
        return None
    insertion = _clean_identifier(insertion_code)
    return sequence if insertion is None else f"{sequence}{insertion}"


def _integer_value(value: Any) -> int | None:
    cleaned = _clean_identifier(value)
    if cleaned is None:
        return None
    match = _INTEGER_PATTERN.match(cleaned)
    return int(match.group()) if match else None


def _set_summary(values: set[str] | set[int], *, preview_size: int = 10) -> dict[str, Any]:
    if not values:
        return {"count": 0, "range": None, "preview": []}
    integers = sorted(
        number for number in (_integer_value(value) for value in values) if number is not None
    )
    preview = sorted(values, key=lambda value: (_integer_value(value) is None, _integer_value(value), str(value)))
    return {
        "count": len(values),
        "range": [integers[0], integers[-1]] if integers else None,
        "preview": [str(value) for value in preview[:preview_size]],
    }


def _ca_residues(atom_site: pd.DataFrame, chain_id: str) -> pd.DataFrame:
    atoms = atom_site.copy()
    group = atoms["group_pdb"].astype(str).str.upper()
    atom_name = atoms["atom_name"].astype(str).str.strip().str.upper()
    auth_chain = atoms["auth_asym_id"].astype(str).str.strip()
    label_chain = atoms["label_asym_id"].astype(str).str.strip()
    mask = (group == "ATOM") & (atom_name == "CA")
    mask &= (auth_chain == chain_id) | (label_chain == chain_id)
    return atoms.loc[mask].copy()


def _fragment_interval(record: dict[str, Any], mapped_positions: set[int]) -> dict[str, Any]:
    start = record.get("uniprotStart", record.get("sequenceStart"))
    end = record.get("uniprotEnd", record.get("sequenceEnd"))
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        start_int = None
        end_int = None

    model_id = str(record.get("modelEntityId") or record.get("entryId") or "")
    mapped_start = min(mapped_positions) if mapped_positions else None
    mapped_end = max(mapped_positions) if mapped_positions else None
    if start_int is None or end_int is None or mapped_start is None or mapped_end is None:
        overlap_count = 0
        fully_covers = False
    else:
        overlap_count = len(
            mapped_positions.intersection(range(start_int, end_int + 1))
        )
        fully_covers = start_int <= mapped_start and end_int >= mapped_end
    return {
        "model_entity_id": model_id,
        "entry_id": record.get("entryId"),
        "version": record.get("latestVersion"),
        "uniprot_start": start_int,
        "uniprot_end": end_int,
        "length": (
            end_int - start_int + 1
            if start_int is not None and end_int is not None
            else None
        ),
        "mapped_interval_overlap_count": overlap_count,
        "fully_covers_mapped_interval": fully_covers,
        "cif_url": record.get("cifUrl"),
        "is_complex": bool(record.get("isComplex", False)),
    }


def diagnose_numbering_intersections(
    *,
    mapping: pd.DataFrame,
    pdb_atom_site: pd.DataFrame,
    afdb_atom_site: pd.DataFrame,
    prediction_records: list[dict[str, Any]],
    selected_model_entity_id: str,
    chain_id: str,
) -> dict[str, Any]:
    """Diagnose numbering/fragment failure modes without changing coordinates."""
    sifts_pdb = {
        key
        for key in (
            _residue_key(value)
            for value in mapping["pdb_residue_number"]
        )
        if key is not None
    }
    sifts_uniprot = {
        number
        for number in (
            _integer_value(value)
            for value in mapping["uniprot_residue_number"]
        )
        if number is not None
    }

    pdb_ca = _ca_residues(pdb_atom_site, chain_id)
    pdb_auth = {
        key
        for key in (
            _residue_key(row.auth_seq_id, row.insertion_code)
            for row in pdb_ca.itertuples(index=False)
        )
        if key is not None
    }
    pdb_label = {
        key
        for key in (
            _residue_key(value)
            for value in pdb_ca["label_seq_id"]
        )
        if key is not None
    }

    afdb_ca = _ca_residues(afdb_atom_site, "A")
    afdb_local = {
        number
        for number in (
            _integer_value(value)
            for value in afdb_ca["auth_seq_id"]
        )
        if number is not None
    }

    fragments = [
        _fragment_interval(record, sifts_uniprot)
        for record in prediction_records
        if not record.get("isComplex", False)
    ]
    selected_fragment = next(
        (
            fragment
            for fragment in fragments
            if fragment["model_entity_id"] == selected_model_entity_id
        ),
        None,
    )
    full_cover_fragments = [
        fragment for fragment in fragments if fragment["fully_covers_mapped_interval"]
    ]

    projected_afdb: set[int] = set()
    if selected_fragment is not None and selected_fragment["uniprot_start"] is not None:
        start = int(selected_fragment["uniprot_start"])
        projected_afdb = {start + position - 1 for position in afdb_local}

    auth_overlap = len(sifts_pdb & pdb_auth)
    label_overlap = len(sifts_pdb & pdb_label)
    raw_afdb_overlap = len(sifts_uniprot & afdb_local)
    projected_afdb_overlap = len(sifts_uniprot & projected_afdb)

    evidence = [
        (
            f"SIFTS PDB keys intersect auth numbering at {auth_overlap}/"
            f"{len(sifts_pdb)} and label numbering at {label_overlap}/{len(sifts_pdb)}."
        ),
        (
            f"Mapped UniProt positions intersect selected AFDB local numbering at "
            f"{raw_afdb_overlap}/{len(sifts_uniprot)} and metadata-projected numbering "
            f"at {projected_afdb_overlap}/{len(sifts_uniprot)}."
        ),
        (
            f"{len(full_cover_fragments)} of {len(fragments)} AFDB prediction records "
            "fully cover the mapped UniProt interval."
        ),
    ]

    if auth_overlap < len(sifts_pdb) and label_overlap > auth_overlap:
        root_cause = "auth_label_numbering_mismatch"
        next_action = (
            "Use explicit auth/label chain and residue identifiers in the mapping join; "
            "preserve insertion codes and do not infer an offset."
        )
    elif not full_cover_fragments:
        root_cause = "unsupported_afdb_coverage"
        next_action = (
            "Mark this pair unsupported_afdb_fragment because no AFDB prediction record "
            "fully covers the mapped UniProt interval."
        )
    elif selected_fragment is None or not selected_fragment["fully_covers_mapped_interval"]:
        root_cause = "afdb_fragment_selection_mismatch"
        next_action = (
            "Select an AFDB prediction record that fully covers the mapped UniProt interval."
        )
    elif projected_afdb_overlap > raw_afdb_overlap:
        root_cause = "afdb_residue_offset_mismatch"
        next_action = (
            "Map AFDB local residue identifiers through the selected fragment metadata "
            "interval; do not hard-code an accession-specific offset."
        )
    else:
        root_cause = "no_numbering_or_fragment_mismatch"
        next_action = "Investigate sequence identity, alternate locations, and CA filtering."

    return {
        "numbering_sets": {
            "sifts_pdb_residue_keys": _set_summary(sifts_pdb),
            "sifts_uniprot": _set_summary(sifts_uniprot),
            "pdb_auth_residue_keys": _set_summary(pdb_auth),
            "pdb_label_residue_keys": _set_summary(pdb_label),
            "afdb_local_residue_numbers": _set_summary(afdb_local),
            "afdb_projected_uniprot_positions": _set_summary(projected_afdb),
        },
        "auth_label_intersections": {
            "sifts_pdb_vs_auth": auth_overlap,
            "sifts_pdb_vs_label": label_overlap,
            "pdb_auth_vs_label": len(pdb_auth & pdb_label),
        },
        "uniprot_afdb_intersections": {
            "mapped_uniprot_vs_afdb_local": raw_afdb_overlap,
            "mapped_uniprot_vs_selected_fragment_projected": projected_afdb_overlap,
        },
        "afdb_fragment_intervals": fragments,
        "selected_model_entity_id": selected_model_entity_id,
        "categorical_root_cause": root_cause,
        "supporting_evidence": evidence,
        "recommended_next_action": next_action,
    }
