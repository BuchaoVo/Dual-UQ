from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa

from .schema import normalize_residue_mapping


class ResidueJoinError(ValueError):
    """Raised when residue namespaces cannot be joined uniquely."""


def residue_name_to_one_letter(name: str) -> str:
    value = str(name).strip().upper()
    if len(value) == 1:
        return value
    return protein_letters_3to1_extended.get(value, "X")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _atom_site_column(data: dict[str, Any], name: str, size: int) -> list[Any]:
    values = _as_list(data.get(name, ["?"] * size))
    if len(values) != size:
        raise ValueError(f"mmCIF column {name} has {len(values)} rows; expected {size}.")
    return values


def _clean_text(value: Any) -> str | pd.NA:
    cleaned = str(value).strip()
    return pd.NA if cleaned in {"", ".", "?"} else cleaned


def load_atom_site_table(cif_path: str | Path) -> pd.DataFrame:
    """Read the atom_site loop without collapsing auth and label namespaces."""
    path = Path(cif_path)
    data = MMCIF2Dict(str(path))
    atom_names = _as_list(data.get("_atom_site.label_atom_id", []))
    if not atom_names:
        raise ValueError(f"No _atom_site rows found in {path}.")
    size = len(atom_names)

    table = pd.DataFrame(
        {
            "group_pdb": _atom_site_column(data, "_atom_site.group_PDB", size),
            "atom_name": atom_names,
            "alt_id": _atom_site_column(data, "_atom_site.label_alt_id", size),
            "residue_name": _atom_site_column(
                data, "_atom_site.label_comp_id", size
            ),
            "auth_asym_id": _atom_site_column(
                data, "_atom_site.auth_asym_id", size
            ),
            "label_asym_id": _atom_site_column(
                data, "_atom_site.label_asym_id", size
            ),
            "auth_seq_id": _atom_site_column(data, "_atom_site.auth_seq_id", size),
            "label_seq_id": _atom_site_column(
                data, "_atom_site.label_seq_id", size
            ),
            "insertion_code": _atom_site_column(
                data, "_atom_site.pdbx_PDB_ins_code", size
            ),
            "x": _atom_site_column(data, "_atom_site.Cartn_x", size),
            "y": _atom_site_column(data, "_atom_site.Cartn_y", size),
            "z": _atom_site_column(data, "_atom_site.Cartn_z", size),
            "occupancy": _atom_site_column(data, "_atom_site.occupancy", size),
            "bfactor": _atom_site_column(
                data, "_atom_site.B_iso_or_equiv", size
            ),
            "model_number": _atom_site_column(
                data, "_atom_site.pdbx_PDB_model_num", size
            ),
        }
    )
    for column in ("auth_asym_id", "label_asym_id"):
        table[column] = table[column].map(_clean_text)
    for column in ("auth_seq_id", "label_seq_id"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    table["insertion_code"] = (
        table["insertion_code"].map(_clean_text).fillna("").astype(str).str.upper()
    )
    for column in ("x", "y", "z", "occupancy", "bfactor"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["model_number"] = pd.to_numeric(
        table["model_number"], errors="coerce"
    ).fillna(1).astype(int)
    return table


def load_chain_ca_table(
    cif_path: str | Path,
    chain_id: str | None = None,
) -> pd.DataFrame:
    """Return one CA row per residue while preserving both mmCIF namespaces."""
    path = Path(cif_path)
    atoms = load_atom_site_table(path)
    atoms = atoms.loc[atoms["model_number"] == atoms["model_number"].min()].copy()

    auth_chains = set(atoms["auth_asym_id"].dropna().astype(str))
    label_chains = set(atoms["label_asym_id"].dropna().astype(str))
    if chain_id is None:
        candidates = auth_chains or label_chains
        if len(candidates) != 1:
            raise ValueError(
                f"{path} contains {len(candidates)} chains; provide chain_id explicitly."
            )
        chain_id = next(iter(candidates))
    if chain_id in auth_chains:
        atoms = atoms.loc[atoms["auth_asym_id"] == chain_id].copy()
    elif chain_id in label_chains:
        matching_auth = atoms.loc[
            atoms["label_asym_id"] == chain_id, "auth_asym_id"
        ].dropna().unique()
        if len(matching_auth) > 1:
            raise ResidueJoinError(
                f"Label chain {chain_id!r} maps to multiple author chains."
            )
        atoms = atoms.loc[atoms["label_asym_id"] == chain_id].copy()
    else:
        raise KeyError(
            f"Chain {chain_id!r} not in {path}. "
            f"Author chains: {sorted(auth_chains)}; label chains: {sorted(label_chains)}"
        )

    amino_acid_atoms = atoms.loc[
        atoms["residue_name"].map(lambda name: is_aa(str(name), standard=False))
    ].copy()
    namespace_columns = [
        "auth_asym_id",
        "auth_seq_id",
        "insertion_code",
        "label_asym_id",
        "label_seq_id",
    ]
    namespace_pairs = amino_acid_atoms[namespace_columns].dropna(
        subset=["auth_asym_id", "auth_seq_id", "label_asym_id", "label_seq_id"]
    ).drop_duplicates()
    auth_keys = ["auth_asym_id", "auth_seq_id", "insertion_code"]
    label_keys = ["label_asym_id", "label_seq_id"]
    if namespace_pairs.duplicated(auth_keys, keep=False).any() or namespace_pairs.duplicated(
        label_keys, keep=False
    ).any():
        raise ResidueJoinError(
            f"mmCIF atom rows contain inconsistent auth/label residue namespaces in {path}."
        )

    ca = amino_acid_atoms.loc[
        amino_acid_atoms["atom_name"].astype(str).str.strip() == "CA"
    ].copy()
    if ca.empty:
        raise ValueError(f"No amino-acid CA atoms found in chain {chain_id!r} of {path}")

    ca["_alt_rank"] = ca["alt_id"].map(
        lambda value: 0 if str(value).strip() in {".", "?", "", "A"} else 1
    )
    keys = [
        "auth_asym_id",
        "label_asym_id",
        "auth_seq_id",
        "label_seq_id",
        "insertion_code",
    ]
    ca = (
        ca.sort_values(["_alt_rank", "occupancy"], ascending=[True, False])
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )
    ca["residue_one_letter"] = ca["residue_name"].map(residue_name_to_one_letter)
    ca["chain_id"] = ca["auth_asym_id"]
    ca["residue_number_int"] = ca["auth_seq_id"]
    ca["pdb_residue_number"] = ca["auth_seq_id"].astype(str) + ca["insertion_code"]
    return ca.drop(columns=["_alt_rank"])


def _duplicate_count(table: pd.DataFrame, keys: list[str]) -> int:
    usable = table.dropna(subset=keys)
    duplicates = usable.loc[usable.duplicated(keys, keep=False), keys]
    return len(duplicates.drop_duplicates())


def join_residue_mapping_to_ca(
    mapping: pd.DataFrame,
    ca_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Join mapping rows to CA rows using unique auth keys, then label fallback."""
    normalized = normalize_residue_mapping(mapping).reset_index(drop=True)
    ca = normalize_residue_mapping(ca_table).reset_index(drop=True)
    normalized["_mapping_row_id"] = np.arange(len(normalized))

    auth_keys = ["auth_asym_id", "auth_seq_id", "insertion_code"]
    label_keys = ["label_asym_id", "label_seq_id"]
    ca_auth_duplicates = _duplicate_count(ca, auth_keys)
    ca_auth_keys = ca.dropna(subset=auth_keys)[auth_keys].drop_duplicates()
    joinable_mapping_auth = normalized.dropna(subset=auth_keys).merge(
        ca_auth_keys,
        on=auth_keys,
        how="inner",
        validate="many_to_one",
    )
    mapping_auth_duplicates = _duplicate_count(joinable_mapping_auth, auth_keys)
    if ca_auth_duplicates:
        raise ResidueJoinError(
            f"CA table contains duplicate author residue keys "
            f"({ca_auth_duplicates} keys)."
        )
    if mapping_auth_duplicates:
        raise ResidueJoinError(
            f"mapping contains duplicate author residue keys "
            f"({mapping_auth_duplicates} keys)."
        )

    # Old single-chain tables may lack chain aliases. Permit a sequence+insertion
    # auth join only when both sides have no chain IDs and the reduced key is unique.
    if (
        normalized["auth_asym_id"].isna().all()
        and ca["auth_asym_id"].isna().all()
    ):
        auth_keys = ["auth_seq_id", "insertion_code"]
        reduced_duplicates = _duplicate_count(ca, auth_keys)
        if reduced_duplicates:
            raise ResidueJoinError(
                "Chainless legacy CA table has duplicate author residue keys."
            )
        ca_auth_keys = ca.dropna(subset=auth_keys)[auth_keys].drop_duplicates()
        joinable_mapping_auth = normalized.dropna(subset=auth_keys).merge(
            ca_auth_keys,
            on=auth_keys,
            how="inner",
            validate="many_to_one",
        )
        mapping_reduced_duplicates = _duplicate_count(
            joinable_mapping_auth, auth_keys
        )
        if mapping_reduced_duplicates:
            raise ResidueJoinError(
                "Chainless legacy mapping contains duplicate author residue keys."
            )

    numbering_columns = {
        "auth_asym_id",
        "label_asym_id",
        "auth_seq_id",
        "label_seq_id",
        "insertion_code",
    }
    ca_payload = [
        column
        for column in ca.columns
        if column not in normalized.columns and column not in numbering_columns
    ]
    ca_payload.extend(column for column in ("x", "y", "z") if column in ca)
    ca_payload = list(dict.fromkeys(ca_payload))
    auth_extra = [
        column
        for column in ("label_asym_id", "label_seq_id")
        if column not in auth_keys
    ]
    auth_source = ca[auth_keys + auth_extra + ca_payload].copy()
    auth_source = auth_source.rename(
        columns={column: f"{column}_ca" for column in auth_extra}
    )
    auth_matches = joinable_mapping_auth.merge(
        auth_source,
        on=auth_keys,
        how="inner",
        validate="one_to_one",
    )
    auth_matches["residue_join_mode"] = "auth"
    matched_ids = set(auth_matches["_mapping_row_id"])

    unmatched = normalized.loc[
        ~normalized["_mapping_row_id"].isin(matched_ids)
    ].copy()
    if auth_keys == ["auth_seq_id", "insertion_code"]:
        author_available = unmatched["auth_seq_id"].notna()
    else:
        author_available = (
            unmatched["auth_asym_id"].notna() & unmatched["auth_seq_id"].notna()
        )
    label_candidates = unmatched.loc[~author_available].dropna(subset=label_keys)
    mapping_label_duplicates = _duplicate_count(label_candidates, label_keys)
    if mapping_label_duplicates:
        raise ResidueJoinError(
            f"mapping contains duplicate label fallback keys "
            f"({mapping_label_duplicates} keys)."
        )
    label_extra = [
        column
        for column in ("auth_asym_id", "auth_seq_id", "insertion_code")
        if column not in label_keys
    ]
    requested_label_keys = label_candidates[label_keys].drop_duplicates()
    label_source = ca.dropna(subset=label_keys).merge(
        requested_label_keys,
        on=label_keys,
        how="inner",
        validate="many_to_one",
    )[
        label_keys + label_extra + ca_payload
    ].rename(columns={column: f"{column}_ca" for column in label_extra})
    ca_label_duplicates = _duplicate_count(label_source, label_keys)
    if ca_label_duplicates:
        raise ResidueJoinError(
            f"CA table contains ambiguous requested label fallback keys "
            f"({ca_label_duplicates} keys)."
        )
    label_matches = label_candidates.merge(
        label_source,
        on=label_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ca"),
    )
    label_matches["residue_join_mode"] = "label_fallback"

    joined = pd.concat([auth_matches, label_matches], ignore_index=True)
    if not joined.empty:
        joined = joined.sort_values("_mapping_row_id").reset_index(drop=True)
        for column in (
            "auth_asym_id",
            "label_asym_id",
            "auth_seq_id",
            "label_seq_id",
            "insertion_code",
        ):
            ca_column = f"{column}_ca"
            if ca_column in joined:
                use_ca = joined[column].isna()
                if column == "insertion_code":
                    use_ca |= joined["residue_join_mode"].eq("label_fallback")
                joined[column] = joined[column].where(~use_ca, joined[ca_column])
                joined = joined.drop(columns=ca_column)
    diagnostics: dict[str, int | str] = {
        "auth_match_count": len(auth_matches),
        "label_fallback_match_count": len(label_matches),
        "unmatched_mapping_count": int(len(normalized) - len(joined)),
        "duplicate_key_count": 0,
        "residue_join_mode": (
            "none"
            if len(auth_matches) == 0 and len(label_matches) == 0
            else "label_fallback"
            if len(auth_matches) == 0
            else "auth"
            if len(label_matches) == 0
            else "auth_with_label_fallback"
        ),
    }
    return joined.drop(columns=["_mapping_row_id"], errors="ignore"), diagnostics


def coordinates_from_table(table: pd.DataFrame) -> np.ndarray:
    return table[["x", "y", "z"]].to_numpy(dtype=float)
