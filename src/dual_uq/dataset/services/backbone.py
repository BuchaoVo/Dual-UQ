"""Adapters for canonical-position masks using the frozen P1 backbone contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dual_uq.structure_io import residue_name_to_one_letter

from ..models import AFDBFragment, ResidueKey, group_residue_records
from ..stages.derivation import (
    P1ValidationError,
    _index_records,
    _load_atom_records,
    _optional_mapping_label,
    _pair_residues,
    _select_backbone,
)


def paired_common_backbone_positions(
    mapping: pd.DataFrame,
    *,
    canonical_sequence: str,
    pair_id: str,
    pdb_path: Any,
    afdb_path: Any,
    fragment: AFDBFragment,
) -> tuple[int, ...]:
    """Return mapped UniProt positions with valid paired N/CA/C/O backbones.

    Missing coordinates are maskable. Coordinate-bearing identity conflicts and
    ambiguous atom/residue identities remain structured P1 validation failures.
    """
    required = {
        "uniprot_position",
        "uniprot_residue_name",
        "auth_asym_id",
        "auth_seq_id",
        "insertion_code",
        "label_asym_id",
        "label_seq_id",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise P1ValidationError(
            "missing_mapping_column",
            "Common-mask mapping lacks frozen provenance columns",
            missing_columns=missing,
        )
    table = mapping.sort_values("uniprot_position", kind="mergesort").copy()
    positions = pd.to_numeric(table["uniprot_position"], errors="coerce")
    if (
        table.empty
        or positions.isna().any()
        or positions.mod(1).ne(0).any()
        or positions.le(0).any()
        or positions.duplicated().any()
    ):
        raise P1ValidationError(
            "invalid_uniprot_mapping", "Common-mask UniProt positions are invalid"
        )
    table["uniprot_residue_number"] = positions.astype(int)

    pdb_records = _load_atom_records(pdb_path, pair_id)
    afdb_records = _load_atom_records(afdb_path, fragment.model_entity_id)
    pdb_index = _index_records(pdb_records)
    afdb_groups = group_residue_records(afdb_records)
    afdb_index = {group.residue.key.auth_seq_id: group.atoms for group in afdb_groups}
    if set(afdb_index) != set(range(1, fragment.model_residue_count + 1)):
        raise P1ValidationError(
            "afdb_model_position_mismatch",
            "AFDB author positions must be exactly model-local 1..N",
        )

    keep: list[int] = []
    canonical_amino_acids: list[str] = []
    for row_index, row in enumerate(table.itertuples(index=False)):
        uniprot_position = int(row.uniprot_position)
        if uniprot_position > len(canonical_sequence):
            raise P1ValidationError(
                "uniprot_position_mismatch",
                "Mapped UniProt position exceeds canonical sequence length",
            )
        canonical_aa = canonical_sequence[uniprot_position - 1]
        mapping_aa = residue_name_to_one_letter(row.uniprot_residue_name)
        if mapping_aa != canonical_aa:
            raise P1ValidationError(
                "mapping_amino_acid_mismatch",
                "SIFTS mapping residue conflicts with canonical sequence",
                uniprot_position=uniprot_position,
            )

        auth_chain = _optional_mapping_label(row.auth_asym_id)
        auth_seq = pd.to_numeric(pd.Series([row.auth_seq_id]), errors="coerce").iloc[0]
        if auth_chain is None or pd.isna(auth_seq):
            continue
        pdb_key = ResidueKey(
            pair_id,
            auth_chain,
            int(auth_seq),
            "" if pd.isna(row.insertion_code) else str(row.insertion_code),
        )
        pdb_atoms = pdb_index.get(pdb_key)
        model_position = uniprot_position - fragment.uniprot_start + 1
        afdb_atoms = afdb_index.get(model_position)
        if pdb_atoms is None or afdb_atoms is None:
            continue
        try:
            _select_backbone(
                pdb_atoms, source="pdb", output_position=uniprot_position
            )
            _select_backbone(
                afdb_atoms, source="afdb", output_position=uniprot_position
            )
        except P1ValidationError as exc:
            if exc.code == "missing_backbone_atom":
                continue
            raise
        keep.append(row_index)
        canonical_amino_acids.append(canonical_aa)

    selected = table.iloc[keep].copy().reset_index(drop=True)
    selected["output_position"] = np.arange(1, len(selected) + 1)
    selected["canonical_aa"] = canonical_amino_acids
    paired = _pair_residues(
        mapping=selected,
        pdb_records=pdb_records,
        afdb_records=afdb_records,
        pair_id=pair_id,
        model_entity_id=fragment.model_entity_id,
        fragment_start=fragment.uniprot_start,
        fragment_end=fragment.uniprot_end,
    )
    return tuple(residue.uniprot_position for residue in paired)
