from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1_extended


def residue_name_to_one_letter(name: str) -> str:
    value = str(name).strip().upper()
    if len(value) == 1:
        return value
    return protein_letters_3to1_extended.get(value, "X")


def load_chain_ca_table(
    cif_path: str | Path,
    chain_id: str | None = None,
) -> pd.DataFrame:
    """Return one row per amino-acid residue with a CA atom."""
    path = Path(cif_path)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())

    if chain_id is None:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                f"{path} contains {len(chains)} chains; provide chain_id explicitly."
            )
        chain = chains[0]
    else:
        if chain_id not in model:
            available = [chain.id for chain in model.get_chains()]
            raise KeyError(f"Chain {chain_id!r} not in {path}. Available: {available}")
        chain = model[chain_id]

    rows: list[dict[str, Any]] = []
    for residue in chain:
        if not is_aa(residue, standard=False) or "CA" not in residue:
            continue
        _, seqnum, insertion_code = residue.id
        ca = residue["CA"]
        rows.append(
            {
                "chain_id": chain.id,
                "pdb_residue_number": str(seqnum)
                + (str(insertion_code).strip() if str(insertion_code).strip() else ""),
                "residue_number_int": int(seqnum),
                "insertion_code": str(insertion_code).strip(),
                "residue_name": residue.resname,
                "residue_one_letter": residue_name_to_one_letter(residue.resname),
                "x": float(ca.coord[0]),
                "y": float(ca.coord[1]),
                "z": float(ca.coord[2]),
                "bfactor": float(ca.bfactor),
                "occupancy": None if ca.occupancy is None else float(ca.occupancy),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No amino-acid CA atoms found in chain {chain_id!r} of {path}")
    return table


def coordinates_from_table(table: pd.DataFrame) -> np.ndarray:
    return table[["x", "y", "z"]].to_numpy(dtype=float)
