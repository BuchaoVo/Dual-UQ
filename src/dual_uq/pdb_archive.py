from __future__ import annotations

from pathlib import Path

from .net import download_file

RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def normalize_pdb_id(pdb_id: str) -> str:
    value = pdb_id.strip().lower()
    if len(value) == 4:
        return value
    if value.startswith("pdb_") and len(value) == 12:
        return value
    raise ValueError(
        "PDB ID must be a legacy 4-character ID or an extended 12-character ID."
    )


def fetch_pdb_mmcif(pdb_id: str, output_dir: str | Path) -> Path:
    normalized = normalize_pdb_id(pdb_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{normalized}.cif"
    url = RCSB_MMCIF_URL.format(pdb_id=normalized)
    return download_file(url, path)
