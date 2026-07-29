from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from .net import download_file
from .pdb_archive import normalize_pdb_id


SIFTS_XML_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz"


def fetch_sifts_xml(pdb_id: str, output_dir: str | Path) -> Path:
    normalized = normalize_pdb_id(pdb_id)
    if len(normalized) != 4:
        raise ValueError(
            "The current SIFTS XML downloader supports legacy 4-character PDB IDs only."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{normalized}.xml.gz"
    return download_file(SIFTS_XML_URL.format(pdb_id=normalized), path)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_crossref(
    residue: ET.Element,
    source: str,
) -> dict[str, str] | None:
    for child in residue:
        if _local_name(child.tag) != "crossRefDb":
            continue
        if child.attrib.get("dbSource") == source:
            return dict(child.attrib)
    return None


def parse_sifts_residue_mapping(
    xml_gz_path: str | Path,
    *,
    chain_id: str,
    uniprot_id: str,
) -> pd.DataFrame:
    chain_id = chain_id.strip()
    uniprot_id = uniprot_id.strip().upper()
    rows: list[dict[str, Any]] = []

    with gzip.open(xml_gz_path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local_name(element.tag) != "residue":
                continue

            pdb_ref = _first_crossref(element, "PDB")
            uniprot_ref = _first_crossref(element, "UniProt")
            if pdb_ref is None or uniprot_ref is None:
                element.clear()
                continue

            pdb_chain = pdb_ref.get("dbChainId")
            accession = str(uniprot_ref.get("dbAccessionId", "")).upper()
            if pdb_chain != chain_id or accession != uniprot_id:
                element.clear()
                continue

            rows.append(
                {
                    "pdb_chain_id": pdb_chain,
                    "pdb_residue_number": pdb_ref.get("dbResNum"),
                    "pdb_residue_name": pdb_ref.get("dbResName"),
                    "uniprot_id": accession,
                    "uniprot_residue_number": uniprot_ref.get("dbResNum"),
                    "uniprot_residue_name": uniprot_ref.get("dbResName"),
                }
            )
            element.clear()

    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise LookupError(
            f"No SIFTS residue mapping found for chain {chain_id} and UniProt {uniprot_id}."
        )

    mapping["uniprot_residue_number"] = pd.to_numeric(
        mapping["uniprot_residue_number"], errors="coerce"
    ).astype("Int64")
    mapping = mapping.dropna(subset=["uniprot_residue_number"]).copy()
    mapping = mapping.sort_values(
        ["uniprot_residue_number", "pdb_residue_number"]
    ).drop_duplicates(
        subset=["pdb_chain_id", "pdb_residue_number", "uniprot_residue_number"]
    )
    return mapping.reset_index(drop=True)
