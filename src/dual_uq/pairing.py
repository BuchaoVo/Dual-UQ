from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended

from .afdb import fetch_afdb_prediction
from .ids import stable_id
from .manifests import initialize_manifests
from .pdb_archive import fetch_pdb_mmcif, normalize_pdb_id
from .sifts import fetch_sifts_xml, parse_sifts_residue_mapping


def _normalise_residue_name(name: Any) -> str | None:
    if name is None:
        return None
    value = str(name).strip().upper()
    if len(value) == 1 and value.isalpha():
        return value
    return protein_letters_3to1_extended.get(value)


def _append_unique(
    path: Path,
    row: dict[str, Any],
    *,
    key_columns: list[str],
) -> None:
    existing = pd.read_parquet(path)
    new = pd.DataFrame([row])

    if existing.empty:
        combined = new
    else:
        mask = pd.Series(True, index=existing.index)
        for column in key_columns:
            mask &= existing[column].astype(str) == str(row[column])
        combined = pd.concat([existing.loc[~mask], new], ignore_index=True)

    combined.to_parquet(path, index=False)


def build_pair(
    *,
    project_root: str | Path,
    pdb_id: str,
    chain_id: str,
    uniprot_id: str,
    min_mapping_coverage: float = 0.90,
    min_sequence_identity: float = 0.95,
) -> dict[str, Any]:
    root = Path(project_root)
    pdb_id = normalize_pdb_id(pdb_id)
    chain_id = chain_id.strip()
    uniprot_id = uniprot_id.strip().upper()

    manifest_dir = root / "data/manifests"
    initialize_manifests(manifest_dir)

    pdb_path = fetch_pdb_mmcif(pdb_id, root / "data/raw/pdb")
    afdb = fetch_afdb_prediction(uniprot_id, root / "data/raw/afdb")
    sifts_path = fetch_sifts_xml(pdb_id, root / "data/raw/mappings")

    mapping = parse_sifts_residue_mapping(
        sifts_path,
        chain_id=chain_id,
        uniprot_id=uniprot_id,
    )

    pair_name = f"{pdb_id}_{chain_id}__{uniprot_id}"
    pair_dir = root / "data/processed/pairs" / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = pair_dir / "residue_mapping.parquet"
    mapping.to_parquet(mapping_path, index=False)

    prediction = afdb["prediction"]
    sequence = str(
        prediction.get("uniprotSequence")
        or prediction.get("sequence")
        or ""
    ).strip().upper()
    if not sequence:
        raise ValueError(f"No UniProt sequence present in AFDB metadata for {uniprot_id}")

    mapped_positions = mapping["uniprot_residue_number"].dropna().astype(int).unique()
    mapping_coverage = len(mapped_positions) / len(sequence)

    pdb_letters = mapping["pdb_residue_name"].map(_normalise_residue_name)
    uniprot_letters = mapping["uniprot_residue_name"].map(_normalise_residue_name)
    comparable = pdb_letters.notna() & uniprot_letters.notna()
    if comparable.any():
        sequence_identity = float(
            (pdb_letters[comparable].values == uniprot_letters[comparable].values).mean()
        )
    else:
        sequence_identity = float("nan")

    if (
        mapping_coverage >= min_mapping_coverage
        and sequence_identity >= min_sequence_identity
    ):
        quality_flag = "pass"
    elif mapping_coverage >= 0.70 and sequence_identity >= 0.90:
        quality_flag = "warn"
    else:
        quality_flag = "fail"

    protein_id = stable_id("protein", uniprot_id, pdb_id, chain_id)
    pdb_structure_id = stable_id("structure", protein_id, "pdb_clean")
    afdb_structure_id = stable_id("structure", protein_id, "afdb_native")

    protein_row = {
        "protein_id": protein_id,
        "uniprot_id": uniprot_id,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "sequence": sequence,
        "length": len(sequence),
        "sequence_cluster": None,
        "domain_count": None,
        "secondary_structure_class": None,
        "split": "a0_smoke",
        "mapping_coverage": mapping_coverage,
        "sequence_identity": sequence_identity,
        "quality_flag": quality_flag,
    }
    _append_unique(
        manifest_dir / "protein_manifest.parquet",
        protein_row,
        key_columns=["protein_id"],
    )

    pdb_structure_row = {
        "structure_id": pdb_structure_id,
        "protein_id": protein_id,
        "source": "pdb_clean",
        "parent_structure_id": None,
        "coordinate_path": str(pdb_path),
        "plddt_path": None,
        "pae_path": None,
        "perturbation_type": None,
        "perturbation_strength": None,
        "perturbed_residues": None,
        "random_seed": None,
        "mapping_quality": mapping_coverage,
        "structure_valid": quality_flag != "fail",
    }
    afdb_structure_row = {
        "structure_id": afdb_structure_id,
        "protein_id": protein_id,
        "source": "afdb_native",
        "parent_structure_id": None,
        "coordinate_path": afdb["model_path"],
        "plddt_path": afdb["plddt_path"],
        "pae_path": afdb["pae_path"],
        "perturbation_type": None,
        "perturbation_strength": None,
        "perturbed_residues": None,
        "random_seed": None,
        "mapping_quality": mapping_coverage,
        "structure_valid": True,
    }
    for row in (pdb_structure_row, afdb_structure_row):
        _append_unique(
            manifest_dir / "structure_manifest.parquet",
            row,
            key_columns=["structure_id"],
        )

    report = {
        "protein_id": protein_id,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "uniprot_length": len(sequence),
        "mapped_residue_count": int(len(mapped_positions)),
        "mapping_coverage": mapping_coverage,
        "sequence_identity": sequence_identity,
        "quality_flag": quality_flag,
        "pdb_path": str(pdb_path),
        "afdb_model_path": afdb["model_path"],
        "plddt_path": afdb["plddt_path"],
        "pae_path": afdb["pae_path"],
        "sifts_path": str(sifts_path),
        "mapping_path": str(mapping_path),
        "afdb_model_entity_id": prediction.get("modelEntityId"),
        "afdb_version": prediction.get("latestVersion"),
    }
    report_path = pair_dir / "pair_qc.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
