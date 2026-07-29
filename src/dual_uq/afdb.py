from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .net import download_file, request_json


AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def _select_monomer_prediction(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise LookupError("AlphaFold DB returned no predictions.")

    monomers = [record for record in records if not record.get("isComplex", False)]
    candidates = monomers or records

    for record in candidates:
        entity_id = str(record.get("modelEntityId") or record.get("entryId") or "")
        if entity_id.endswith("-F1"):
            return record

    return candidates[0]


def get_afdb_prediction_metadata(accession: str) -> dict[str, Any]:
    accession = accession.strip().upper()
    records = request_json(AFDB_PREDICTION_API.format(accession=accession))
    if not isinstance(records, list):
        raise TypeError(f"Unexpected AlphaFold DB response for {accession}: expected a list")
    return _select_monomer_prediction(records)


def fetch_afdb_prediction(accession: str, output_dir: str | Path) -> dict[str, Any]:
    accession = accession.strip().upper()
    output = Path(output_dir) / accession
    output.mkdir(parents=True, exist_ok=True)

    prediction = get_afdb_prediction_metadata(accession)
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    required_urls = {
        "model_cif": prediction.get("cifUrl"),
        "plddt_json": prediction.get("plddtDocUrl"),
        "pae_json": prediction.get("paeDocUrl"),
    }
    missing = [name for name, url in required_urls.items() if not url]
    if missing:
        raise KeyError(
            f"AlphaFold DB metadata lacks required URL fields for {accession}: {missing}"
        )

    model_path = download_file(required_urls["model_cif"], output / "model.cif")
    plddt_path = download_file(required_urls["plddt_json"], output / "plddt.json")
    pae_path = download_file(required_urls["pae_json"], output / "pae.json")

    return {
        "accession": accession,
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
        "plddt_path": str(plddt_path),
        "pae_path": str(pae_path),
        "prediction": prediction,
    }
