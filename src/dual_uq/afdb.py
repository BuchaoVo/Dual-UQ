from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .net import download_file, request_json


AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


class UnsupportedAFDBFragment(LookupError):
    def __init__(
        self,
        mapped_interval: tuple[int, int],
        fragment_intervals: tuple[tuple[int, int], ...],
    ) -> None:
        self.mapped_interval = mapped_interval
        self.fragment_intervals = fragment_intervals
        super().__init__(
            "No AlphaFold DB prediction fully covers mapped UniProt interval "
            f"{mapped_interval[0]}-{mapped_interval[1]}; available fragments: "
            f"{fragment_intervals}"
        )


def prediction_interval(record: dict[str, Any]) -> tuple[int, int] | None:
    start = record.get("uniprotStart", record.get("sequenceStart"))
    end = record.get("uniprotEnd", record.get("sequenceEnd"))
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None
    if start_int < 1 or end_int < start_int:
        return None
    return start_int, end_int


def canonical_uniprot_length(records: list[dict[str, Any]]) -> int:
    interval_ends = [
        interval[1]
        for record in records
        if not record.get("isComplex", False)
        if (interval := prediction_interval(record)) is not None
    ]
    if interval_ends:
        return max(interval_ends)

    sequence_lengths = [
        len(str(record.get("uniprotSequence") or record.get("sequence") or ""))
        for record in records
        if not record.get("isComplex", False)
    ]
    length = max(sequence_lengths, default=0)
    if length == 0:
        raise ValueError("AlphaFold DB metadata contains no canonical length evidence.")
    return length


def prediction_fragment_length(record: dict[str, Any]) -> int:
    interval = prediction_interval(record)
    if interval is not None:
        return interval[1] - interval[0] + 1
    sequence = str(record.get("uniprotSequence") or record.get("sequence") or "")
    if not sequence:
        raise ValueError("AlphaFold DB prediction contains no fragment length evidence.")
    return len(sequence)


def select_prediction_for_interval(
    records: list[dict[str, Any]],
    mapped_interval: tuple[int, int],
) -> dict[str, Any]:
    mapped_start, mapped_end = (int(mapped_interval[0]), int(mapped_interval[1]))
    if mapped_start < 1 or mapped_end < mapped_start:
        raise ValueError(f"Invalid mapped UniProt interval: {mapped_interval}")

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fragment_intervals: list[tuple[int, int]] = []
    for record in records:
        if record.get("isComplex", False):
            continue
        interval = prediction_interval(record)
        if interval is None:
            continue
        start, end = interval
        fragment_intervals.append(interval)
        if start > mapped_start or end < mapped_end:
            continue
        redundancy = (mapped_start - start) + (end - mapped_end)
        model_id = str(record.get("modelEntityId") or record.get("entryId") or "")
        version = int(record.get("latestVersion") or 0)
        rank = (redundancy, start, end, model_id, -version)
        candidates.append((rank, record))

    if not candidates:
        raise UnsupportedAFDBFragment(
            (mapped_start, mapped_end),
            tuple(sorted(set(fragment_intervals))),
        )
    return min(candidates, key=lambda item: item[0])[1]


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


def get_afdb_prediction_records(accession: str) -> list[dict[str, Any]]:
    accession = accession.strip().upper()
    records = request_json(AFDB_PREDICTION_API.format(accession=accession))
    if not isinstance(records, list):
        raise TypeError(f"Unexpected AlphaFold DB response for {accession}: expected a list")
    return records


def get_afdb_prediction_metadata(
    accession: str,
    *,
    mapped_interval: tuple[int, int] | None = None,
) -> dict[str, Any]:
    records = get_afdb_prediction_records(accession)
    if mapped_interval is None:
        return _select_monomer_prediction(records)
    return select_prediction_for_interval(records, mapped_interval)


def fetch_afdb_prediction(
    accession: str,
    output_dir: str | Path,
    *,
    mapped_interval: tuple[int, int] | None = None,
) -> dict[str, Any]:
    accession = accession.strip().upper()
    records = get_afdb_prediction_records(accession)
    prediction = (
        _select_monomer_prediction(records)
        if mapped_interval is None
        else select_prediction_for_interval(records, mapped_interval)
    )
    model_id = str(
        prediction.get("modelEntityId") or prediction.get("entryId") or "unknown-model"
    )
    safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    output = Path(output_dir) / accession / safe_model_id
    output.mkdir(parents=True, exist_ok=True)

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
        "prediction_records": records,
        "canonical_uniprot_length": canonical_uniprot_length(records),
        "fragment_length": prediction_fragment_length(prediction),
        "fragment_interval": prediction_interval(prediction),
    }
