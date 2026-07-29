from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.afdb import AFDB_PREDICTION_API
from dual_uq.indexing_diagnostics import (
    diagnose_numbering_intersections,
    parse_mmcif_atom_site,
)
from dual_uq.net import request_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose PDB auth/label and AFDB fragment numbering intersections."
    )
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to reports/index9_numbering_diagnosis.json under the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir).expanduser().resolve()
    project_root = pair_dir.parents[3]
    pair_report_path = pair_dir / "pair_qc.json"
    mapping_path = pair_dir / "residue_mapping.parquet"
    report = json.loads(pair_report_path.read_text(encoding="utf-8"))
    metadata_path = Path(report["afdb_model_path"]).with_name("metadata.json")
    selected_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    prediction_url = AFDB_PREDICTION_API.format(accession=report["uniprot_id"])
    prediction_records = request_json(prediction_url)
    if not isinstance(prediction_records, list):
        raise TypeError("AlphaFold DB prediction API did not return a list.")

    diagnosis = diagnose_numbering_intersections(
        mapping=pd.read_parquet(mapping_path),
        pdb_atom_site=parse_mmcif_atom_site(report["pdb_path"]),
        afdb_atom_site=parse_mmcif_atom_site(report["afdb_model_path"]),
        prediction_records=prediction_records,
        selected_model_entity_id=str(report["afdb_model_entity_id"]),
        chain_id=str(report["chain_id"]),
    )
    diagnosis = {
        "pair_name": pair_dir.name,
        "inputs": {
            "pair_qc": str(pair_report_path),
            "residue_mapping": str(mapping_path),
            "pdb_mmcif": str(report["pdb_path"]),
            "afdb_metadata": str(metadata_path),
            "afdb_model_mmcif": str(report["afdb_model_path"]),
            "afdb_prediction_api": prediction_url,
        },
        "model_versions": {
            "selected_model_entity_id": report["afdb_model_entity_id"],
            "pair_qc_afdb_version": report.get("afdb_version"),
            "metadata_latest_version": selected_metadata.get("latestVersion"),
            "prediction_record_versions": {
                str(record.get("modelEntityId") or record.get("entryId")): record.get(
                    "latestVersion"
                )
                for record in prediction_records
            },
        },
        **diagnosis,
    }

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else project_root / "reports/index9_numbering_diagnosis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
