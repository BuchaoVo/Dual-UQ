"""Prepare or score the EvoEF2 physical robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dual_uq.evaluation.physical_robustness import (
    ESMFOLD_SAMPLE_INDICES,
    EvoEF2Protocol,
    manifest_payload,
    prepare_inputs,
    score_prepared_inputs,
    screen_prepared_inputs,
)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "screen", "score"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--evoef2", type=Path, default=Path("third_party/EvoEF2/EvoEF2"))
    parser.add_argument("--output", type=Path, default=Path("runs/physical_validation/evoef2/pilot"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--all-clean", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="select four IDs by stable hash")
    parser.add_argument("--protein-id", action="append", dest="protein_ids")
    parser.add_argument("--sample-index", action="append", type=int, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.phase == "prepare" and args.all_clean and (args.protein_ids or args.pilot):
        raise SystemExit("--all-clean and --protein-id are mutually exclusive")
    if args.phase == "prepare" and args.all_clean:
        import pandas as pd

        generation = pd.read_parquet(
            root / "experiments/dataset/analysis/generative_propagation/generated_sequences.parquet",
            columns=["protein_id"],
        )
        protein_ids = tuple(sorted(generation["protein_id"].astype(str).unique()))
        if len(protein_ids) != 68:
            raise SystemExit(f"clean generation cohort must contain 68 proteins, found {len(protein_ids)}")
    elif args.phase == "prepare":
        if args.protein_ids and args.pilot:
            raise SystemExit("--pilot and --protein-id are mutually exclusive")
        if args.pilot:
            generation = __import__("pandas").read_parquet(
                root / "experiments/dataset/analysis/generative_propagation/generated_sequences.parquet",
                columns=["protein_id"],
            )
            all_ids = tuple(sorted(generation["protein_id"].astype(str).unique()))
            protein_ids = tuple(sorted(all_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:4])
        else:
            protein_ids = tuple(args.protein_ids or ())
            if not protein_ids:
                raise SystemExit("specify --pilot, --all-clean, or --protein-id")
    if args.sample_index is not None:
        sample_indices = tuple(args.sample_index)
    else:
        sample_indices = ESMFOLD_SAMPLE_INDICES
    if args.phase == "prepare":
        manifest = prepare_inputs(
            project_root=root, protein_ids=protein_ids,
            sample_indices=sample_indices, output_root=output,
        )
        print(manifest)
        return 0
    protocol = EvoEF2Protocol(root / args.evoef2 if not args.evoef2.is_absolute() else args.evoef2)
    manifest = args.manifest.expanduser().resolve() if args.manifest is not None else output / "input_manifest.json"
    input_root = None if args.input_root is None else args.input_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "screen":
        screen = screen_prepared_inputs(protocol, manifest, input_root=input_root)
        screen_path = output / "evaluability.parquet"
        screen.to_parquet(screen_path, index=False)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        (output / "screen_manifest.json").write_text(
            json.dumps({
                "schema": "evoef2_physical_evaluability_screen_v1",
                "input_manifest": manifest.name,
                "cohort": payload.get("cohort"),
                "protein_ids": payload.get("protein_ids", []),
                "sample_indices": payload.get("sample_indices", []),
                "case_count": len(screen),
                "successful_cases": int((screen["status"] == "success").sum()),
                "failed_cases": int((screen["status"] != "success").sum()),
                "failed_proteins": sorted(screen.loc[screen["status"] != "success", "protein_id"].astype(str).unique()),
                "evaluator": "EvoEF2",
                "evaluator_revision": protocol.source_revision,
                "executable_sha256": protocol.executable_sha256,
            }, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(screen_path)
        return 0
    records = score_prepared_inputs(protocol, manifest, input_root=input_root)
    records_path = output / "energy_records.parquet"
    records.to_parquet(records_path, index=False)
    summary = manifest_payload(protocol, cohort=json.loads(manifest.read_text())["cohort"], row_count=len(records))
    summary["input_manifest"] = "input_manifest.json"
    summary["energy_records"] = records_path.name
    summary["energy_columns"] = sorted(records.columns.tolist())
    (output / "manifest.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(records_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
