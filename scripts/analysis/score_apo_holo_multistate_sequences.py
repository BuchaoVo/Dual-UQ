"""Score true multi-state ProteinMPNN sequences on both Apo/Holo states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations
from dual_uq.evaluation.multi_state_baseline import compatibility_endpoints
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    load_authorized_proteinmpnn_adapter,
)
from scripts.analysis.score_apo_holo_multistate_baseline import (
    ESM_IF1_CHECKPOINT_SHA256,
    ESM_IF1_REVISION,
    _case_map,
    _read_cases,
    _score_esm_batch,
    _score_pnn,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cases-pickle", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--evaluator", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--implementation-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args(argv)


def _shards(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted((Path(root) / "shards").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "apo_holo_multistate_generation_v1":
            raise ValueError(f"unexpected multi-state shard schema: {path}")
        result[str(payload["protein_id"])] = payload
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.project_root).resolve()
    case_map = _case_map(_read_cases(Path(args.cases_pickle)))
    shards = _shards(Path(args.generation_root))
    proteins = sorted(set(case_map).intersection(shards))
    if args.evaluator == "ProteinMPNN":
        adapter = load_authorized_proteinmpnn_adapter(
            implementation_path=Path(args.implementation_path or project_root / "third_party/ProteinMPNN"),
            checkpoint_path=Path(args.checkpoint_path or project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
            device_name=args.device,
            backbone_noise=0.0,
        )
    else:
        checkpoint_path = args.checkpoint_path or os.environ.get(
            "ESM_IF1_CHECKPOINT_PATH"
        )
        if checkpoint_path is None:
            raise ValueError(
                "ESM-IF1 requires --checkpoint-path or ESM_IF1_CHECKPOINT_PATH"
            )
        adapter = load_esm_if1(
            Path(args.source_root or project_root / "third_party/esm"),
            Path(checkpoint_path),
            expected_revision=ESM_IF1_REVISION,
            expected_checkpoint_sha256=ESM_IF1_CHECKPOINT_SHA256,
            device=args.device,
        )
    rows: list[dict[str, object]] = []
    for index, protein in enumerate(proteins, start=1):
        states = case_map[protein]
        positions = states["APO"]["positions"]
        full_wt = states["APO"]["sequence"]
        if positions != states["HOLO"]["positions"] or full_wt != states["HOLO"]["sequence"]:
            raise ValueError(f"state axes differ for {protein}")
        projection_indices = np.arange(len(positions), dtype=np.int64)
        if args.evaluator == "ESM-IF1":
            finite = np.isfinite(states["APO"]["coordinates"]).all(axis=(1, 2)) & np.isfinite(states["HOLO"]["coordinates"]).all(axis=(1, 2))
            projection_indices = np.flatnonzero(finite)
        if len(projection_indices) < 3:
            continue
        wt = "".join(full_wt[i] for i in projection_indices)
        if args.evaluator == "ProteinMPNN":
            realization = build_decoding_realizations(protein, len(positions), count=1)[0]
            structures = {
                condition: ProteinMPNNStructureInput(
                    protein_id=protein,
                    backbone_condition="PDB" if condition == "APO" else "AFDB",
                    uniprot_positions=positions,
                    wt_sequence_projection=full_wt,
                    coordinates=states[condition]["coordinates"],
                    structure_sha256=states[condition]["structure_sha256"],
                )
                for condition in ("APO", "HOLO")
            }
        else:
            structures = {
                # The shared case cache retains ProteinMPNN's N/CA/C/O
                # projection.  ESM-IF1 consumes the same common positions but
                # its native coordinate contract is N/CA/C.
                condition: states[condition]["coordinates"][projection_indices, :3]
                for condition in ("APO", "HOLO")
            }
        records = sorted(shards[protein]["records"], key=lambda row: int(row["sample_index"]))
        sequences = tuple(
            (str(row["sequence"]) if args.evaluator == "ProteinMPNN" else "".join(str(row["sequence"])[i] for i in projection_indices))
            for row in records
        )
        if args.evaluator == "ProteinMPNN":
            score_vectors = {condition: _score_pnn(adapter, structure, sequences, realization, args.batch_size) for condition, structure in structures.items()}
            wt_scores = {condition: _score_pnn(adapter, structure, (full_wt,), realization, args.batch_size)[0] for condition, structure in structures.items()}
        else:
            score_vectors = {condition: _score_esm_batch(adapter, sequences, coords, args.batch_size) for condition, coords in structures.items()}
            wt_scores = {condition: _score_esm_batch(adapter, (wt,), coords, args.batch_size)[0] for condition, coords in structures.items()}
        for row_index, record in enumerate(records):
            endpoint = compatibility_endpoints(
                apo_sequence_score=score_vectors["APO"][row_index],
                holo_sequence_score=score_vectors["HOLO"][row_index],
                apo_wt_score=wt_scores["APO"], holo_wt_score=wt_scores["HOLO"],
            )
            rows.append({
                "protein_id": protein, "pair_id": states["APO"]["pair_id"],
                "evaluator": args.evaluator, "source_state": "MULTI",
                "sample_index": int(record["sample_index"]),
                "sequence_hash": str(record["sequence_hash"]),
                "score_apo": score_vectors["APO"][row_index], "score_holo": score_vectors["HOLO"][row_index],
                "wt_score_apo": wt_scores["APO"], "wt_score_holo": wt_scores["HOLO"],
                "c_apo": endpoint.c_apo, "c_holo": endpoint.c_holo,
                "mean_compat": endpoint.mean_compat, "worst_compat": endpoint.worst_compat,
                "state_gap": endpoint.state_gap,
            })
        if index % 10 == 0:
            print(f"{args.evaluator} multi-state {index}/{len(proteins)}", flush=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["protein_id", "sample_index"], kind="mergesort")
    frame.to_parquet(output, index=False)
    manifest = {"schema_version": "multi_state_baseline_scores_v1", "evaluator": args.evaluator, "protein_count": int(frame.protein_id.nunique()), "row_count": int(len(frame)), "source_state": "MULTI"}
    atomic_write_new_bytes(output.with_suffix(".json"), (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


if __name__ == "__main__":
    run(parse_args())
