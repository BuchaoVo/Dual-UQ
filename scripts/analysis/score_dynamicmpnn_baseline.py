"""Score DynamicMPNN sequences against both APO and HOLO evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.evaluation.multi_state_baseline import compatibility_endpoints
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput, load_authorized_proteinmpnn_adapter
from scripts.analysis.score_apo_holo_multistate_baseline import (
    ESM_IF1_CHECKPOINT_SHA256,
    ESM_IF1_REVISION,
    _case_map,
    _read_cases,
    _score_esm_batch,
    _score_pnn,
)
from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--cases-pickle", type=Path, required=True)
    parser.add_argument("--evaluator", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--implementation-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def _project_sequence(sequence: str, source_positions: tuple[int, ...], target_positions: tuple[int, ...]) -> str:
    index = {position: offset for offset, position in enumerate(source_positions)}
    if any(position not in index for position in target_positions):
        raise ValueError("target evaluator positions are not covered by DynamicMPNN input")
    return "".join(sequence[index[position]] for position in target_positions)


def _dynamic_shards(root: Path) -> dict[str, dict]:
    """Read worker-nested or merged flat DynamicMPNN shards."""
    paths = set((Path(root) / "shards").glob("*.json"))
    paths.update((Path(root) / "shards").glob("*/*.json"))
    result: dict[str, dict] = {}
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "apo_holo_multistate_generation_v1":
            raise ValueError(f"unexpected DynamicMPNN shard schema: {path}")
        protein_id = str(payload["protein_id"])
        if protein_id in result and result[protein_id] != payload:
            raise ValueError(f"conflicting DynamicMPNN shards for {protein_id}")
        result[protein_id] = payload
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.project_root).resolve()
    cases = _read_cases(Path(args.cases_pickle))
    case_map = _case_map(cases)
    shards = _dynamic_shards(Path(args.generation_root))
    proteins = sorted(set(case_map).intersection(shards))
    proteins = proteins[args.offset :]
    if args.limit is not None:
        proteins = proteins[: args.limit]
    if not proteins:
        raise ValueError("no DynamicMPNN/case intersection")
    if args.evaluator == "ProteinMPNN":
        adapter = load_authorized_proteinmpnn_adapter(
            implementation_path=Path(args.implementation_path or root / "third_party/ProteinMPNN"),
            checkpoint_path=Path(args.checkpoint_path or root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
            device_name=args.device,
            backbone_noise=0.0,
        )
    else:
        adapter = load_esm_if1(
            Path(args.source_root or root / "third_party/esm"),
            Path(args.checkpoint_path or root / "third_party/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt"),
            expected_revision=ESM_IF1_REVISION,
            expected_checkpoint_sha256=ESM_IF1_CHECKPOINT_SHA256,
            device=args.device,
        )

    rows: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []
    for index, protein in enumerate(proteins, start=1):
        dynamic_state = case_map[protein]
        dynamic_positions = tuple(dynamic_state["APO"]["positions"])
        payload = shards[protein]
        records = sorted(payload["records"], key=lambda record: int(record["sample_index"]))
        try:
            states = case_map[protein]
            target_positions = tuple(states["APO"]["positions"])
            if target_positions != tuple(states["HOLO"]["positions"]):
                raise ValueError("evaluator state axes differ")
            if args.evaluator == "ESM-IF1":
                finite_apo = np.isfinite(states["APO"]["coordinates"]).all(axis=(1, 2))
                finite_holo = np.isfinite(states["HOLO"]["coordinates"]).all(axis=(1, 2))
                common = finite_apo & finite_holo
                target_positions = tuple(position for position, keep in zip(target_positions, common, strict=True) if keep)
            sequences = tuple(
                _project_sequence(str(record["sequence"]), dynamic_positions, target_positions)
                for record in records
            )
            wt = _project_sequence(str(states["APO"]["sequence"]), dynamic_positions, target_positions)
            if args.evaluator == "ProteinMPNN":
                realization = build_decoding_realizations(protein, len(target_positions), count=1)[0]
                structures = {
                    condition: ProteinMPNNStructureInput(
                        protein_id=protein,
                        backbone_condition="PDB" if condition == "APO" else "AFDB",
                        uniprot_positions=target_positions,
                        wt_sequence_projection=str(states[condition]["sequence"]),
                        coordinates=states[condition]["coordinates"],
                        structure_sha256=str(states[condition]["structure_sha256"]),
                    )
                    for condition in ("APO", "HOLO")
                }
                wt_scores = {
                    condition: _score_pnn(adapter, structure, (wt,), realization, args.batch_size)[0]
                    for condition, structure in structures.items()
                }
                scores = {
                    condition: _score_pnn(adapter, structure, sequences, realization, args.batch_size)
                    for condition, structure in structures.items()
                }
            else:
                coords = {
                    condition: states[condition]["coordinates"][
                        [dynamic_positions.index(position) for position in target_positions]
                    ]
                    for condition in ("APO", "HOLO")
                }
                wt_scores = {
                    condition: _score_esm_batch(adapter, (wt,), value, args.batch_size)[0]
                    for condition, value in coords.items()
                }
                scores = {
                    condition: _score_esm_batch(adapter, sequences, value, args.batch_size)
                    for condition, value in coords.items()
                }
            for row_index, record in enumerate(records):
                endpoint = compatibility_endpoints(
                    apo_sequence_score=scores["APO"][row_index],
                    holo_sequence_score=scores["HOLO"][row_index],
                    apo_wt_score=wt_scores["APO"],
                    holo_wt_score=wt_scores["HOLO"],
                )
                rows.append(
                    {
                        "protein_id": protein,
                        "pair_id": str(payload["pair_id"]),
                        "evaluator": args.evaluator,
                        "source_state": "DYNAMICMPNN",
                        "sample_index": int(record["sample_index"]),
                        "sequence_hash": str(record["sequence_hash"]),
                        "score_apo": scores["APO"][row_index],
                        "score_holo": scores["HOLO"][row_index],
                        "wt_score_apo": wt_scores["APO"],
                        "wt_score_holo": wt_scores["HOLO"],
                        "c_apo": endpoint.c_apo,
                        "c_holo": endpoint.c_holo,
                        "mean_compat": endpoint.mean_compat,
                        "worst_compat": endpoint.worst_compat,
                        "state_gap": endpoint.state_gap,
                        "scoring_position_count": len(target_positions),
                    }
                )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            unavailable.append({"protein_id": protein, "reason": str(exc)})
        if index % 10 == 0:
            print(f"{args.evaluator} DynamicMPNN {index}/{len(proteins)}", flush=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["protein_id", "sample_index"], kind="mergesort")
    frame.to_parquet(output, index=False)
    manifest = {
        "schema_version": "dynamicmpnn_compatibility_scores_v1",
        "evaluator": args.evaluator,
        "protein_count": int(frame.protein_id.nunique()) if len(frame) else 0,
        "row_count": int(len(frame)),
        "unavailable_count": len(unavailable),
        "unavailable": unavailable,
        "source_state": "DYNAMICMPNN",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    run(parse_args())
