"""Run deterministic native greedy generation on frozen StructCal paired views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.artifacts import write_immutable_json, write_immutable_parquet
from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations
from dual_uq.evaluation.cross_model_representation_cases import make_degenerate_dynamic_case
from dual_uq.evaluation.generation_consequence import (
    STANDARD_AMINO_ACIDS,
    GreedyGeneration,
    build_generation_response,
    greedy_multinomial,
    proteinmpnn_greedy,
)
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput
from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    CHECKPOINTS,
    DEFAULT_RUN_ROOT,
    MODEL_SPECS,
    REGIMES,
    coordinates_from_case_group,
    filter_dynamicmpnn_contiguous_cases,
    load_structcal_model,
    safe_output_path,
)

def _seed(protein_id: str) -> int:
    return int.from_bytes(hashlib.sha256(protein_id.encode()).digest()[:4], "big")


def _generate(
    model_name: str,
    adapter: Any,
    *,
    protein_id: str,
    pair_id: str,
    condition: str,
    sequence: str,
    coordinates: np.ndarray,
    positions: tuple[int, ...],
) -> GreedyGeneration:
    order = tuple(range(len(sequence)))
    if model_name == "esm_if1":
        import torch

        with greedy_multinomial(torch):
            generated = adapter.sample(
                coordinates[:, :3], temperature=1.0, seed=_seed(protein_id)
            )
        return GreedyGeneration(generated, order)
    if model_name == "dynamicmpnn":
        case = make_degenerate_dynamic_case(
            protein_id=protein_id,
            pair_id=f"{pair_id}::{condition}",
            canonical_positions=positions,
            sequence=sequence,
            coordinates=coordinates[:, :3],
            chain_id="A",
        )
        with greedy_multinomial(adapter._torch or __import__("torch")):
            generated = adapter.sample(case, n_samples=1, seed=_seed(protein_id)).sequences[0]
        return GreedyGeneration(generated, order)
    if model_name == "proteinmpnn":
        realization = build_decoding_realizations(protein_id, len(sequence), count=1)[0]
        return proteinmpnn_greedy(
            adapter,
            ProteinMPNNStructureInput(
                protein_id=protein_id,
                backbone_condition=condition,
                uniprot_positions=positions,
                wt_sequence_projection=sequence,
                coordinates=coordinates,
                structure_sha256=None,
            ),
            realization.order,
        )
    raise ValueError(f"unsupported autoregressive model: {model_name}")


def _pifold_generations(
    output_root: Path, cases: pd.DataFrame, regime: str
) -> dict[tuple[str, str], GreedyGeneration]:
    residue = pd.read_parquet(
        safe_output_path(
            output_root,
            f"shards/pifold/{CHECKPOINTS['pifold']}/{regime}/residue_response.parquet",
        )
    )
    result: dict[tuple[str, str], GreedyGeneration] = {}
    for pair_id, group in residue.groupby("pair_id", sort=True):
        ordered = group.sort_values("canonical_position", kind="mergesort")
        left = "".join(ordered["top1_left"].astype(str))
        right = "".join(ordered["top1_right"].astype(str))
        if set(left + right).difference(STANDARD_AMINO_ACIDS):
            raise ValueError(f"PiFold top-1 output is not standard-AA: {pair_id}")
        order = tuple(range(len(left)))
        result[(str(pair_id), "CONDITION_1")] = GreedyGeneration(left, order)
        result[(str(pair_id), "CONDITION_2")] = GreedyGeneration(right, order)
    expected = set(cases["pair_id"].astype(str))
    observed = {pair_id for pair_id, _condition in result}
    if observed != expected:
        raise ValueError("PiFold frozen residue response differs from the generation cohort")
    return result


def score_regime(
    *,
    project_root: Path,
    output_root: Path,
    model_name: str,
    regime: str,
    adapter: Any,
    checkpoint_id: str,
    generation_cache: dict[tuple[Any, ...], GreedyGeneration] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    atom_key = "ncao" if len(MODEL_SPECS[model_name]["atoms"]) == 4 else "nca"
    cases = pd.read_parquet(safe_output_path(output_root, f"cases/{atom_key}/{regime}.parquet"))
    excluded = pd.DataFrame()
    if model_name == "dynamicmpnn":
        cases, excluded = filter_dynamicmpnn_contiguous_cases(cases, regime=regime)
    if shard_count is not None:
        if shard_index is None or shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("shard index must be in [0, shard count)")
        protein_ids = sorted(cases["protein_id"].astype(str).unique())
        selected = set(protein_ids[shard_index::shard_count])
        cases = cases.loc[cases["protein_id"].astype(str).isin(selected)].copy()
    if model_name == "pifold":
        generations = _pifold_generations(output_root, cases, regime)
        unique_inputs = len(generations)
    else:
        generations = {}
        cache = generation_cache if generation_cache is not None else {}
        requests: dict[tuple[Any, ...], dict[str, Any]] = {}
        generation_keys: dict[tuple[str, str], tuple[Any, ...]] = {}
        for (pair_id, condition), group in cases.groupby(["pair_id", "condition"], sort=True):
            ordered = group.sort_values("canonical_position", kind="mergesort")
            sequence_values = ordered["wt_sequence_projection"].astype(str).drop_duplicates()
            if len(sequence_values) != 1:
                raise ValueError(f"case sequence is not constant: {pair_id}/{condition}")
            coordinates = coordinates_from_case_group(ordered)
            positions = tuple(int(value) for value in ordered["canonical_position"])
            protein_id = str(ordered.iloc[0]["protein_id"])
            cache_key = (
                protein_id,
                positions,
                str(sequence_values.iloc[0]),
                hashlib.sha256(coordinates.astype("<f4", copy=False).tobytes()).digest(),
            )
            if cache_key not in cache:
                requests[cache_key] = {
                    "protein_id": protein_id,
                    "pair_id": str(pair_id),
                    "condition": str(condition),
                    "sequence": str(sequence_values.iloc[0]),
                    "coordinates": coordinates,
                    "positions": positions,
                }
            generation_keys[(str(pair_id), str(condition))] = cache_key
        if model_name == "esm_if1":
            import torch

            pending = list(requests.items())
            for length in sorted({len(item[1]["sequence"]) for item in pending}):
                same_length = [item for item in pending if len(item[1]["sequence"]) == length]
                for start in range(0, len(same_length), 8):
                    chunk = same_length[start : start + 8]
                    metadata = [item[1] for item in chunk]
                    with greedy_multinomial(torch):
                        sequences = adapter.sample_batch(
                            np.stack([item["coordinates"][:, :3] for item in metadata]),
                            tuple(1.0 for _ in metadata),
                            tuple(_seed(item["protein_id"]) for item in metadata),
                        )
                    for (cache_key, _item), sequence in zip(chunk, sequences, strict=True):
                        cache[cache_key] = GreedyGeneration(sequence, tuple(range(length)))
        else:
            for cache_key, metadata in requests.items():
                cache[cache_key] = _generate(model_name, adapter, **metadata)
        generations.update({key: cache[cache_key] for key, cache_key in generation_keys.items()})
        unique_inputs = len(set(generation_keys.values()))
    response = build_generation_response(
        cases,
        generations,
        model_id=model_name,
        checkpoint_id=checkpoint_id,
        regime=regime,
    )
    filename = (
        "greedy_generation_response.parquet"
        if shard_count is None
        else f"greedy_generation_response.part-{shard_index:02d}-of-{shard_count:02d}.parquet"
    )
    path = safe_output_path(
        output_root, f"generation_shards/{model_name}/{checkpoint_id}/{regime}/{filename}"
    )
    write_immutable_parquet(path, response)
    summary = {
        "status": "COMPLETE",
        "model": model_name,
        "checkpoint": checkpoint_id,
        "regime": regime,
        "pairs": len(response),
        "proteins": int(response["protein_id"].nunique()),
        "unique_model_inputs": unique_inputs,
        "exclusions": len(excluded),
        "decode_policy": "native_autoregressive_argmax" if model_name != "pifold" else "native_one_shot_argmax",
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    summary_name = "summary.json" if shard_count is None else filename.replace(
        "greedy_generation_response", "summary"
    ).replace(".parquet", ".json")
    write_immutable_json(path.with_name(summary_name), summary)
    return summary


def merge_shards(
    *, output_root: Path, model_name: str, regime: str, shard_count: int
) -> dict[str, Any]:
    checkpoint_id = CHECKPOINTS[model_name]
    root = safe_output_path(
        output_root, f"generation_shards/{model_name}/{checkpoint_id}/{regime}"
    )
    paths = [
        root / f"greedy_generation_response.part-{index:02d}-of-{shard_count:02d}.parquet"
        for index in range(shard_count)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"generation shards are incomplete: {missing}")
    response = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if response["pair_id"].duplicated().any():
        raise ValueError("generation shards contain duplicate pair IDs")
    atom_key = "ncao" if len(MODEL_SPECS[model_name]["atoms"]) == 4 else "nca"
    cases = pd.read_parquet(output_root / f"cases/{atom_key}/{regime}.parquet")
    if model_name == "dynamicmpnn":
        cases, _excluded = filter_dynamicmpnn_contiguous_cases(cases, regime=regime)
    expected_pairs = set(cases["pair_id"].astype(str))
    observed_pairs = set(response["pair_id"].astype(str))
    if observed_pairs != expected_pairs:
        raise ValueError("generation shard union differs from the frozen evaluable cohort")
    response = response.sort_values(
        ["model_id", "regime", "pair_id"], kind="mergesort", ignore_index=True
    )
    write_immutable_parquet(root / "greedy_generation_response.parquet", response)
    summary = {
        "status": "COMPLETE",
        "model": model_name,
        "checkpoint": checkpoint_id,
        "regime": regime,
        "pairs": len(response),
        "proteins": int(response["protein_id"].nunique()),
        "merged_shards": shard_count,
    }
    write_immutable_json(root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--regime", choices=REGIMES, action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    regimes = tuple(args.regime or REGIMES)
    if args.merge:
        if len(regimes) != 1 or args.shard_count is None:
            raise ValueError("merge requires one --regime and --shard-count")
        result = merge_shards(
            output_root=output_root,
            model_name=args.model,
            regime=regimes[0],
            shard_count=args.shard_count,
        )
        print(json.dumps({"status": "COMPLETE", "result": result}, indent=2, sort_keys=True))
        return 0
    if args.model == "pifold":
        adapter, checkpoint_id = None, CHECKPOINTS["pifold"]
    else:
        checkpoint = "v_48_020" if args.model == "proteinmpnn" else "default"
        adapter, checkpoint_id = load_structcal_model(
            args.model, checkpoint, project_root, output_root, args.device
        )
    generation_cache: dict[tuple[Any, ...], GreedyGeneration] = {}
    summaries = [
        score_regime(
            project_root=project_root,
            output_root=output_root,
            model_name=args.model,
            regime=regime,
            adapter=adapter,
            checkpoint_id=checkpoint_id,
            generation_cache=generation_cache,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        for regime in regimes
    ]
    print(json.dumps({"status": "COMPLETE", "results": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
