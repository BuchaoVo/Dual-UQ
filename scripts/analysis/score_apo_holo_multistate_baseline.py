"""Score frozen single-state Apo/Holo ensembles on both structural states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.evaluation.multi_state_baseline import compatibility_endpoints
from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations
from dual_uq.inference.apo_holo_generation import load_apo_holo_generation_records
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    load_authorized_proteinmpnn_adapter,
)
from dual_uq.models.esm_if1 import load_esm_if1

ESM_IF1_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESM_IF1_CHECKPOINT_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"
AA = tuple("ACDEFGHIKLMNPQRSTVWY")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--shared-generation-root", type=Path)
    parser.add_argument("--cases-pickle", type=Path, required=True)
    parser.add_argument("--evaluator", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--implementation-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def _case_map(cases: pd.DataFrame) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for (protein, condition), group in cases.groupby(["protein_id", "condition"], sort=False):
        row = group.sort_values("canonical_position", kind="mergesort").iloc[0]
        positions = tuple(int(v) for v in group.sort_values("canonical_position")["canonical_position"])
        result.setdefault(str(protein), {})[str(condition)] = {
            "pair_id": str(row.pair_id),
            "positions": positions,
            "sequence": str(row.wt_sequence_projection),
            "coordinates": np.asarray(row.coordinates, dtype=np.float32),
            "structure_sha256": str(row.structure_sha256),
        }
    return result


def _read_cases(path: Path) -> pd.DataFrame:
    """Read the shared case cache across the two model environments."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        # NumPy 1.x/2.x pickle module spelling differs between the frozen
        # analysis and isolated ESM-IF1 environments.
        if "numpy._core" not in str(exc):
            raise
        import numpy

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
        return pd.read_pickle(path)


def _score_esm(adapter, sequence: str, coordinates: np.ndarray) -> float:
    probabilities = adapter.score_teacher_forced(sequence, coordinates)
    indices = np.asarray([AA.index(aa) for aa in sequence], dtype=np.int64)
    values = probabilities[np.arange(len(indices)), indices]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("ESM-IF1 returned invalid target probabilities")
    return float(np.log(values).sum(dtype=np.float64))


def _score_esm_batch(adapter, sequences: tuple[str, ...], coordinates: np.ndarray, batch_size: int) -> list[float]:
    current = batch_size
    while current >= 1:
        try:
            probabilities = adapter.score_teacher_forced_batch(sequences, coordinates, batch_size=current)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current == 1:
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, AttributeError, RuntimeError):
                pass
            current //= 2
    else:
        raise RuntimeError("ESM-IF1 scoring batch could not be reduced")
    result = []
    for sequence, table in zip(sequences, probabilities, strict=True):
        indices = np.asarray([AA.index(aa) for aa in sequence], dtype=np.int64)
        values = table[np.arange(len(indices)), indices]
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("ESM-IF1 returned invalid target probabilities")
        result.append(float(np.log(values).sum(dtype=np.float64)))
    return result


def _score_pnn(adapter, structure: ProteinMPNNStructureInput, sequences: tuple[str, ...], realization, batch_size: int) -> list[float]:
    current = batch_size
    while current >= 1:
        try:
            scores = adapter.score_sequences(structure, sequences, realization, batch_size=current)
            return [float(item.score_sum_logp_mask) for item in scores]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current == 1:
                raise
            try:
                adapter.torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass
            current //= 2
    raise RuntimeError("ProteinMPNN scoring batch could not be reduced")


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.project_root).resolve()
    cases = _read_cases(Path(args.cases_pickle))
    case_map = _case_map(cases)
    records = load_apo_holo_generation_records(Path(args.generation_root))
    grouped: dict[str, dict[str, list]] = {}
    for record in records:
        grouped.setdefault(record.condition.protein_id, {}).setdefault(record.condition.state, []).append(record)
    proteins = sorted(set(case_map).intersection(grouped))
    if args.shared_generation_root is not None:
        shared_records = load_apo_holo_generation_records(Path(args.shared_generation_root))
        shared_ids = {record.condition.protein_id for record in shared_records}
        proteins = [protein for protein in proteins if protein in shared_ids]
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    proteins = proteins[args.offset :]
    if args.limit is not None:
        proteins = proteins[: args.limit]
    if not proteins:
        raise ValueError("no generation/case intersection")
    if args.evaluator == "ProteinMPNN":
        implementation = Path(args.implementation_path or project_root / "third_party/ProteinMPNN")
        checkpoint = Path(args.checkpoint_path or project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt")
        adapter = load_authorized_proteinmpnn_adapter(
            implementation_path=implementation,
            checkpoint_path=checkpoint,
            device_name=args.device,
            backbone_noise=0.0,
        )
    else:
        source_root = Path(args.source_root or project_root / "third_party/esm")
        checkpoint = Path(args.checkpoint_path or project_root / "third_party/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt")
        adapter = load_esm_if1(
            source_root,
            checkpoint,
            expected_revision=ESM_IF1_REVISION,
            expected_checkpoint_sha256=ESM_IF1_CHECKPOINT_SHA256,
            device=args.device,
        )
    rows: list[dict[str, object]] = []
    for index, protein in enumerate(proteins, start=1):
        states = case_map[protein]
        if set(states) != {"APO", "HOLO"}:
            continue
        positions = states["APO"]["positions"]
        if positions != states["HOLO"]["positions"]:
            raise ValueError(f"state axes differ for {protein}")
        full_wt = states["APO"]["sequence"]
        if full_wt != states["HOLO"]["sequence"]:
            raise ValueError(f"WT projections differ for {protein}")
        projection_indices = np.arange(len(positions), dtype=np.int64)
        if args.evaluator == "ESM-IF1":
            finite_apo = np.isfinite(states["APO"]["coordinates"]).all(axis=(1, 2))
            finite_holo = np.isfinite(states["HOLO"]["coordinates"]).all(axis=(1, 2))
            projection_indices = np.flatnonzero(finite_apo & finite_holo)
            if len(projection_indices) < 3:
                continue
        wt = "".join(full_wt[index] for index in projection_indices)
        if args.evaluator == "ProteinMPNN":
            realization = build_decoding_realizations(protein, len(positions), count=1)[0]
            structures = {
                condition: ProteinMPNNStructureInput(
                    protein_id=protein,
                    backbone_condition="PDB" if condition == "APO" else "AFDB",
                    uniprot_positions=positions,
                    wt_sequence_projection=wt,
                    coordinates=states[condition]["coordinates"],
                    structure_sha256=states[condition]["structure_sha256"],
                )
                for condition in ("APO", "HOLO")
            }
            wt_scores = {
                condition: _score_pnn(adapter, structure, (wt,), realization, args.batch_size)[0]
                for condition, structure in structures.items()
            }
        else:
            structures = {
                condition: states[condition]["coordinates"][projection_indices]
                for condition in ("APO", "HOLO")
            }
            wt_scores = {
                condition: _score_esm_batch(adapter, (wt,), coords, args.batch_size)[0]
                for condition, coords in structures.items()
            }
        all_records = [item for state in ("APO", "HOLO") for item in sorted(grouped[protein].get(state, []), key=lambda item: item.sample_index)]
        if args.evaluator == "ProteinMPNN":
            sequences = tuple([wt] + [str(item.sequence) for item in all_records])
            score_vectors = {
                condition: _score_pnn(adapter, structure, sequences, realization, args.batch_size)
                for condition, structure in structures.items()
            }
        else:
            sequences = tuple(
                [wt]
                + [
                    "".join(str(item.sequence)[index] for index in projection_indices)
                    for item in all_records
                ]
            )
            score_vectors = {
                condition: _score_esm_batch(adapter, sequences, coords, args.batch_size)
                for condition, coords in structures.items()
            }
        for row_index, record in enumerate([None] + all_records):
            source_state = "WT" if record is None else str(record.condition.state)
            scores = {condition: values[row_index] for condition, values in score_vectors.items()}
            endpoint = compatibility_endpoints(
                apo_sequence_score=scores["APO"], holo_sequence_score=scores["HOLO"],
                apo_wt_score=wt_scores["APO"], holo_wt_score=wt_scores["HOLO"],
            )
            rows.append({
                "protein_id": protein,
                "pair_id": states["APO"]["pair_id"],
                "evaluator": args.evaluator,
                "source_state": source_state,
                "sample_index": -1 if record is None else int(record.sample_index),
                "sequence_hash": "WT" if record is None else str(record.sequence_hash),
                "score_apo": scores["APO"], "score_holo": scores["HOLO"],
                "wt_score_apo": wt_scores["APO"], "wt_score_holo": wt_scores["HOLO"],
                "c_apo": endpoint.c_apo, "c_holo": endpoint.c_holo,
                "mean_compat": endpoint.mean_compat, "worst_compat": endpoint.worst_compat,
                "state_gap": endpoint.state_gap,
            })
        if index % 10 == 0:
            print(f"{args.evaluator} {index}/{len(proteins)}", flush=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["protein_id", "source_state", "sample_index"], kind="mergesort")
    frame.to_parquet(output, index=False)
    manifest = {
        "schema_version": "multi_state_baseline_scores_v1",
        "evaluator": args.evaluator,
        "protein_count": int(frame["protein_id"].nunique()),
        "row_count": int(len(frame)),
        "wt_normalization": "score_state(sequence)-score_state(WT)",
        "output": str(output.relative_to(project_root)),
    }
    atomic_write_new_bytes(output.with_suffix(".json"), (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


if __name__ == "__main__":
    run(parse_args())
