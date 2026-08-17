"""Generate independent APO/HOLO sequence ensembles for one frozen model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.evaluation.apo_holo_local_response import build_apo_holo_cases
from dual_uq.inference.apo_holo_generation import (
    ApoHoloGenerationCondition,
    ESMIF1ApoHoloGenerationAdapter,
    ProteinMPNNApoHoloGenerationAdapter,
    generate_apo_holo_conditions,
)
from dual_uq.inference.generative_propagation import _full_chain_generation_structure
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    load_authorized_proteinmpnn_adapter,
)
from dual_uq.models.proteinmpnn_generation import ProteinMPNNGenerationAdapter

ESM_IF1_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESM_IF1_CHECKPOINT_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"


def _generation_exclusion(
    *,
    model: str,
    protein_id: str,
    pair_id: str,
    failure_state: str,
    error: Exception,
) -> dict[str, object]:
    """Describe a whole-protein build exclusion without changing the cohort."""
    detail = str(error)
    reason = (
        "generation_projection_order_incompatible"
        if "does not preserve full-chain order" in detail
        else "generation_build_incompatibility"
    )
    return {
        "model": model,
        "protein_id": protein_id,
        "pair_id": pair_id,
        "states_attempted": ["APO", "HOLO"],
        "failure_state": failure_state,
        "reason": reason,
        "detail": detail,
    }


def _write_worker_exclusions(
    output_root: Path,
    *,
    model: str,
    worker_index: int,
    exclusions: tuple[dict[str, object], ...],
) -> None:
    """Materialize deterministic, immutable evidence for build exclusions."""
    payload = {
        "schema_version": "apo_holo_generation_exclusions_v1",
        "model": model,
        "worker_index": worker_index,
        "exclusions": list(exclusions),
    }
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = Path(output_root) / f"generation_exclusions_worker_{worker_index}.json"
    try:
        atomic_write_new_bytes(path, rendered)
    except FileExistsError:
        if path.read_bytes() != rendered:
            raise ValueError(f"immutable generation exclusion conflict: {path}") from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-protein-id")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args(argv)


def _primary_pairs(release_root: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(Path(release_root) / "primary_pairs.parquet")
    if pairs.empty or not pairs["admitted"].eq(True).all():
        raise ValueError("release primary_pairs is empty or contains non-admitted rows")
    return pairs.set_index(pairs["pair_id"].astype(str), drop=False)


def _case_condition(
    group: pd.DataFrame,
    *,
    state: str,
    pair: pd.Series,
    mapping: pd.DataFrame,
    project_root: Path,
    model: str,
) -> ApoHoloGenerationCondition:
    ordered = group.sort_values("canonical_position", kind="mergesort")
    positions = tuple(int(value) for value in ordered["canonical_position"])
    sequence = str(ordered["wt_sequence_projection"].iloc[0])
    coordinates = np.asarray(ordered["coordinates"].iloc[0], dtype=np.float32)
    structure_sha256 = str(ordered["structure_sha256"].iloc[0])
    prefix = "apo" if state == "APO" else "holo"
    structure_path = project_root / str(pair[f"{prefix}_mmcif_relative_path"])
    chain_id = str(pair[f"{prefix}_chain_id"])
    proteinmpnn_structure = None
    if model == "ProteinMPNN":
        projection = ProteinMPNNStructureInput(
            protein_id=str(ordered["protein_id"].iloc[0]),
            backbone_condition="PDB" if state == "APO" else "AFDB",
            uniprot_positions=positions,
            wt_sequence_projection=sequence,
            coordinates=coordinates,
            structure_sha256=structure_sha256,
        )
        proteinmpnn_structure = _full_chain_generation_structure(
            projection,
            structure_path=structure_path,
            source_id=str(pair["pair_id"]),
            chain_id=chain_id,
            canonicalize_projected_sequence=True,
            projection_chain_auth_keys=tuple(
                (
                    int(row[f"{prefix}_auth_seq_id"]),
                    str(row[f"insertion_code_{prefix}"] or "").strip().upper(),
                )
                for _, row in mapping.sort_values("canonical_position", kind="mergesort").iterrows()
                if int(row["canonical_position"]) in set(positions)
                and pd.notna(row[f"{prefix}_auth_seq_id"])
            ),
            rebind_projection_coordinates=True,
        )
        coordinates = np.asarray(proteinmpnn_structure.projection.coordinates, dtype=np.float32)
    return ApoHoloGenerationCondition(
        protein_id=str(ordered["protein_id"].iloc[0]),
        pair_id=str(pair["pair_id"]),
        state=state,
        structure_sha256=structure_sha256,
        canonical_positions=positions,
        wt_sequence_projection=sequence,
        coordinates=coordinates,
        structure_path=structure_path,
        chain_id=chain_id,
        proteinmpnn_structure=proteinmpnn_structure,
    )


def build_conditions(
    project_root: Path,
    release_root: Path,
    *,
    model: str,
    workers: int,
    smoke_protein_id: str | None,
    worker_index: int,
    worker_count: int,
) -> tuple[tuple[ApoHoloGenerationCondition, ...], tuple[dict[str, object], ...]]:
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError("worker-index must be in 0..worker-count-1")
    proteinmpnn_cases, esm_cases, _exclusions = build_apo_holo_cases(
        project_root,
        release_root,
        require_model_evaluable=True,
        require_contiguous_esm_if1=True,
        workers=workers,
    )
    cases = proteinmpnn_cases if model == "ProteinMPNN" else esm_cases
    pairs = _primary_pairs(release_root)
    mappings = pd.read_parquet(Path(release_root) / "residue_mappings.parquet")
    shared = set(esm_cases["protein_id"].astype(str))
    cases = cases.loc[cases["protein_id"].astype(str).isin(shared)].copy()
    protein_ids = sorted(cases["protein_id"].astype(str).unique())
    if smoke_protein_id is not None:
        if smoke_protein_id not in protein_ids:
            raise ValueError(f"smoke protein is not in the shared cohort: {smoke_protein_id}")
        protein_ids = [smoke_protein_id]
    else:
        protein_ids = protein_ids[worker_index::worker_count]
    conditions: list[ApoHoloGenerationCondition] = []
    exclusions: list[dict[str, object]] = []
    for protein_id in protein_ids:
        protein_cases = cases.loc[cases["protein_id"].astype(str).eq(protein_id)]
        pair_id = str(protein_cases["pair_id"].iloc[0])
        if pair_id not in pairs.index:
            raise ValueError(f"case pair is absent from the frozen release: {pair_id}")
        pair = pairs.loc[pair_id]
        protein_conditions: list[ApoHoloGenerationCondition] = []
        protein_failure: tuple[str, Exception] | None = None
        for state in ("APO", "HOLO"):
            state_cases = protein_cases.loc[protein_cases["condition"].astype(str).eq(state)]
            try:
                if state_cases.empty:
                    raise ValueError(f"missing {state} case for {protein_id}")
                condition = _case_condition(
                    state_cases,
                    state=state,
                    pair=pair,
                    mapping=mappings.loc[mappings["pair_id"].astype(str).eq(pair_id)],
                    project_root=project_root,
                    model=model,
                )
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                protein_failure = (state, exc)
                break
            protein_conditions.append(condition)
        if protein_failure is not None:
            failure_state, error = protein_failure
            exclusions.append(
                _generation_exclusion(
                    model=model,
                    protein_id=protein_id,
                    pair_id=pair_id,
                    failure_state=failure_state,
                    error=error,
                )
            )
            continue
        conditions.extend(protein_conditions)
    return tuple(conditions), tuple(exclusions)


def _load_model_adapter(args: argparse.Namespace):
    project_root = args.project_root.resolve()
    if args.model == "ProteinMPNN":
        checkpoint = args.checkpoint_path or (
            project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
        )
        model = load_authorized_proteinmpnn_adapter(
            implementation_path=project_root / "third_party/ProteinMPNN",
            checkpoint_path=checkpoint,
            device_name=args.device,
            backbone_noise=0.0,
        )
        return ProteinMPNNApoHoloGenerationAdapter(
            ProteinMPNNGenerationAdapter(model, batch_size=args.batch_size)
        )
    checkpoint = args.checkpoint_path
    if checkpoint is None:
        configured = os.environ.get("ESM_IF1_CHECKPOINT_PATH")
        if configured:
            checkpoint = Path(configured)
    if checkpoint is None:
        raise ValueError("ESM-IF1 requires --checkpoint-path or ESM_IF1_CHECKPOINT_PATH")
    model = load_esm_if1(
        project_root / "third_party/esm",
        checkpoint,
        expected_revision=ESM_IF1_REVISION,
        expected_checkpoint_sha256=ESM_IF1_CHECKPOINT_SHA256,
        device=args.device,
    )
    return ESMIF1ApoHoloGenerationAdapter(model, batch_size=args.batch_size)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        conditions, exclusions = build_conditions(
            args.project_root.resolve(),
            args.release_root.resolve(),
            model=args.model,
            workers=args.workers,
            smoke_protein_id=args.smoke_protein_id,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
        payload: dict[str, object] = {
            "status": "READY_FOR_MODEL_BINDING" if not args.execute else "EXECUTING",
            "model": args.model,
            "protein_count": len({condition.protein_id for condition in conditions}),
            "condition_count": len(conditions),
            "excluded_protein_count": len(exclusions),
            "excluded_proteins": [item["protein_id"] for item in exclusions],
            "sample_count_per_condition": 64,
            "output_root": str(args.output_root),
        }
        if args.execute:
            _write_worker_exclusions(
                args.output_root,
                model=args.model,
                worker_index=args.worker_index,
                exclusions=exclusions,
            )
            adapter = _load_model_adapter(args)
            records, executed, reused = generate_apo_holo_conditions(
                conditions,
                adapter,
                output_root=args.output_root,
                resume=args.resume,
            )
            payload.update(
                {
                    "status": "COMPLETE",
                    "record_count": len(records),
                    "executed_conditions": executed,
                    "reused_conditions": reused,
                }
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
