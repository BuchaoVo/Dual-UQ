"""Resolve the frozen generative-propagation cohort for controlled execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.inference.generative_propagation import (
    GenerationExecutionError,
    GenerationInputs,
    generate_clean_cohort,
    load_generation_inputs,
)
from dual_uq.models.proteinmpnn import load_authorized_proteinmpnn_adapter
from dual_uq.models.proteinmpnn_generation import ProteinMPNNGenerationAdapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/analysis/generative_propagation"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-protein-id", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="load the authorized model and generate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args(argv)


def select_smoke_inputs(inputs: GenerationInputs, protein_ids: list[str]) -> GenerationInputs:
    """Return only explicitly requested complete PDB/AFDB smoke pairs."""
    if not protein_ids:
        return inputs
    requested = set(protein_ids)
    observed = {row.protein_id for row in inputs.conditions}
    missing = sorted(requested - observed)
    if missing:
        raise GenerationExecutionError(f"smoke protein IDs are not in the clean cohort: {missing}")
    return GenerationInputs(
        tuple(row for row in inputs.conditions if row.protein_id in requested),
        expected_protein_count=len(requested),
        input_provenance=inputs.input_provenance,
    )


def select_worker_inputs(
    inputs: GenerationInputs, *, worker_index: int, worker_count: int
) -> GenerationInputs:
    """Partition proteins deterministically; each worker owns disjoint shards."""
    if (
        isinstance(worker_index, bool)
        or not isinstance(worker_index, int)
        or isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count <= 0
        or worker_index < 0
        or worker_index >= worker_count
    ):
        raise GenerationExecutionError("worker index/count must satisfy 0 <= index < count")
    proteins = sorted({row.protein_id for row in inputs.conditions})
    selected = set(proteins[worker_index::worker_count])
    if not selected:
        raise GenerationExecutionError("worker owns no proteins")
    return GenerationInputs(
        tuple(row for row in inputs.conditions if row.protein_id in selected),
        expected_protein_count=len(selected),
        input_provenance={
            **dict(inputs.input_provenance or {}),
            "worker_index": worker_index,
            "worker_count": worker_count,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve(),
        anchor=Path(__file__),
    )
    project_root = paths.repository_root
    try:
        inputs = select_smoke_inputs(load_generation_inputs(project_root), args.smoke_protein_id)
        inputs = select_worker_inputs(
            inputs,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
    except GenerationExecutionError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    if not args.execute:
        print(json.dumps({
            "status": "RESOLVED_NO_MODEL_EXECUTION",
            "conditions": len(inputs.conditions),
            "output_root": output_root.resolve().as_posix(),
            "resume": bool(args.resume),
            "smoke_protein_ids": list(args.smoke_protein_id),
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
        }, sort_keys=True))
        return 0
    try:
        raw_adapter = load_authorized_proteinmpnn_adapter(
            implementation_path=project_root / "third_party/ProteinMPNN",
            checkpoint_path=project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt",
            device_name=args.device,
            backbone_noise=0.0,
        )
        adapter = ProteinMPNNGenerationAdapter(raw_adapter, batch_size=args.batch_size)
        run = generate_clean_cohort(
            inputs,
            adapter,
            output_root=output_root,
            resume=args.resume,
        )
    except (GenerationExecutionError, ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    print(json.dumps({
        "status": "GENERATION_COMPLETE",
        "conditions": len(inputs.conditions),
        "executed_conditions": run.executed_conditions,
        "reused_conditions": run.reused_conditions,
        "records": len(run.records),
        "output_root": output_root.resolve().as_posix(),
        "device": args.device,
        "batch_size": args.batch_size,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
