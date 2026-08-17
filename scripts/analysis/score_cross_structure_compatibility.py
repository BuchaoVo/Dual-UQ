"""Score generated sequences under both paired structural representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.inference.cross_structure_compatibility import (
    CrossStructureScoringError,
    build_cross_structure_inputs,
    score_cross_structure,
)
from dual_uq.models.proteinmpnn import load_authorized_proteinmpnn_adapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, default=Path("runs/analysis/generative_propagation"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/analysis/cross_structure_compatibility"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="load the authorized model and score")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(project_root=args.project_root.resolve(), anchor=Path(__file__))
    root = paths.repository_root
    generation_root = args.generation_root if args.generation_root.is_absolute() else root / args.generation_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    try:
        inputs = build_cross_structure_inputs(root, generation_root)
        if not args.execute:
            print(json.dumps({
                "status": "RESOLVED_NO_MODEL_EXECUTION",
                "protein_count": inputs.expected_protein_count,
                "generated_records": len(inputs.records),
                "output_root": output_root.resolve().as_posix(),
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
            }, sort_keys=True))
            return 0
        adapter = load_authorized_proteinmpnn_adapter(
            implementation_path=root / "third_party/ProteinMPNN",
            checkpoint_path=root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt",
            device_name=args.device,
            backbone_noise=0.0,
        )
        run = score_cross_structure(
            inputs,
            adapter,
            output_root=output_root,
            batch_size=args.batch_size,
            resume=args.resume,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
    except (CrossStructureScoringError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "COMPLETE",
        "protein_count": inputs.expected_protein_count,
        "rows": len(run.rows),
        "executed_proteins": run.executed_proteins,
        "reused_proteins": run.reused_proteins,
        "output_root": output_root.resolve().as_posix(),
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
