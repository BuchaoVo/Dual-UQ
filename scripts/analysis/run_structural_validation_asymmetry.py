#!/usr/bin/env python3
"""Predict one WT/reference structure per clean PDB/AFDB pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.inference.generative_propagation import load_generation_inputs
from dual_uq.inference.independent_structure_validation import (
    LocalESMFoldAdapter,
    StructuralValidationAsymmetryError,
    predict_wt_sequences,
    select_wt_sequences,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/analysis/independent_structure_validation/asymmetry/wt"),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root.expanduser().resolve())
        if (
            isinstance(args.worker_index, bool)
            or isinstance(args.worker_count, bool)
            or args.worker_count <= 0
            or args.worker_index < 0
            or args.worker_index >= args.worker_count
        ):
            raise StructuralValidationAsymmetryError("invalid worker partition")
        inputs = load_generation_inputs(paths.repository_root)
        projections = {
            (condition.protein_id, condition.backbone_condition): condition.structure.projection
            for condition in inputs.conditions
        }
        requests = select_wt_sequences(projections)
        selected = requests[args.worker_index :: args.worker_count]
        if not args.execute:
            print(f"RESOLVED_NO_MODEL_EXECUTION rows={len(selected)} proteins={len({row.protein_id for row in selected})}")
            return 0
        if args.model_path is None:
            raise StructuralValidationAsymmetryError("--model-path is required with --execute")
        output_root = args.output_root if args.output_root.is_absolute() else paths.repository_root / args.output_root
        adapter = LocalESMFoldAdapter(args.model_path, device=args.device, chunk_size=args.chunk_size)
        rows = predict_wt_sequences(
            selected,
            projections,
            adapter,
            output_root=output_root,
            resume=args.resume,
        )
        print(f"COMPLETE rows={len(rows)} output={output_root}")
        return 0
    except StructuralValidationAsymmetryError as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
