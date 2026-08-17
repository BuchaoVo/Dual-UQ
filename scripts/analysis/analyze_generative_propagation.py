"""Materialize analysis-ready generative propagation measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.evaluation.generative_propagation import (
    GenerativePropagationError,
    GenerativePropagationInputs,
    build_analysis_result,
    materialize_generative_propagation,
)
from dual_uq.inference.generative_propagation import (
    GenerationExecutionError,
    load_generation_records,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--generation-root", type=Path, default=Path("runs/analysis/generative_propagation")
    )
    parser.add_argument(
        "--descriptors",
        type=Path,
        default=Path("experiments/dataset/analysis/inverse_folding_remodeling/position_remodeling.parquet"),
    )
    parser.add_argument("--cross-compatibility", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/dataset/analysis/generative_propagation"),
    )
    parser.add_argument("--expected-proteins", type=int, default=68)
    return parser.parse_args(argv)


def _project_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    root = paths.repository_root
    generation_root = _project_path(root, args.generation_root)
    descriptor_path = _project_path(root, args.descriptors)
    output_root = _project_path(root, args.output_root)
    cross_path = _project_path(root, args.cross_compatibility) if args.cross_compatibility else None
    try:
        records = load_generation_records(generation_root)
        descriptors = pd.read_parquet(descriptor_path)
        cross = pd.read_parquet(cross_path) if cross_path is not None else None
        provenance = {
            "generation_root": generation_root.relative_to(root).as_posix(),
            "descriptor_path": descriptor_path.relative_to(root).as_posix(),
            "descriptor_sha256": sha256_file(descriptor_path),
        }
        if cross_path is not None:
            provenance.update(
                {
                    "cross_compatibility_path": cross_path.relative_to(root).as_posix(),
                    "cross_compatibility_sha256": sha256_file(cross_path),
                }
            )
        inputs = GenerativePropagationInputs(
            records=records,
            position_descriptors=descriptors,
            cross_compatibility=cross,
            expected_protein_count=args.expected_proteins,
            input_provenance=provenance,
        )
        result = build_analysis_result(inputs)
        materialized = materialize_generative_propagation(result, output_root)
    except (OSError, ValueError, GenerationExecutionError, GenerativePropagationError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output_root": output_root.relative_to(root).as_posix(),
                "protein_count": len(result.protein_summary),
                "generated_rows": len(result.generated_sequences),
                "manifest_status": materialized["manifest_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
