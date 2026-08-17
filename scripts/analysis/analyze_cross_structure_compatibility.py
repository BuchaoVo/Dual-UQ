"""Materialize cross-structure ProteinMPNN compatibility measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.evaluation.cross_structure_compatibility import (
    CrossStructureCompatibilityError,
    build_cross_structure_analysis,
    materialize_cross_structure_analysis,
)
from dual_uq.inference.cross_structure_compatibility import (
    cross_score_rows_frame,
    load_cross_score_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, default=Path("runs/analysis/cross_structure_compatibility"))
    parser.add_argument("--generation-summary", type=Path, default=Path("experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet"))
    parser.add_argument("--remodeling-summary", type=Path, default=Path("experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/dataset/analysis/cross_structure_compatibility"))
    return parser.parse_args(argv)


def _project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = ProjectPaths.discover(project_root=args.project_root.resolve(), anchor=Path(__file__)).repository_root
    score_root = _project_path(root, args.score_root)
    generation_path = _project_path(root, args.generation_summary)
    remodeling_path = _project_path(root, args.remodeling_summary)
    output_root = _project_path(root, args.output_root)
    try:
        rows = load_cross_score_rows(score_root)
        generation = pd.read_parquet(generation_path)
        remodeling = pd.read_parquet(remodeling_path)
        provenance = {
            "score_root": score_root.relative_to(root).as_posix(),
            "generation_summary": generation_path.relative_to(root).as_posix(),
            "generation_summary_sha256": sha256_file(generation_path),
            "remodeling_summary": remodeling_path.relative_to(root).as_posix(),
            "remodeling_summary_sha256": sha256_file(remodeling_path),
        }
        result = build_cross_structure_analysis(
            cross_score_rows_frame(rows), generation, remodeling, input_provenance=provenance
        )
        materialized = materialize_cross_structure_analysis(result, output_root)
    except (CrossStructureCompatibilityError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "COMPLETE",
        "protein_count": len(result.protein_summary),
        "score_rows": len(result.score_rows),
        "directional_rows": len(result.directional_loss),
        "manifest_status": materialized["manifest_status"],
        "output_root": output_root.relative_to(root).as_posix(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
