"""Run the offline descriptive PDB/AFDB structural-response analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.evaluation.structural_response import (
    StructuralResponseError,
    StructuralResponseResult,
    build_paired_structural_response,
    load_structural_response_inputs,
    materialize_structural_response,
    summarize_structural_response,
)

DEFAULT_OUTPUT = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: experiments/dataset/analysis/structural_response)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    output_dir = args.output_dir or DatasetExperimentPaths.from_project(paths).structural_response
    if not output_dir.is_absolute():
        output_dir = paths.repository_root / output_dir
    try:
        inputs = load_structural_response_inputs(paths.repository_root)
        paired = build_paired_structural_response(inputs)
        summary = summarize_structural_response(paired)
        report = materialize_structural_response(
            StructuralResponseResult(inputs=inputs, paired=paired, summary=summary),
            output_dir,
        )
    except StructuralResponseError as exc:
        print(json.dumps({"status": exc.outcome, "failure_code": exc.code, "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "COMPLETE", "manifest_path": paths.logical_ref(report["manifest_path"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
