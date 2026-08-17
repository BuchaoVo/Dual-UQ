"""Analyze local inverse-folding compatibility remodeling from frozen paired scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.evaluation.inverse_folding_remodeling import (
    RemodelingAnalysisError,
    build_remodeling_result,
    load_remodeling_inputs,
    materialize_remodeling_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    output_dir = args.output_dir or paths.repository_root / "experiments/dataset/analysis/inverse_folding_remodeling"
    if not output_dir.is_absolute():
        output_dir = paths.repository_root / output_dir
    try:
        inputs = load_remodeling_inputs(paths.repository_root)
        result = build_remodeling_result(inputs)
        report = materialize_remodeling_result(result, output_dir)
    except RemodelingAnalysisError as exc:
        print(json.dumps({"status": exc.outcome, "failure_code": exc.code, "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "COMPLETE", "output_dir": str(report["output_dir"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
