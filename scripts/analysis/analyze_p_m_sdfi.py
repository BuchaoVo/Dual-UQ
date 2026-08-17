"""Run the offline confirmatory P/M/SDFI characterization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.evaluation.p_m_sdfi import (
    PMSDFIError,
    analyze_p_m_sdfi,
    load_p_m_sdfi_inputs,
    materialize_p_m_sdfi,
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
    output_dir = args.output_dir or (DatasetExperimentPaths.from_project(paths).structural_response / "p_m_sdfi")
    if not output_dir.is_absolute():
        output_dir = paths.repository_root / output_dir
    try:
        inputs = load_p_m_sdfi_inputs(paths.repository_root)
        result = analyze_p_m_sdfi(inputs)
        report = materialize_p_m_sdfi(result, output_dir)
    except PMSDFIError as exc:
        print(json.dumps({"status": exc.outcome, "failure_code": exc.code, "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "COMPLETE", "output_dir": paths.logical_ref(report["output_dir"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
