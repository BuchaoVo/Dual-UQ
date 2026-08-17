#!/usr/bin/env python3
"""Run the descriptive frozen PDB/AFDB pair-validity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.evaluation.pair_validity import analyze_pair_validity, materialize_pair_validity
from dual_uq.dataset.paths import DatasetExperimentPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/dataset/analysis/pair_validity)",
    )
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir or DatasetExperimentPaths.from_project(
        ProjectPaths.discover(project_root=project_root)
    ).pair_validity
    result = analyze_pair_validity(project_root)
    materialize_pair_validity(result, output_dir)
    print(result.summary)
    print(f"outputs={output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
