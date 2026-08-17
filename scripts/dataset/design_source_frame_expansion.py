#!/usr/bin/env python3
"""Design an immutable source-frame expansion release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.expansion import ExpansionConfig
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.expansion import (
    ExpansionError,
    run_expansion_design,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design the outcome-blind Scale-1 source-frame expansion."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Explicit Dual-UQ repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(
        project_root=args.project_root,
        anchor=Path(__file__),
    )
    config = ExpansionConfig(
        output_root_ref=DatasetExperimentPaths.from_project(paths).logical_ref("expansion_planning")
    )
    try:
        report = run_expansion_design(paths, config=config)
    except ExpansionError as exc:
        report = {
            "scale1_expansion_design_status": exc.status,
            "failure_code": exc.code,
            "message": str(exc),
            "details": exc.details,
        }
        print(json.dumps(report, sort_keys=True, indent=2))
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
