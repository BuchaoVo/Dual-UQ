"""Run offline protein-level feasibility and empirical power planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.planning import (
    PowerPlanningConfig,
    PowerPlanningError,
    run_power_planning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic planning path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("expansion_planning")
        result = run_power_planning(
            paths,
            config=PowerPlanningConfig(output_root_ref=output_root),
        )
    except PowerPlanningError as exc:
        print(
            json.dumps(
                {
                    "planning_status": "BLOCKED_INPUT_INTEGRITY",
                    "failure_code": exc.code,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
