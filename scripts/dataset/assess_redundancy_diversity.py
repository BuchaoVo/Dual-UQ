"""Assess frozen redundancy and diversity rules for the admitted frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.redundancy_diversity import (
    BLOCKED_INPUT_INTEGRITY,
    RedundancyDiversityConfig,
    RedundancyDiversityError,
    run_redundancy_diversity_census,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic redundancy path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("redundancy")
        result = run_redundancy_diversity_census(
            paths,
            config=RedundancyDiversityConfig(output_root_ref=output_root),
        )
    except RedundancyDiversityError as exc:
        print(
            json.dumps(
                {
                    "scale1a3_status": exc.status,
                    "failure_code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                },
                sort_keys=True,
            )
        )
        return 2
    except ProjectPathError as exc:
        print(
            json.dumps(
                {
                    "scale1a3_status": BLOCKED_INPUT_INTEGRITY,
                    "failure_code": "project_root_unresolved",
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
