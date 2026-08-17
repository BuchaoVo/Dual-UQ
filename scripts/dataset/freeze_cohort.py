"""Freeze the pre-outcome scoring panel and redundancy core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.scale1b_cohort_freeze import (
    Scale1BCohortFreezeConfig,
    Scale1BCohortFreezeError,
    run_scale1b_cohort_freeze,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic cohort release path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("cohort_release")
        result = run_scale1b_cohort_freeze(
            paths,
            config=Scale1BCohortFreezeConfig(output_root_ref=output_root),
        )
    except Scale1BCohortFreezeError as exc:
        print(
            json.dumps(
                {
                    "freeze_status": exc.status,
                    "failure_code": exc.code,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    except ProjectPathError as exc:
        print(
            json.dumps(
                {
                    "freeze_status": "SCALE1B_BLOCKED_INPUT_INTEGRITY",
                    "failure_code": "project_root_unresolved",
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["freeze_status"] == "SCALE1B_COHORT_FREEZE_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
