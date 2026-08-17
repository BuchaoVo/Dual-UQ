"""Build the offline formal admission census and common masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.admission import (
    FormalAdmissionConfig,
    FormalAdmissionError,
    run_formal_admission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic admission path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("admission")
        result = run_formal_admission(
            paths,
            config=FormalAdmissionConfig(output_root_ref=output_root),
        )
    except FormalAdmissionError as exc:
        print(
            json.dumps(
                {
                    "scale1a1_status": (
                        "SCALE1A1_BLOCKED_ADMISSION_REGRESSION"
                        if exc.code == "blocked_admission_regression"
                        else "SCALE1A1_BLOCKED_INPUT_INTEGRITY"
                    ),
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
