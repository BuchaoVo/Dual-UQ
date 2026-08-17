"""Audit the offline sampling frame and its local-readiness evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.sampling_frame import (
    SamplingFrameAuditConfig,
    SamplingFrameAuditError,
    run_sampling_frame_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic sampling-frame path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("sampling_frame")
        result = run_sampling_frame_audit(
            paths,
            config=SamplingFrameAuditConfig(output_root_ref=output_root),
        )
    except SamplingFrameAuditError as exc:
        print(
            json.dumps(
                {
                    "scale1a0_status": "SCALE1A0_BLOCKED_LOCAL_CENSUS",
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
