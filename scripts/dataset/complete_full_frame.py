#!/usr/bin/env python3
"""Complete the frozen full-frame acquisition and admission census."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.completion import (
    FullFrameConfig,
    resolve_asset_requirements,
    run_full_frame_completion,
    validate_full_frame_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate frozen inputs and print independent acquisition counts only.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    paths = ProjectPaths.discover(project_root=args.project_root)
    config = FullFrameConfig(
        output_root_ref=DatasetExperimentPaths.from_project(paths).logical_ref("full_frame")
    )
    if args.plan_only:
        inputs = validate_full_frame_inputs(paths, config)
        requirements = resolve_asset_requirements(inputs, paths)
        payload = {
            "status": "PLAN_ONLY",
            "source_frame_count": len(inputs.source_frame),
            "preexisting_local_ready_count": len(inputs.preexisting_local_ready),
            "acquisition_target_count": len(inputs.acquisition_targets),
            "independent_asset_counts": dict(
                sorted(Counter(item.asset_type for item in requirements).items())
            ),
            "network_requests_executed": 0,
        }
    else:
        payload = run_full_frame_completion(
            paths, config=config, max_attempts=args.max_attempts
        )
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
