"""Freeze the fixed-probe space and static scoring protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.dataset.scale1b_protocol_freeze import (
    PROTOCOL_FREEZE_COMPLETE,
    Scale1BProtocolFreezeConfig,
    Scale1BProtocolFreezeError,
    run_scale1b_protocol_freeze,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic scoring-protocol path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("scoring_protocol_release")
        result = run_scale1b_protocol_freeze(
            paths,
            config=Scale1BProtocolFreezeConfig(output_root_ref=output_root),
        )
    except Scale1BProtocolFreezeError as exc:
        print(
            json.dumps(
                {
                    "protocol_status": exc.status,
                    "failure_code": exc.code,
                    "message": str(exc),
                    "proteinmpnn_forward_executions": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    except ProjectPathError as exc:
        print(
            json.dumps(
                {
                    "protocol_status": "SCALE1B_BLOCKED_INPUT_INTEGRITY",
                    "failure_code": "project_root_unresolved",
                    "message": str(exc),
                    "proteinmpnn_forward_executions": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["protocol_status"] == PROTOCOL_FREEZE_COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
