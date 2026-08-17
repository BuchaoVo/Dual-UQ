"""Freeze the final outcome-blind confirmatory cohort and scoring protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.workflows.final_confirmatory_protocol import (
    FINAL_PROTOCOL_FROZEN,
    FinalConfirmatoryProtocolConfig,
    FinalConfirmatoryProtocolError,
    run_final_confirmatory_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Canonical output directory (defaults to the semantic confirmatory release path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.discover(project_root=args.project_root)
        output_root = args.output_root or DatasetExperimentPaths.from_project(paths).logical_ref("confirmatory_release")
        result = run_final_confirmatory_protocol(
            paths,
            FinalConfirmatoryProtocolConfig(output_root_ref=output_root),
        )
    except FinalConfirmatoryProtocolError as exc:
        print(
            json.dumps(
                {
                    "status": exc.status,
                    "failure_code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
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
                    "status": "BLOCKED_INPUT_INTEGRITY",
                    "failure_code": "project_root_unresolved",
                    "message": str(exc),
                    "proteinmpnn_forward_executions": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == FINAL_PROTOCOL_FROZEN else 2


if __name__ == "__main__":
    raise SystemExit(main())
