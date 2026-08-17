#!/usr/bin/env python3
"""Execute the prospectively frozen source-frame expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.workflows.dataset_expansion import (
    DatasetExpansionConfig,
    build_expansion_freeze,
    materialize_expansion_freeze,
    run_dataset_expansion,
    validate_expansion_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Validate inputs and freeze/hash amendment plus declaration without acquisition.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    paths = ProjectPaths.discover(project_root=args.project_root)
    config = DatasetExpansionConfig(
        output_root_ref=DatasetExperimentPaths.from_project(paths).logical_ref("expansion_wave_1")
    )
    if args.freeze_only:
        frozen = build_expansion_freeze(validate_expansion_inputs(paths, config))
        result = materialize_expansion_freeze(frozen, paths, config=config)
    else:
        result = run_dataset_expansion(
            paths, config=config, max_attempts=args.max_attempts
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
