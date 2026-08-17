#!/usr/bin/env python3
"""Execute the prospectively frozen expansion capacity completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetExperimentPaths
from dual_uq.workflows.dataset_expansion import DatasetExpansionWave2Config
from dual_uq.workflows.dataset_expansion import run_dataset_expansion_wave2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(project_root=args.project_root)
    config = DatasetExpansionWave2Config(
        output_root_ref=DatasetExperimentPaths.from_project(paths).logical_ref("expansion_wave_2")
    )
    result = run_dataset_expansion_wave2(
        paths, config=config, max_attempts=args.max_attempts
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "capacity_status": result["capacity_status"],
                "write_status": result["write_status"],
                "outputs": result["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
