"""Analyze frozen Stage0-2A local amino-acid decision sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.fixed_probe_decision_sensitivity import (
    STAGE0_ROOT,
    DecisionSensitivityError,
    materialize_fixed_probe_decision_sensitivity,
    run_fixed_probe_decision_sensitivity,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    try:
        result = run_fixed_probe_decision_sensitivity(paths.repository_root)
        report = materialize_fixed_probe_decision_sensitivity(
            result,
            paths.repository_root / STAGE0_ROOT,
        )
    except DecisionSensitivityError as exc:
        print(
            json.dumps(
                {
                    "status": exc.outcome,
                    "failure_code": exc.code,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    rendered = dict(report)
    rendered["manifest_path"] = paths.logical_ref(report["manifest_path"])
    print(json.dumps(rendered, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
