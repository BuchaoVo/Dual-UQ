#!/usr/bin/env python3
"""CLI adapter for the frozen Stage0-2A boundary representation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.dataset.fixed_probe_boundary_audit import (
    RepresentationAuditError,
    materialize_fixed_probe_boundary_audit,
    run_fixed_probe_boundary_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/p2_design_baseline/stage0"),
    )
    args = parser.parse_args()
    try:
        result = run_fixed_probe_boundary_audit(args.project_root)
        payload = materialize_fixed_probe_boundary_audit(result, args.output_root.resolve())
    except RepresentationAuditError as exc:
        print(json.dumps({"status": exc.outcome, "code": exc.code, "message": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "conclusion": payload["conclusion"],
                "manifest_path": str(payload["manifest_path"]),
                "write_status": payload["write_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
