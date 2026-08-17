#!/usr/bin/env python3
"""Render the frozen Stage0 H1 confirmatory audit release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dual_uq.dataset.h1_confirmatory_audit import (
    H1AuditError,
    materialize_h1_confirmatory_audit,
    run_h1_confirmatory_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = args.output_root
    if output_root is None:
        output_root = project_root / "experiments/p2_design_baseline/stage0"
    elif not output_root.is_absolute():
        output_root = project_root / output_root
    try:
        result = run_h1_confirmatory_audit(project_root)
        payload = materialize_h1_confirmatory_audit(result, output_root.resolve())
    except H1AuditError as exc:
        print(
            json.dumps(
                {
                    "status": exc.outcome,
                    "failure_code": exc.code,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                **payload,
                "manifest_path": str(payload["manifest_path"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
