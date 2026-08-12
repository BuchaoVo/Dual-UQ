"""Run frozen Stage-0-2A G2 audit or formal fixed-probe scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.fixed_probe_scoring import (
    FixedProbeScoringError,
    execute_formal_scoring,
    execute_g2_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-python",
        type=Path,
        help="Python executable for the Torch model environment",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="run the frozen G2 audit without formal fixed-probe scoring",
    )
    mode.add_argument(
        "--formal",
        action="store_true",
        help="execute/resume the frozen 30-repeat formal scoring run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="validate and reuse exact formal shards (enabled by default)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = paths.repository_root / config_path
    g2_path = paths.runs_root / "design_baseline/stage0-2a/g2_audit.json"
    try:
        if args.audit_only:
            result = execute_g2_audit(
                config_path=config_path.resolve(),
                project_root=paths.repository_root,
                device_name=args.device,
                output_path=g2_path,
                model_python=(
                    args.model_python.resolve()
                    if args.model_python is not None
                    else None
                ),
            )
        else:
            model_python = (
                args.model_python.resolve()
                if args.model_python is not None
                else Path(sys.executable).resolve()
            )
            result = execute_formal_scoring(
                config_path=config_path.resolve(),
                project_root=paths.repository_root,
                device_name=args.device,
                model_python=model_python,
                worker_path=(
                    paths.repository_root
                    / "scripts/dataset/proteinmpnn_g2_worker.py"
                ),
                g2_audit_path=g2_path,
                run_root=paths.runs_root / "design_baseline/stage0-2a/formal",
                output_root=(
                    paths.repository_root
                    / "experiments/p2_design_baseline/stage0"
                ),
                batch_size=args.batch_size,
                resume=args.resume,
            )
    except FixedProbeScoringError as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "failure_code": exc.code, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    rendered = dict(result)
    for field in ("output_path", "manifest_path"):
        if field in rendered:
            rendered[field] = paths.logical_ref(rendered[field])
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
