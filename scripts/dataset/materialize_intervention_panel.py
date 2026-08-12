"""Materialize the frozen Stage-0 intervention panel from prior human review."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.releases.intervention_panel import materialize_intervention_panel


def _git_value(repository_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    root = paths.repository_root
    dirty_output = _git_value(root, "status", "--porcelain")
    result = materialize_intervention_panel(
        repository_root=root,
        source_path=root / "docs/design/DATASET_SCALE_AND_VALIDATION_PLAN_V1.md",
        inventory_path=(
            root / "artifacts/dataset/reports/census/candidate_inventory_v1.json"
        ),
        binding_path=root / "configs/experiments/design_baseline/stage0.yaml",
        panel_path=(
            root
            / "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_panel_v1.jsonl"
        ),
        metadata_path=(
            root
            / "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_panel_v1.meta.json"
        ),
        git_commit=_git_value(root, "rev-parse", "HEAD"),
        git_dirty=None if dirty_output is None else bool(dirty_output),
    )
    print(
        json.dumps(
            {
                "panel_path": result["panel_path"],
                "panel_sha256": result["metadata"]["panel_jsonl_sha256"],
                "panel_write_status": result["panel_write_status"],
                "record_count": len(result["records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
