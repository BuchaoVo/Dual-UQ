from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dual_uq.a0_selection import (
    MINIMUM_COMPLETE_QUALITY_PASS_DIAGNOSTICS,
    PRIMARY_CATEGORIES,
    select_final_a0_panel,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit gates and deterministically select the final A0 panel."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--summary",
        default="reports/a0_candidate_summary.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/legacy/a0_screening/a0_selection.yaml",
    )
    parser.add_argument(
        "--panel",
        default="data/manifests/a0_final_panel.tsv",
    )
    parser.add_argument(
        "--audit",
        default="reports/a0_final_selection_audit.json",
    )
    return parser.parse_args(argv)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _load_selection_config(path: Path) -> dict[str, int]:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A0 selection config must be a mapping")
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise TypeError("A0 selection config missing selection mapping")
    requirements = selection.get("target_per_primary_category")
    if not isinstance(requirements, dict):
        raise TypeError("A0 selection config missing category requirements")
    normalized = {
        category: int(requirements[category])
        for category in PRIMARY_CATEGORIES
    }
    target_total = int(selection.get("target_total", 0))
    if target_total != sum(normalized.values()):
        raise ValueError(
            "selection target_total must equal category requirement sum"
        )
    return normalized


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    summary_path = _resolve(root, args.summary)
    config_path = _resolve(root, args.config)
    panel_path = _resolve(root, args.panel)
    audit_path = _resolve(root, args.audit)

    summary = pd.read_csv(summary_path)
    requirements = _load_selection_config(config_path)
    result = select_final_a0_panel(
        summary,
        minimum_complete_diagnostics=(
            MINIMUM_COMPLETE_QUALITY_PASS_DIAGNOSTICS
        ),
        category_requirements=requirements,
    )

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    panel_temporary = _atomic_temporary(panel_path)
    audit_temporary = _atomic_temporary(audit_path)
    result.panel.to_csv(panel_temporary, sep="\t", index=False)
    audit_temporary.write_text(
        json.dumps(
            result.audit,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    panel_temporary.replace(panel_path)
    audit_temporary.replace(audit_path)

    print(result.panel.to_string(index=False))
    print("\nFinal A0 selection audit")
    print(json.dumps(result.audit, indent=2, sort_keys=True))
    print(f"\nPanel: {panel_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
