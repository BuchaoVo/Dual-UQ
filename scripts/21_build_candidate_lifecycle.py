from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.lifecycle import build_candidate_lifecycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited candidate lifecycle table.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--pool", default="data/manifests/screening_pool.tsv")
    parser.add_argument(
        "--preflight",
        default="data/manifests/screening_pool_preflight.tsv",
    )
    parser.add_argument(
        "--status",
        default="reports/screening_pool_status_latest.csv",
    )
    parser.add_argument(
        "--candidate-summary",
        default="reports/a0_candidate_summary.csv",
    )
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    pool = pd.read_csv(_resolve(root, args.pool), sep="\t")
    preflight = pd.read_csv(_resolve(root, args.preflight), sep="\t")
    status_path = _resolve(root, args.status)
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    summary_path = _resolve(root, args.candidate_summary)
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    lifecycle, audit = build_candidate_lifecycle(
        pool=pool,
        preflight=preflight,
        status=status,
        pair_root=root / "data/processed/pairs",
        candidate_summary=summary,
    )
    output_csv = root / "reports/candidate_lifecycle.csv"
    output_json = root / "reports/candidate_lifecycle_audit.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(lifecycle.to_string(index=False))
    print(f"\nSaved: {output_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
