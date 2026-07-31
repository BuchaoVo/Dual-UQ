from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from dual_uq.lifecycle import build_candidate_lifecycle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--output-csv",
        default="reports/candidate_lifecycle.csv",
    )
    parser.add_argument(
        "--output-audit",
        default="reports/candidate_lifecycle_audit.json",
    )
    return parser.parse_args(argv)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _atomic_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        return Path(handle.name)


def write_lifecycle_outputs(
    lifecycle: pd.DataFrame,
    audit: dict[str, object],
    *,
    output_csv: Path,
    output_audit: Path,
) -> None:
    csv_temporary = _atomic_path(output_csv)
    audit_temporary = _atomic_path(output_audit)
    try:
        lifecycle.to_csv(csv_temporary, index=False)
        audit_temporary.write_text(
            json.dumps(audit, indent=2),
            encoding="utf-8",
        )
        csv_temporary.replace(output_csv)
        audit_temporary.replace(output_audit)
    finally:
        csv_temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)


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
    output_csv = _resolve(root, args.output_csv)
    output_audit = _resolve(root, args.output_audit)
    write_lifecycle_outputs(
        lifecycle,
        audit,
        output_csv=output_csv,
        output_audit=output_audit,
    )

    print(lifecycle.to_string(index=False))
    print(f"\nSaved: {output_csv}")
    print(f"Saved: {output_audit}")


if __name__ == "__main__":
    main()
