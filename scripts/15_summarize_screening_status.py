from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the latest screening status per pair.")
    parser.add_argument("--project-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    path = root / "reports/screening_pool_status.jsonl"
    if not path.exists():
        print("No screening status log found.")
        return

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    table = pd.DataFrame(rows)
    latest = (
        table.sort_index()
        .drop_duplicates(subset=["screening_index", "pair_name"], keep="last")
        .sort_values("screening_index")
    )
    columns = [
        "screening_index",
        "pair_name",
        "provisional_stratum",
        "status",
        "failed_log",
    ]
    columns = [column for column in columns if column in latest.columns]
    print(latest[columns].to_string(index=False))

    output = root / "reports/screening_pool_status_latest.csv"
    latest.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
