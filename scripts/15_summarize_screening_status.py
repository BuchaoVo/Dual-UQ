from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dual_uq.screening_runner import summarize_status_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize structured screening stage history."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--status-path", default=None)
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: ignoring invalid JSONL line {line_number}.")
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    path = (
        Path(args.status_path).expanduser().resolve()
        if args.status_path
        else root / "reports/screening_pool_status.jsonl"
    )
    if not path.exists():
        print("No screening status log found.")
        return

    summary, statistics = summarize_status_records(_read_records(path))
    print(summary.to_string(index=False))
    print("\nStatistics:")
    print(json.dumps(statistics, indent=2, sort_keys=True))

    output = root / "reports/screening_pool_status_latest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
