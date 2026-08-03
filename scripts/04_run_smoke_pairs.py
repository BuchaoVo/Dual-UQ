from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.pairing import build_pair


DEFAULT_PROJECT_ROOT = "/home/zbc/data/AI4S/ProteinDesign/Dual-UQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all PDB–AFDB smoke pairs.")
    parser.add_argument(
        "--targets",
        default="tests/fixtures/manifests/smoke_pairs.tsv",
        help="TSV containing pdb_id, chain_id and uniprot_id.",
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    targets_path = root / args.targets
    targets = pd.read_csv(targets_path, sep="\t")

    required = {"pdb_id", "chain_id", "uniprot_id"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Targets table lacks columns: {sorted(missing)}")

    reports = []
    for row in targets.itertuples(index=False):
        try:
            report = build_pair(
                project_root=root,
                pdb_id=row.pdb_id,
                chain_id=row.chain_id,
                uniprot_id=row.uniprot_id,
            )
            report["run_status"] = "success"
        except Exception as exc:
            report = {
                "pdb_id": row.pdb_id,
                "chain_id": row.chain_id,
                "uniprot_id": row.uniprot_id,
                "run_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        reports.append(report)
        print(json.dumps(report, indent=2))

    output = root / "reports/pair_smoke_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    failed = [r for r in reports if r["run_status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} smoke pair(s) failed. See {output}")

    print(f"Smoke pair summary written to {output}")


if __name__ == "__main__":
    main()
