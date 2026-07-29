from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Steps 1–4 over a selected screening pool with resume support."
    )
    parser.add_argument(
        "--config",
        default="configs/screening_pool.yaml",
    )
    parser.add_argument(
        "--pool",
        default="data/manifests/screening_pool.tsv",
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated screening indices, e.g. 1,4,7.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    ended = datetime.now(timezone.utc)
    return {
        "command": command,
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "runtime_seconds": (ended - started).total_seconds(),
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(args.project_root or config["project_root"]).expanduser().resolve()

    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = root / pool_path
    pool = pd.read_csv(pool_path, sep="\t")

    if args.only:
        requested = {int(value) for value in args.only.split(",") if value.strip()}
        pool = pool[pool["screening_index"].astype(int).isin(requested)]
    else:
        pool = pool[pool["screening_index"].astype(int) >= args.start_index]
        if args.limit is not None:
            pool = pool.head(args.limit)

    batch_cfg = config["batch"]
    status_path = root / "reports/screening_pool_status.jsonl"
    python = sys.executable

    for row in pool.itertuples(index=False):
        index = int(row.screening_index)
        pdb_id = str(row.pdb_id).lower()
        chain_id = str(row.chain_id)
        uniprot_id = str(row.uniprot_id).upper()
        pair_name = f"{pdb_id}_{chain_id}__{uniprot_id}"
        pair_dir = root / "data/processed/pairs" / pair_name
        final_output = pair_dir / "robust_pair_diagnostics.json"

        record: dict[str, Any] = {
            "screening_index": index,
            "pair_name": pair_name,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "uniprot_id": uniprot_id,
            "provisional_stratum": getattr(row, "provisional_stratum", None),
            "status": "pending",
            "steps": [],
        }

        if final_output.exists() and not args.force:
            record["status"] = "skipped_complete"
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[{index}] {pair_name}: already complete")
            continue

        commands = [
            (
                "pair",
                [
                    python,
                    "scripts/03_build_pair.py",
                    "--pdb-id",
                    pdb_id,
                    "--chain-id",
                    chain_id,
                    "--uniprot-id",
                    uniprot_id,
                    "--project-root",
                    str(root),
                ],
            ),
            (
                "geometry",
                [
                    python,
                    "scripts/06_analyze_pair_geometry.py",
                    "--pair-report",
                    str(pair_dir / "pair_qc.json"),
                    "--pair-min-separation",
                    str(batch_cfg["pair_min_separation"]),
                ],
            ),
            (
                "robust",
                [
                    python,
                    "scripts/09_robust_pair_diagnostics.py",
                    "--pair-dir",
                    str(pair_dir),
                    "--local-permutations",
                    str(batch_cfg["local_permutations"]),
                    "--pair-permutations",
                    str(batch_cfg["pair_permutations"]),
                    "--min-separation",
                    str(batch_cfg["pair_min_separation"]),
                ],
            ),
            (
                "segment_context",
                [
                    python,
                    "scripts/11_characterize_disagreement_segments.py",
                    "--pair-dir",
                    str(pair_dir),
                    "--threshold",
                    str(batch_cfg["segment_threshold"]),
                    "--min-length",
                    str(batch_cfg["segment_min_length"]),
                    "--flank-size",
                    str(batch_cfg["segment_flank_size"]),
                ],
            ),
        ]

        failed = False
        for step_name, command in commands:
            log_path = root / "reports/screening_logs" / f"{index:03d}_{pair_name}_{step_name}.log"
            result = run_command(command, cwd=root, log_path=log_path)
            result["step"] = step_name
            record["steps"].append(result)
            if result["returncode"] != 0:
                record["status"] = f"failed:{step_name}"
                record["failed_log"] = str(log_path)
                failed = True
                print(f"[{index}] {pair_name}: FAILED at {step_name}; see {log_path}")
                break

        if not failed:
            record["status"] = "complete"
            print(f"[{index}] {pair_name}: complete")

        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if failed and not bool(batch_cfg["continue_on_error"]):
            raise SystemExit(1)

    summary_command = [
        python,
        "scripts/12_build_a0_candidate_summary.py",
        "--project-root",
        str(root),
    ]
    subprocess.run(summary_command, cwd=root, check=False)
    print(f"\nStatus log: {status_path}")


if __name__ == "__main__":
    main()
