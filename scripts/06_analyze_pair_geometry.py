from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.pair_geometry import analyze_pair_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse AFDB confidence, align PDB–AFDB CA atoms and quantify disagreement."
    )
    parser.add_argument(
        "--pair-report",
        required=True,
        help="Path to data/processed/pairs/<pair>/pair_qc.json",
    )
    parser.add_argument("--high-confidence-threshold", type=float, default=70.0)
    parser.add_argument("--pair-min-separation", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.pair_report).expanduser().resolve()
    summary = analyze_pair_geometry(
        report_path,
        high_confidence_threshold=args.high_confidence_threshold,
        pair_min_sequence_separation=args.pair_min_separation,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
