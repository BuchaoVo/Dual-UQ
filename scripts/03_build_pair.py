from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.pairing import build_pair


DEFAULT_PROJECT_ROOT = "/home/zbc/data/AI4S/ProteinDesign/Dual-UQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and register one PDB–AlphaFoldDB residue-mapped pair."
    )
    parser.add_argument("--pdb-id", required=True)
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--uniprot-id", required=True)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--min-mapping-coverage", type=float, default=0.90)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    report = build_pair(
        project_root=root,
        pdb_id=args.pdb_id,
        chain_id=args.chain_id,
        uniprot_id=args.uniprot_id,
        min_mapping_coverage=args.min_mapping_coverage,
        min_sequence_identity=args.min_sequence_identity,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
