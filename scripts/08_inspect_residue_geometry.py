from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the largest local PDB–AFDB disagreements."
    )
    parser.add_argument("--residue-table", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(Path(args.residue_table))
    columns = [
        "uniprot_residue_number",
        "pdb_residue_number",
        "pdb_residue_one_letter",
        "afdb_residue_one_letter",
        "plddt",
        "aligned_ca_distance",
        "is_high_confidence",
    ]
    output = table.sort_values("aligned_ca_distance", ascending=False).head(args.top_k)
    print(output[columns].to_string(index=False))


if __name__ == "__main__":
    main()
