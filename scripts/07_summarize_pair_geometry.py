from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all pair geometry QC files.")
    parser.add_argument("--project-root", required=True)
    return parser.parse_args()


def nested_value(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return None


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    rows = []
    for path in sorted((root / "data/processed/pairs").glob("*/pair_geometry_qc.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "protein_id": record["protein_id"],
                "pdb_id": record["pdb_id"],
                "chain_id": record["chain_id"],
                "uniprot_id": record["uniprot_id"],
                "mapped_ca_count": record["mapped_ca_count"],
                "mapped_ca_coverage": record["mapped_ca_coverage"],
                "alignment_mode": record["alignment_mode"],
                "all_mapped_fit_rmsd": record["all_mapped_fit_rmsd"],
                "high_confidence_fit_rmsd": record["high_confidence_fit_rmsd"],
                "median_aligned_ca_distance": record["median_aligned_ca_distance"],
                "p90_aligned_ca_distance": record["p90_aligned_ca_distance"],
                "plddt_disagreement_rho": nested_value(
                    record["plddt_vs_local_disagreement_spearman"], "rho"
                ),
                "pae_pair_error_rho": nested_value(
                    record["symmetric_pae_vs_pairwise_error_spearman"], "rho"
                ),
            }
        )

    if not rows:
        print("No pair_geometry_qc.json files found.")
        return

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    output = root / "reports/pair_geometry_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
