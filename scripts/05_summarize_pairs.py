from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/zbc/data/AI4S/ProteinDesign/Dual-UQ")


def main() -> None:
    protein_path = PROJECT_ROOT / "data/manifests/protein_manifest.parquet"
    structure_path = PROJECT_ROOT / "data/manifests/structure_manifest.parquet"

    proteins = pd.read_parquet(protein_path)
    structures = pd.read_parquet(structure_path)

    if proteins.empty:
        print("No protein pairs registered.")
        return

    summary = proteins[
        [
            "protein_id",
            "uniprot_id",
            "pdb_id",
            "chain_id",
            "length",
            "mapping_coverage",
            "sequence_identity",
            "quality_flag",
        ]
    ].copy()
    structure_counts = structures.groupby("protein_id").size().rename("structure_count")
    summary = summary.merge(structure_counts, on="protein_id", how="left")
    summary = summary.sort_values(["quality_flag", "mapping_coverage"], ascending=[True, False])

    print(summary.to_string(index=False))
    output = PROJECT_ROOT / "reports/pair_manifest_summary.csv"
    summary.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
