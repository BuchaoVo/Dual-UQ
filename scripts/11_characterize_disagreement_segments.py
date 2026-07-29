from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.context_analysis import (
    characterize_segment,
    nearest_nonwater_hetero,
    residue_sasa_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize whether disagreement segments are rigid shifts or local deformations."
    )
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--flank-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir).expanduser().resolve()

    report = json.loads((pair_dir / "pair_qc.json").read_text(encoding="utf-8"))
    residues = pd.read_parquet(pair_dir / "residue_geometry.parquet")
    segments = pd.read_csv(pair_dir / "disagreement_segments.csv")
    pairwise = np.load(pair_dir / "pairwise_geometry.npz")

    selected = segments[
        (segments["threshold"] == args.threshold)
        & (segments["residue_count"] >= args.min_length)
    ].copy()
    if selected.empty:
        print("No segments satisfy the requested threshold and minimum length.")
        return

    sasa = residue_sasa_table(report["pdb_path"], report["chain_id"])
    residues = residues.copy()
    residues["pdb_residue_number_norm"] = residues["pdb_residue_number"].astype(str).str.strip()
    residues = residues.merge(
        sasa,
        on="pdb_residue_number_norm",
        how="left",
        validate="many_to_one",
    )

    outputs = []
    for row in selected.itertuples(index=False):
        start = int(row.start_position)
        end = int(row.end_position)
        result = characterize_segment(
            residues,
            pairwise,
            start_position=start,
            end_position=end,
            flank_size=args.flank_size,
        )

        segment_rows = residues[
            residues["uniprot_residue_number"].astype(int).between(start, end)
        ]
        result["median_pdb_residue_sasa"] = float(
            np.nanmedian(segment_rows["pdb_residue_sasa"])
        )

        residue_numbers = set(
            segment_rows["pdb_residue_number"].astype(str).str.strip()
        )
        result.update(
            nearest_nonwater_hetero(
                report["pdb_path"],
                report["chain_id"],
                residue_numbers,
            )
        )
        outputs.append(result)

    output_table = pd.DataFrame(outputs)
    output_path = pair_dir / "segment_context.csv"
    output_table.to_csv(output_path, index=False)

    summary_path = pair_dir / "segment_context.json"
    summary_path.write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )

    print(output_table.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
