from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.robust_stats import (
    circular_shift_permutation_test,
    contiguous_segments,
    matrix_label_permutation_test,
    safe_spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robust single-pair diagnostics with dependency-aware null tests."
    )
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--local-permutations", type=int, default=1000)
    parser.add_argument("--pair-permutations", type=int, default=500)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.nanmin(values)),
        "q10": float(np.nanquantile(values, 0.10)),
        "q25": float(np.nanquantile(values, 0.25)),
        "median": float(np.nanmedian(values)),
        "q75": float(np.nanquantile(values, 0.75)),
        "q90": float(np.nanquantile(values, 0.90)),
        "max": float(np.nanmax(values)),
    }


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir).expanduser().resolve()

    residue_path = pair_dir / "residue_geometry.parquet"
    pairwise_path = pair_dir / "pairwise_geometry.npz"
    if not residue_path.exists() or not pairwise_path.exists():
        raise FileNotFoundError(
            "Run scripts/06_analyze_pair_geometry.py before robust diagnostics."
        )

    residues = pd.read_parquet(residue_path).sort_values(
        "uniprot_residue_number"
    ).reset_index(drop=True)
    pairwise = np.load(pairwise_path)

    positions = residues["uniprot_residue_number"].astype(int).to_numpy()
    plddt = residues["plddt"].to_numpy(dtype=float)
    uncertainty = 100.0 - plddt
    local_disagreement = residues["aligned_ca_distance"].to_numpy(dtype=float)

    local_test = circular_shift_permutation_test(
        uncertainty,
        local_disagreement,
        n_permutations=args.local_permutations,
        seed=args.seed,
    )
    pair_test = matrix_label_permutation_test(
        pairwise["symmetric_pae"],
        pairwise["absolute_pairwise_error"],
        pairwise["uniprot_positions"],
        min_sequence_separation=args.min_separation,
        n_permutations=args.pair_permutations,
        seed=args.seed,
    )

    segments = []
    for threshold in (1.0, 2.0):
        segments.extend(
            contiguous_segments(
                positions,
                local_disagreement,
                plddt,
                threshold=threshold,
            )
        )
    segment_table = pd.DataFrame(segments)
    segment_path = pair_dir / "disagreement_segments.csv"
    segment_table.to_csv(segment_path, index=False)

    separation = np.abs(positions[:, None] - positions[None, :])
    upper = np.triu(np.ones_like(separation, dtype=bool), k=1)
    pair_error = pairwise["absolute_pairwise_error"]
    symmetric_pae = pairwise["symmetric_pae"]

    bins = [
        ("6-11", 6, 12),
        ("12-23", 12, 24),
        ("24-47", 24, 48),
        ("48-95", 48, 96),
        ("96+", 96, None),
    ]
    strata = []
    for label, lower, upper_bound in bins:
        mask = upper & (separation >= lower)
        if upper_bound is not None:
            mask &= separation < upper_bound
        result = safe_spearman(symmetric_pae[mask], pair_error[mask])
        strata.append(
            {
                "sequence_separation_bin": label,
                **result.as_dict(),
                "median_pae": float(np.nanmedian(symmetric_pae[mask])),
                "median_pairwise_error": float(np.nanmedian(pair_error[mask])),
            }
        )

    strata_table = pd.DataFrame(strata)
    strata_path = pair_dir / "pairwise_strata.csv"
    strata_table.to_csv(strata_path, index=False)

    high_confidence_disagreement = residues[
        (residues["plddt"] >= 90.0)
        & (residues["aligned_ca_distance"] >= 1.0)
    ][
        [
            "uniprot_residue_number",
            "pdb_residue_number",
            "pdb_residue_one_letter",
            "plddt",
            "aligned_ca_distance",
        ]
    ].copy()
    high_conf_path = pair_dir / "high_confidence_disagreement.csv"
    high_confidence_disagreement.to_csv(high_conf_path, index=False)

    summary = {
        "pair_dir": str(pair_dir),
        "residue_count": int(len(residues)),
        "plddt_distribution": quantiles(plddt),
        "local_disagreement_distribution": quantiles(local_disagreement),
        "local_plddt_disagreement_test": local_test,
        "pae_pairwise_error_test": pair_test,
        "high_confidence_disagreement_count": int(len(high_confidence_disagreement)),
        "high_confidence_disagreement_definition": "pLDDT >= 90 and aligned CA distance >= 1 A",
        "largest_segment_residue_count_at_1A": (
            int(segment_table.loc[segment_table["threshold"] == 1.0, "residue_count"].max())
            if not segment_table.empty and (segment_table["threshold"] == 1.0).any()
            else 0
        ),
        "interpretation_guardrails": [
            "A low permutation p-value is evidence of association, not proof that AFDB is wrong.",
            "All residue pairs within one protein are dependent; use matrix-label permutation rather than naive pairwise p-values.",
            "High pLDDT with large PDB-AFDB disagreement may indicate alternative conformational or experimental states.",
            "One protein cannot calibrate pLDDT or PAE for the full benchmark.",
        ],
        "outputs": {
            "segments": str(segment_path),
            "pairwise_strata": str(strata_path),
            "high_confidence_disagreement": str(high_conf_path),
        },
    }

    output = pair_dir / "robust_pair_diagnostics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
