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
from dual_uq.schema import (
    RESIDUE_NUMBERING_COLUMNS,
    AmbiguousLegacyResidueIdentifier,
    normalize_residue_mapping,
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


def prepare_residue_geometry(
    residues: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prepared = residues.copy()
    if (
        "aligned_ca_distance" not in prepared
        and "ca_disagreement" in prepared
    ):
        prepared["aligned_ca_distance"] = prepared["ca_disagreement"]

    required_values = {
        "uniprot_residue_number",
        "plddt",
        "aligned_ca_distance",
    }
    missing_values = sorted(required_values - set(prepared.columns))
    if missing_values:
        raise ValueError(
            "Residue geometry missing required columns: "
            + ",".join(missing_values)
        )

    explicit_author = {
        "auth_asym_id",
        "auth_seq_id",
    }.issubset(prepared.columns)
    if explicit_author:
        missing_numbering = sorted(
            set(RESIDUE_NUMBERING_COLUMNS) - set(prepared.columns)
        )
        if missing_numbering:
            raise ValueError(
                "Explicit residue geometry missing numbering columns: "
                + ",".join(missing_numbering)
            )
        mode = "explicit_auth_label"
    else:
        if "pdb_residue_number" not in prepared:
            raise AmbiguousLegacyResidueIdentifier(
                "Residue geometry has neither explicit author numbering "
                "nor a legacy pdb_residue_number."
            )
        chain_columns = [
            column
            for column in ("pdb_chain_id", "chain_id")
            if column in prepared
        ]
        if not chain_columns or prepared[chain_columns].isna().all(axis=1).any():
            raise AmbiguousLegacyResidueIdentifier(
                "Legacy residue numbering requires an unambiguous chain alias."
            )
        mode = "legacy_alias"

    prepared = normalize_residue_mapping(prepared)
    uniprot = pd.to_numeric(
        prepared["uniprot_residue_number"],
        errors="coerce",
    )
    invalid_uniprot = uniprot.isna() | uniprot.mod(1).ne(0)
    if invalid_uniprot.any():
        raise ValueError("Residue geometry contains invalid UniProt positions.")
    prepared["uniprot_residue_number"] = uniprot.astype(int)
    if prepared["uniprot_residue_number"].duplicated().any():
        raise ValueError("Residue geometry contains duplicate UniProt positions.")

    if prepared["auth_asym_id"].isna().any() or prepared["auth_seq_id"].isna().any():
        raise AmbiguousLegacyResidueIdentifier(
            "Residue geometry contains an incomplete author residue identity."
        )
    author_key = [
        "auth_asym_id",
        "auth_seq_id",
        "insertion_code",
    ]
    if prepared.duplicated(author_key).any():
        raise ValueError("Residue geometry contains a duplicate author residue key.")

    if mode == "explicit_auth_label":
        if (
            prepared["label_asym_id"].isna().any()
            or prepared["label_seq_id"].isna().any()
        ):
            raise ValueError(
                "Explicit residue geometry contains an incomplete label identity."
            )
        if prepared.duplicated(["label_asym_id", "label_seq_id"]).any():
            raise ValueError(
                "Residue geometry contains a duplicate label residue key."
            )

    prepared = prepared.sort_values(
        "uniprot_residue_number",
        kind="mergesort",
    ).reset_index(drop=True)
    provenance = {
        "mode": mode,
        "residue_count": len(prepared),
        "join_modes": sorted(
            prepared["residue_join_mode"].dropna().astype(str).unique().tolist()
        )
        if "residue_join_mode" in prepared
        else [],
    }
    return prepared, provenance


def run_robust_diagnostics(
    pair_dir: str | Path,
    *,
    local_permutations: int,
    pair_permutations: int,
    min_separation: int,
    seed: int,
) -> dict[str, object]:
    pair_dir = Path(pair_dir).expanduser().resolve()

    residue_path = pair_dir / "residue_geometry.parquet"
    pairwise_path = pair_dir / "pairwise_geometry.npz"
    if not residue_path.exists() or not pairwise_path.exists():
        raise FileNotFoundError(
            "Run scripts/06_analyze_pair_geometry.py before robust diagnostics."
        )

    residues, numbering_provenance = prepare_residue_geometry(
        pd.read_parquet(residue_path)
    )
    pairwise = np.load(pairwise_path)

    positions = residues["uniprot_residue_number"].astype(int).to_numpy()
    plddt = residues["plddt"].to_numpy(dtype=float)
    uncertainty = 100.0 - plddt
    local_disagreement = residues["aligned_ca_distance"].to_numpy(dtype=float)

    local_test = circular_shift_permutation_test(
        uncertainty,
        local_disagreement,
        n_permutations=local_permutations,
        seed=seed,
    )
    pair_test = matrix_label_permutation_test(
        pairwise["symmetric_pae"],
        pairwise["absolute_pairwise_error"],
        pairwise["uniprot_positions"],
        min_sequence_separation=min_separation,
        n_permutations=pair_permutations,
        seed=seed,
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

    provenance_columns = [
        column
        for column in (
            *RESIDUE_NUMBERING_COLUMNS,
            "residue_mapping_provenance",
            "residue_join_mode",
        )
        if column in residues
    ]
    descriptive_columns = [
        column
        for column in ("pdb_residue_one_letter",)
        if column in residues
    ]
    high_confidence_disagreement = residues[
        (residues["plddt"] >= 90.0)
        & (residues["aligned_ca_distance"] >= 1.0)
    ][
        [
            "uniprot_residue_number",
            *provenance_columns,
            *descriptive_columns,
            "plddt",
            "aligned_ca_distance",
        ]
    ].copy()
    high_conf_path = pair_dir / "high_confidence_disagreement.csv"
    high_confidence_disagreement.to_csv(high_conf_path, index=False)

    summary = {
        "pair_dir": str(pair_dir),
        "residue_count": len(residues),
        "plddt_distribution": quantiles(plddt),
        "local_disagreement_distribution": quantiles(local_disagreement),
        "local_plddt_disagreement_test": local_test,
        "pae_pairwise_error_test": pair_test,
        "residue_numbering_provenance": numbering_provenance,
        "high_confidence_disagreement_count": len(high_confidence_disagreement),
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
    return summary


def main() -> None:
    args = parse_args()
    summary = run_robust_diagnostics(
        args.pair_dir,
        local_permutations=args.local_permutations,
        pair_permutations=args.pair_permutations,
        min_separation=args.min_separation,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
