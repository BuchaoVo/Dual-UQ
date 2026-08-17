"""Characterize disagreement segments as shifts or local deformations."""

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
from dual_uq.schema import (
    RESIDUE_NUMBERING_COLUMNS,
    AmbiguousLegacyResidueIdentifier,
    normalize_residue_mapping,
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
    partial_explicit = (
        set(RESIDUE_NUMBERING_COLUMNS).intersection(prepared.columns)
        and not explicit_author
    )
    if partial_explicit:
        raise AmbiguousLegacyResidueIdentifier(
            "Legacy residue geometry contains a partial explicit residue identity."
        )
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
    if mode == "legacy_alias" and prepared["auth_asym_id"].nunique() != 1:
        raise AmbiguousLegacyResidueIdentifier(
            "Legacy residue numbering must resolve to exactly one chain."
        )
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
    author_key = ["auth_asym_id", "auth_seq_id", "insertion_code"]
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
            raise ValueError("Residue geometry contains a duplicate label residue key.")

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


def author_residue_keys(
    residues: pd.DataFrame,
) -> set[tuple[str, int, str]]:
    return {
        (
            str(row.auth_asym_id),
            int(row.auth_seq_id),
            str(row.insertion_code),
        )
        for row in residues.itertuples(index=False)
    }


def segment_rows_for_interval(
    residues: pd.DataFrame,
    *,
    start_position: int,
    end_position: int,
) -> pd.DataFrame:
    positions = residues["uniprot_residue_number"].astype(int)
    return (
        residues.loc[positions.between(start_position, end_position)]
        .sort_values("uniprot_residue_number", kind="mergesort")
        .copy()
    )


def run_segment_context(
    pair_dir: str | Path,
    *,
    threshold: float,
    min_length: int,
    flank_size: int,
) -> pd.DataFrame:
    pair_dir = Path(pair_dir).expanduser().resolve()

    report = json.loads((pair_dir / "pair_qc.json").read_text(encoding="utf-8"))
    residues, numbering_provenance = prepare_residue_geometry(
        pd.read_parquet(pair_dir / "residue_geometry.parquet")
    )
    author_chains = set(residues["auth_asym_id"].astype(str))
    expected_chain = str(report["chain_id"])
    if author_chains != {expected_chain}:
        raise ValueError(
            "Residue geometry author chain does not match pair_qc chain: "
            f"expected={expected_chain!r}, observed={sorted(author_chains)!r}"
        )
    segments = pd.read_csv(pair_dir / "disagreement_segments.csv")
    pairwise = np.load(pair_dir / "pairwise_geometry.npz")

    selected = segments[
        (segments["threshold"] == threshold)
        & (segments["residue_count"] >= min_length)
    ].copy()
    if selected.empty:
        print("No segments satisfy the requested threshold and minimum length.")
        return pd.DataFrame()

    sasa = residue_sasa_table(report["pdb_path"], report["chain_id"])
    residues = residues.merge(
        sasa,
        on=["auth_asym_id", "auth_seq_id", "insertion_code"],
        how="left",
        validate="many_to_one",
    )
    residues = residues.sort_values(
        "uniprot_residue_number",
        kind="mergesort",
    ).reset_index(drop=True)

    outputs = []
    for row in selected.itertuples(index=False):
        start = int(row.start_position)
        end = int(row.end_position)
        result = characterize_segment(
            residues,
            pairwise,
            start_position=start,
            end_position=end,
            flank_size=flank_size,
        )

        segment_rows = segment_rows_for_interval(
            residues,
            start_position=start,
            end_position=end,
        )
        result["median_pdb_residue_sasa"] = float(
            np.nanmedian(segment_rows["pdb_residue_sasa"])
        )
        result["residue_numbering_mode"] = numbering_provenance["mode"]

        result.update(
            nearest_nonwater_hetero(
                report["pdb_path"],
                report["chain_id"],
                author_residue_keys(segment_rows),
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
    return output_table


def main() -> None:
    args = parse_args()
    run_segment_context(
        args.pair_dir,
        threshold=args.threshold,
        min_length=args.min_length,
        flank_size=args.flank_size,
    )


if __name__ == "__main__":
    main()
