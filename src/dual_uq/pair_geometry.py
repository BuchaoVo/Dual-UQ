from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .confidence import load_pae, load_plddt
from .geometry import kabsch_align, pairwise_distances, rmsd
from .structure_io import join_residue_mapping_to_ca, load_chain_ca_table


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return {"rho": None, "pvalue": None, "n": int(mask.sum())}
    result = spearmanr(x[mask], y[mask])
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": int(mask.sum()),
    }


def analyze_pair_geometry(
    pair_report_path: str | Path,
    *,
    high_confidence_threshold: float = 70.0,
    pair_min_sequence_separation: int = 6,
) -> dict[str, Any]:
    report_path = Path(pair_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pair_dir = report_path.parent

    mapping = pd.read_parquet(report["mapping_path"]).copy()

    pdb_table = load_chain_ca_table(report["pdb_path"], report["chain_id"])
    mapped_pdb, join_diagnostics = join_residue_mapping_to_ca(mapping, pdb_table)

    afdb_table = load_chain_ca_table(report["afdb_model_path"], chain_id="A")
    afdb_table = afdb_table.rename(
        columns={
            "residue_number_int": "uniprot_residue_number",
            "residue_one_letter": "afdb_residue_one_letter",
            "x": "afdb_x",
            "y": "afdb_y",
            "z": "afdb_z",
            "bfactor": "afdb_bfactor",
        }
    )

    pdb_keep = mapped_pdb.rename(
        columns={
            "residue_one_letter": "pdb_residue_one_letter",
            "x": "pdb_x",
            "y": "pdb_y",
            "z": "pdb_z",
            "bfactor": "pdb_bfactor",
        }
    )[
        [
            "uniprot_residue_number",
            "auth_asym_id",
            "label_asym_id",
            "auth_seq_id",
            "label_seq_id",
            "insertion_code",
            "residue_join_mode",
            "pdb_residue_one_letter",
            "pdb_x",
            "pdb_y",
            "pdb_z",
            "pdb_bfactor",
        ]
    ]

    merged = pdb_keep.merge(
        afdb_table[
            [
                "uniprot_residue_number",
                "afdb_residue_one_letter",
                "afdb_x",
                "afdb_y",
                "afdb_z",
                "afdb_bfactor",
            ]
        ],
        on="uniprot_residue_number",
        how="inner",
        validate="many_to_one",
    )
    merged = merged.sort_values("uniprot_residue_number").reset_index(drop=True)

    if len(merged) < 3:
        raise ValueError(f"Only {len(merged)} mapped CA pairs found.")

    sequence_length = int(report["uniprot_length"])
    plddt = load_plddt(report["plddt_path"], expected_length=sequence_length)
    pae = load_pae(report["pae_path"], expected_length=sequence_length)

    indices = merged["uniprot_residue_number"].astype(int).to_numpy() - 1
    if indices.min() < 0 or indices.max() >= sequence_length:
        raise IndexError("Mapped UniProt residue indices fall outside AFDB sequence length.")

    merged["plddt"] = plddt[indices]
    merged["plddt_bfactor_delta"] = merged["plddt"] - merged["afdb_bfactor"]

    pdb_coordinates = merged[["pdb_x", "pdb_y", "pdb_z"]].to_numpy(dtype=float)
    afdb_coordinates = merged[["afdb_x", "afdb_y", "afdb_z"]].to_numpy(dtype=float)

    aligned_all, rotation_all, translation_all = kabsch_align(
        afdb_coordinates, pdb_coordinates
    )
    all_rmsd = rmsd(aligned_all, pdb_coordinates)

    high_conf_mask = merged["plddt"].to_numpy(dtype=float) >= high_confidence_threshold
    if high_conf_mask.sum() >= 3:
        _, rotation_core, translation_core = kabsch_align(
            afdb_coordinates[high_conf_mask],
            pdb_coordinates[high_conf_mask],
        )
        aligned_core = afdb_coordinates @ rotation_core + translation_core
        core_fit_rmsd = rmsd(
            aligned_core[high_conf_mask],
            pdb_coordinates[high_conf_mask],
        )
        alignment_mode = "high_confidence_core"
    else:
        aligned_core = aligned_all
        rotation_core = rotation_all
        translation_core = translation_all
        core_fit_rmsd = all_rmsd
        alignment_mode = "all_mapped"

    per_residue_distance = np.sqrt(
        np.sum((aligned_core - pdb_coordinates) ** 2, axis=1)
    )
    merged["aligned_ca_distance"] = per_residue_distance
    merged["is_high_confidence"] = high_conf_mask
    merged["pdb_aa_matches_afdb"] = (
        merged["pdb_residue_one_letter"] == merged["afdb_residue_one_letter"]
    )

    pdb_pairwise = pairwise_distances(pdb_coordinates)
    afdb_pairwise = pairwise_distances(afdb_coordinates)
    pairwise_error = np.abs(pdb_pairwise - afdb_pairwise)
    pae_submatrix = pae[np.ix_(indices, indices)]
    symmetric_pae = 0.5 * (pae_submatrix + pae_submatrix.T)

    n = len(merged)
    row_idx, col_idx = np.triu_indices(n, k=pair_min_sequence_separation)
    pair_errors = pairwise_error[row_idx, col_idx]
    pair_pae = symmetric_pae[row_idx, col_idx]
    pair_correlation = _safe_spearman(pair_pae, pair_errors)
    local_correlation = _safe_spearman(
        100.0 - merged["plddt"].to_numpy(dtype=float),
        per_residue_distance,
    )

    residue_output = pair_dir / "residue_geometry.parquet"
    merged.to_parquet(residue_output, index=False)

    pairwise_output = pair_dir / "pairwise_geometry.npz"
    np.savez_compressed(
        pairwise_output,
        uniprot_positions=merged["uniprot_residue_number"].astype(int).to_numpy(),
        pdb_pairwise_distance=pdb_pairwise.astype(np.float32),
        afdb_pairwise_distance=afdb_pairwise.astype(np.float32),
        absolute_pairwise_error=pairwise_error.astype(np.float32),
        pae=pae_submatrix.astype(np.float32),
        symmetric_pae=symmetric_pae.astype(np.float32),
    )

    summary = {
        "protein_id": report["protein_id"],
        "pdb_id": report["pdb_id"],
        "chain_id": report["chain_id"],
        "uniprot_id": report["uniprot_id"],
        "mapped_ca_count": len(merged),
        "uniprot_length": sequence_length,
        "mapped_ca_coverage": float(len(merged) / sequence_length),
        "alignment_mode": alignment_mode,
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_ca_count": int(high_conf_mask.sum()),
        "all_mapped_fit_rmsd": all_rmsd,
        "high_confidence_fit_rmsd": core_fit_rmsd,
        "median_aligned_ca_distance": float(np.median(per_residue_distance)),
        "p90_aligned_ca_distance": float(np.quantile(per_residue_distance, 0.90)),
        "max_aligned_ca_distance": float(np.max(per_residue_distance)),
        "pdb_afdb_aa_match_fraction": float(merged["pdb_aa_matches_afdb"].mean()),
        "plddt_json_bfactor_median_abs_delta": float(
            np.nanmedian(np.abs(merged["plddt_bfactor_delta"]))
        ),
        "plddt_vs_local_disagreement_spearman": local_correlation,
        "symmetric_pae_vs_pairwise_error_spearman": pair_correlation,
        "pair_min_sequence_separation": pair_min_sequence_separation,
        **join_diagnostics,
        "residue_output": str(residue_output),
        "pairwise_output": str(pairwise_output),
        "terminology_note": (
            "PDB–AFDB coordinate disagreement is not assumed to be AlphaFold error; "
            "it may include conformational, construct, ligand-state, or experimental differences."
        ),
    }

    summary_path = pair_dir / "pair_geometry_qc.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
