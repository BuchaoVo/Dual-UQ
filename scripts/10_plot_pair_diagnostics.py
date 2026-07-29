from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot residue-level pair diagnostics.")
    parser.add_argument("--pair-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_dir = Path(args.pair_dir).expanduser().resolve()
    table = pd.read_parquet(pair_dir / "residue_geometry.parquet").sort_values(
        "uniprot_residue_number"
    )

    output_dir = pair_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(table["uniprot_residue_number"], table["aligned_ca_distance"])
    ax.set_xlabel("UniProt residue position")
    ax.set_ylabel("Aligned Cα disagreement (Å)")
    ax.set_title("PDB–AFDB local structural disagreement")
    fig.tight_layout()
    fig.savefig(output_dir / "local_disagreement.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(100.0 - table["plddt"], table["aligned_ca_distance"], s=18)
    ax.set_xlabel("100 − pLDDT")
    ax.set_ylabel("Aligned Cα disagreement (Å)")
    ax.set_title("Confidence versus local disagreement")
    fig.tight_layout()
    fig.savefig(output_dir / "plddt_vs_disagreement.png", dpi=200)
    plt.close(fig)

    pairwise = np.load(pair_dir / "pairwise_geometry.npz")
    positions = pairwise["uniprot_positions"]
    separation = np.abs(positions[:, None] - positions[None, :])
    mask = np.triu(np.ones_like(separation, dtype=bool), k=1) & (separation >= 6)

    pae = pairwise["symmetric_pae"][mask]
    error = pairwise["absolute_pairwise_error"][mask]
    finite = np.isfinite(pae) & np.isfinite(error)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(pae[finite], error[finite], gridsize=45, mincnt=1)
    ax.set_xlabel("Symmetric PAE (Å)")
    ax.set_ylabel("|PDB distance − AFDB distance| (Å)")
    ax.set_title("PAE versus residue-pair distance disagreement")
    fig.tight_layout()
    fig.savefig(output_dir / "pae_vs_pairwise_error.png", dpi=200)
    plt.close(fig)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
