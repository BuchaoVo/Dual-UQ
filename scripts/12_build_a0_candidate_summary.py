from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and categorize all completed PDB–AFDB pairs for A0 selection."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/a0_selection.yaml",
    )
    return parser.parse_args()


def load_config(root: Path, config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def long_range_pae_features(
    pairwise_path: Path,
    min_separation: int,
) -> dict[str, float]:
    data = np.load(pairwise_path)
    positions = np.asarray(data["uniprot_positions"], dtype=int)
    pae = np.asarray(data["symmetric_pae"], dtype=float)
    separation = np.abs(positions[:, None] - positions[None, :])
    mask = np.triu(np.ones_like(separation, dtype=bool), k=1)
    mask &= separation >= min_separation
    values = pae[mask]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "long_range_pae_median": float("nan"),
            "long_range_pae_q90": float("nan"),
            "long_range_pae_above_10_fraction": float("nan"),
        }
    return {
        "long_range_pae_median": float(np.median(values)),
        "long_range_pae_q90": float(np.quantile(values, 0.90)),
        "long_range_pae_above_10_fraction": float(np.mean(values >= 10.0)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config = load_config(root, args.config)
    thresholds = config["thresholds"]
    quality = config["quality"]

    rows = []
    for pair_dir in sorted((root / "data/processed/pairs").iterdir()):
        if not pair_dir.is_dir():
            continue
        required = [
            pair_dir / "pair_qc.json",
            pair_dir / "pair_geometry_qc.json",
            pair_dir / "robust_pair_diagnostics.json",
            pair_dir / "residue_geometry.parquet",
            pair_dir / "pairwise_geometry.npz",
        ]
        if not all(path.exists() for path in required):
            continue

        pair_qc = json.loads(required[0].read_text(encoding="utf-8"))
        geometry_qc = json.loads(required[1].read_text(encoding="utf-8"))
        robust = json.loads(required[2].read_text(encoding="utf-8"))
        residues = pd.read_parquet(required[3])
        plddt = residues["plddt"].to_numpy(dtype=float)

        long_range = long_range_pae_features(
            required[4],
            int(thresholds["high_pae_long_range"]["min_sequence_separation"]),
        )

        segment_path = pair_dir / "disagreement_segments.csv"
        high_conf_segment_length = 0
        if segment_path.exists():
            segments = pd.read_csv(segment_path)
            state_cfg = thresholds["high_conf_state_disagreement"]
            eligible = segments[
                (segments["threshold"] == state_cfg["disagreement_threshold"])
                & (segments["median_plddt"] >= state_cfg["plddt_min"])
            ]
            if not eligible.empty:
                high_conf_segment_length = int(eligible["residue_count"].max())

        tags = []
        easy = thresholds["easy_control"]
        if (
            np.median(plddt) >= easy["median_plddt_min"]
            and geometry_qc["p90_aligned_ca_distance"] <= easy["p90_disagreement_max"]
            and high_conf_segment_length <= easy["high_conf_segment_max_length"]
        ):
            tags.append("easy_control")

        low = thresholds["low_conf_local"]
        if np.min(plddt) <= low["min_plddt_max"] or np.quantile(plddt, 0.10) <= low["q10_plddt_max"]:
            tags.append("low_conf_local")

        high_pae = thresholds["high_pae_long_range"]
        if (
            long_range["long_range_pae_q90"] >= high_pae["pae_q90_min"]
            or long_range["long_range_pae_above_10_fraction"]
            >= high_pae["pae_above_10_fraction_min"]
        ):
            tags.append("high_pae_long_range")

        state = thresholds["high_conf_state_disagreement"]
        if high_conf_segment_length >= state["min_contiguous_length"]:
            tags.append("high_conf_state_disagreement")

        quality_pass = (
            pair_qc["mapping_coverage"] >= quality["min_mapping_coverage"]
            and geometry_qc["pdb_afdb_aa_match_fraction"] >= quality["min_aa_match_fraction"]
            and geometry_qc["plddt_json_bfactor_median_abs_delta"]
            <= quality["max_plddt_bfactor_median_abs_delta"]
        )

        priority = "unclassified"
        for candidate in (
            "high_conf_state_disagreement",
            "low_conf_local",
            "high_pae_long_range",
            "easy_control",
        ):
            if candidate in tags:
                priority = candidate
                break

        rows.append(
            {
                "pair_name": pair_dir.name,
                "protein_id": pair_qc["protein_id"],
                "pdb_id": pair_qc["pdb_id"],
                "chain_id": pair_qc["chain_id"],
                "uniprot_id": pair_qc["uniprot_id"],
                "length": pair_qc["uniprot_length"],
                "mapping_coverage": pair_qc["mapping_coverage"],
                "quality_pass": quality_pass,
                "plddt_min": float(np.min(plddt)),
                "plddt_q10": float(np.quantile(plddt, 0.10)),
                "plddt_median": float(np.median(plddt)),
                "disagreement_median": geometry_qc["median_aligned_ca_distance"],
                "disagreement_p90": geometry_qc["p90_aligned_ca_distance"],
                "high_conf_segment_max_length": high_conf_segment_length,
                **long_range,
                "candidate_tags": ";".join(tags) if tags else "unclassified",
                "primary_category": priority,
            }
        )

    if not rows:
        print("No completed pair directories found.")
        return

    table = pd.DataFrame(rows).sort_values(
        ["quality_pass", "primary_category", "pdb_id"],
        ascending=[False, True, True],
    )
    output = root / "reports/a0_candidate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
