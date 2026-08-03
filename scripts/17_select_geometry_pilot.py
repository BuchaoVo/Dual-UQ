from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one quality-passing pilot per provisional stratum."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--include-negative-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    table = pd.read_csv(
        root / "data/manifests/screening_pool_preflight.tsv",
        sep="\t",
    )

    passed = table[table["preflight_status"] == "pass_full_length"].copy()
    selected = []
    for stratum in [
        "high_global_confidence",
        "lower_global_confidence",
        "long_backbone_proxy",
        "balanced_background",
    ]:
        candidates = passed[passed["provisional_stratum"] == stratum].copy()
        if candidates.empty:
            continue
        candidates["coverage_score"] = (
            candidates["full_length_mapping_coverage"]
            + candidates["observed_ca_fraction_of_mapped"]
        )
        candidates = candidates.sort_values(
            ["coverage_score", "sequence_identity"],
            ascending=False,
        )
        record = candidates.iloc[0].to_dict()
        record["pilot_role"] = f"positive:{stratum}"
        selected.append(record)

    if args.include_negative_control:
        warnings = table[
            table["preflight_status"] == "warn_construct_difference"
        ].copy()
        if not warnings.empty:
            warnings = warnings.sort_values(
                "full_length_mapping_coverage",
                ascending=False,
            )
            record = warnings.iloc[0].to_dict()
            record["pilot_role"] = "negative_control:construct_difference"
            selected.append(record)

    output = pd.DataFrame(selected)
    output_path = root / "data/manifests/geometry_pilot.tsv"
    output.to_csv(output_path, sep="\t", index=False)

    indices = output["screening_index"].astype(int).tolist() if not output.empty else []
    command = (
        "python scripts/14_run_screening_pool.py "
        "--config configs/legacy/a0_screening/screening_pool.yaml "
        f"--only {','.join(str(i) for i in indices)}"
    )
    summary_path = root / "reports/geometry_pilot_selection.json"
    summary_path.write_text(
        json.dumps(
            {
                "selected_indices": indices,
                "run_command": command,
                "output_path": str(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not output.empty:
        print(
            output[
                [
                    "screening_index", "pdb_id", "chain_id", "uniprot_id",
                    "provisional_stratum", "full_length_mapping_coverage",
                    "preflight_status", "pilot_role",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved: {output_path}")
    print(f"Run: {command}")


if __name__ == "__main__":
    main()
