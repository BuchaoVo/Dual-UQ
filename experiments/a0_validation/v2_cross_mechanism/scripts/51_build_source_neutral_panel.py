#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALPHABET = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_single_fasta(path: Path, candidate_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f">{candidate_id}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start:start + 80] + "\n")


def mutate_sequence(
    canonical: str,
    distance: int,
    rng: np.random.Generator,
) -> tuple[str, list[int], list[str]]:
    positions = sorted(
        int(value)
        for value in rng.choice(
            len(canonical),
            size=distance,
            replace=False,
        )
    )
    sequence = list(canonical)
    substitutions: list[str] = []

    for zero_based in positions:
        original = canonical[zero_based]
        choices = ALPHABET[ALPHABET != original]
        replacement = str(rng.choice(choices))
        sequence[zero_based] = replacement
        substitutions.append(
            f"{zero_based + 1}:{original}>{replacement}"
        )

    return "".join(sequence), positions, substitutions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a structure-blind source-neutral candidate panel."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--hamming-distances", default="1,2,4,8,16,32")
    parser.add_argument("--variants-per-distance", type=int, default=8)
    parser.add_argument("--panel-seed", type=int, default=20260731)
    args = parser.parse_args()

    distances = [
        int(value.strip())
        for value in args.hamming_distances.split(",")
        if value.strip()
    ]
    if not distances or any(distance <= 0 for distance in distances):
        raise ValueError("Hamming distances must be positive integers.")
    if len(set(distances)) != len(distances):
        raise ValueError("Hamming distances must be unique.")
    if args.variants_per_distance <= 0:
        raise ValueError("variants-per-distance must be positive.")

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    plan_path = (
        v2 / f"manifests/index{args.screening_index}_paired_input_plan.tsv"
    )
    panel_root = (
        v2 / f"inputs/index{args.screening_index}/source_neutral_panel"
    )
    fasta_root = panel_root / "candidate_fastas"

    plan = pd.read_csv(plan_path, sep="\t").sort_values(
        "output_position",
        kind="mergesort",
    )
    canonical = "".join(plan["canonical_aa"].astype(str)).upper()
    uniprot_positions = plan["uniprot_position"].astype(int).to_numpy()
    expected_length = len(canonical)

    if max(distances) > expected_length:
        raise ValueError(
            f"Maximum Hamming distance={max(distances)} exceeds "
            f"sequence length={expected_length}."
        )

    rng = np.random.default_rng(args.panel_seed)
    rows: list[dict[str, Any]] = [{
        "candidate_order": 1,
        "candidate_id": f"index{args.screening_index}_neutral_canonical",
        "panel_role": "canonical",
        "hamming_distance": 0,
        "variant_index": 0,
        "mutation_output_positions": "",
        "mutation_uniprot_positions": "",
        "substitutions": "",
        "sequence_length": expected_length,
        "sequence_sha256": sha256_text(canonical),
        "sequence": canonical,
    }]
    seen = {canonical}
    errors: list[str] = []

    for distance in distances:
        accepted = 0
        attempts = 0
        max_attempts = args.variants_per_distance * 1000

        while accepted < args.variants_per_distance:
            attempts += 1
            if attempts > max_attempts:
                errors.append(
                    f"Could not create {args.variants_per_distance} unique "
                    f"variants at Hamming distance {distance}."
                )
                break

            sequence, zero_based_positions, substitutions = mutate_sequence(
                canonical,
                distance,
                rng,
            )
            if sequence in seen:
                continue

            actual_distance = sum(
                left != right
                for left, right in zip(canonical, sequence)
            )
            if actual_distance != distance:
                raise AssertionError(
                    f"Generated distance={actual_distance}, expected={distance}."
                )

            accepted += 1
            seen.add(sequence)
            output_positions = [
                position + 1 for position in zero_based_positions
            ]
            mutation_uniprot = [
                int(uniprot_positions[position])
                for position in zero_based_positions
            ]
            rows.append({
                "candidate_order": len(rows) + 1,
                "candidate_id": (
                    f"index{args.screening_index}_neutral_"
                    f"h{distance:03d}_v{accepted:02d}"
                ),
                "panel_role": "neutral_variant",
                "hamming_distance": distance,
                "variant_index": accepted,
                "mutation_output_positions": ";".join(
                    str(value) for value in output_positions
                ),
                "mutation_uniprot_positions": ";".join(
                    str(value) for value in mutation_uniprot
                ),
                "substitutions": ";".join(substitutions),
                "sequence_length": expected_length,
                "sequence_sha256": sha256_text(sequence),
                "sequence": sequence,
            })

    expected_count = 1 + len(distances) * args.variants_per_distance
    if len(rows) != expected_count:
        errors.append(
            f"Candidate rows={len(rows)}, expected {expected_count}."
        )
    if len(seen) != len(rows):
        errors.append("Source-neutral panel contains duplicate sequences.")

    fasta_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        fasta_path = fasta_root / f"{row['candidate_id']}.fasta"
        write_single_fasta(
            fasta_path,
            str(row["candidate_id"]),
            str(row["sequence"]),
        )
        row["candidate_fasta"] = str(fasta_path)

    panel_path = (
        v2 / f"manifests/index{args.screening_index}_source_neutral_candidates.tsv"
    )
    jobs_path = (
        v2 / f"manifests/index{args.screening_index}_source_neutral_score_jobs.tsv"
    )
    metrics_path = (
        v2 / f"metrics/index{args.screening_index}_source_neutral_panel.json"
    )

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    with panel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with jobs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "candidate_fasta"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "candidate_fasta": row["candidate_fasta"],
            })

    panel_hash_payload = "\n".join(
        f"{row['candidate_id']}\t{row['sequence_sha256']}"
        for row in rows
    )
    audit = {
        "screening_index": args.screening_index,
        "canonical_length": expected_length,
        "hamming_distances": distances,
        "variants_per_distance": args.variants_per_distance,
        "panel_seed": args.panel_seed,
        "candidate_count": len(rows),
        "unique_sequence_count": len(seen),
        "panel_sha256": sha256_text(panel_hash_payload),
        "uses_structure_information": False,
        "uses_generation_outputs": False,
        "candidate_manifest": str(panel_path),
        "score_jobs_manifest": str(jobs_path),
        "errors": errors,
        "panel_pass": not errors,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Canonical length:", expected_length)
    print("Hamming distances:", distances)
    print("Variants per distance:", args.variants_per_distance)
    print("Candidate count:", len(rows))
    print("Unique sequences:", len(seen))
    print("Panel SHA-256:", audit["panel_sha256"])
    print("Uses structure information:", False)
    print("Errors:", errors)
    print("Panel pass:", not errors)
    print("Wrote:", panel_path)
    print("Wrote:", jobs_path)
    print("Wrote:", metrics_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
