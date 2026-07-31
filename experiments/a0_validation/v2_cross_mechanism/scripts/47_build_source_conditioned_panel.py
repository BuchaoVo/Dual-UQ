#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def parse_fasta(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").upper()
        records.append({"header": header, "sequence": sequence})
        header = None
        sequence_parts = []

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence before FASTA header.")
                sequence_parts.append(line)
    flush()
    return records


def find_generation_fasta(output_dir: Path) -> Path:
    candidates = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".fa", ".fasta", ".faa"}
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{output_dir}: found {len(candidates)} FASTA files, expected exactly 1: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def parse_header_metrics(header: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for token in header.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def write_single_fasta(path: Path, candidate_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f">{candidate_id}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start:start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a canonical + source-conditioned candidate panel."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--generated-per-backbone", type=int, default=4)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    plan_path = (
        v2 / f"manifests/index{args.screening_index}_paired_input_plan.tsv"
    )
    generation_root = (
        v2 / f"outputs/index{args.screening_index}/source_conditioned_generation"
    )
    panel_root = (
        v2 / f"inputs/index{args.screening_index}/source_conditioned_panel"
    )
    fasta_root = panel_root / "candidate_fastas"

    plan = pd.read_csv(plan_path, sep="\t").sort_values(
        "output_position",
        kind="mergesort",
    )
    canonical = "".join(plan["canonical_aa"].astype(str)).upper()
    expected_length = len(canonical)
    errors: list[str] = []
    candidate_rows: list[dict[str, Any]] = [{
        "candidate_order": 1,
        "candidate_id": f"index{args.screening_index}_canonical",
        "candidate_origin": "canonical",
        "generation_source": "none",
        "generation_sample": 0,
        "generation_header": "canonical_from_paired_input_plan",
        "generation_score": None,
        "generation_global_score": None,
        "sequence_length": expected_length,
        "sequence_sha256": sha256_text(canonical),
        "sequence": canonical,
    }]

    generation_fastas = {}
    source_sequences: dict[str, list[str]] = {}

    for source in ("pdb", "afdb"):
        fasta = find_generation_fasta(generation_root / source)
        generation_fastas[source] = str(fasta)
        records = parse_fasta(fasta)
        if len(records) < args.generated_per_backbone + 1:
            errors.append(
                f"{source}: FASTA records={len(records)}, expected at least "
                f"{args.generated_per_backbone + 1}."
            )
            continue
        if records[0]["sequence"] != canonical:
            errors.append(
                f"{source}: first FASTA record does not match frozen canonical sequence."
            )

        generated = records[1:1 + args.generated_per_backbone]
        source_sequences[source] = [record["sequence"] for record in generated]

        for sample_index, record in enumerate(generated, start=1):
            sequence = record["sequence"]
            if len(sequence) != expected_length:
                errors.append(
                    f"{source} sample {sample_index}: length={len(sequence)}, "
                    f"expected {expected_length}."
                )
            if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                errors.append(
                    f"{source} sample {sample_index}: unsupported residues "
                    f"{sorted(set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'))}."
                )
            metrics = parse_header_metrics(record["header"])
            candidate_rows.append({
                "candidate_order": len(candidate_rows) + 1,
                "candidate_id": (
                    f"index{args.screening_index}_gen_{source}_{sample_index:02d}"
                ),
                "candidate_origin": f"{source}_generated",
                "generation_source": source,
                "generation_sample": sample_index,
                "generation_header": record["header"],
                "generation_score": metrics.get("score"),
                "generation_global_score": metrics.get("global_score"),
                "sequence_length": len(sequence),
                "sequence_sha256": sha256_text(sequence),
                "sequence": sequence,
            })

    sequences = [row["sequence"] for row in candidate_rows]
    expected_candidate_count = 1 + 2 * args.generated_per_backbone
    if len(candidate_rows) != expected_candidate_count:
        errors.append(
            f"candidate rows={len(candidate_rows)}, expected {expected_candidate_count}."
        )
    if len(set(sequences)) != len(sequences):
        errors.append("Candidate panel contains duplicate sequences.")

    pdb_set = set(source_sequences.get("pdb", []))
    afdb_set = set(source_sequences.get("afdb", []))
    exact_overlap = sorted(pdb_set & afdb_set)
    if exact_overlap:
        errors.append(
            f"PDB- and AFDB-generated sequence sets overlap exactly: "
            f"{len(exact_overlap)} sequence(s)."
        )

    panel_root.mkdir(parents=True, exist_ok=True)
    fasta_root.mkdir(parents=True, exist_ok=True)

    panel_path = (
        v2 / f"manifests/index{args.screening_index}_source_conditioned_candidates.tsv"
    )
    jobs_path = (
        v2 / f"manifests/index{args.screening_index}_source_conditioned_score_jobs.tsv"
    )
    metrics_path = (
        v2 / f"metrics/index{args.screening_index}_source_conditioned_generation.json"
    )

    for row in candidate_rows:
        fasta_path = fasta_root / f"{row['candidate_id']}.fasta"
        write_single_fasta(
            fasta_path,
            str(row["candidate_id"]),
            str(row["sequence"]),
        )
        row["candidate_fasta"] = str(fasta_path)

    with panel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(candidate_rows[0].keys()),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    with jobs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "candidate_fasta"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "candidate_fasta": row["candidate_fasta"],
            })

    audit = {
        "screening_index": args.screening_index,
        "canonical_length": expected_length,
        "generated_per_backbone": args.generated_per_backbone,
        "candidate_count": len(candidate_rows),
        "unique_sequence_count": len(set(sequences)),
        "pdb_generated_unique_count": len(pdb_set),
        "afdb_generated_unique_count": len(afdb_set),
        "pdb_afdb_exact_overlap_count": len(exact_overlap),
        "generation_fastas": generation_fastas,
        "candidate_panel": str(panel_path),
        "score_jobs": str(jobs_path),
        "errors": errors,
        "generation_validation_pass": not errors,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Canonical length:", expected_length)
    print("Candidate count:", len(candidate_rows))
    print("Unique sequences:", len(set(sequences)))
    print("PDB-generated unique:", len(pdb_set))
    print("AFDB-generated unique:", len(afdb_set))
    print("PDB/AFDB exact overlap:", len(exact_overlap))
    print("Errors:", errors)
    print("Generation validation pass:", not errors)
    print("Wrote:", panel_path)
    print("Wrote:", jobs_path)
    print("Wrote:", metrics_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
