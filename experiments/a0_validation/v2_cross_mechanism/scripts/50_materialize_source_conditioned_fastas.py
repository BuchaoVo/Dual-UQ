#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_single_fasta(path: Path, candidate_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f">{candidate_id}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start:start + 80] + "\n")


def read_single_fasta(path: Path) -> tuple[str, str]:
    header = None
    sequence_parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(f"{path}: multiple FASTA records.")
                header = line[1:]
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence before header.")
                sequence_parts.append(line)
    if header is None:
        raise ValueError(f"{path}: no FASTA record.")
    return header, "".join(sequence_parts).upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize and validate V2C candidate FASTA files."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--expected-candidates", type=int, default=9)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    candidates_path = (
        v2
        / f"manifests/index{args.screening_index}_source_conditioned_candidates.tsv"
    )
    jobs_path = (
        v2
        / f"manifests/index{args.screening_index}_source_conditioned_score_jobs.tsv"
    )
    fasta_root = (
        v2
        / f"inputs/index{args.screening_index}/"
          "source_conditioned_panel/candidate_fastas"
    )
    audit_path = (
        v2
        / f"metrics/index{args.screening_index}_candidate_fasta_repair.json"
    )

    candidates = pd.read_csv(candidates_path, sep="\t", dtype=str)
    required = {
        "candidate_id",
        "sequence",
        "sequence_length",
        "sequence_sha256",
    }
    missing = required - set(candidates.columns)
    errors: list[str] = []
    rows: list[dict[str, str]] = []

    if missing:
        errors.append(f"Candidate manifest missing columns: {sorted(missing)}.")
    if len(candidates) != args.expected_candidates:
        errors.append(
            f"Candidate rows={len(candidates)}, "
            f"expected {args.expected_candidates}."
        )
    if "candidate_id" in candidates and candidates["candidate_id"].duplicated().any():
        errors.append("Duplicate candidate IDs.")
    if "sequence" in candidates and candidates["sequence"].duplicated().any():
        errors.append("Duplicate candidate sequences.")

    fasta_root.mkdir(parents=True, exist_ok=True)

    if not errors:
        for _, row in candidates.iterrows():
            candidate_id = str(row["candidate_id"]).strip()
            sequence = str(row["sequence"]).strip().upper()
            expected_length = int(row["sequence_length"])
            expected_sha = str(row["sequence_sha256"]).strip()

            if len(sequence) != expected_length:
                errors.append(
                    f"{candidate_id}: sequence length={len(sequence)}, "
                    f"manifest length={expected_length}."
                )
                continue
            invalid = sorted(set(sequence) - VALID_AA)
            if invalid:
                errors.append(
                    f"{candidate_id}: invalid amino acids {invalid}."
                )
                continue
            actual_sha = sha256_text(sequence)
            if actual_sha != expected_sha:
                errors.append(
                    f"{candidate_id}: sequence SHA-256 mismatch."
                )
                continue

            fasta_path = fasta_root / f"{candidate_id}.fasta"
            write_single_fasta(fasta_path, candidate_id, sequence)
            observed_header, observed_sequence = read_single_fasta(fasta_path)
            if observed_header != candidate_id:
                errors.append(
                    f"{candidate_id}: FASTA header mismatch after writing."
                )
                continue
            if observed_sequence != sequence:
                errors.append(
                    f"{candidate_id}: FASTA sequence mismatch after writing."
                )
                continue

            rows.append({
                "candidate_id": candidate_id,
                "candidate_fasta": str(fasta_path),
            })

    if len(rows) != args.expected_candidates:
        errors.append(
            f"Validated FASTA rows={len(rows)}, "
            f"expected {args.expected_candidates}."
        )

    if not errors:
        path_map = {
            row["candidate_id"]: row["candidate_fasta"]
            for row in rows
        }
        candidates["candidate_fasta"] = candidates["candidate_id"].map(path_map)
        candidates.to_csv(
            candidates_path,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        pd.DataFrame(rows).to_csv(
            jobs_path,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )

    audit = {
        "screening_index": args.screening_index,
        "candidate_manifest": str(candidates_path),
        "score_jobs_manifest": str(jobs_path),
        "candidate_fasta_root": str(fasta_root),
        "expected_candidate_count": args.expected_candidates,
        "manifest_candidate_count": len(candidates),
        "validated_fasta_count": len(rows),
        "candidate_fastas": rows,
        "errors": errors,
        "repair_pass": not errors,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Manifest candidates:", len(candidates))
    print("Validated FASTAs:", len(rows))
    print("FASTA root:", fasta_root)
    print("Errors:", errors)
    print("Repair pass:", not errors)
    print("Wrote:", jobs_path)
    print("Wrote:", audit_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
