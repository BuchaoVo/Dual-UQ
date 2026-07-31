#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def read_single_fasta(path: Path) -> str:
    headers = 0
    sequence_parts: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                headers += 1
            else:
                if headers == 0:
                    raise ValueError(f"{path}: sequence before FASTA header.")
                sequence_parts.append(line)

    if headers != 1:
        raise ValueError(f"{path}: FASTA headers={headers}, expected 1.")
    sequence = "".join(sequence_parts).upper()
    if not sequence:
        raise ValueError(f"{path}: empty sequence.")
    return sequence


def locate_fasta_npz(output_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in output_dir.rglob("*.npz")
        if path.is_file() and "score_only" in path.parts
    )
    fasta_candidates = [
        path for path in candidates
        if path.stem.endswith("_fasta_1")
    ]
    if len(fasta_candidates) == 1:
        return fasta_candidates[0]
    if len(fasta_candidates) == 0:
        raise ValueError(
            f"{output_dir}: no *_fasta_1.npz found; "
            f"all NPZ files={len(candidates)}."
        )
    raise ValueError(
        f"{output_dir}: multiple *_fasta_1.npz files: "
        f"{[str(path) for path in fasta_candidates]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one source-neutral score-only output."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-fasta", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, required=True)
    args = parser.parse_args()

    try:
        sequence = read_single_fasta(args.candidate_fasta)
        expected_s = np.asarray(
            [ALPHABET.index(amino_acid) for amino_acid in sequence],
            dtype=int,
        )
        npz_path = locate_fasta_npz(args.output_dir)

        with np.load(npz_path, allow_pickle=False) as data:
            required = {"score", "global_score", "S"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"{npz_path}: missing arrays {sorted(missing)}."
                )
            score = np.asarray(data["score"], dtype=float).reshape(-1)
            global_score = np.asarray(
                data["global_score"], dtype=float
            ).reshape(-1)
            observed_s = np.asarray(data["S"], dtype=int).reshape(-1)

        if len(score) != args.expected_repeats:
            raise ValueError(
                f"{npz_path}: score repeats={len(score)}, "
                f"expected {args.expected_repeats}."
            )
        if len(global_score) != args.expected_repeats:
            raise ValueError(
                f"{npz_path}: global_score repeats={len(global_score)}, "
                f"expected {args.expected_repeats}."
            )
        if not np.array_equal(observed_s, expected_s):
            raise ValueError(
                f"{npz_path}: S does not match candidate FASTA."
            )
        if not np.isfinite(score).all() or not np.isfinite(global_score).all():
            raise ValueError(f"{npz_path}: non-finite scores.")

        print(npz_path)
        return 0
    except Exception as exc:
        print(f"INCOMPLETE: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
