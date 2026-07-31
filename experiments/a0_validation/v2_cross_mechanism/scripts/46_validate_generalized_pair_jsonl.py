#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ATOMS = ("N", "CA", "C", "O")


def read_one_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 1:
        raise ValueError(f"{path}: records={len(rows)}, expected 1.")
    return rows[0]


def atom_coords(record: dict[str, Any], atom: str) -> np.ndarray:
    key = f"{atom}_chain_A"
    values = np.asarray(record["coords_chain_A"][key], dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{key}: invalid shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{key}: non-finite coordinates.")
    return values


def kabsch_distances(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mc, tc = mobile.mean(0), target.mean(0)
    x, y = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    transformed = x @ rotation + tc
    return np.linalg.norm(transformed - target, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    plan = pd.read_csv(
        v2 / f"manifests/index{args.screening_index}_paired_input_plan.tsv",
        sep="\t",
    ).sort_values("output_position")
    build = json.loads(
        (
            v2 / f"metrics/index{args.screening_index}_paired_pdb_build.json"
        ).read_text(encoding="utf-8")
    )

    jsonl_dir = v2 / f"inputs/index{args.screening_index}/jsonl"
    paths = {
        "pdb": jsonl_dir / f"index{args.screening_index}_pdb.jsonl",
        "afdb": jsonl_dir / f"index{args.screening_index}_afdb.jsonl",
    }
    expected_sequence = "".join(plan["canonical_aa"].astype(str))
    expected_length = len(plan)
    errors: list[str] = []
    records: dict[str, dict[str, Any]] = {}

    for side, path in paths.items():
        try:
            record = read_one_jsonl(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        records[side] = record

        if record.get("num_of_chains") != 1:
            errors.append(
                f"{side}: num_of_chains={record.get('num_of_chains')}, expected 1."
            )
        if record.get("seq") != expected_sequence:
            errors.append(f"{side}: seq mismatch.")
        if record.get("seq_chain_A") != expected_sequence:
            errors.append(f"{side}: seq_chain_A mismatch.")
        if len(record.get("seq", "")) != expected_length:
            errors.append(
                f"{side}: length={len(record.get('seq', ''))}, "
                f"expected {expected_length}."
            )
        if "coords_chain_A" not in record:
            errors.append(f"{side}: coords_chain_A missing.")
            continue
        for atom in ATOMS:
            try:
                values = atom_coords(record, atom)
            except Exception as exc:
                errors.append(f"{side}/{atom}: {type(exc).__name__}: {exc}")
                continue
            if len(values) != expected_length:
                errors.append(
                    f"{side}/{atom}: rows={len(values)}, expected {expected_length}."
                )

    ca_metrics = None
    if set(records) == {"pdb", "afdb"} and not errors:
        pdb_ca = atom_coords(records["pdb"], "CA")
        afdb_ca = atom_coords(records["afdb"], "CA")
        distances = kabsch_distances(afdb_ca, pdb_ca)
        rmsd = float(np.sqrt(np.mean(np.square(distances))))
        builder_rmsd = float(build["all_residue_ca_kabsch_rmsd"])
        ca_metrics = {
            "rmsd": rmsd,
            "distance_min": float(distances.min()),
            "distance_median": float(np.median(distances)),
            "distance_mean": float(distances.mean()),
            "distance_p90": float(np.quantile(distances, 0.9)),
            "distance_max": float(distances.max()),
            "builder_rmsd": builder_rmsd,
            "builder_parser_rmsd_abs_delta": abs(rmsd - builder_rmsd),
        }
        if ca_metrics["builder_parser_rmsd_abs_delta"] > 0.002:
            errors.append(
                "Builder and official-parser CA RMSD differ by more than 0.002 Å."
            )

    result = {
        "screening_index": args.screening_index,
        "expected_length": expected_length,
        "pdb_name": records.get("pdb", {}).get("name"),
        "afdb_name": records.get("afdb", {}).get("name"),
        "pdb_sequence_length": len(records.get("pdb", {}).get("seq", "")),
        "afdb_sequence_length": len(records.get("afdb", {}).get("seq", "")),
        "sequences_equal": (
            records.get("pdb", {}).get("seq")
            == records.get("afdb", {}).get("seq")
            == expected_sequence
        ),
        "ca_kabsch": ca_metrics,
        "errors": errors,
        "validation_pass": not errors,
        "next_stage": (
            "V2C_source_conditioned_generation_and_cross_scoring"
            if not errors
            else "repair_derived_pdb_or_parser_input"
        ),
    }

    metrics = v2 / f"metrics/index{args.screening_index}_paired_input_validation.json"
    metrics.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown = (
        v2 / f"V2B2_INDEX{args.screening_index}_PAIRED_INPUT_VALIDATION.md"
    )
    lines = [
        f"# A0-V2B2 Index {args.screening_index} Paired Input Validation",
        "",
        f"- Expected length: `{expected_length}`",
        f"- PDB parsed length: `{result['pdb_sequence_length']}`",
        f"- AFDB parsed length: `{result['afdb_sequence_length']}`",
        f"- Sequences equal: `{result['sequences_equal']}`",
        f"- PDB parser name: `{result['pdb_name']}`",
        f"- AFDB parser name: `{result['afdb_name']}`",
        f"- CA Kabsch metrics: `{ca_metrics}`",
        f"- Validation pass: `{result['validation_pass']}`",
        f"- Next stage: `{result['next_stage']}`",
        "",
        "## Interpretation boundary",
        "",
        "较大的CA RMSD或局部距离不会导致V2B2失败。"
        "本阶段只要求两份派生结构被ProteinMPNN一致解析，"
        "序列、长度、链和坐标维度完全匹配。",
        "",
    ]
    if errors:
        lines += ["## Errors", ""]
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    markdown.write_text("\n".join(lines), encoding="utf-8")

    print("PDB name/length:", result["pdb_name"], result["pdb_sequence_length"])
    print("AFDB name/length:", result["afdb_name"], result["afdb_sequence_length"])
    print("Sequences equal:", result["sequences_equal"])
    print("CA Kabsch:", ca_metrics)
    print("Errors:", errors)
    print("Validation pass:", result["validation_pass"])
    print("Next stage:", result["next_stage"])
    print("Wrote:", metrics)
    print("Wrote:", markdown)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
