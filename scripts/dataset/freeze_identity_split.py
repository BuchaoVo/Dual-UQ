#!/usr/bin/env python3
"""Freeze the internal Dual-UQ 30%-identity cluster split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.dataset.identity_split import (
    build_canonical_sequence_table,
    build_connected_identity_assignments,
    characterize_identity_clusters,
    write_canonical_fasta,
)
from dual_uq.dataset.splits import build_identity_disjoint_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--mmseqs-tmp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--clustering-mode", choices=("connected_components",), default="connected_components")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--model-evaluability", type=Path)
    parser.add_argument("--proteinmpnn-local", type=Path)
    parser.add_argument("--esm-if1-local", type=Path)
    parser.add_argument("--generative-summary", type=Path)
    parser.add_argument("--dynamicmpnn-summary", type=Path)
    parser.add_argument("--geometry-summary", type=Path)
    parser.add_argument("--ligand-sites", type=Path)
    return parser


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _read_ids(
    path: Path | None, *, valid: pd.Series | None = None, suffix_after_double_underscore: bool = False
) -> set[str]:
    if path is None:
        return set()
    table = pd.read_parquet(path)
    if "protein_id" not in table.columns:
        raise ValueError(f"coverage source lacks protein_id: {path}")
    if valid is not None and "model_evaluable" in table.columns:
        table = table.loc[table["model_evaluable"].astype(bool)]
    identifiers = table["protein_id"].astype(str)
    if suffix_after_double_underscore:
        identifiers = identifiers.str.rsplit("__", n=1).str[-1]
    return set(identifiers)


def _coverage_table(
    sequence_table: pd.DataFrame,
    assignments: pd.DataFrame,
    sources: dict[str, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, identifiers in sources.items():
        selected = assignments.loc[assignments["protein_id"].isin(identifiers)]
        rows.append(
            {
                "cohort": name,
                "source_protein_count": int(len(identifiers)),
                "primary_admitted_overlap": int(len(set(sequence_table["protein_id"]) & identifiers)),
                "train_count": int((selected["split"] == "TRAIN").sum()),
                "validation_count": int((selected["split"] == "VALIDATION").sum()),
                "locked_test_count": int((selected["split"] == "LOCKED_TEST").sum()),
            }
        )
    return pd.DataFrame(rows)


def _read_hits(path: Path) -> pd.DataFrame:
    columns = ["query", "target", "pident", "alnlen", "qcov", "tcov"]
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"MMseqs all-vs-all output is missing or empty: {path}")
    return pd.read_csv(path, sep="\t", header=None, names=columns)


def _leakage_audit_from_hits(hits: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    split_by_protein = assignments.set_index("protein_id")["split"]
    audit = hits.loc[hits["query"].ne(hits["target"])].copy()
    audit["left_split"] = audit["query"].map(split_by_protein)
    audit["right_split"] = audit["target"].map(split_by_protein)
    if audit[["left_split", "right_split"]].isna().any().any():
        raise RuntimeError("MMseqs leakage audit references an unassigned protein")
    audit = audit.loc[audit["left_split"].ne(audit["right_split"])]
    return audit[["left_split", "right_split", "query", "target", "pident", "alnlen", "qcov", "tcov"]].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mmseqs = Path(args.mmseqs)
    if not mmseqs.is_file():
        resolved = shutil.which(str(args.mmseqs))
        if resolved is None:
            raise FileNotFoundError(f"MMseqs2 executable not found: {args.mmseqs}")
        mmseqs = Path(resolved)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.candidates)
    sequence_table = build_canonical_sequence_table(candidates, args.sequence_dir)
    sequence_table.to_parquet(output_dir / "canonical_sequence_manifest.parquet", index=False)
    fasta = output_dir / "canonical_sequences.fasta"
    write_canonical_fasta(sequence_table, fasta)
    hits_path = args.mmseqs_tmp / "identity_hits.tsv"
    cluster_command = [
        str(mmseqs),
        "easy-search",
        str(fasta),
        str(fasta),
        str(hits_path),
        str(args.mmseqs_tmp / "all_vs_all_tmp"),
        "--min-seq-id",
        str(args.threshold),
        "-c",
        str(args.coverage),
        "--cov-mode",
        str(args.cov_mode),
        "--max-seqs",
        "100000",
        "--add-self-matches",
        "1",
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov",
    ]
    _run(cluster_command, output_dir / "mmseqs_all_vs_all.log")
    hits = _read_hits(hits_path)
    cluster_edges = build_connected_identity_assignments(hits, sequence_table)
    cluster_tsv = output_dir / "identity_clusters.tsv"
    cluster_edges.rename(columns={"sequence_cluster_id": "identity_cluster_id"}).to_csv(
        cluster_tsv, sep="\t", header=False, index=False
    )
    cluster_table = cluster_edges.merge(
        sequence_table, on="protein_id", how="left", validate="one_to_one"
    )
    cluster_table["identity_threshold"] = float(args.threshold)
    cluster_table["coverage_requirement"] = float(args.coverage)
    cluster_table["cov_mode"] = int(args.cov_mode)
    cluster_table["clustering_mode"] = args.clustering_mode
    cluster_table["cluster_source"] = f"MMseqs2-{subprocess.check_output([str(mmseqs), 'version'], text=True).strip()}-all-vs-all"
    cluster_summary = characterize_identity_clusters(cluster_table)
    cluster_table.rename(columns={"sequence_cluster_id": "identity_cluster_id"}).to_parquet(
        output_dir / "cluster_assignments.parquet", index=False
    )
    (output_dir / "clustering_summary.json").write_text(
        json.dumps(cluster_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_result = build_identity_disjoint_split(
        sequence_table,
        cluster_table[["protein_id", "sequence_cluster_id", "identity_threshold", "cluster_source"]],
        threshold=args.threshold,
        seed=args.seed,
    )
    split_assignments = split_result.assignments.rename(columns={"sequence_cluster_id": "identity_cluster_id"})
    leakage = _leakage_audit_from_hits(
        hits,
        split_assignments.rename(columns={"identity_cluster_id": "sequence_cluster_id"}),
    )
    leakage.to_parquet(output_dir / "leakage_audit.parquet", index=False)
    leakage_violations = int(len(leakage))
    if leakage_violations == 0:
        split_assignments.to_parquet(output_dir / "method_split_assignments.parquet", index=False)
        status = "READY"
    else:
        status = "LEAKAGE_CONTRACT_FAIL"
    split_summary = dict(split_result.summary)
    split_summary["status"] = status
    split_summary["leakage_violation_count"] = leakage_violations
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sources = {
        "PRIMARY_ADMITTED": set(sequence_table["protein_id"]),
        "MODEL_EVALUABLE": _read_ids(args.model_evaluability, valid=pd.Series(dtype=bool)),
        "PROTEINMPNN_LOCAL_RESPONSE": _read_ids(args.proteinmpnn_local),
        "ESM_IF1_LOCAL_RESPONSE": _read_ids(args.esm_if1_local),
        "SHARED_GENERATIVE_COHORT": _read_ids(args.generative_summary),
        "DYNAMICMPNN_SUBSET": _read_ids(args.dynamicmpnn_summary),
        "GEOMETRY_EVALUABLE": _read_ids(args.geometry_summary, suffix_after_double_underscore=True),
        "LIGAND_LOCALIZATION": _read_ids(args.ligand_sites),
    }
    coverage_table = _coverage_table(sequence_table, split_assignments, sources)
    coverage_table.to_parquet(output_dir / "cohort_coverage.parquet", index=False)
    command_text = " ".join(cluster_command)
    sequence_summary = {
        "protein_count": int(len(sequence_table)),
        "unique_uniprot_count": int(sequence_table["uniprot_id"].nunique()),
        "unique_sequence_hash_count": int(sequence_table["sequence_hash"].nunique()),
        "conflicting_duplicate_identity": False,
        "nonstandard_character_records": [
            str(row.protein_id)
            for row in sequence_table.itertuples(index=False)
            if set(str(row.sequence)).difference(set("ACDEFGHIKLMNPQRSTVWY"))
        ],
    }
    manifest = {
        "schema_version": "dual_uq_identity_split_freeze_v1",
        "status": status,
        "terminology": "Dual-UQ 30%-identity cluster; internal locked test",
        "tool": {
            "executable": str(mmseqs),
            "version": subprocess.check_output([str(mmseqs), "version"], text=True).strip(),
            "cluster_command": command_text,
            "min_sequence_identity": float(args.threshold),
            "coverage": float(args.coverage),
            "cov_mode": int(args.cov_mode),
            "cov_mode_semantics": "coverage of query and target",
            "clustering_mode": args.clustering_mode,
            "clustering_mode_semantics": "connected components of thresholded MMseqs2 all-vs-all hits",
            "algorithm_selection_note": "A diagnostic set-cover run produced 17 cross-split threshold hits; it was not frozen. The final contract uses connected components of the same MMseqs2 identity/coverage hits, with no protein deletion or manual cluster splitting.",
        },
        "inputs": {
            "candidate_path": str(args.candidates),
            "canonical_sequence_dir": str(args.sequence_dir),
            "canonical_fasta": str(fasta),
            "canonical_fasta_sha256": _sha256(fasta),
            "cluster_tsv": str(cluster_tsv),
            "cluster_tsv_sha256": _sha256(cluster_tsv),
            "mmseqs_hits_tsv": str(hits_path),
            "mmseqs_hits_tsv_sha256": _sha256(hits_path),
            "seed": int(args.seed),
        },
        "canonical_sequence_summary": sequence_summary,
        "cluster_summary": cluster_summary,
        "split_summary": split_summary,
        "leakage_audit": {"violation_count": leakage_violations, "pairs_checked": 3},
        "coverage_source_counts": {name: int(len(values)) for name, values in sources.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    largest = cluster_summary["largest_clusters"][0]
    report = "\n".join(
        [
            "# Dual-UQ Identity-Disjoint Split Freeze",
            "",
            f"- VERDICT: `{status}`",
            f"- Clustering: MMseqs2 `{manifest['tool']['version']}`, min identity `{args.threshold}`, coverage `{args.coverage}`, cov-mode `{args.cov_mode}` ({manifest['tool']['cov_mode_semantics']}), mode `{args.clustering_mode}` ({manifest['tool']['clustering_mode_semantics']}).",
            f"- Proteins: `{cluster_summary['protein_count']}`; clusters: `{cluster_summary['cluster_count']}`; singleton fraction: `{cluster_summary['singleton_fraction']}`; largest cluster: `{largest}`.",
            f"- Canonical sequence freeze: `{sequence_summary['protein_count']}` proteins, `{sequence_summary['unique_sequence_hash_count']}` unique sequence hashes, conflicting duplicate identity: `{sequence_summary['conflicting_duplicate_identity']}`, nonstandard-character records retained exactly: `{sequence_summary['nonstandard_character_records']}`.",
            f"- Split target: deterministic `{args.seed}` assignment of complete clusters to TRAIN/VALIDATION/LOCKED_TEST.",
            f"- Leakage audit: `{leakage_violations}` cross-split violations across TRAIN/VALIDATION, TRAIN/LOCKED_TEST, and VALIDATION/LOCKED_TEST.",
            f"- Algorithm selection: `{manifest['tool']['algorithm_selection_note']}`",
            "- Terminology: this is an internal Dual-UQ 30%-identity cluster split, not UniRef30 and not an external prospective benchmark.",
            "- No model response, geometry, generation, or downstream outcome was used for clustering or balancing.",
            "- V1 training was not started.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"{status} proteins={cluster_summary['protein_count']} clusters={cluster_summary['cluster_count']} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
