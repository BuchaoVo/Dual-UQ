"""Canonical sequence and identity-cluster utilities for the internal split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class IdentitySplitInputError(ValueError):
    """Raised when a canonical sequence or cluster artifact is not usable."""


_REQUIRED_CANDIDATE_COLUMNS = ("protein_id", "pair_id", "uniprot_id")


def _sha256_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def build_canonical_sequence_table(candidates: pd.DataFrame, sequence_dir: Path) -> pd.DataFrame:
    """Bind exactly one local UniProt canonical sequence to each protein."""

    missing = sorted(set(_REQUIRED_CANDIDATE_COLUMNS).difference(candidates.columns))
    if missing:
        raise IdentitySplitInputError(f"candidates missing required columns: {missing}")
    if candidates.empty:
        raise IdentitySplitInputError("candidates is empty")
    rows = candidates[list(_REQUIRED_CANDIDATE_COLUMNS)].copy()
    rows = rows.astype({"protein_id": str, "pair_id": str, "uniprot_id": str})
    mapping_counts = rows.groupby("protein_id")["uniprot_id"].nunique()
    if (mapping_counts > 1).any():
        raise IdentitySplitInputError("duplicate protein identity maps to conflicting UniProt accessions")
    rows = rows.drop_duplicates("protein_id", keep="first")
    records: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        source = Path(sequence_dir) / f"{row.uniprot_id}.json"
        if not source.is_file():
            raise IdentitySplitInputError(f"missing canonical sequence for {row.uniprot_id}: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            accession = payload.get("primaryAccession")
            sequence_payload = payload["sequence"]
            sequence = sequence_payload["value"]
            length = int(sequence_payload["length"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise IdentitySplitInputError(f"invalid canonical sequence record: {source}") from exc
        if accession != row.uniprot_id:
            raise IdentitySplitInputError(
                f"canonical accession mismatch for {row.uniprot_id}: returned {accession}"
            )
        if type(sequence) is not str or not sequence or sequence != sequence.strip():
            raise IdentitySplitInputError(f"invalid canonical sequence text for {row.uniprot_id}")
        if len(sequence) != length:
            raise IdentitySplitInputError(f"canonical sequence length mismatch for {row.uniprot_id}")
        records.append(
            {
                "protein_id": row.protein_id,
                "pair_id": row.pair_id,
                "uniprot_id": row.uniprot_id,
                "sequence": sequence,
                "canonical_length": length,
                "sequence_hash": _sha256_sequence(sequence),
                "sequence_source_relative_path": source.as_posix(),
            }
        )
    result = pd.DataFrame(records).sort_values("protein_id", kind="mergesort").reset_index(drop=True)
    conflict = result.groupby("uniprot_id")["sequence_hash"].nunique()
    if (conflict > 1).any():
        raise IdentitySplitInputError("one UniProt accession maps to conflicting canonical sequences")
    return result


def write_canonical_fasta(sequence_table: pd.DataFrame, path: Path) -> None:
    """Write deterministic FASTA records keyed by frozen protein identity."""

    required = {"protein_id", "sequence"}
    missing = sorted(required.difference(sequence_table.columns))
    if missing:
        raise IdentitySplitInputError(f"sequence table missing columns: {missing}")
    ordered = sequence_table.sort_values("protein_id", kind="mergesort")
    lines: list[str] = []
    for row in ordered.itertuples(index=False):
        lines.extend([f">{row.protein_id}", str(row.sequence)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_mmseqs_cluster_assignments(cluster_tsv: Path, sequence_table: pd.DataFrame) -> pd.DataFrame:
    """Convert MMseqs2 representative/member adjacency output to cluster rows."""

    try:
        edges = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["identity_cluster_id", "protein_id"], dtype=str)
    except (OSError, pd.errors.ParserError) as exc:
        raise IdentitySplitInputError(f"cannot read MMseqs2 cluster output: {cluster_tsv}") from exc
    if edges.empty or edges["protein_id"].duplicated().any():
        raise IdentitySplitInputError("MMseqs2 cluster output is empty or has duplicate members")
    expected = set(sequence_table["protein_id"].astype(str))
    observed = set(edges["protein_id"])
    if observed != expected:
        raise IdentitySplitInputError(
            f"MMseqs2 cluster coverage mismatch: missing={len(expected - observed)} extra={len(observed - expected)}"
        )
    assignments = edges.rename(columns={"identity_cluster_id": "sequence_cluster_id"})
    return assignments[["protein_id", "sequence_cluster_id"]].sort_values("protein_id", kind="mergesort").reset_index(drop=True)


def build_connected_identity_assignments(
    hits: pd.DataFrame, sequence_table: pd.DataFrame
) -> pd.DataFrame:
    """Build deterministic connected components from thresholded MMseqs hits."""

    required = {"query", "target"}
    missing = sorted(required.difference(hits.columns))
    if missing:
        raise IdentitySplitInputError(f"MMseqs hits missing columns: {missing}")
    identifiers = sorted(sequence_table["protein_id"].astype(str))
    expected = set(identifiers)
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for query, target in hits[["query", "target"]].astype(str).itertuples(index=False, name=None):
        if query not in expected or target not in expected:
            raise IdentitySplitInputError("MMseqs hit references a protein outside the frozen sequence table")
        query_root = find(query)
        target_root = find(target)
        if query_root != target_root:
            parent[target_root] = query_root
    components: dict[str, list[str]] = {}
    for identifier in identifiers:
        components.setdefault(find(identifier), []).append(identifier)
    canonical_ids = {root: min(members) for root, members in components.items()}
    return pd.DataFrame(
        {
            "protein_id": identifiers,
            "sequence_cluster_id": [canonical_ids[find(identifier)] for identifier in identifiers],
        }
    )


def characterize_identity_clusters(assignments: pd.DataFrame) -> dict[str, Any]:
    """Return descriptive cluster-size and length summaries without balancing on outcomes."""

    required = {"protein_id", "sequence_cluster_id", "canonical_length"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise IdentitySplitInputError(f"cluster assignments missing columns: {missing}")
    if assignments.empty:
        raise IdentitySplitInputError("cluster assignments are empty")
    sizes = assignments.groupby("sequence_cluster_id", sort=True).size().rename("protein_count")
    largest = [
        {"cluster_id": str(cluster_id), "protein_count": int(size)}
        for cluster_id, size in sizes.sort_values(ascending=False, kind="stable").head(10).items()
    ]
    length_summary: dict[str, dict[str, float | int]] = {}
    grouped = assignments.assign(cluster_size=assignments["sequence_cluster_id"].map(sizes)).groupby("cluster_size")
    for cluster_size, group in grouped:
        lengths = pd.to_numeric(group["canonical_length"], errors="coerce").dropna()
        length_summary[str(int(cluster_size))] = {
            "cluster_count": int(group["sequence_cluster_id"].nunique()),
            "protein_count": int(len(group)),
            "length_median": float(lengths.median()) if not lengths.empty else None,
            "length_min": int(lengths.min()) if not lengths.empty else None,
            "length_max": int(lengths.max()) if not lengths.empty else None,
        }
    return {
        "protein_count": int(len(assignments)),
        "cluster_count": int(len(sizes)),
        "singleton_cluster_count": int((sizes == 1).sum()),
        "singleton_fraction": float((sizes == 1).mean()),
        "cluster_size_quantiles": {
            "q10": float(sizes.quantile(0.10)),
            "median": float(sizes.median()),
            "q90": float(sizes.quantile(0.90)),
        },
        "largest_clusters": largest,
        "length_by_cluster_size": length_summary,
    }
