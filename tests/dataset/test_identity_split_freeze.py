from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.dataset.identity_split import (
    IdentitySplitInputError,
    build_canonical_sequence_table,
    build_connected_identity_assignments,
    characterize_identity_clusters,
    load_mmseqs_cluster_assignments,
)


def test_canonical_sequence_table_binds_one_sequence_per_protein(tmp_path: Path) -> None:
    (tmp_path / "U1.json").write_text(
        json.dumps({"primaryAccession": "U1", "sequence": {"value": "ACDE", "length": 4}}),
        encoding="utf-8",
    )
    candidates = pd.DataFrame({"protein_id": ["P1"], "pair_id": ["pair1"], "uniprot_id": ["U1"]})
    table = build_canonical_sequence_table(candidates, tmp_path)
    assert table.loc[0, "sequence"] == "ACDE"
    assert table.loc[0, "canonical_length"] == 4
    assert len(table.loc[0, "sequence_hash"]) == 64


def test_canonical_sequence_table_rejects_conflicting_duplicate_identity(tmp_path: Path) -> None:
    (tmp_path / "U1.json").write_text(
        json.dumps({"primaryAccession": "U1", "sequence": {"value": "ACDE", "length": 4}}),
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        {"protein_id": ["P1", "P1"], "pair_id": ["pair1", "pair2"], "uniprot_id": ["U1", "U1"]}
    )
    table = build_canonical_sequence_table(candidates, tmp_path)
    assert table["protein_id"].is_unique


def test_canonical_sequence_table_reports_missing_source(tmp_path: Path) -> None:
    candidates = pd.DataFrame({"protein_id": ["P1"], "pair_id": ["pair1"], "uniprot_id": ["U1"]})
    with pytest.raises(IdentitySplitInputError, match="missing canonical sequence"):
        build_canonical_sequence_table(candidates, tmp_path)


def test_cluster_characterization_reports_singletons_and_largest() -> None:
    assignments = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "sequence_cluster_id": ["C1", "C1", "C2"],
            "canonical_length": [100, 120, 80],
        }
    )
    summary = characterize_identity_clusters(assignments)
    assert summary["protein_count"] == 3
    assert summary["cluster_count"] == 2
    assert summary["singleton_cluster_count"] == 1
    assert summary["largest_clusters"][0]["cluster_id"] == "C1"


def test_mmseqs_cluster_parser_requires_exact_member_coverage(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("P1\tP1\nP1\tP2\n", encoding="utf-8")
    sequence_table = pd.DataFrame({"protein_id": ["P1", "P2"]})
    assignments = load_mmseqs_cluster_assignments(path, sequence_table)
    assert assignments.to_dict("records") == [
        {"protein_id": "P1", "sequence_cluster_id": "P1"},
        {"protein_id": "P2", "sequence_cluster_id": "P1"},
    ]


def test_connected_identity_assignments_join_transitive_hits() -> None:
    sequence_table = pd.DataFrame({"protein_id": ["P1", "P2", "P3"]})
    hits = pd.DataFrame({"query": ["P1", "P2"], "target": ["P2", "P3"]})
    assignments = build_connected_identity_assignments(hits, sequence_table)
    assert assignments["sequence_cluster_id"].nunique() == 1
    assert assignments["sequence_cluster_id"].iloc[0] == "P1"
