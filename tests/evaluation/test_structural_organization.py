from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.structural_organization import (
    StructuralOrganizationError,
    _circular_delta_degrees,
    _derive_pair_descriptors,
    _spatial_null_for_protein,
)


def _backbone(offset: float = 0.0) -> dict[int, dict[str, np.ndarray]]:
    """Small four-residue backbone with a controlled local-distance change."""
    result: dict[int, dict[str, np.ndarray]] = {}
    for position in range(1, 5):
        x = float(position) * 3.8 + offset
        result[position] = {
            "N": np.array([x - 1.2, 0.0, 0.0]),
            "CA": np.array([x, 0.0, 0.0]),
            "C": np.array([x + 1.2, 0.1, 0.0]),
            "O": np.array([x + 1.6, 0.1, 0.0]),
        }
    return result


def test_circular_deltas_use_shortest_arc_and_keep_missing_missing() -> None:
    assert _circular_delta_degrees(179.0, -179.0) == pytest.approx(2.0)
    assert np.isnan(_circular_delta_degrees(np.nan, 20.0))
    assert np.isnan(_circular_delta_degrees(20.0, np.nan))


def test_pair_descriptors_include_alignment_and_relational_geometry() -> None:
    pdb = _backbone()
    afdb = _backbone()
    afdb[3] = {name: value.copy() for name, value in afdb[3].items()}
    afdb[3]["CA"][1] = 2.0
    afdb[3]["N"][1] = 1.8
    afdb[3]["C"][1] = 2.1
    afdb[3]["O"][1] = 2.1
    descriptors = _derive_pair_descriptors(
        protein_id="p1",
        common_positions=(1, 2, 3, 4),
        pdb_backbone=pdb,
        afdb_backbone=afdb,
        torsions_pdb={},
        torsions_afdb={},
    )
    assert len(descriptors) == 4
    assert descriptors.loc[descriptors["position"] == 3, "ca_displacement"].iloc[0] > 0
    assert "local_pairwise_distance_change" in descriptors
    assert "contact_turnover_fraction" in descriptors
    assert descriptors["geometry_status"].eq("available").all()


def test_missing_torsions_are_nan_not_zero() -> None:
    descriptors = _derive_pair_descriptors(
        protein_id="p1",
        common_positions=(1, 2),
        pdb_backbone=_backbone(),
        afdb_backbone=_backbone(),
        torsions_pdb={1: {"phi": np.nan, "psi": 30.0, "omega": np.nan}},
        torsions_afdb={1: {"phi": 0.0, "psi": np.nan, "omega": 20.0}},
    )
    row = descriptors.loc[descriptors["position"] == 1].iloc[0]
    assert np.isnan(row["torsion_delta_phi"])
    assert np.isnan(row["torsion_delta_psi"])
    assert np.isnan(row["torsion_delta_omega"])


def test_spatial_null_is_within_protein_and_deterministic() -> None:
    positions = pd.DataFrame(
        {
            "protein_id": ["p1"] * 4,
            "position": [1, 2, 3, 4],
            "magnitude_p": [0.1, 0.2, 0.8, 0.9],
            "breadth_b": [0.2, 0.3, 0.7, 0.8],
            "rank_displacement": [0.1, 0.2, 0.8, 0.9],
        }
    )
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)
    first = _spatial_null_for_protein(positions, edges, "magnitude_p", permutations=25, seed=7)
    second = _spatial_null_for_protein(positions, edges, "magnitude_p", permutations=25, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "protein_id"] == "p1"
    assert first.loc[0, "metric"] == "magnitude_p"
    assert first.loc[0, "permutation_count"] == 25


def test_spatial_null_rejects_cross_protein_edges() -> None:
    positions = pd.DataFrame(
        {
            "protein_id": ["p1", "p1", "p2", "p2"],
            "position": [1, 2, 1, 2],
            "magnitude_p": [0.1, 0.2, 0.8, 0.9],
        }
    )
    with pytest.raises(StructuralOrganizationError, match="same protein"):
        _spatial_null_for_protein(positions, np.array([[0, 2]], dtype=int), "magnitude_p")


def test_analysis_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analysis/analyze_structural_organization.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
