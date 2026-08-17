from __future__ import annotations

import numpy as np

from dual_uq.dataset.apo_holo import (
    _pair_common_coordinates,
    characterize_coordinates,
)


def test_geometry_uses_common_positions_and_ligand_partition() -> None:
    apo = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
    holo = apo.copy()
    holo[1, 1] = 1.0
    result = characterize_coordinates(
        apo,
        holo,
        canonical_positions=[1, 2, 3],
        ligand_proximal=[False, True, False],
        contact_cutoff_angstrom=4.5,
    )
    assert len(result.residue_table) == 3
    assert result.residue_table.loc[result.residue_table["canonical_position"] == 2, "ligand_proximal"].item()
    assert result.pair_summary["common_residue_count"] == 3
    assert result.pair_summary["aligned_ca_rmsd"] >= 0.0


def test_pair_common_coordinates_uses_sorted_canonical_intersection() -> None:
    apo = {
        4: np.array([4.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        3: np.array([3.0, 0.0, 0.0]),
    }
    holo = {
        3: np.array([3.0, 1.0, 0.0]),
        2: np.array([2.0, 1.0, 0.0]),
        1: np.array([1.0, 1.0, 0.0]),
        4: np.array([4.0, 1.0, 0.0]),
    }
    positions, apo_array, holo_array = _pair_common_coordinates(apo, holo)
    assert positions == [1, 3, 4]
    assert apo_array.shape == (3, 3)
    assert holo_array.shape == (3, 3)
