import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.controlled_operational_geometry import (
    cluster_bootstrap_spearman,
    controlled_signature_projection,
    knn_coverage_distances,
    nearest_global_rmsd_matches,
    pair_geometry_summaries,
)


def test_pair_geometry_uses_fixed_quantiles_and_global_ca_rms() -> None:
    local = pd.DataFrame(
        {
            "pair_id": ["p1"] * 4,
            "protein_id": ["P1"] * 4,
            "identity_cluster_id": ["C1"] * 4,
            "canonical_position": [1, 2, 3, 4],
            "requested_dose": [0.25] * 4,
            "ca_displacement": [0.0, 1.0, 2.0, 3.0],
            "fragment_7_rmsd": [np.nan, 2.0, 4.0, np.nan],
            "neighborhood_distance_deformation": [1.0, 1.0, 2.0, 4.0],
            "torsion_phi_psi_change": [0.0, 10.0, 20.0, 30.0],
        }
    )

    result = pair_geometry_summaries(local, regime="controlled")

    assert len(result) == 1
    assert result.loc[0, "ca_displacement_median"] == 1.5
    assert result.loc[0, "ca_displacement_q75"] == 2.25
    assert result.loc[0, "ca_displacement_q90"] == 2.7
    assert result.loc[0, "fragment_7_rmsd_median"] == 3.0
    assert result.loc[0, "aligned_ca_rmsd"] == pytest.approx(np.sqrt(3.5))
    assert result.loc[0, "mapped_residue_count"] == 4


def _features(values: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=["pair_id", "x", "y"]).assign(
        protein_id=lambda x: x["pair_id"].str.upper(),
        identity_cluster_id=lambda x: "C" + x["pair_id"],
    )


def test_signature_fit_uses_controlled_only_and_projects_without_refitting() -> None:
    controlled = _features([("c1", 0.0, 0.0), ("c2", 1.0, 2.0), ("c3", 2.0, 1.0), ("c4", 3.0, 3.0)])
    operational = _features([("o1", 4.0, 5.0)])
    shifted_operational = _features([("o1", 400.0, 500.0)])

    first, first_fit = controlled_signature_projection(
        controlled, operational, feature_columns=("x", "y")
    )
    second, second_fit = controlled_signature_projection(
        controlled, shifted_operational, feature_columns=("x", "y")
    )

    assert first_fit == second_fit
    pd.testing.assert_frame_equal(
        first.loc[first["structural_regime"].eq("controlled")].reset_index(drop=True),
        second.loc[second["structural_regime"].eq("controlled")].reset_index(drop=True),
    )
    assert first_fit["standardization_fit_regime"] == "controlled"
    assert sum(first_fit["explained_variance_ratio"]) == pytest.approx(1.0)


def test_knn_coverage_uses_operational_to_controlled_and_controlled_leave_one_out() -> None:
    controlled = np.asarray([[0.0], [1.0], [3.0]])
    operational = np.asarray([[2.0], [10.0]])

    controlled_loo, operational_distance = knn_coverage_distances(controlled, operational, k=1)

    assert controlled_loo.tolist() == pytest.approx([1.0, 1.0, 2.0])
    assert operational_distance.tolist() == pytest.approx([1.0, 7.0])


def test_global_rmsd_matching_uses_only_operational_pairs_in_controlled_range() -> None:
    controlled = _features([("c1", 0.2, 1.0), ("c2", 0.5, 4.0)]).rename(
        columns={"x": "aligned_ca_rmsd", "y": "feature"}
    )
    operational = _features(
        [("o-low", 0.1, 2.0), ("o-mid", 0.41, 3.0), ("o-high", 0.8, 5.0)]
    ).rename(columns={"x": "aligned_ca_rmsd", "y": "feature"})

    result = nearest_global_rmsd_matches(controlled, operational, feature_columns=("feature",))

    assert result["operational_pair_id"].tolist() == ["o-mid"]
    assert result.loc[0, "controlled_pair_id"] == "c2"
    assert result.loc[0, "feature_difference"] == -1.0
    assert result.loc[0, "overlap_min"] == 0.2
    assert result.loc[0, "overlap_max"] == 0.5


def test_cluster_bootstrap_spearman_is_deterministic() -> None:
    proteins = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4"],
            "identity_cluster_id": ["C1", "C2", "C3", "C4"],
            "coverage": [1.0, 2.0, 3.0, 4.0],
            "response": [2.0, 4.0, 6.0, 8.0],
        }
    )

    first = cluster_bootstrap_spearman(
        proteins, x_column="coverage", y_column="response", replicates=100, seed=7
    )
    second = cluster_bootstrap_spearman(
        proteins, x_column="coverage", y_column="response", replicates=100, seed=7
    )

    assert first == second
    assert first["spearman_rho"] == pytest.approx(1.0)
    assert first["bootstrap_replicates"] == 100
