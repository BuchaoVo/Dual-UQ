import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.cross_model_representation_mechanism import (
    cross_model_localization,
    paired_local_geometry,
    within_protein_spearman,
)


def _backbone(length: int = 9) -> np.ndarray:
    coordinates = []
    for index in range(length):
        origin = np.array([3.7 * index, np.sin(index), np.cos(index)])
        coordinates.append(
            np.stack(
                [
                    origin + np.array([-1.2, 0.3, 0.1]),
                    origin,
                    origin + np.array([1.3, 0.2, -0.2]),
                ]
            )
        )
    return np.asarray(coordinates)


def test_local_geometry_has_a_rigid_transform_floor_at_zero() -> None:
    reference = _backbone()
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = reference @ rotation.T + np.array([10.0, -4.0, 7.0])

    result = paired_local_geometry(
        reference,
        transformed,
        canonical_positions=tuple(range(1, 10)),
    )

    assert result["ca_displacement"].max() == pytest.approx(0.0, abs=1e-10)
    assert result["fragment_7_rmsd"].dropna().max() == pytest.approx(0.0, abs=1e-10)
    assert result["torsion_phi_psi_change"].dropna().max() == pytest.approx(
        0.0, abs=1e-10
    )
    assert result["neighborhood_distance_deformation"].max() == pytest.approx(
        0.0, abs=1e-10
    )
    assert result["fragment_7_rmsd"].notna().sum() == 3


def test_within_protein_association_and_cross_model_hotspots_preserve_protein_unit() -> None:
    local = pd.DataFrame(
        [
            {
                "model_id": model,
                "checkpoint_id": "c",
                "protein_id": protein,
                "identity_cluster_id": cluster,
                "pair_id": "pair",
                "canonical_position": position,
                "jsd_bits": multiplier * position,
                "ca_displacement": position,
            }
            for model, multiplier in (("m1", 1.0), ("m2", 2.0))
            for protein, cluster in (("P1", "C1"), ("P2", "C2"))
            for position in range(1, 6)
        ]
    )
    # A position present in only one model is not a shared hotspot observation.
    local = pd.concat(
        [
            local,
            pd.DataFrame(
                [
                    {
                        "model_id": "m1",
                        "checkpoint_id": "c",
                        "protein_id": "P1",
                        "identity_cluster_id": "C1",
                        "pair_id": "pair",
                        "canonical_position": 99,
                        "jsd_bits": 100.0,
                        "ca_displacement": 100.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    associations = within_protein_spearman(
        local, descriptors=("ca_displacement",)
    )
    localization = cross_model_localization(local)

    assert len(associations) == 4
    assert associations["spearman_rho"].tolist() == pytest.approx([1.0] * 4)
    assert len(localization) == 2
    assert localization["spearman_rho"].tolist() == pytest.approx([1.0, 1.0])
    assert localization["n_matched_observations"].tolist() == [5, 5]
