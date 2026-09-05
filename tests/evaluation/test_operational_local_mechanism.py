import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.cross_model_representation_mechanism import (
    local_geometry_from_cases,
)
from dual_uq.evaluation.pair_conditioned_local_mechanism import (
    pair_conditioned_cross_model_hotspots,
    pair_conditioned_local_associations,
)


def _backbone(length: int = 9) -> np.ndarray:
    return np.asarray(
        [
            np.stack(
                [
                    np.array([3.7 * index - 1.2, np.sin(index) + 0.3, np.cos(index) + 0.1]),
                    np.array([3.7 * index, np.sin(index), np.cos(index)]),
                    np.array([3.7 * index + 1.3, np.sin(index) + 0.2, np.cos(index) - 0.2]),
                ]
            )
            for index in range(length)
        ]
    )


def test_operational_geometry_reuses_rigid_invariant_descriptor_protocol() -> None:
    reference = _backbone()
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    comparison = reference @ rotation.T + np.asarray([4.0, -2.0, 7.0])
    rows = []
    for condition, coordinates in (
        ("CONDITION_1", reference),
        ("CONDITION_2", comparison),
    ):
        for index, position in enumerate(range(1, 10)):
            rows.append(
                {
                    "pair_id": "p1",
                    "protein_id": "P1",
                    "identity_cluster_id": "C1",
                    "condition": condition,
                    "canonical_position": position,
                    "coordinates": coordinates.tolist() if index == 0 else None,
                }
            )

    geometry = local_geometry_from_cases(pd.DataFrame(rows))

    assert len(geometry) == 9
    assert geometry["ca_displacement"].max() == pytest.approx(0.0, abs=1e-10)
    assert geometry["fragment_7_rmsd"].dropna().max() == pytest.approx(0.0, abs=1e-10)
    assert geometry["torsion_phi_psi_change"].dropna().max() == pytest.approx(0.0, abs=1e-10)
    assert geometry["neighborhood_distance_deformation"].max() == pytest.approx(0.0, abs=1e-10)


def _response_rows(model: str, pair_id: str) -> list[dict[str, object]]:
    return [
        {
            "model_id": model,
            "checkpoint_id": f"{model}-checkpoint",
            "protein_id": "P1",
            "identity_cluster_id": "C1",
            "pair_id": pair_id,
            "canonical_position": position,
            "jsd_bits": float(position),
            "ca_displacement": float(position),
        }
        for position in range(1, 5)
    ]


def test_pair_conditioned_analysis_does_not_require_controlled_dose_metadata() -> None:
    local = pd.DataFrame(_response_rows("m1", "p1"))

    result = pair_conditioned_local_associations(local, descriptors=("ca_displacement",))

    assert result.loc[0, "spearman_rho"] == pytest.approx(1.0)
    assert "requested_dose" not in result


def test_cross_model_hotspots_use_each_model_pairs_actual_shared_subset() -> None:
    local = pd.DataFrame(
        _response_rows("m1", "p1")
        + _response_rows("m1", "p2")
        + _response_rows("m2", "p1")
        + _response_rows("m2", "p2")
        + _response_rows("m3", "p2")
    )

    result = pair_conditioned_cross_model_hotspots(local)
    counts = result.groupby(["left_model_id", "right_model_id"])["pair_id"].nunique()

    assert counts.to_dict() == {
        ("m1", "m2"): 2,
        ("m1", "m3"): 1,
        ("m2", "m3"): 1,
    }
    assert result["status"].eq("VALID").all()
