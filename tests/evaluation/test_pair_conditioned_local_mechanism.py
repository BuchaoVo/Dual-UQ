import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.pair_conditioned_local_mechanism import (
    aggregate_pair_correlations_by_protein,
    hotspot_permutation_null,
    pair_conditioned_cross_model_hotspots,
    pair_conditioned_local_associations,
    pair_fixed_effect_local_associations,
    permuted_pair_values,
    summarize_protein_correlations,
)


def _local_rows(
    *,
    model: str,
    pair_id: str,
    descriptor: list[float],
    response: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "model_id": model,
            "checkpoint_id": f"{model}-checkpoint",
            "protein_id": "P1",
            "identity_cluster_id": "C1",
            "pair_id": pair_id,
            "canonical_position": position,
            "requested_dose": 0.25 if pair_id == "pair-1" else 0.5,
            "jsd_bits": jsd,
            "ca_displacement": deformation,
        }
        for position, (deformation, jsd) in enumerate(
            zip(descriptor, response, strict=True), start=1
        )
    ]


def test_pair_conditioned_association_never_pools_distinct_pairs() -> None:
    local = pd.DataFrame(
        _local_rows(
            model="m1",
            pair_id="pair-1",
            descriptor=[0.0, 1.0, 2.0],
            response=[0.0, 1.0, 2.0],
        )
        + _local_rows(
            model="m1",
            pair_id="pair-2",
            descriptor=[0.0, 1.0, 2.0],
            response=[2.0, 1.0, 0.0],
        )
    )

    result = pair_conditioned_local_associations(local, descriptors=("ca_displacement",))

    assert result[["pair_id", "spearman_rho", "status"]].to_dict("records") == [
        {"pair_id": "pair-1", "spearman_rho": 1.0, "status": "VALID"},
        {"pair_id": "pair-2", "spearman_rho": -1.0, "status": "VALID"},
    ]


def test_undefined_pair_correlation_is_retained_with_reason() -> None:
    local = pd.DataFrame(
        _local_rows(
            model="m1",
            pair_id="pair-1",
            descriptor=[1.0, 1.0, 1.0],
            response=[0.0, 1.0, 2.0],
        )
    )

    result = pair_conditioned_local_associations(local, descriptors=("ca_displacement",))

    assert len(result) == 1
    assert result.loc[0, "status"] == "UNDEFINED"
    assert result.loc[0, "reason"] == "CONSTANT_DESCRIPTOR"
    assert np.isnan(result.loc[0, "spearman_rho"])


def test_protein_aggregation_uses_median_over_valid_pairs_and_median_bootstrap() -> None:
    pair_correlations = pd.DataFrame(
        [
            {
                "model_id": "m1",
                "checkpoint_id": "c1",
                "protein_id": protein,
                "identity_cluster_id": cluster,
                "descriptor": "ca_displacement",
                "pair_id": pair,
                "spearman_rho": rho,
                "status": "VALID",
                "reason": None,
            }
            for protein, cluster, pair, rho in (
                ("P1", "C1", "p1", -0.8),
                ("P1", "C1", "p2", 0.2),
                ("P1", "C1", "p3", 0.9),
                ("P2", "C2", "p4", 0.5),
            )
        ]
    )

    proteins = aggregate_pair_correlations_by_protein(
        pair_correlations,
        group_columns=(
            "model_id",
            "checkpoint_id",
            "protein_id",
            "identity_cluster_id",
            "descriptor",
        ),
    )
    summary = summarize_protein_correlations(
        proteins,
        group_columns=("model_id", "checkpoint_id", "descriptor"),
        bootstrap_replicates=100,
        seed=7,
    )

    assert proteins.set_index("protein_id").loc["P1", "spearman_rho"] == 0.2
    assert summary.loc[0, "bootstrap_statistic"] == "median"
    assert summary.loc[0, "n_valid_pairs"] == 4
    assert summary.loc[0, "n_proteins"] == 2


def test_pair_fixed_effect_centers_each_pair_before_protein_pooling() -> None:
    local = pd.DataFrame(
        _local_rows(
            model="m1",
            pair_id="pair-1",
            descriptor=[0.0, 1.0, 2.0],
            response=[100.0, 101.0, 102.0],
        )
        + _local_rows(
            model="m1",
            pair_id="pair-2",
            descriptor=[100.0, 101.0, 102.0],
            response=[0.0, 1.0, 2.0],
        )
    )

    result = pair_fixed_effect_local_associations(local, descriptors=("ca_displacement",))

    assert result.loc[0, "spearman_rho"] == pytest.approx(1.0)
    assert result.loc[0, "n_valid_pairs"] == 2
    assert result.loc[0, "n_centered_residues"] == 6


def test_cross_model_hotspots_match_only_pair_and_canonical_position() -> None:
    local = pd.DataFrame(
        _local_rows(
            model="m1",
            pair_id="pair-1",
            descriptor=[0.0, 1.0, 2.0],
            response=[0.1, 0.2, 0.3],
        )
        + list(
            reversed(
                _local_rows(
                    model="m2",
                    pair_id="pair-1",
                    descriptor=[0.0, 1.0, 2.0],
                    response=[0.2, 0.4, 0.6],
                )
            )
        )
    )
    extra = local.iloc[[0]].assign(
        canonical_position=99,
        jsd_bits=99.0,
        model_local_index=0,
    )
    local["model_local_index"] = np.arange(len(local))[::-1]
    local = pd.concat([local, extra], ignore_index=True)

    result = pair_conditioned_cross_model_hotspots(local)

    assert result.loc[0, "spearman_rho"] == pytest.approx(1.0)
    assert result.loc[0, "n_shared_residues"] == 3
    assert result.loc[0, "status"] == "VALID"


def test_pair_permutation_preserves_values_and_is_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 5.0])

    first = permuted_pair_values(values, permutations=8, rng=np.random.default_rng(11))
    second = permuted_pair_values(values, permutations=8, rng=np.random.default_rng(11))

    assert np.array_equal(first, second)
    assert all(np.array_equal(np.sort(row), np.sort(values)) for row in first)


def test_hotspot_permutation_null_is_deterministic_and_keeps_pair_hierarchy() -> None:
    local = pd.DataFrame(
        _local_rows(
            model="m1",
            pair_id="pair-1",
            descriptor=[0.0, 1.0, 2.0, 3.0],
            response=[0.1, 0.2, 0.3, 0.4],
        )
        + _local_rows(
            model="m2",
            pair_id="pair-1",
            descriptor=[0.0, 1.0, 2.0, 3.0],
            response=[0.4, 0.1, 0.3, 0.2],
        )
    )
    hotspots = pair_conditioned_cross_model_hotspots(local)

    first_null, first_summary = hotspot_permutation_null(local, hotspots, permutations=20, seed=13)
    second_null, second_summary = hotspot_permutation_null(
        local, hotspots, permutations=20, seed=13
    )

    pd.testing.assert_frame_equal(first_null, second_null)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert len(first_null) == 20
    assert first_null["n_valid_pairs"].eq(1).all()
    assert first_null["n_valid_proteins"].eq(1).all()
