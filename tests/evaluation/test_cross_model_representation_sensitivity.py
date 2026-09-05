import math

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.cross_model_representation_sensitivity import (
    attach_controlled_metadata,
    build_response_rows,
    cluster_bootstrap_summary,
    controlled_severity_differences,
    paired_probability_metrics,
    paired_protein_differences,
    protein_level_response,
    standard_probability_quality,
)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _rows(*indices: int, confidence: float = 0.9) -> np.ndarray:
    result = np.full((len(indices), 20), (1.0 - confidence) / 19.0)
    result[np.arange(len(indices)), indices] = confidence
    return result


def test_identical_probability_views_have_zero_response() -> None:
    probabilities = _rows(0, 1, 2)

    result = paired_probability_metrics(probabilities, probabilities, "ACD")

    assert np.array_equal(result["residue_jsd_bits"], np.zeros(3))
    assert result["r_jsd_bits"] == 0.0
    assert result["r_flip"] == 0.0
    assert result["r_prob"] == 0.0
    assert result["abs_delta_nll"] == 0.0


def test_probability_pair_metrics_preserve_bits_and_native_probability_semantics() -> None:
    left = _rows(0, confidence=0.9)
    right = _rows(1, confidence=0.9)

    result = paired_probability_metrics(left, right, "A")

    expected_js = 0.9 * math.log2(0.9 / ((0.9 + 0.1 / 19.0) / 2.0))
    expected_js += (0.1 / 19.0) * math.log2(
        (0.1 / 19.0) / ((0.9 + 0.1 / 19.0) / 2.0)
    )
    assert result["r_jsd_bits"] == pytest.approx(expected_js)
    assert result["r_flip"] == 1.0
    assert result["r_prob"] == pytest.approx(0.9 - 0.1 / 19.0)
    assert result["abs_delta_nll"] == pytest.approx(
        abs(-math.log(0.9) + math.log(0.1 / 19.0))
    )


def test_standard_quality_uses_native_top1_and_mean_nll() -> None:
    probabilities = _rows(0, 2, confidence=0.8)

    result = standard_probability_quality(probabilities, "AC")

    expected_nll = (-math.log(0.8) - math.log(0.2 / 19.0)) / 2.0
    assert result == pytest.approx(
        {"recovery": 0.5, "nll": expected_nll, "perplexity": math.exp(expected_nll)}
    )


@pytest.mark.parametrize(
    ("left", "right", "sequence", "message"),
    [
        (_rows(0), _rows(0) * 2.0, "A", "normalized"),
        (_rows(0), np.full((1, 20), np.nan), "A", "finite"),
        (_rows(0), _rows(0, 1), "A", "same shape"),
        (_rows(0), _rows(0), "X", "standard-AA"),
    ],
)
def test_probability_pair_metrics_reject_invalid_contracts(
    left: np.ndarray,
    right: np.ndarray,
    sequence: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_probability_metrics(left, right, sequence)


def test_response_rows_preserve_pair_identity_and_standard_quality() -> None:
    cases = pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "protein_id": "P1",
                "identity_cluster_id": "C1",
                "split": "LOCKED_TEST",
                "track_or_diagnostic": "TRACK_I",
                "state_family": None,
                "condition": condition,
                "condition_label": label,
                "canonical_position": position,
                "wt_sequence_projection": "AC",
                "n_canonical_positions": 2,
            }
            for condition, label in (("CONDITION_1", "PDB"), ("CONDITION_2", "AFDB"))
            for position in (1, 2)
        ]
    )
    left = _rows(0, 1)
    right = _rows(1, 1)

    residue, pair, quality = build_response_rows(
        cases,
        {("pair-1", "CONDITION_1"): left, ("pair-1", "CONDITION_2"): right},
        model_id="model",
        checkpoint_id="checkpoint",
        semantic_class="L2",
        regime="operational_pdb_afdb",
    )

    assert residue[["pair_id", "canonical_position"]].values.tolist() == [
        ["pair-1", 1],
        ["pair-1", 2],
    ]
    assert pair.loc[0, "r_flip"] == 0.5
    assert pair.loc[0, "n_evaluable_positions"] == 2
    assert pair.loc[0, "reference_label"] == "PDB"
    assert pair.loc[0, "comparison_label"] == "AFDB"
    assert quality.loc[0, "recovery"] == 1.0
    assert quality.loc[0, "reference_label"] == "PDB"


def test_controlled_metadata_join_is_complete_and_one_to_one() -> None:
    response = pd.DataFrame(
        [{"pair_id": "p1", "protein_id": "P1", "requested_dose": np.nan}]
    )
    metadata = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "protein_id": "P1",
                "perturbation_family": "SMOOTH_COORDINATE_FIELD",
                "requested_dose": 0.25,
                "requested_dose_unit": "angstrom",
            }
        ]
    )

    enriched = attach_controlled_metadata(response, metadata)

    assert enriched.loc[0, "perturbation_family"] == "SMOOTH_COORDINATE_FIELD"
    assert enriched.loc[0, "requested_dose"] == 0.25
    with pytest.raises(ValueError, match="complete"):
        attach_controlled_metadata(response, metadata.iloc[0:0])


def test_protein_aggregation_precedes_cluster_bootstrap() -> None:
    pairs = pd.DataFrame(
        [
            {"model_id": "m", "checkpoint_id": "c", "regime": "controlled", "protein_id": "P1", "identity_cluster_id": "C1", "pair_id": "p1", "r_jsd_bits": 1.0},
            {"model_id": "m", "checkpoint_id": "c", "regime": "controlled", "protein_id": "P1", "identity_cluster_id": "C1", "pair_id": "p2", "r_jsd_bits": 3.0},
            {"model_id": "m", "checkpoint_id": "c", "regime": "controlled", "protein_id": "P2", "identity_cluster_id": "C2", "pair_id": "p3", "r_jsd_bits": 10.0},
        ]
    )

    proteins = protein_level_response(pairs, metrics=("r_jsd_bits",))
    summary = cluster_bootstrap_summary(
        proteins, value_column="r_jsd_bits", bootstrap_replicates=100, seed=7
    )

    assert proteins[["protein_id", "r_jsd_bits", "n_pairs"]].to_dict("records") == [
        {"protein_id": "P1", "r_jsd_bits": 2.0, "n_pairs": 2},
        {"protein_id": "P2", "r_jsd_bits": 10.0, "n_pairs": 1},
    ]
    assert summary["mean"] == 6.0
    assert summary["median"] == 6.0
    assert summary["n_proteins"] == 2
    assert summary["n_identity_clusters"] == 2

    median_summary = cluster_bootstrap_summary(
        proteins,
        value_column="r_jsd_bits",
        bootstrap_replicates=100,
        seed=7,
        bootstrap_statistic="median",
    )
    assert median_summary["bootstrap_statistic"] == "median"
    assert median_summary["median"] == 6.0


def test_median_cluster_bootstrap_changes_the_ci_statistic_not_the_point_fields() -> None:
    proteins = pd.DataFrame(
        [
            {"protein_id": "P0", "identity_cluster_id": "C0", "r_jsd_bits": 12.0},
            {"protein_id": "P1", "identity_cluster_id": "C0", "r_jsd_bits": 13.0},
            {"protein_id": "P2", "identity_cluster_id": "C1", "r_jsd_bits": 15.0},
            {"protein_id": "P3", "identity_cluster_id": "C1", "r_jsd_bits": 12.0},
            {"protein_id": "P4", "identity_cluster_id": "C1", "r_jsd_bits": 14.0},
            {"protein_id": "P5", "identity_cluster_id": "C1", "r_jsd_bits": 18.0},
            {"protein_id": "P6", "identity_cluster_id": "C2", "r_jsd_bits": 18.0},
            {"protein_id": "P7", "identity_cluster_id": "C2", "r_jsd_bits": 18.0},
            {"protein_id": "P8", "identity_cluster_id": "C2", "r_jsd_bits": 17.0},
            {"protein_id": "P9", "identity_cluster_id": "C2", "r_jsd_bits": 14.0},
            {"protein_id": "P10", "identity_cluster_id": "C3", "r_jsd_bits": 18.0},
            {"protein_id": "P11", "identity_cluster_id": "C3", "r_jsd_bits": 0.0},
            {"protein_id": "P12", "identity_cluster_id": "C4", "r_jsd_bits": 0.0},
        ]
    )

    mean_summary = cluster_bootstrap_summary(
        proteins,
        value_column="r_jsd_bits",
        bootstrap_replicates=1_000,
        seed=7,
    )
    median_summary = cluster_bootstrap_summary(
        proteins,
        value_column="r_jsd_bits",
        bootstrap_replicates=1_000,
        seed=7,
        bootstrap_statistic="median",
    )

    assert median_summary["median"] == 14.0
    assert median_summary["ci_low"] == pytest.approx(6.0)
    assert median_summary["ci_high"] == pytest.approx(17.0)
    assert (median_summary["ci_low"], median_summary["ci_high"]) != (
        mean_summary["ci_low"],
        mean_summary["ci_high"],
    )


def test_paired_floor_and_severity_differences_use_matched_proteins() -> None:
    exact = pd.DataFrame(
        [
            {"protein_id": "P1", "identity_cluster_id": "C1", "r_jsd_bits": 0.1},
            {"protein_id": "P2", "identity_cluster_id": "C2", "r_jsd_bits": 0.2},
        ]
    )
    response = exact.assign(r_jsd_bits=[0.4, 0.8])

    difference = paired_protein_differences(
        response, exact, value_column="r_jsd_bits", difference_column="above_floor"
    )

    assert difference[["protein_id", "above_floor"]].to_dict("records") == [
        {"protein_id": "P1", "above_floor": pytest.approx(0.3)},
        {"protein_id": "P2", "above_floor": pytest.approx(0.6)},
    ]

    controlled = pd.DataFrame(
        [
            {"model_id": "m", "checkpoint_id": "c", "protein_id": protein, "identity_cluster_id": cluster, "pair_id": f"{protein}-{dose}-{replicate}", "requested_dose": dose, "r_jsd_bits": value}
            for protein, cluster, values in (("P1", "C1", (1.0, 3.0)), ("P2", "C2", (2.0, 6.0)))
            for dose, value in zip((0.25, 0.5), values, strict=True)
            for replicate in range(2)
        ]
    )
    severity = controlled_severity_differences(
        controlled, metrics=("r_jsd_bits",)
    )
    assert severity[["protein_id", "r_jsd_bits_high_minus_low"]].to_dict("records") == [
        {"protein_id": "P1", "r_jsd_bits_high_minus_low": 2.0},
        {"protein_id": "P2", "r_jsd_bits_high_minus_low": 4.0},
    ]
