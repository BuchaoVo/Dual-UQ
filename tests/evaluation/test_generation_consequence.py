from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from dual_uq.evaluation.generation_consequence import (
    GreedyGeneration,
    build_generation_response,
    generation_dose_response,
    greedy_multinomial,
    normalized_hamming,
    protein_generation_response,
    summarize_generation,
)


def _cases(pair_id: str = "pair-1", *, dose: float = 0.25) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "protein_id": "P1",
                "identity_cluster_id": "C1",
                "condition": condition,
                "condition_label": label,
                "canonical_position": position,
                "wt_sequence_projection": "ACD",
                "requested_dose": dose,
                "perturbation_family": "noise",
            }
            for condition, label in (("CONDITION_1", "reference"), ("CONDITION_2", "comparison"))
            for position in (1, 2, 3)
        ]
    )


def test_normalized_hamming_uses_the_matched_canonical_axis() -> None:
    assert normalized_hamming("ACDE", "ACAE") == 0.25
    with pytest.raises(ValueError, match="equal positive length"):
        normalized_hamming("AC", "A")
    with pytest.raises(ValueError, match="standard 20-AA"):
        normalized_hamming("AX", "AC")


def test_greedy_multinomial_restores_the_official_draw_function() -> None:
    original = lambda *_args, **_kwargs: "sampled"
    module = SimpleNamespace(multinomial=original)
    probabilities = SimpleNamespace(argmax=lambda **_kwargs: "argmax")

    with greedy_multinomial(module):
        assert module.multinomial(probabilities, 1) == "argmax"
    assert module.multinomial is original


def test_generation_response_checks_shared_order_and_identical_floor() -> None:
    order = (2, 0, 1)
    generation = GreedyGeneration("ACD", order)
    result = build_generation_response(
        _cases(),
        {
            ("pair-1", "CONDITION_1"): generation,
            ("pair-1", "CONDITION_2"): generation,
        },
        model_id="m",
        checkpoint_id="c",
        regime="identical",
    )

    assert result.loc[0, "r_greedy"] == 0.0
    assert result.loc[0, "sequence_identity"] == 1.0
    assert result.loc[0, "decoding_order"] == [2, 0, 1]

    with pytest.raises(ValueError, match="decoding orders differ"):
        build_generation_response(
            _cases(),
            {
                ("pair-1", "CONDITION_1"): generation,
                ("pair-1", "CONDITION_2"): GreedyGeneration("ACD", (0, 1, 2)),
            },
            model_id="m",
            checkpoint_id="c",
            regime="identical",
        )


def test_dose_pairing_and_protein_aggregation_do_not_impute() -> None:
    cases = pd.concat([_cases("low", dose=0.25), _cases("high", dose=0.50)], ignore_index=True)
    generations = {
        ("low", "CONDITION_1"): GreedyGeneration("ACD", (0, 1, 2)),
        ("low", "CONDITION_2"): GreedyGeneration("AAD", (0, 1, 2)),
        ("high", "CONDITION_1"): GreedyGeneration("ACD", (0, 1, 2)),
        ("high", "CONDITION_2"): GreedyGeneration("AAA", (0, 1, 2)),
    }
    response = build_generation_response(
        cases,
        generations,
        model_id="m",
        checkpoint_id="c",
        regime="controlled",
    )
    proteins = protein_generation_response(response)
    dose = generation_dose_response(response)

    assert proteins.loc[0, "r_greedy"] == pytest.approx(0.5)
    assert proteins.loc[0, "n_pairs"] == 2
    assert dose.loc[0, "r_greedy_high_minus_low"] == pytest.approx(1 / 3)


def test_cluster_bootstrap_summary_is_deterministic() -> None:
    rows = []
    for index, drift in enumerate((0.0, 0.25, 0.5)):
        row = {
            "model_id": "m",
            "checkpoint_id": "c",
            "regime": "controlled",
            "pair_id": f"pair-{index}",
            "protein_id": f"P{index}",
            "identity_cluster_id": f"C{index}",
            "r_greedy": drift,
            "recovery_left": 0.5,
            "recovery_right": 0.5,
        }
        rows.append(row)
    response = pd.DataFrame(rows)

    first = summarize_generation(response, replicates=100)
    second = summarize_generation(response, replicates=100)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "n_proteins"] == 3

