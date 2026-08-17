import numpy as np

from dual_uq.evaluation.multi_state_baseline import (
    aggregate_equal_weight_log_probabilities,
    compatibility_endpoints,
)


def test_compatibility_endpoints_are_wt_normalized_and_directional():
    result = compatibility_endpoints(
        apo_sequence_score=-12.0,
        holo_sequence_score=-15.0,
        apo_wt_score=-10.0,
        holo_wt_score=-11.0,
    )
    assert result.c_apo == -2.0
    assert result.c_holo == -4.0
    assert result.mean_compat == -3.0
    assert result.worst_compat == -4.0
    assert result.state_gap == 2.0


def test_equal_weight_log_probabilities_are_finite_normalized_and_deterministic():
    apo_values = np.array([0.2, 0.3, 0.5] + [0.01] * 17, dtype=float)
    holo_values = np.array([0.5, 0.25, 0.25] + [0.01] * 17, dtype=float)
    apo_values /= apo_values.sum()
    holo_values /= holo_values.sum()
    apo = np.log(apo_values)
    holo = np.log(holo_values)
    first = aggregate_equal_weight_log_probabilities(apo, holo)
    second = aggregate_equal_weight_log_probabilities(apo, holo)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.isclose(np.exp(first).sum(), 1.0)
    expected = np.sqrt(apo_values * holo_values)
    expected /= expected.sum()
    assert np.allclose(np.exp(first), expected)
