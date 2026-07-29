import numpy as np

from dual_uq.robust_stats import contiguous_segments, safe_spearman


def test_contiguous_segments() -> None:
    positions = np.array([1, 2, 3, 5, 6])
    values = np.array([0.2, 1.2, 1.5, 2.1, 0.3])
    confidence = np.array([95.0, 90.0, 88.0, 92.0, 96.0])
    segments = contiguous_segments(
        positions,
        values,
        confidence,
        threshold=1.0,
    )
    assert len(segments) == 2
    assert segments[0]["start_position"] == 2
    assert segments[0]["end_position"] == 3
    assert segments[1]["start_position"] == 5


def test_safe_spearman() -> None:
    result = safe_spearman(np.array([1, 2, 3]), np.array([2, 4, 6]))
    assert result.rho == 1.0
    assert result.n == 3
