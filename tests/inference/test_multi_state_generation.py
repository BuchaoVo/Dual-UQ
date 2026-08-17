import pytest

from dual_uq.inference.multi_state_generation import validate_shared_decode_context


def test_shared_decode_context_requires_same_axis_and_order():
    context = validate_shared_decode_context(
        apo_positions=(1, 2, 3),
        holo_positions=(1, 2, 3),
        decoding_order=(2, 0, 1),
        prefix=("A", "C", "D"),
    )
    assert context.decoding_order == (2, 0, 1)
    assert context.prefix == ("A", "C", "D")


def test_shared_decode_context_rejects_axis_mismatch():
    with pytest.raises(ValueError, match="position axes"):
        validate_shared_decode_context(
            apo_positions=(1, 2, 3),
            holo_positions=(1, 4, 3),
            decoding_order=(0, 1, 2),
            prefix=("A", "C", "D"),
        )
