from __future__ import annotations

import numpy as np
import pytest

from dual_uq.models.pifold import (
    PiFoldAdapter,
    PiFoldModelError,
    PiFoldStructureInput,
    call_legacy_pifold_featurizer,
    normalize_pifold_logits,
    validate_pifold_structure,
)


def test_pifold_structure_requires_complete_n_ca_c_o_coordinates() -> None:
    valid = PiFoldStructureInput(
        coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        sequence_length=3,
    )
    validate_pifold_structure(valid)

    with pytest.raises(PiFoldModelError, match="shape"):
        validate_pifold_structure(
            PiFoldStructureInput(coordinates=np.zeros((3, 3, 3)), sequence_length=3)
        )

    with pytest.raises(PiFoldModelError, match="finite"):
        validate_pifold_structure(
            PiFoldStructureInput(
                coordinates=np.full((3, 4, 3), np.nan), sequence_length=3
            )
        )


def test_pifold_logits_normalize_to_standard_aa_probabilities() -> None:
    logits = np.array([[0.0] * 20, [1.0] + [0.0] * 19], dtype=np.float64)

    probabilities = normalize_pifold_logits(logits)

    assert probabilities.shape == (2, 20)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[1, 0] > probabilities[1, 1]


def test_pifold_adapter_has_l0_geometry_only_binding() -> None:
    adapter = PiFoldAdapter(
        model=object(),
        torch_module=object(),
        device="cpu",
        featurizer=lambda batch: batch,
        implementation_id="b28a0994ae02d7770733bf0aee792ffab99bdf68",
        checkpoint_id="checkpoint.pth",
    )

    binding = adapter.binding()

    assert binding["semantic_class"] == "L0"
    assert binding["probe_semantics"] == "geometry_only"
    assert binding["native_sequence_leakage"] is False
    assert binding["checkpoint_id"] == "checkpoint.pth"


def test_legacy_pifold_featurizer_gets_numpy_int_compatibility_without_persisting_patch() -> None:
    def legacy(_batch):
        return np.int

    assert call_legacy_pifold_featurizer(legacy, []) is int
    assert not hasattr(np, "int")
