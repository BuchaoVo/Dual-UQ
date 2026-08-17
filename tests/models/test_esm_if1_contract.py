from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dual_uq.models.esm_if1 import (
    ESMIF1ContractError,
    FakeESMIF1Adapter,
    validate_esm_if1_source,
)


def test_missing_official_esm_source_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(ESMIF1ContractError, match="official ESM source"):
        validate_esm_if1_source(tmp_path, tmp_path / "esm_if1.pt")


def test_model_binding_records_revision_and_checkpoint_sha() -> None:
    adapter = FakeESMIF1Adapter(
        implementation_revision="official-revision",
        checkpoint_sha256="a" * 64,
    )
    binding = adapter.binding()
    assert binding["implementation_revision"] == "official-revision"
    assert binding["checkpoint_sha256"] == "a" * 64


def test_fake_adapter_has_native_distribution_and_deterministic_sample() -> None:
    adapter = FakeESMIF1Adapter(implementation_revision="r", checkpoint_sha256="b" * 64)
    coords = np.zeros((4, 3, 3), dtype=np.float32)
    distributions = adapter.score_teacher_forced("ACDE", coords)
    assert distributions.shape == (4, 20)
    np.testing.assert_allclose(distributions.sum(axis=1), 1.0)
    assert adapter.sample(coords, temperature=0.1, seed=7) == adapter.sample(
        coords, temperature=0.1, seed=7
    )
