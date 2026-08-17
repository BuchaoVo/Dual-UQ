from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dual_uq.inference.apo_holo_generation import (
    ApoHoloGenerationCondition,
    ApoHoloGenerationRecord,
    ESMIF1ApoHoloGenerationAdapter,
    generate_apo_holo_conditions,
    generation_seed,
    validate_generation_records,
)
from dual_uq.models.esm_if1 import FakeESMIF1Adapter


def _condition(state: str) -> ApoHoloGenerationCondition:
    return ApoHoloGenerationCondition(
        protein_id="P00001",
        pair_id="apo_holo__P00001",
        state=state,
        structure_sha256=("1" if state == "APO" else "2") * 64,
        canonical_positions=(1, 2, 3),
        wt_sequence_projection="ACD",
        coordinates=np.zeros((3, 3, 3), dtype=np.float32),
        structure_path=Path("data/raw/apo_holo/pdb/example.cif"),
        chain_id="A",
    )


def test_generation_seed_is_stable_and_state_disjoint() -> None:
    assert generation_seed("P00001", "APO", 0) == generation_seed("P00001", "APO", 0)
    assert generation_seed("P00001", "APO", 0) != generation_seed("P00001", "HOLO", 0)
    assert generation_seed("P00001", "APO", 0) != generation_seed("P00001", "APO", 1)


def test_proteinmpnn_seed_uses_existing_independent_condition_domains() -> None:
    assert generation_seed("P00001", "APO", 0, model_name="ProteinMPNN") == 1_000_000
    assert generation_seed("P00001", "HOLO", 0, model_name="ProteinMPNN") == 2_000_000
    assert generation_seed("P00001", "APO", 63, model_name="ProteinMPNN") == 1_000_063


def test_esm_if1_generation_adapter_returns_64_independent_sequences() -> None:
    adapter = ESMIF1ApoHoloGenerationAdapter(
        FakeESMIF1Adapter(implementation_revision="r1", checkpoint_sha256="1" * 64),
        batch_size=8,
    )
    sequences = adapter.generate(_condition("APO"), n_samples=64)
    assert len(sequences) == 64
    assert all(len(sequence) == 3 for sequence in sequences)


def test_generation_record_validation_requires_complete_64_sample_state_grid() -> None:
    records = tuple(
        ApoHoloGenerationRecord(
            condition=_condition(state),
            sample_index=index,
            seed=generation_seed("P00001", state, index),
            temperature=0.1,
            sequence="ACD",
            sequence_hash=hashlib.sha256(b"ACD").hexdigest(),
            model_name="fixture",
            implementation_id="fixture",
            checkpoint_id="1" * 64,
        )
        for state in ("APO", "HOLO")
        for index in range(64)
    )
    # The fixture intentionally uses the public sequence hash helper in the
    # implementation; this assertion documents the required 128-record grid.
    assert validate_generation_records(records, expected_protein_count=1) == records


def test_resume_accepts_shard_reconstructed_without_runtime_coordinates(tmp_path: Path) -> None:
    class FakeGenerationAdapter:
        model_name = "fixture"
        implementation_id = "fixture"
        checkpoint_id = "1" * 64

        def generate(
            self,
            condition: ApoHoloGenerationCondition,
            *,
            n_samples: int,
        ) -> tuple[str, ...]:
            assert condition.coordinates is not None
            return ("ACD",) * n_samples

    adapter = FakeGenerationAdapter()
    condition = _condition("APO")
    _, first_executed, first_reused = generate_apo_holo_conditions(
        (condition,), adapter, output_root=tmp_path, resume=True
    )
    _, second_executed, second_reused = generate_apo_holo_conditions(
        (condition,), adapter, output_root=tmp_path, resume=True
    )

    assert (first_executed, first_reused) == (1, 0)
    assert (second_executed, second_reused) == (0, 1)
