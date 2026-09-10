from __future__ import annotations

import hashlib

from dual_uq.evaluation.apo_holo_generative_propagation import (
    summarize_generation_propagation,
)
from dual_uq.inference.apo_holo_generation import (
    ApoHoloGenerationCondition,
    ApoHoloGenerationRecord,
    generation_seed,
)


def _records() -> tuple[ApoHoloGenerationRecord, ...]:
    condition = {
        state: ApoHoloGenerationCondition(
            protein_id="P00001",
            pair_id="apo_holo__P00001",
            state=state,
            structure_sha256=("1" if state == "APO" else "2") * 64,
            canonical_positions=(1, 2),
            wt_sequence_projection="AA",
            coordinates=None,
            structure_path=None,
            chain_id=None,
        )
        for state in ("APO", "HOLO")
    }
    rows = []
    for state in ("APO", "HOLO"):
        for index in range(64):
            sequence = "AC" if state == "APO" or index % 2 == 0 else "CA"
            rows.append(
                ApoHoloGenerationRecord(
                    condition=condition[state],
                    sample_index=index,
                    seed=generation_seed("P00001", state, index),
                    temperature=0.1,
                    sequence=sequence,
                    sequence_hash=hashlib.sha256(sequence.encode()).hexdigest(),
                    model_name="fixture",
                    implementation_id="fixture",
                    checkpoint_id="1" * 64,
                )
            )
    return tuple(rows)


def test_generation_summary_reports_nested_divergence_and_position_js() -> None:
    records = _records()
    result = summarize_generation_propagation(records, prefixes=(16, 32, 64))
    assert set(result.protein_summary["protein_id"]) == {"P00001"}
    assert {"16", "32", "64"}.issubset(set(result.convergence["sample_count"].astype(str)))
    row = result.protein_summary.iloc[0]
    assert row["d_excess_64"] > 0
    assert result.position_shift["js_bits_64"].gt(0).all()
