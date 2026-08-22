from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ROOT = Path(__file__).parents[2] / "schemas"
CORE = (
    "proteins",
    "structures",
    "condition_pairs",
    "residue_mappings",
    "benchmark_instances",
    "splits",
)


def _schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_ROOT / f"{name}.schema.json").read_text(encoding="utf-8"))


def test_structcal_public_core_has_arm_and_no_hash_or_model_specific_fields() -> None:
    prohibited_exact = {"model_score", "model_response", "R_local", "D_excess", "J_full"}
    for name in CORE:
        properties = set(_schema(name)["properties"])
        assert not any(field.endswith(("_sha256", "_checksum")) for field in properties)
        assert not any(field.startswith(("proteinmpnn_", "esm_if1_")) for field in properties)
        assert properties.isdisjoint(prohibited_exact)

    assert "arm" in _schema("structures")["required"]
    assert "arm" in _schema("condition_pairs")["required"]
    assert "arm" in _schema("benchmark_instances")["required"]
    assert "structural_condition_semantics" not in _schema("structures")["properties"]


def test_structcal_split_schema_has_one_authoritative_release_assignment() -> None:
    schema = _schema("splits")
    required = set(schema["required"])
    assert {"split_seed", "split_protocol_version"} <= required
    assert schema["properties"]["split"]["enum"] == ["TRAIN", "VALIDATION", "LOCKED_TEST"]
    assert schema["x-primary-key"] == ["protein_id"]


def test_structcal_instance_schema_uses_generic_task_names() -> None:
    properties = set(_schema("benchmark_instances")["properties"])
    assert {
        "local_sensitivity_eligible",
        "generation_eligible",
        "compatibility_eligible",
        "multistate_eligible",
    } <= properties
    assert properties.isdisjoint(
        {
            "generative_propagation_eligible",
            "sequence_scoring_eligible",
            "multistate_generation_eligible",
        }
    )
