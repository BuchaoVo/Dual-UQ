from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.benchmark.condition_semantics import orient_pair
from dual_uq.benchmark.ids import canonical_id, pair_id, protein_id
from dual_uq.benchmark.instances import BenchmarkInstance
from dual_uq.benchmark.schema_registry import SchemaRegistry
from dual_uq.benchmark.tables import canonical_columns, validate_frame, write_parquet_transactional
from dual_uq.benchmark.validation import validate_primary_key

ROOT = Path(__file__).parents[2]


def test_schema_registry_reads_frozen_schema_metadata_without_duplicate_field_lists():
    registry = SchemaRegistry(ROOT / "schemas")
    schema = registry.get("condition_pairs")
    assert schema.filename == "condition_pairs.schema.json"
    assert "pair_id" in schema.required
    assert schema.primary_key == ("pair_id",)
    assert canonical_columns("proteins", registry) == tuple(
        json.loads((ROOT / "schemas/proteins.schema.json").read_text())["properties"]
    )


def test_canonical_ids_are_order_and_path_independent():
    assert protein_id("P00001") == protein_id("P00001")
    assert canonical_id("pair", {"protein_id": "p", "left": "a", "right": "b"}) == canonical_id(
        "pair", {"right": "b", "left": "a", "protein_id": "p"}
    )
    assert pair_id("p", "representation_variation", "s1", "s2") != pair_id(
        "p", "representation_variation", "s2", "s1"
    )


def test_orientation_is_semantic_and_independent_of_input_order():
    oriented = orient_pair(
        "functional_state",
        "OPEN_CLOSED",
        {"structure_id": "closed-id", "condition_label": "CLOSED"},
        {"structure_id": "open-id", "condition_label": "OPEN"},
    )
    assert oriented.condition_1_label == "OPEN"
    assert oriented.condition_1_structure_id == "open-id"
    assert oriented.condition_2_label == "CLOSED"
    assert oriented.condition_2_structure_id == "closed-id"


def test_core_validation_rejects_model_result_contamination():
    registry = SchemaRegistry(ROOT / "schemas")
    frame = pd.DataFrame(
        [{
            "protein_id": "p",
            "uniprot_id": "P00001",
            "canonical_sequence": "AC",
            "canonical_length": 2,
            "canonical_sequence_status": "STANDARD_20AA",
            "model_score": 0.1,
        }]
    )
    with pytest.raises(ValueError, match="outside frozen schema"):
        validate_frame(frame, "proteins", registry)


def test_primary_key_validation_is_explicit():
    with pytest.raises(ValueError, match="duplicate"):
        validate_primary_key(pd.DataFrame({"pair_id": ["p", "p"]}), ("pair_id",))


def test_benchmark_instance_has_only_generic_eligibility():
    instance = BenchmarkInstance(
        instance_id="i",
        pair_id="p",
        protein_id="protein",
        arm="functional_state",
        local_sensitivity_eligible=True,
        generation_eligible=False,
        compatibility_eligible=False,
        multistate_eligible=False,
        geometry_evaluable=True,
        ligand_evaluable=False,
    )
    assert instance.model_id is None


def test_table_writer_orders_schema_columns_and_replaces_transactionally(tmp_path):
    registry = SchemaRegistry(ROOT / "schemas")
    row = {
        "protein_id": "p",
        "uniprot_id": "P00001",
        "canonical_sequence": "AC",
        "canonical_length": 2,
        "canonical_sequence_status": "STANDARD_20AA",
    }
    destination = tmp_path / "proteins.parquet"
    write_parquet_transactional(pd.DataFrame([row]), "proteins", destination, registry)
    assert list(pd.read_parquet(destination).columns) == list(canonical_columns("proteins", registry))
