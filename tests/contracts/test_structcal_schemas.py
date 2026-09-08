from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SCHEMA_ROOT = Path(__file__).parents[2] / "schemas"

CORE_SCHEMAS = {
    "proteins.schema.json",
    "structures.schema.json",
    "condition_pairs.schema.json",
    "residue_mappings.schema.json",
    "benchmark_instances.schema.json",
    "splits.schema.json",
}
ANNOTATION_SCHEMAS = {
    "pair_structural_descriptors.schema.json",
    "residue_structural_descriptors.schema.json",
    "ligand_annotations.schema.json",
    "perturbation_descriptors.schema.json",
}
RAW_SCHEMAS = {
    "local_scores.schema.json",
    "generated_sequences.schema.json",
    "sequence_scores.schema.json",
}
SEQUENCE_ENSEMBLE_SCHEMAS = {
    "sequence_ensemble_coverage.schema.json",
    "sequence_ensemble_exclusions.schema.json",
    "sequence_ensemble_manifest.schema.json",
    "sequence_ensemble_readiness.schema.json",
}
OTHER_SCHEMAS = {
    "model_capabilities.schema.json",
    "model_instance_eligibility.schema.json",
    "report_card.schema.json",
    "benchmark_version.schema.json",
    "schema_version.schema.json",
    "metric_version.schema.json",
    "annotation_version.schema.json",
    "release_manifest.schema.json",
}
ALL_SCHEMAS = (
    CORE_SCHEMAS
    | ANNOTATION_SCHEMAS
    | RAW_SCHEMAS
    | SEQUENCE_ENSEMBLE_SCHEMAS
    | OTHER_SCHEMAS
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "anyOf" in schema:
        errors: list[str] = []
        for branch in schema["anyOf"]:
            try:
                _validate(value, branch, path)
                break
            except AssertionError as exc:
                errors.append(str(exc))
        else:
            raise AssertionError(f"{path} did not match anyOf: {errors}")
        return

    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        assert any(_matches_type(value, item) for item in expected_types), (
            f"{path} has type {type(value).__name__}, expected {expected_types}"
        )

    if "enum" in schema:
        assert value in schema["enum"], f"{path} value {value!r} is outside enum"

    if isinstance(value, str):
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], path
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"], path

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            assert value >= schema["minimum"], path
        if "maximum" in schema:
            assert value <= schema["maximum"], path

    if isinstance(value, dict):
        for field in schema.get("required", []):
            assert field in value, f"{path} missing required field {field!r}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            assert not unknown, f"{path} has unknown fields {sorted(unknown)}"
        for field, child in properties.items():
            if field in value:
                _validate(value[field], child, f"{path}.{field}")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


def _assert_valid(name: str, value: dict[str, Any]) -> None:
    _validate(value, _load(name))


def _base_core_rows() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protein = {
        "protein_id": "protein-1",
        "uniprot_id": "P00001",
        "canonical_sequence": "ACDE",
        "canonical_length": 4,
        "canonical_sequence_status": "STANDARD_20AA",
    }
    structure_a = {
        "structure_id": "structure-a",
        "source_structure_id": "1abc:A",
        "protein_id": "protein-1",
        "arm": "representation_variation",
        "condition_type": "experimental",
        "condition_label": "PDB",
        "state_family": None,
        "source_type": "PDB",
        "source_accession": "1abc",
        "chain_id": "A",
        "entity_id": "1abc_1",
        "assembly_id": None,
        "experimental_method": "X-ray",
        "source_file_ref": "data/raw/mock/1abc.cif",
        "state_evidence_tier": None,
        "state_evidence_source": None,
        "parent_structure_id": None,
    }
    structure_b = {
        "structure_id": "structure-b",
        "source_structure_id": "AF-P00001-F1",
        "protein_id": "protein-1",
        "arm": "representation_variation",
        "condition_type": "predicted",
        "condition_label": "AFDB",
        "state_family": None,
        "source_type": "AFDB",
        "source_accession": "AF-P00001-F1",
        "chain_id": "A",
        "entity_id": None,
        "assembly_id": None,
        "experimental_method": None,
        "source_file_ref": "data/raw/mock/AF-P00001-F1.cif",
        "state_evidence_tier": None,
        "state_evidence_source": None,
        "parent_structure_id": None,
    }
    return protein, structure_a, structure_b


def test_all_stable_schema_paths_exist_and_are_json() -> None:
    assert ALL_SCHEMAS == {path.name for path in SCHEMA_ROOT.glob("*.schema.json")}
    for name in sorted(ALL_SCHEMAS):
        schema = _load(name)
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["title"]
        assert schema["$id"].rsplit("/", 1)[-1] == name


def test_core_rows_validate_and_foreign_keys_are_consistent() -> None:
    protein, structure_a, structure_b = _base_core_rows()
    _assert_valid("proteins.schema.json", protein)
    _assert_valid("structures.schema.json", structure_a)
    _assert_valid("structures.schema.json", structure_b)

    pair = {
        "pair_id": "pair-1",
        "protein_id": "protein-1",
        "arm": "representation_variation",
        "condition_1_structure_id": "structure-a",
        "condition_2_structure_id": "structure-b",
        "condition_1_label": "PDB",
        "condition_2_label": "AFDB",
        "pair_role": "PRIMARY",
        "admission_status": "ADMITTED",
        "sequence_comparable": True,
        "construct_comparable": True,
        "assembly_comparable": True,
        "mapping_comparable": True,
        "pair_provenance": "mock://pair-1",
        "common_mapped_count": 4,
        "common_mapped_fraction": 1.0,
        "common_coordinate_visible_count": 4,
        "common_coordinate_visible_fraction": 1.0,
    }
    _assert_valid("condition_pairs.schema.json", pair)
    assert pair["protein_id"] == protein["protein_id"]
    assert {pair["condition_1_structure_id"], pair["condition_2_structure_id"]} == {
        structure_a["structure_id"],
        structure_b["structure_id"],
    }
    with pytest.raises(AssertionError):
        assert "missing-structure" in {structure_a["structure_id"], structure_b["structure_id"]}


def test_primary_key_uniqueness_is_required_for_core_rows() -> None:
    rows = [{"protein_id": "p-1"}, {"protein_id": "p-1"}]
    keys = [row["protein_id"] for row in rows]
    with pytest.raises(AssertionError):
        assert len(keys) == len(set(keys)), "duplicate primary key"
    composite = [("pair-1", 1), ("pair-1", 1)]
    with pytest.raises(AssertionError):
        assert len(composite) == len(set(composite)), "duplicate composite key"


def test_canonical_length_and_enum_constraints_are_scientific_contracts() -> None:
    protein, _, _ = _base_core_rows()
    bad_length = dict(protein, canonical_length=5)
    with pytest.raises(AssertionError):
        assert bad_length["canonical_length"] == len(bad_length["canonical_sequence"])
    bad_status = dict(protein, canonical_sequence_status="IMPUTED")
    with pytest.raises(AssertionError):
        _assert_valid("proteins.schema.json", bad_status)


def test_residue_mapping_is_mapping_only_and_has_pair_position_key() -> None:
    mapping = {
        "pair_id": "pair-1",
        "protein_id": "protein-1",
        "canonical_position": 1,
        "canonical_aa": "A",
        "condition_1_residue_id": "A:1",
        "condition_2_residue_id": "A:1",
        "condition_1_aa": "A",
        "condition_2_aa": "A",
        "condition_1_mapped": True,
        "condition_2_mapped": True,
        "condition_1_coordinate_visible": True,
        "condition_2_coordinate_visible": True,
        "common_mapped": True,
        "common_coordinate_visible": True,
        "condition_1_missing_reason": None,
        "condition_2_missing_reason": None,
        "mapping_status": "COMMON_VISIBLE",
    }
    _assert_valid("residue_mappings.schema.json", mapping)
    prohibited = set(_load("residue_mappings.schema.json")["x-prohibited-fields"])
    assert prohibited.isdisjoint(mapping)
    with pytest.raises(AssertionError):
        assert prohibited.isdisjoint(mapping | {"ca_displacement": 0.2})


def test_benchmark_instance_and_model_eligibility_are_separate_layers() -> None:
    instance = {
        "instance_id": "instance-1",
        "pair_id": "pair-1",
        "protein_id": "protein-1",
        "arm": "representation_variation",
        "local_sensitivity_eligible": True,
        "generation_eligible": True,
        "compatibility_eligible": True,
        "multistate_eligible": False,
        "geometry_evaluable": True,
        "ligand_evaluable": False,
        "eligibility_reason": None,
    }
    eligibility = {
        "instance_id": "instance-1",
        "model_id": "proteinmpnn",
        "capability": "LOCAL_SCORING",
        "eligible": True,
        "reason": None,
    }
    _assert_valid("benchmark_instances.schema.json", instance)
    _assert_valid("model_instance_eligibility.schema.json", eligibility)
    assert "model_id" not in _load("benchmark_instances.schema.json")["properties"]
    assert "model_result" not in _load("benchmark_instances.schema.json")["properties"]


def test_functional_state_and_controlled_perturbation_semantics_are_explicit() -> None:
    functional = {
        "structure_id": "functional-a",
        "source_structure_id": "9xyz:A",
        "protein_id": "protein-1",
        "arm": "functional_state",
        "condition_type": "experimental",
        "condition_label": "OPEN",
        "state_family": "OPEN_CLOSED",
        "source_type": "PDB",
        "source_accession": "9xyz",
        "chain_id": "A",
        "entity_id": "9xyz_1",
        "assembly_id": "1",
        "experimental_method": "X-ray",
        "source_file_ref": "data/raw/mock/9xyz.cif",
        "state_evidence_tier": "TIER_2",
        "state_evidence_source": "mock",
        "parent_structure_id": None,
    }
    perturbation = {
        "structure_id": "perturbed-a",
        "source_structure_id": "perturbed:1",
        "protein_id": "protein-1",
        "arm": "controlled_perturbation",
        "condition_type": "perturbed",
        "condition_label": "PERTURBED",
        "state_family": None,
        "source_type": "CONTROLLED_PERTURBATION",
        "source_accession": "perturbed:1",
        "chain_id": "A",
        "entity_id": None,
        "assembly_id": None,
        "experimental_method": None,
        "source_file_ref": "data/interim/mock/perturbed.cif",
        "state_evidence_tier": None,
        "state_evidence_source": "mock",
        "parent_structure_id": "structure-a",
    }
    _assert_valid("structures.schema.json", functional)
    _assert_valid("structures.schema.json", perturbation)
    assert functional["state_family"]
    assert perturbation["parent_structure_id"]
    with pytest.raises(AssertionError):
        assert dict(functional, state_family=None)["state_family"] is not None
    with pytest.raises(AssertionError):
        assert dict(perturbation, parent_structure_id=None)["parent_structure_id"] is not None
    descriptor = {
        "pair_id": "pair-1",
        "perturbation_family": "COORDINATE_NOISE",
        "requested_dose": 0.1,
        "requested_dose_unit": "angstrom",
        "realized_ca_rmsd": 0.12,
        "realized_pairwise_distance_change": 0.04,
        "realized_contact_change": 0.01,
        "random_seed": 7,
        "perturbation_method": "mock",
        "perturbation_version": "1.0.0",
    }
    _assert_valid("perturbation_descriptors.schema.json", descriptor)
    assert descriptor["requested_dose"] != descriptor["realized_ca_rmsd"]


def test_raw_outputs_keep_model_identity_and_do_not_become_core_rows() -> None:
    local = {
        "model_id": "proteinmpnn",
        "model_version": "frozen",
        "checkpoint_id": "checkpoint-sha",
        "pair_id": "pair-1",
        "condition_structure_id": "structure-a",
        "canonical_position": 1,
        "aa": "A",
        "score_type": "log_probability",
        "score": -0.2,
        "probability": 0.8,
    }
    generated = {
        "model_id": "proteinmpnn",
        "model_version": "frozen",
        "checkpoint_id": "checkpoint-sha",
        "pair_id": "pair-1",
        "condition_structure_id": "structure-a",
        "sample_id": "sample-1",
        "seed": 1,
        "temperature": 0.1,
        "decoding_protocol": "autoregressive",
        "sequence": "ACDE",
        "sequence_sha256": "b" * 64,
    }
    score = {
        "sequence_id": "sample-1",
        "sequence_sha256": "b" * 64,
        "generator_model_id": "proteinmpnn",
        "evaluator_model_id": "esm_if1",
        "evaluator_role": "REFERENCE_ESM_IF1",
        "pair_id": "pair-1",
        "target_condition_structure_id": "structure-b",
        "raw_score": -3.2,
        "score_direction": "higher_is_better",
        "wt_reference_score": -3.0,
        "baseline_adjusted_score": -0.2,
        "score_semantics": "mean_log_probability",
    }
    _assert_valid("local_scores.schema.json", local)
    _assert_valid("generated_sequences.schema.json", generated)
    _assert_valid("sequence_scores.schema.json", score)
    assert set(local) - set(_load("proteins.schema.json")["properties"])
    assert "canonical_sequence" not in local
    assert score["generator_model_id"] != score["evaluator_model_id"]


def test_model_capabilities_and_report_card_are_multidimensional() -> None:
    capabilities = {
        "model_id": "proteinmpnn",
        "model_version": "frozen",
        "checkpoint_id": "checkpoint-sha",
        "source_revision": "commit-sha",
        "LOCAL_SCORING": True,
        "GENERATION": True,
        "SEQUENCE_SCORING": True,
        "MULTISTATE_GENERATION": False,
        "local_score_semantics": "native_log_probability",
        "generation_semantics": "autoregressive",
        "sequence_score_semantics": "masked_mean_log_probability",
        "supported_structure_requirements": "mapped coordinates",
        "missing_coordinate_behavior": "unavailable",
        "sequence_gap_behavior": "explicit",
        "nonstandard_residue_behavior": "unavailable",
    }
    report = {
        "model_identity": {"model_id": "proteinmpnn", "model_version": "frozen", "checkpoint_id": "checkpoint-sha", "source_revision": "commit-sha"},
        "capability_coverage": [{"capability": "LOCAL_SCORING", "declared": True, "evaluated_instances": 1, "unavailable_reason": None}],
        "structural_condition_coverage": [{"structural_condition_semantics": "representation_variation", "tasks_evaluated": ["local_sensitivity"], "primary_cohort": 1, "unavailable_cases": []}],
        "local_sensitivity": {},
        "generative_propagation": {},
        "cross_condition_compatibility": {},
        "multistate_robustness": {},
        "structural_grounding": {},
        "controlled_perturbation_response": {},
        "cross_evaluator_agreement": {},
        "model_specific_unavailable_cases": [],
    }
    _assert_valid("model_capabilities.schema.json", capabilities)
    _assert_valid("report_card.schema.json", report)
    assert "scalar_score" in _load("report_card.schema.json")["x-forbidden-top-level-fields"]


def test_structcal_core_keeps_arm_only_where_structural_condition_membership_lives() -> None:
    for name in ("structures.schema.json", "condition_pairs.schema.json", "benchmark_instances.schema.json"):
        assert "arm" in _load(name)["properties"]
    for name in ("proteins.schema.json", "residue_mappings.schema.json", "splits.schema.json"):
        assert "arm" not in _load(name)["properties"]
    instance_properties = set(_load("benchmark_instances.schema.json")["properties"])
    assert instance_properties.isdisjoint({"task_1", "task_2", "task_3", "task_4"})


def test_metadata_schemas_are_independently_versioned() -> None:
    benchmark = {"benchmark_id": "dual_uq_benchmark", "benchmark_version": "0.1.0", "status": "SPECIFICATION_ONLY", "core_schema_family": "1.0.0", "release_id": None, "legacy_ids": []}
    schema = {"schema_family": "dual_uq.structcal", "schema_version": "1.0.0", "schemas": {"proteins": "1.0.0", "structures": "1.0.0"}}
    metric = {"metric_family": "dual_uq.canonical_metrics", "metric_version": "1.0.0", "definitions": {"local_js": "central evaluator definition"}}
    annotation = {"annotation_family": "dual_uq.structural_annotations", "annotation_version": "1.0.0", "definitions": {"residue": "descriptor contract"}}
    _assert_valid("benchmark_version.schema.json", benchmark)
    _assert_valid("schema_version.schema.json", schema)
    _assert_valid("metric_version.schema.json", metric)
    _assert_valid("annotation_version.schema.json", annotation)
    assert benchmark["benchmark_version"] != schema["schema_version"] or benchmark["benchmark_id"] != schema["schema_family"]


def test_initial_metadata_definitions_match_the_metadata_schemas() -> None:
    metadata_root = SCHEMA_ROOT.parent / "structcal" / "metadata"
    for filename, schema_name in (
        ("benchmark_version.json", "benchmark_version.schema.json"),
        ("schema_version.json", "schema_version.schema.json"),
        ("metric_version.json", "metric_version.schema.json"),
        ("annotation_version.json", "annotation_version.schema.json"),
    ):
        value = json.loads((metadata_root / filename).read_text(encoding="utf-8"))
        _assert_valid(schema_name, value)
    assert not (metadata_root / "release_manifest.json").exists(), (
        "a release manifest would falsely imply that a benchmark release was materialized"
    )


def test_canonical_names_do_not_use_development_stage_labels() -> None:
    forbidden = {"v1", "v2", "a1", "a2", "stage1", "final", "new", "latest", "test2"}
    canonical_paths = [
        "structcal/core",
        "structcal/annotations",
        "structcal/results/proteinmpnn/local_sensitivity",
        "structcal/results/esm_if1/generative_propagation",
        "structcal/metadata",
    ]
    for path in canonical_paths:
        assert not any(part.lower() in forbidden for part in Path(path).parts)
