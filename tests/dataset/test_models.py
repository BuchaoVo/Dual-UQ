from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dual_uq.dataset.models import (
    BiologicalIdentity,
    CandidateContext,
    CandidateDerivationResult,
    DatasetRelease,
    DerivationConfig,
    DerivationRunResult,
    LogicalAssetRef,
    MappingRecord,
    ProteinRecord,
    StageResult,
    StructureRecord,
    skipped_dependency_results,
)


def _identity(index: int = 1) -> BiologicalIdentity:
    return BiologicalIdentity(
        pair_id=f"1abc_A__P{index:05d}",
        pdb_id="1abc",
        chain_id="A",
        uniprot_accession=f"P{index:05d}",
        polymer_entity_id="1ABC_1",
    )


def _asset(asset_type: str = "pdb_mmcif") -> LogicalAssetRef:
    return LogicalAssetRef(
        asset_type=asset_type,
        logical_path="data/raw/pdb/1abc.cif",
        sha256="a" * 64,
        provenance="manifest_bound",
    )


def test_candidate_context_is_lightweight_immutable_and_identity_bound() -> None:
    context = CandidateContext(
        candidate_index=1,
        identity=_identity(),
        exact_afdb_accession="P00001",
        expected_afdb_model_identity="AF-P00001-F1",
        assets=(_asset(),),
        source_bindings=(("inventory_sha256", "b" * 64),),
        protocol_binding="c" * 64,
        selection_roles=("A_short_simple",),
        prederivation_evidence=(("canonical_length", 100),),
    )

    assert context.asset("pdb_mmcif") == _asset()
    assert not hasattr(context, "parsed_pdb")
    with pytest.raises(FrozenInstanceError):
        context.candidate_index = 2  # type: ignore[misc]


def test_context_rejects_filesystem_identity_and_duplicate_assets() -> None:
    with pytest.raises(ValueError, match="exact AFDB accession"):
        CandidateContext(
            candidate_index=1,
            identity=_identity(),
            exact_afdb_accession="Q99999",
            expected_afdb_model_identity="AF-Q99999-F1",
            assets=(_asset(),),
            source_bindings=(),
            protocol_binding="binding",
        )
    with pytest.raises(ValueError, match="duplicate asset type"):
        CandidateContext(
            candidate_index=1,
            identity=_identity(),
            exact_afdb_accession="P00001",
            expected_afdb_model_identity="AF-P00001-F1",
            assets=(_asset(), _asset()),
            source_bindings=(),
            protocol_binding="binding",
        )
    with pytest.raises(ValueError, match="portable logical path"):
        LogicalAssetRef(
            asset_type="pdb_mmcif",
            logical_path="/" + "home/developer/project/1abc.cif",
            sha256=None,
            provenance="invalid",
        )


@pytest.mark.parametrize("status", ["failed", "unobservable"])
def test_stage_failure_states_require_structured_code(status: str) -> None:
    with pytest.raises(ValueError, match="failure code"):
        StageResult(stage="mapping", status=status)


def test_stage_result_keeps_stage_specific_metrics_and_warnings() -> None:
    result = StageResult(
        stage="pair_qc",
        status="complete",
        warnings=("pair_qc_threshold_not_met",),
        metrics={"full_length_mapping_coverage": 0.8},
        artifacts=(_asset(),),
    )

    assert result.primary_failure_code is None
    assert result.metrics == {"full_length_mapping_coverage": 0.8}
    assert result.warnings == ("pair_qc_threshold_not_met",)


def test_mapping_failure_propagates_dependency_skips_without_new_failures() -> None:
    results = skipped_dependency_results(
        upstream_stage="mapping",
        stages=("fragment", "pae", "confidence", "observability"),
    )

    assert [result.status for result in results] == ["skipped_dependency"] * 4
    assert [result.primary_failure_code for result in results] == [None] * 4
    assert all(result.dependent_unavailable == ("mapping",) for result in results)


def test_run_result_accepts_one_candidate_without_uniform_stage_metrics() -> None:
    context = CandidateContext(
        candidate_index=1,
        identity=_identity(),
        exact_afdb_accession="P00001",
        expected_afdb_model_identity="AF-P00001-F1",
        assets=(_asset(),),
        source_bindings=(),
        protocol_binding="binding",
    )
    candidate = CandidateDerivationResult(
        context=context,
        stages=(StageResult(stage="identity", status="complete", metrics={"length": 10}),),
        evidence={"canonical_sequence_complete": True},
        report_record={"candidate_index": 1},
    )
    run = DerivationRunResult(
        schema_version="dataset-a.derive-pilot.v1",
        preflight={"preflight_pass": True},
        candidates=(candidate,),
        summary={"pilot_candidate_count": 1},
        attrition_bias_probe={"scope": "fixture"},
        scope={"candidate_admission_changed": False},
    )
    config = DerivationConfig(
        protocol_version="v1",
        protocol_binding="d" * 64,
        preflight_thresholds={"min_sequence_identity": 0.95},
        observability_thresholds={"easy_control": {}},
    )

    assert run.candidates[0].stages[0].metrics == {"length": 10}
    assert config.protocol_binding == "d" * 64


def test_scientific_configuration_is_deeply_immutable() -> None:
    thresholds = {"easy_control": {"minimum": 0.9}}
    config = DerivationConfig(
        protocol_version="v1",
        protocol_binding="binding",
        preflight_thresholds={},
        observability_thresholds=thresholds,
    )

    thresholds["easy_control"]["minimum"] = 0.1

    assert config.observability_thresholds["easy_control"]["minimum"] == 0.9
    with pytest.raises(TypeError):
        config.observability_thresholds["easy_control"]["minimum"] = 0.2


def test_candidate_result_rejects_duplicate_or_out_of_order_stage_records() -> None:
    context = CandidateContext(
        candidate_index=1,
        identity=_identity(),
        exact_afdb_accession="P00001",
        expected_afdb_model_identity="AF-P00001-F1",
        assets=(),
        source_bindings=(),
        protocol_binding="binding",
    )

    with pytest.raises(ValueError, match="duplicate stage"):
        CandidateDerivationResult(
            context=context,
            stages=(
                StageResult(stage="raw", status="complete"),
                StageResult(stage="raw", status="complete"),
            ),
            evidence={},
            report_record={"candidate_index": 1},
        )
    with pytest.raises(ValueError, match="stage order"):
        CandidateDerivationResult(
            context=context,
            stages=(
                StageResult(stage="mapping", status="complete"),
                StageResult(stage="identity", status="complete"),
            ),
            evidence={},
            report_record={"candidate_index": 1},
        )


def test_shared_manifest_models_require_portable_hashed_resources() -> None:
    protein = ProteinRecord("protein-001", {"nested": {"value": 1}})
    structure = StructureRecord(
        "protein-001", "pdb:1abc:A", "data/raw/pdb/1abc.cif", "a" * 64
    )
    mapping = MappingRecord(
        "protein-001", "sifts:1abc:A", "data/raw/mappings/1abc.xml.gz", "b" * 64
    )
    release = DatasetRelease(
        "dataset-a",
        "dataset-a-dev-v1",
        "data/dataset/releases/dataset-a-dev-v1.parquet",
        "c" * 64,
        1,
    )

    assert protein.record_id == structure.record_id == mapping.record_id
    assert release.record_count == 1
    with pytest.raises(ValueError, match="SHA-256"):
        StructureRecord("protein-001", "pdb:1abc:A", "data/pdb.cif", "invalid")
    with pytest.raises(ValueError, match="portable"):
        DatasetRelease("dataset-a", "v1", "/tmp/release.parquet", "c" * 64, 1)
