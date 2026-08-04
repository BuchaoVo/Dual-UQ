from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from dual_uq.dataset.models import (
    BiologicalIdentity,
    CandidateContext,
    CandidateDerivationResult,
    DerivationRunResult,
    StageResult,
)
from dual_uq.dataset.pipeline import LifecycleStatus, TaskExecutionResult
from dual_uq.dataset.reporting import (
    build_report,
    execution_summary,
    write_execution_report,
    write_reports,
)


def _result() -> DerivationRunResult:
    context = CandidateContext(
        candidate_index=1,
        identity=BiologicalIdentity(
            pair_id="1abc_A__P12345",
            pdb_id="1abc",
            chain_id="A",
            uniprot_accession="P12345",
            polymer_entity_id="1ABC_1",
        ),
        exact_afdb_accession="P12345",
        expected_afdb_model_identity="AF-P12345-F1",
        assets=(),
        source_bindings=(),
        protocol_binding="binding",
    )
    candidate = CandidateDerivationResult(
        context=context,
        stages=(StageResult(stage="observability", status="complete"),),
        evidence={},
        report_record={
            "candidate_index": 1,
            "pair_id": "1abc_A__P12345",
            "raw_complete": True,
            "canonical_sequence_complete": True,
            "pair_qc_complete": True,
            "mapping_complete": True,
            "fragment_resolved": True,
            "pae_bound": True,
            "confidence_bound": True,
            "mechanism_observable": True,
            "primary_failure_stage": None,
            "primary_failure_code": None,
            "selection_roles": ["A"],
            "sampling_stratum_prior_recomputed": "observed_no_positive_prior",
            "z_optional_failure_detail": None,
        },
    )
    return DerivationRunResult(
        schema_version="dataset-a.derive-pilot.v1",
        preflight={"preflight_pass": True},
        candidates=(candidate,),
        summary={
            "candidate_count": 1,
            "pipeline_verdict": "DERIVATION_PIPELINE_PASS",
        },
        attrition_bias_probe={"scope": "fixture", "groups": {}},
        scope={"candidate_admission_changed": False},
        selection_policy={"candidate_indices": [1]},
        report_metadata={
            "candidate_count_field": "pilot_candidate_count",
            "summary_fields": {"derive_48_started": False},
        },
    )


def test_all_formats_consume_one_canonical_run_result_deterministically(
    tmp_path: Path,
) -> None:
    result = _result()
    canonical = build_report(result)

    write_reports(result, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    write_reports(result, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert json.loads(first["derive_pilot_v1.json"])["summary"] == canonical["summary"]
    assert b"\r" not in first["derive_pilot_v1.tsv"]
    assert all(
        not line.endswith((b"\t", b" "))
        for line in first["derive_pilot_v1.tsv"].splitlines()
    )
    markdown = first["derive_pilot_v1.md"].decode()
    assert "## Stage outcomes" in markdown
    assert "## Candidate audit" in markdown


def test_output_profile_is_run_configuration_not_a_new_scientific_pipeline(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        report_metadata={
            "output_basename": "derivation_batch1_v1",
            "markdown_title": "Dataset derivation Batch-1",
            "markdown_status": "Pre-registered derivation run.",
        },
    )

    write_reports(result, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "derivation_batch1_v1.json",
        "derivation_batch1_v1.tsv",
        "derivation_batch1_v1.md",
    }
    markdown = (tmp_path / "derivation_batch1_v1.md").read_text()
    assert markdown.startswith("# Dataset derivation Batch-1\n")
    assert "Pre-registered derivation run." in markdown


def test_batch_execution_reports_only_consume_structured_results(tmp_path: Path) -> None:
    result = TaskExecutionResult(
        record_id="protein-001",
        task_key="a" * 64,
        task_digest="b" * 64,
        stage="validation",
        stage_version="v1",
        input_digest="c" * 64,
        config_digest="d" * 64,
        dependency_digest="e" * 64,
        execution_status=LifecycleStatus.COMPLETE,
        metrics={"checked": 3},
        scientific_disposition="eligible",
    )
    table_path = tmp_path / "results.parquet"
    summary_path = tmp_path / "summary.json"

    write_execution_report(
        (result,), table_path=table_path, summary_path=summary_path
    )

    assert execution_summary((result,)) == {
        "record_count": 1,
        "execution_status_counts": {"complete": 1},
        "failure_code_counts": {},
    }
    assert json.loads(summary_path.read_text())["record_count"] == 1
    assert table_path.is_file()
