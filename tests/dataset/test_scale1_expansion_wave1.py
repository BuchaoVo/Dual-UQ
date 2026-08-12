from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.completion import (
    AcquisitionRecord,
    acquisition_records_frame,
)
from dual_uq.workflows import dataset_expansion as expansion_workflow
from dual_uq.workflows.dataset_expansion import (
    BLOCKED_INPUT_INTEGRITY,
    CORE_REACHED_PRIMARY_TARGET_PENDING,
    PRIMARY_CONFIRMATORY_CAPACITY_REACHED,
    WAVE2_REQUIRED_CORE_NOT_REACHED,
    DatasetExpansionConfig,
    DatasetExpansionError,
    DatasetExpansionReadinessResult,
    assemble_expansion_result,
    bind_acquisition_ledger,
    build_cluster_capacity,
    build_expansion_freeze,
    build_initial_asset_requirements,
    build_initial_readiness_frame,
    capacity_status,
    evaluate_expansion_admission,
    execute_expansion_readiness,
    load_prior_acquisition_records,
    materialize_expansion_freeze,
    materialize_expansion_result,
    run_dataset_expansion,
    validate_expansion_inputs,
    validate_frozen_declaration,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _paths(**overrides: Path) -> ProjectPaths:
    return ProjectPaths.discover(project_root=REPOSITORY_ROOT, **overrides)


def test_frozen_inputs_reconstruct_the_approved_e0_state() -> None:
    inputs = validate_expansion_inputs(_paths(), DatasetExpansionConfig())

    assert inputs.e0_summary["scale1_expansion_design_status"] == (
        "SCALE1_EXPANSION_DESIGN_READY"
    )
    assert inputs.e0_summary["current_capacity"] == {
        "formally_admitted_proteins": 135,
        "admitted_30pct_clusters": 63,
        "pending_proteins": 51,
        "pending_new_clusters": 15,
        "n_nr_upper": 78,
    }
    assert inputs.e0_summary["expansion_universe"]["n_actionable_target_clusters"] == 124
    assert inputs.e0_summary["expansion_universe"]["n_new_external_clusters"] == 100
    assert inputs.e0_summary["expansion_universe"]["n_unadmitted_existing_clusters"] == 24
    assert inputs.e0_summary["expansion_universe"]["n_primary_candidates"] == 124
    assert inputs.e0_summary["expansion_universe"]["n_total_reserve_candidates"] == 123
    assert len(inputs.targets) == 124


def test_upstream_sha_drift_blocks_before_any_freeze() -> None:
    config = replace(
        DatasetExpansionConfig(), expected_e0_manifest_sha256="0" * 64
    )

    with pytest.raises(DatasetExpansionError) as caught:
        validate_expansion_inputs(_paths(), config)

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "upstream_sha256_mismatch"


def test_freeze_consumes_first_100_persisted_ranks_without_recomputing(
    tmp_path: Path,
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    inputs = validate_expansion_inputs(paths, config)
    result = build_expansion_freeze(inputs)

    assert result.amendment == {
        "schema_version": "dual-uq.scale1e0a-wave1-amendment.v1",
        "amendment_status": "FROZEN",
        "original_recommended_wave1_n": 83,
        "frozen_wave1_primary_n": 100,
        "reason": "ADD_PRE_OUTCOME_ACQUISITION_AND_ADMISSION_ATTRITION_BUFFER",
        "changes_wave1_size_only": True,
        "target_classes_unchanged": True,
        "target_ordering_unchanged": True,
        "candidate_ranking_unchanged": True,
        "admission_contract_unchanged": True,
        "redundancy_contract_unchanged": True,
        "wave1_reserve_activation_authorized": False,
        "e0_manifest_sha256": config.expected_e0_manifest_sha256,
    }
    declaration = pd.DataFrame(result.declaration)
    assert len(declaration) == 100
    assert declaration["wave1_candidate_index"].tolist() == list(range(1, 101))
    assert declaration["prospective_global_primary_rank"].tolist() == list(
        range(1, 101)
    )
    assert declaration["sequence_cluster"].nunique() == 100
    assert declaration["pair_id"].nunique() == 100
    assert declaration["selection_role"].eq("PRIMARY_CANDIDATE").all()
    assert not declaration["reserve_activated"].any()
    assert declaration["amendment_sha256"].eq(result.amendment_sha256).all()

    materialized = materialize_expansion_freeze(result, paths, config=config)
    second = materialize_expansion_freeze(result, paths, config=config)
    assert materialized["write_status"] == {
        "amendment": "created",
        "declaration": "created",
    }
    assert second["write_status"] == {
        "amendment": "reused_identical",
        "declaration": "reused_identical",
    }
    declaration_path = paths.resolve_logical(
        "runs/scale1-expansion-wave1/scale1_expansion_wave1_candidates.jsonl"
    )
    assert hashlib.sha256(declaration_path.read_bytes()).hexdigest() == (
        result.declaration_sha256
    )
    records = [json.loads(line) for line in declaration_path.read_text().splitlines()]
    assert records == list(result.declaration)


def test_initial_readiness_frame_is_declaration_bound_and_conservative() -> None:
    inputs = validate_expansion_inputs(_paths(), DatasetExpansionConfig())
    frozen = build_expansion_freeze(inputs)
    frame = build_initial_readiness_frame(
        frozen.declaration,
        declaration_ref=(
            "experiments/p2_design_baseline/scale1/expansion_wave1/"
            "scale1_expansion_wave1_candidates.jsonl"
        ),
        declaration_sha256=frozen.declaration_sha256,
    )

    assert len(frame) == 100
    assert frame["sampling_frame_index"].tolist() == list(range(1, 101))
    assert frame["candidate_id"].is_unique
    assert frame["acquisition_bound_declaration_sha256"].eq(
        frozen.declaration_sha256
    ).all()
    assert frame["identity_metadata_present"].all()
    assert frame["provenance_metadata_present"].all()
    assert frame["missing_pdb"].all()
    assert frame["missing_afdb"].all()
    assert frame["missing_mapping_metadata"].all()
    assert frame["local_availability_status"].eq(
        "MISSING_PDB_AND_AFDB"
    ).all()


def test_initial_asset_requirements_cover_only_the_frozen_declaration() -> None:
    paths = _paths()
    frozen = build_expansion_freeze(
        validate_expansion_inputs(paths, DatasetExpansionConfig())
    )
    frame = build_initial_readiness_frame(
        frozen.declaration,
        declaration_ref="runs/wave1/candidates.jsonl",
        declaration_sha256=frozen.declaration_sha256,
    )

    requirements = build_initial_asset_requirements(frame, paths)
    counts = pd.Series([item.asset_type for item in requirements]).value_counts()
    assert counts.to_dict() == {
        "pdb_mmcif": 100,
        "sifts": 100,
        "afdb_metadata_collection": 100,
    }
    assert {item.candidate_id for item in requirements} == set(frame["pair_id"])
    assert {item.sampling_frame_index for item in requirements} == set(range(1, 101))


def test_acquisition_ledger_is_bound_to_the_frozen_declaration() -> None:
    raw = pd.DataFrame(
        [
            {
                "sampling_frame_index": 1,
                "candidate_id": "1abc_A__P12345",
                "asset_type": "pdb_mmcif",
                "retrieval_status": "ALREADY_PRESENT_FROZEN",
            }
        ]
    )

    bound = bind_acquisition_ledger(raw, "a" * 64)

    assert bound["acquisition_bound_declaration_sha256"].tolist() == ["a" * 64]
    assert bound.loc[0, "candidate_id"] == "1abc_A__P12345"


def test_prior_acquisition_ledger_rejects_declaration_drift(
    tmp_path: Path,
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    ledger_path = paths.resolve_logical(
        "runs/scale1-expansion-wave1/"
        "scale1_expansion_wave1_acquisition_ledger.parquet"
    )
    ledger_path.parent.mkdir(parents=True)
    record = AcquisitionRecord(
        sampling_frame_index=1,
        candidate_id="1abc_A__P12345",
        asset_type="pdb_mmcif",
        canonical_identifier="1abc",
        source_authority="RCSB_PDB",
        source_record_identifier="1abc",
        source_url="https://example.test/1abc.cif",
        retrieval_status="ALREADY_PRESENT_FROZEN",
        retrieval_timestamp="2026-08-11T00:00:00Z",
        retrieval_method="existing_file_validation",
        attempt_count=0,
        local_path_relative="data/raw/pdb/1abc.cif",
        file_sha256="b" * 64,
        file_size=10,
        source_metadata_version_if_available=None,
        validation_status="VALID",
        failure_code=None,
        http_status=None,
        dependency_asset_type=None,
    )
    bind_acquisition_ledger(
        acquisition_records_frame((record,)), "a" * 64
    ).to_parquet(ledger_path, index=False)

    with pytest.raises(DatasetExpansionError) as caught:
        load_prior_acquisition_records(paths, "c" * 64, config=config)

    assert caught.value.code == "prior_acquisition_declaration_mismatch"


def test_cluster_capacity_has_one_terminal_outcome_per_declared_cluster() -> None:
    frozen = build_expansion_freeze(
        validate_expansion_inputs(_paths(), DatasetExpansionConfig())
    )
    readiness = pd.DataFrame(
        {
            "sampling_frame_index": range(1, 101),
            "candidate_id": [row["candidate_id"] for row in frozen.declaration],
            "local_availability_status": ["LOCAL_READY_FOR_SCALE1A"]
            + ["MISSING_PDB"] * 99,
            "terminal_local_missing_reason": [None] + ["pdb_structure_absent"] * 99,
        }
    )
    admission = pd.DataFrame(
        [
            {
                "sampling_frame_index": 1,
                "pair_id": frozen.declaration[0]["pair_id"],
                "admission_status": "FORMALLY_ADMITTED",
                "terminal_reason_code": "exact_canonical_identity",
            }
        ]
    )

    capacity = build_cluster_capacity(frozen.declaration, readiness, admission)

    assert len(capacity) == 100
    assert capacity["sequence_cluster"].is_unique
    assert capacity["cluster_contribution"].isin([0, 1]).all()
    assert int(capacity["cluster_contribution"].sum()) == 1
    assert capacity.loc[0, "terminal_wave1_cluster_outcome"] == "FORMALLY_ADMITTED"
    assert capacity.loc[1:, "terminal_wave1_cluster_outcome"].eq(
        "READINESS_FAILED"
    ).all()


def test_declaration_drift_blocks_before_acquisition(tmp_path: Path) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    frozen = build_expansion_freeze(validate_expansion_inputs(paths, config))
    materialize_expansion_freeze(frozen, paths, config=config)
    declaration_path = paths.resolve_logical(
        "runs/scale1-expansion-wave1/scale1_expansion_wave1_candidates.jsonl"
    )
    declaration_path.write_bytes(declaration_path.read_bytes() + b"\n")

    with pytest.raises(DatasetExpansionError) as caught:
        validate_frozen_declaration(frozen, paths, config=config)

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "declaration_hash_mismatch"


def test_readiness_execution_scopes_rejected_evidence_to_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    inputs = validate_expansion_inputs(paths, config)
    frozen = build_expansion_freeze(inputs)
    materialize_expansion_freeze(frozen, paths, config=config)
    rejected_refs: list[str] = []

    def fake_execute(*_args: object, **kwargs: object) -> tuple[()]:
        rejected_refs.append(str(kwargs["rejected_evidence_ref"]))
        return ()

    monkeypatch.setattr(
        expansion_workflow, "execute_initial_acquisition", fake_execute
    )
    monkeypatch.setattr(
        expansion_workflow,
        "recompute_full_frame_readiness",
        lambda frame, _records, _paths: frame.copy(deep=True),
    )
    monkeypatch.setattr(
        expansion_workflow, "resolve_uniprot_requirements", lambda *_args: ()
    )
    monkeypatch.setattr(
        expansion_workflow,
        "resolve_afdb_structure_requirements",
        lambda *_args, **_kwargs: ((), ()),
    )

    result = execute_expansion_readiness(
        inputs=inputs,
        frozen=frozen,
        paths=paths,
        config=config,
        sleeper=lambda _delay: None,
    )

    expected_ref = (
        "artifacts/dataset/audits/acquisition_rejected/"
        f"scale1-expansion-wave1/{frozen.declaration_sha256}"
    )
    assert rejected_refs == [expected_ref, expected_ref, expected_ref]
    assert len(result.readiness) == 100
    assert result.acquisition_ledger.empty
    assert "acquisition_bound_declaration_sha256" in result.acquisition_ledger


def test_no_ready_candidates_produce_no_scientific_admission_rows() -> None:
    paths = _paths()
    frozen = build_expansion_freeze(
        validate_expansion_inputs(paths, DatasetExpansionConfig())
    )
    readiness = build_initial_readiness_frame(
        frozen.declaration,
        declaration_ref="runs/wave1/candidates.jsonl",
        declaration_sha256=frozen.declaration_sha256,
    )

    admission, masks = evaluate_expansion_admission(readiness, paths=paths)

    assert admission.empty
    assert "admission_status" in admission.columns
    assert masks.empty
    assert "common_mask" in masks.columns


def test_single_result_renders_remaining_outputs_and_manifest_last(
    tmp_path: Path,
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    inputs = validate_expansion_inputs(paths, config)
    frozen = build_expansion_freeze(inputs)
    materialize_expansion_freeze(frozen, paths, config=config)
    readiness = pd.DataFrame(
        {
            "sampling_frame_index": range(1, 101),
            "candidate_id": [row["candidate_id"] for row in frozen.declaration],
            "pair_id": [row["pair_id"] for row in frozen.declaration],
            "local_availability_status": ["MISSING_PDB"] * 100,
            "terminal_local_missing_reason": ["pdb_structure_absent"] * 100,
        }
    )
    admission = pd.DataFrame(
        columns=["sampling_frame_index", "pair_id", "admission_status"]
    )
    masks = pd.DataFrame()
    ledger = bind_acquisition_ledger(
        pd.DataFrame(
            columns=["sampling_frame_index", "candidate_id", "asset_type"]
        ),
        frozen.declaration_sha256,
    )
    capacity = build_cluster_capacity(
        frozen.declaration, readiness, admission
    )
    result = assemble_expansion_result(
        inputs=inputs,
        frozen=frozen,
        acquisition_ledger=ledger,
        readiness=readiness,
        admission=admission,
        common_masks=masks,
        cluster_capacity=capacity,
    )

    assert result.summary["n_ready"] == 0
    assert result.summary["n_scientifically_evaluated"] == 0
    assert result.summary["n_wave1_new_admitted_clusters"] == 0
    assert result.summary["n_nr_combined"] == 63
    assert result.summary["capacity_status"] == WAVE2_REQUIRED_CORE_NOT_REACHED
    assert result.manifest["WAVE1_DECLARATION_FROZEN_BEFORE_ACQUISITION"] is True
    assert result.manifest["NO_PROTEINMPNN"] is True
    assert result.manifest["scale1a1_admission_contract"] == {
        "implementation": (
            "dual_uq.dataset.admission."
            "evaluate_admission_candidates"
        ),
        "candidate_evaluator": (
            "dual_uq.dataset.admission._evaluate_candidate"
        ),
        "changed": False,
    }
    assert result.manifest["acquisition_contract"] == {
        "authorities": [
            "RCSB_PDB",
            "PDBe_SIFTS",
            "AlphaFold_DB",
            "UniProtKB",
        ],
        "candidate_identity": "frozen_declaration_exact_identity",
        "afdb_record_identity": "exact_uniprot_accession",
        "fragment_selection": "unique_full_cover_of_sifts_mapped_interval",
        "readiness_failure_is_scientific_rejection": False,
        "forbidden_fallbacks": [
            "F1_fallback",
            "nearest_fragment",
            "first_record",
            "manual_choice",
            "offset_inference",
            "closest_coverage_heuristic",
        ],
    }

    first = materialize_expansion_result(result, paths, config=config)
    second = materialize_expansion_result(result, paths, config=config)
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    manifest_path = paths.resolve_logical(
        "runs/scale1-expansion-wave1/scale1_expansion_wave1_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["readiness"]["rows"] == 100
    assert manifest["outputs"]["cluster_capacity"]["rows"] == 100
    assert manifest["outputs"]["manifest"]["self_hash_policy"].startswith(
        "reported_externally"
    )


def test_top_level_run_freezes_declaration_before_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionConfig(), output_root_ref="runs/scale1-expansion-wave1"
    )
    observed: dict[str, str] = {}

    def fake_readiness(**kwargs: object) -> DatasetExpansionReadinessResult:
        frozen = kwargs["frozen"]
        assert hasattr(frozen, "declaration_sha256")
        validate_frozen_declaration(frozen, paths, config=config)
        observed["declaration_sha256"] = frozen.declaration_sha256
        readiness = build_initial_readiness_frame(
            frozen.declaration,
            declaration_ref=(
                "runs/scale1-expansion-wave1/"
                "scale1_expansion_wave1_candidates.jsonl"
            ),
            declaration_sha256=frozen.declaration_sha256,
        )
        ledger = bind_acquisition_ledger(
            pd.DataFrame(
                columns=["sampling_frame_index", "candidate_id", "asset_type"]
            ),
            frozen.declaration_sha256,
        )
        return DatasetExpansionReadinessResult((), ledger, readiness)

    monkeypatch.setattr(
        expansion_workflow, "execute_expansion_readiness", fake_readiness
    )
    monkeypatch.setattr(
        expansion_workflow,
        "evaluate_expansion_admission",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                columns=["sampling_frame_index", "pair_id", "admission_status"]
            ),
            pd.DataFrame(),
        ),
    )

    output = run_dataset_expansion(paths, config=config, max_attempts=1)

    assert observed["declaration_sha256"] == output["outputs"]["declaration"][
        "sha256"
    ]
    assert output["verdict"] == "PASS"
    assert output["capacity_status"] == WAVE2_REQUIRED_CORE_NOT_REACHED


@pytest.mark.parametrize(
    ("new_clusters", "expected"),
    [
        (57, PRIMARY_CONFIRMATORY_CAPACITY_REACHED),
        (56, CORE_REACHED_PRIMARY_TARGET_PENDING),
        (37, CORE_REACHED_PRIMARY_TARGET_PENDING),
        (36, WAVE2_REQUIRED_CORE_NOT_REACHED),
    ],
)
def test_capacity_status_uses_only_the_frozen_63_plus_new_clusters(
    new_clusters: int, expected: str
) -> None:
    assert capacity_status(starting_capacity=63, new_clusters=new_clusters) == expected
