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
    PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED,
    PRIMARY_CONFIRMATORY_CAPACITY_REACHED,
    DatasetExpansionError,
    DatasetExpansionWave2Config,
    DatasetExpansionWave2ReadinessResult,
    assemble_wave2_result,
    bind_acquisition_ledger,
    build_initial_readiness_frame,
    build_wave2_cluster_capacity,
    build_wave2_freeze,
    execute_wave2_readiness,
    load_wave2_prior_acquisition_records,
    materialize_wave2_declaration,
    materialize_wave2_result,
    run_dataset_expansion_wave2,
    validate_wave2_declaration,
    validate_wave2_inputs,
    wave2_capacity_status,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _paths(**overrides: Path) -> ProjectPaths:
    return ProjectPaths.discover(project_root=REPOSITORY_ROOT, **overrides)


def test_wave2_inputs_reconstruct_frozen_start_and_exact_set_difference() -> None:
    inputs = validate_wave2_inputs(_paths(), DatasetExpansionWave2Config())

    assert len(inputs.remaining_targets) == 24
    assert inputs.remaining_targets["prospective_global_primary_rank"].tolist() == list(
        range(101, 125)
    )
    assert inputs.remaining_targets["sequence_cluster"].nunique() == 24
    assert len(inputs.clusters_before) == 112
    assert set(inputs.remaining_targets["sequence_cluster"]).isdisjoint(
        inputs.clusters_before
    )
    assert inputs.wave1_summary["capacity_status"] == (
        "CORE_REACHED_PRIMARY_TARGET_PENDING"
    )


def test_wave2_upstream_drift_blocks_before_declaration() -> None:
    config = replace(
        DatasetExpansionWave2Config(), expected_wave1_manifest_sha256="0" * 64
    )

    with pytest.raises(DatasetExpansionError) as caught:
        validate_wave2_inputs(_paths(), config)

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "upstream_sha256_mismatch"


def test_wave2_freeze_uses_only_remaining_persisted_ranks(tmp_path: Path) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    inputs = validate_wave2_inputs(paths, config)
    frozen = build_wave2_freeze(inputs)

    assert len(frozen.declaration) == 24
    assert [row["wave2_candidate_index"] for row in frozen.declaration] == list(
        range(1, 25)
    )
    assert [
        row["prospective_global_primary_rank"] for row in frozen.declaration
    ] == list(range(101, 125))
    assert len({row["sequence_cluster"] for row in frozen.declaration}) == 24
    assert len({row["pair_id"] for row in frozen.declaration}) == 24
    assert all(row["selection_role"] == "PRIMARY_CANDIDATE" for row in frozen.declaration)
    assert not any(row["reserve_activated"] for row in frozen.declaration)

    first = materialize_wave2_declaration(frozen, paths, config=config)
    second = materialize_wave2_declaration(frozen, paths, config=config)
    assert first["write_status"] == "created"
    assert second["write_status"] == "reused_identical"
    declaration_path = paths.resolve_logical(
        "runs/scale1-expansion-wave2/scale1_expansion_wave2_candidates.jsonl"
    )
    assert hashlib.sha256(declaration_path.read_bytes()).hexdigest() == (
        frozen.declaration_sha256
    )
    assert [json.loads(line) for line in declaration_path.read_text().splitlines()] == list(
        frozen.declaration
    )
    validate_wave2_declaration(frozen, paths, config=config)


def test_wave2_declaration_drift_blocks_before_realization(tmp_path: Path) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    frozen = build_wave2_freeze(validate_wave2_inputs(paths, config))
    materialize_wave2_declaration(frozen, paths, config=config)
    declaration_path = paths.resolve_logical(
        "runs/scale1-expansion-wave2/scale1_expansion_wave2_candidates.jsonl"
    )
    declaration_path.write_bytes(declaration_path.read_bytes() + b"\n")

    with pytest.raises(DatasetExpansionError) as caught:
        validate_wave2_declaration(frozen, paths, config=config)

    assert caught.value.status == BLOCKED_INPUT_INTEGRITY
    assert caught.value.code == "declaration_hash_mismatch"


@pytest.mark.parametrize(
    ("new_clusters", "expected", "remaining_gap"),
    [
        (0, PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED, 8),
        (7, PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED, 1),
        (8, PRIMARY_CONFIRMATORY_CAPACITY_REACHED, 0),
        (24, PRIMARY_CONFIRMATORY_CAPACITY_REACHED, 0),
    ],
)
def test_wave2_capacity_status_is_evaluated_after_all_declared_clusters(
    new_clusters: int, expected: str, remaining_gap: int
) -> None:
    status, gap = wave2_capacity_status(
        starting_capacity=112,
        new_clusters=new_clusters,
        target=120,
    )

    assert status == expected
    assert gap == remaining_gap


def test_wave2_readiness_is_declaration_bound_and_scopes_rejected_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    inputs = validate_wave2_inputs(paths, config)
    frozen = build_wave2_freeze(inputs)
    materialize_wave2_declaration(frozen, paths, config=config)
    rejected_refs: list[str] = []

    def fake_execute(*_args: object, **kwargs: object) -> tuple[()]:
        rejected_refs.append(str(kwargs["rejected_evidence_ref"]))
        return ()

    monkeypatch.setattr(expansion_workflow, "execute_initial_acquisition", fake_execute)
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

    result = execute_wave2_readiness(
        inputs=inputs,
        frozen=frozen,
        paths=paths,
        config=config,
        sleeper=lambda _delay: None,
    )

    expected_ref = (
        "artifacts/dataset/audits/acquisition_rejected/"
        f"scale1-expansion-wave2/{frozen.declaration_sha256}"
    )
    assert rejected_refs == [expected_ref, expected_ref, expected_ref]
    assert len(result.readiness) == 24
    assert result.acquisition_ledger.empty
    assert result.acquisition_ledger[
        "acquisition_bound_declaration_sha256"
    ].empty


def test_wave2_prior_ledger_rejects_declaration_drift(tmp_path: Path) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    ledger_path = paths.resolve_logical(
        "runs/scale1-expansion-wave2/"
        "scale1_expansion_wave2_acquisition_ledger.parquet"
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
        load_wave2_prior_acquisition_records(paths, "c" * 64, config=config)

    assert caught.value.code == "prior_acquisition_declaration_mismatch"


def test_wave2_all_24_clusters_finish_even_when_gate_is_reached_early() -> None:
    frozen = build_wave2_freeze(
        validate_wave2_inputs(_paths(), DatasetExpansionWave2Config())
    )
    readiness = pd.DataFrame(
        {
            "sampling_frame_index": range(1, 25),
            "candidate_id": [row["candidate_id"] for row in frozen.declaration],
            "pair_id": [row["pair_id"] for row in frozen.declaration],
            "local_availability_status": ["LOCAL_READY_FOR_SCALE1A"] * 24,
            "terminal_local_missing_reason": [None] * 24,
        }
    )
    admission = pd.DataFrame(
        [
            {
                "sampling_frame_index": index,
                "pair_id": row["pair_id"],
                "admission_status": (
                    "FORMALLY_ADMITTED"
                    if index <= 8
                    else "PENDING_HUMAN_VARIANT_REVIEW"
                ),
                "terminal_reason_code": "synthetic_terminal_outcome",
            }
            for index, row in enumerate(frozen.declaration, 1)
        ]
    )

    capacity = build_wave2_cluster_capacity(
        frozen.declaration, readiness, admission
    )

    assert len(capacity) == 24
    assert capacity["wave2_candidate_index"].tolist() == list(range(1, 25))
    assert int(capacity["capacity_contribution"].sum()) == 8
    assert capacity["terminal_wave2_cluster_outcome"].notna().all()
    assert capacity.loc[:7, "terminal_wave2_cluster_outcome"].eq(
        "FORMALLY_ADMITTED"
    ).all()
    assert capacity.loc[8:, "terminal_wave2_cluster_outcome"].eq(
        "PENDING_HUMAN_VARIANT_REVIEW"
    ).all()


def test_wave2_single_result_renders_outputs_and_manifest_last(tmp_path: Path) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    inputs = validate_wave2_inputs(paths, config)
    frozen = build_wave2_freeze(inputs)
    materialize_wave2_declaration(frozen, paths, config=config)
    adapted = [
        {**record, "wave1_candidate_index": record["wave2_candidate_index"]}
        for record in frozen.declaration
    ]
    readiness = build_initial_readiness_frame(
        adapted,
        declaration_ref=(
            "runs/scale1-expansion-wave2/scale1_expansion_wave2_candidates.jsonl"
        ),
        declaration_sha256=frozen.declaration_sha256,
    )
    admission = pd.DataFrame(
        columns=["sampling_frame_index", "pair_id", "admission_status"]
    )
    masks = pd.DataFrame()
    ledger = bind_acquisition_ledger(
        pd.DataFrame(columns=["sampling_frame_index", "candidate_id", "asset_type"]),
        frozen.declaration_sha256,
    )
    capacity = build_wave2_cluster_capacity(
        frozen.declaration, readiness, admission
    )
    result = assemble_wave2_result(
        inputs=inputs,
        frozen=frozen,
        acquisition_ledger=ledger,
        readiness=readiness,
        admission=admission,
        common_masks=masks,
        cluster_capacity=capacity,
    )

    assert result.summary["declared_clusters"] == 24
    assert result.summary["ready"] == 0
    assert result.summary["scientifically_evaluated"] == 0
    assert result.summary["starting_N_NR"] == 112
    assert result.summary["new_admitted_clusters"] == 0
    assert result.summary["final_N_NR"] == 112
    assert result.summary["remaining_gap_to_120"] == 8
    assert result.capacity_status == PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED
    assert result.manifest["NO_EARLY_STOP_AT_120"] is True
    assert result.manifest["WAVE2_RESERVE_ACTIVATION_AUTHORIZED"] is False

    first = materialize_wave2_result(result, paths, config=config)
    second = materialize_wave2_result(result, paths, config=config)
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    manifest_path = paths.resolve_logical(
        "runs/scale1-expansion-wave2/scale1_expansion_wave2_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["readiness"]["rows"] == 24
    assert manifest["outputs"]["cluster_capacity"]["rows"] == 24


def test_wave2_top_level_freezes_before_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(runs_root=tmp_path / "runs")
    config = replace(
        DatasetExpansionWave2Config(),
        output_root_ref="runs/scale1-expansion-wave2",
    )
    observed: dict[str, str] = {}

    def fake_readiness(**kwargs: object) -> DatasetExpansionWave2ReadinessResult:
        frozen = kwargs["frozen"]
        validate_wave2_declaration(frozen, paths, config=config)
        observed["declaration_sha256"] = frozen.declaration_sha256
        adapted = [
            {**record, "wave1_candidate_index": record["wave2_candidate_index"]}
            for record in frozen.declaration
        ]
        readiness = build_initial_readiness_frame(
            adapted,
            declaration_ref=(
                "runs/scale1-expansion-wave2/"
                "scale1_expansion_wave2_candidates.jsonl"
            ),
            declaration_sha256=frozen.declaration_sha256,
        )
        ledger = bind_acquisition_ledger(
            pd.DataFrame(
                columns=["sampling_frame_index", "candidate_id", "asset_type"]
            ),
            frozen.declaration_sha256,
        )
        return DatasetExpansionWave2ReadinessResult((), ledger, readiness)

    monkeypatch.setattr(expansion_workflow, "execute_wave2_readiness", fake_readiness)
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

    output = run_dataset_expansion_wave2(paths, config=config, max_attempts=1)

    assert observed["declaration_sha256"] == output["outputs"]["declaration"][
        "sha256"
    ]
    assert output["verdict"] == "PASS"
    assert output["capacity_status"] == PRIMARY_CONFIRMATORY_CAPACITY_NOT_REACHED
