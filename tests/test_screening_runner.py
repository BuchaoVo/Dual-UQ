from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dual_uq.screening_runner import (
    GEOMETRY,
    PAIR,
    ROBUST,
    SEGMENT_CONTEXT,
    OutputValidation,
    build_stage_plan,
    classify_candidate_eligibility,
    derive_stage_seed,
    execute_stage,
    make_terminal_status_record,
    reduce_status_history,
    summarize_status_records,
    validate_stage_outputs,
)


def _candidate(
    *,
    index: int = 6,
    pdb_id: str = "1gci",
    chain_id: str = "A",
    uniprot_id: str = "P29600",
    preflight_status: str = "pass_full_length",
) -> dict[str, object]:
    return {
        "screening_index": index,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": f"{pdb_id}_{chain_id}__{uniprot_id}",
        "provisional_stratum": "test",
        "preflight_status": preflight_status,
        "segment_threshold": 1.0,
        "segment_min_length": 3,
    }


def _validation(valid: bool, status: str = "complete") -> OutputValidation:
    return OutputValidation(valid=valid, status=status)


def _output_state(
    *,
    pair: bool = False,
    geometry: bool = False,
    robust: bool = False,
    segment: bool = False,
) -> dict[str, OutputValidation]:
    return {
        PAIR: _validation(pair),
        GEOMETRY: _validation(geometry),
        ROBUST: _validation(robust),
        SEGMENT_CONTEXT: _validation(segment),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _pair_dir(root: Path, candidate: dict[str, object]) -> Path:
    return root / "data/processed/pairs" / str(candidate["pair_name"])


def _write_valid_pair(root: Path, candidate: dict[str, object]) -> Path:
    pair_dir = _pair_dir(root, candidate)
    mapping_path = pair_dir / "residue_mapping.parquet"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "auth_asym_id": [candidate["chain_id"]],
            "auth_seq_id": [1],
            "insertion_code": [""],
            "uniprot_residue_number": [1],
        }
    ).to_parquet(mapping_path, index=False)
    _write_json(
        pair_dir / "pair_qc.json",
        {
            "pdb_id": candidate["pdb_id"],
            "chain_id": candidate["chain_id"],
            "uniprot_id": candidate["uniprot_id"],
            "mapped_residue_count": 1,
            "mapping_path": str(mapping_path),
            "afdb_model_entity_id": f"AF-{candidate['uniprot_id']}-F1",
            "afdb_version": 6,
            "quality_flag": "pass",
        },
    )
    return pair_dir


def _write_valid_geometry(root: Path, candidate: dict[str, object]) -> Path:
    pair_dir = _write_valid_pair(root, candidate)
    residue_path = pair_dir / "residue_geometry.parquet"
    pd.DataFrame(
        {
            "uniprot_residue_number": [1],
            "aligned_ca_distance": [0.1],
            "plddt": [95.0],
        }
    ).to_parquet(residue_path, index=False)
    np.savez_compressed(
        pair_dir / "pairwise_geometry.npz",
        symmetric_pae=np.ones((1, 1)),
        absolute_pairwise_error=np.zeros((1, 1)),
        uniprot_positions=np.array([1]),
    )
    _write_json(
        pair_dir / "pair_geometry_qc.json",
        {
            "pdb_id": candidate["pdb_id"],
            "chain_id": candidate["chain_id"],
            "uniprot_id": candidate["uniprot_id"],
            "mapped_ca_count": 1,
            "residue_output": str(residue_path),
        },
    )
    return pair_dir


def _write_valid_robust(
    root: Path,
    candidate: dict[str, object],
    *,
    with_segments: bool = True,
) -> Path:
    pair_dir = _write_valid_geometry(root, candidate)
    segments = pd.DataFrame(
        (
            [
                {
                    "threshold": 1.0,
                    "start_position": 1,
                    "end_position": 3,
                    "residue_count": 3,
                }
            ]
            if with_segments
            else []
        ),
        columns=[
            "threshold",
            "start_position",
            "end_position",
            "residue_count",
        ],
    )
    segments.to_csv(pair_dir / "disagreement_segments.csv", index=False)
    pd.DataFrame({"sequence_separation_bin": ["6-11"]}).to_csv(
        pair_dir / "pairwise_strata.csv", index=False
    )
    pd.DataFrame({"uniprot_residue_number": []}).to_csv(
        pair_dir / "high_confidence_disagreement.csv", index=False
    )
    _write_json(
        pair_dir / "robust_pair_diagnostics.json",
        {
            "pair_dir": str(pair_dir),
            "residue_count": 1,
            "local_plddt_disagreement_test": {"n_permutations": 1000},
            "pae_pairwise_error_test": {"n_permutations": 500},
        },
    )
    return pair_dir


def test_default_eligibility_is_pass_only() -> None:
    assert classify_candidate_eligibility("pass_full_length").eligible
    assert not classify_candidate_eligibility("warn_construct_difference").eligible
    assert not classify_candidate_eligibility("fail_preflight").eligible


def test_warnings_require_explicit_inclusion() -> None:
    assert classify_candidate_eligibility(
        "warn_construct_difference", include_warnings=True
    ).eligible
    assert not classify_candidate_eligibility(
        "fail_preflight", include_warnings=True
    ).eligible


def test_preflight_failure_is_skipped_not_failed_pair() -> None:
    plan = build_stage_plan(
        _candidate(preflight_status="fail_preflight"),
        [],
        _output_state(),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    assert {item.action for item in plan} == {"skip_preflight"}
    assert all(item.reason == "skipped_preflight" for item in plan)


def test_unsupported_fragment_never_runs_downstream() -> None:
    plan = build_stage_plan(
        _candidate(
            index=9,
            pdb_id="7kr0",
            uniprot_id="P0DTD1",
            preflight_status="unsupported_afdb_fragment",
        ),
        [],
        _output_state(pair=True),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    assert {item.action for item in plan} == {"skip_unsupported"}
    assert all(item.reason == "unsupported_afdb_fragment" for item in plan)
    assert not any(
        item.stage in {GEOMETRY, ROBUST, SEGMENT_CONTEXT} and item.action == "run"
        for item in plan
    )


def test_stage_resume_runs_only_first_missing_stage_then_unlocks_next() -> None:
    candidate = _candidate()
    state = _output_state(pair=True, geometry=True)

    first = build_stage_plan(
        candidate,
        [],
        state,
        resume=True,
        force_stages=set(),
        base_seed=1,
    )
    assert [item.stage for item in first if item.action == "run"] == [ROBUST]
    assert next(item for item in first if item.stage == PAIR).action == "skip_complete"
    assert next(item for item in first if item.stage == GEOMETRY).action == "skip_complete"
    assert (
        next(item for item in first if item.stage == SEGMENT_CONTEXT).action
        == "blocked_upstream"
    )

    state[ROBUST] = _validation(True)
    second = build_stage_plan(
        candidate,
        [],
        state,
        resume=True,
        force_stages=set(),
        base_seed=1,
    )
    assert [item.stage for item in second if item.action == "run"] == [
        SEGMENT_CONTEXT
    ]


@pytest.mark.parametrize(
    ("corruption", "stage"),
    [
        ("invalid_pair_json", PAIR),
        ("empty_mapping", PAIR),
        ("wrong_candidate", PAIR),
        ("missing_geometry_field", GEOMETRY),
        ("invalid_geometry_parquet", GEOMETRY),
    ],
)
def test_invalid_existing_outputs_are_not_complete(
    tmp_path: Path,
    corruption: str,
    stage: str,
) -> None:
    candidate = _candidate()
    pair_dir = (
        _write_valid_geometry(tmp_path, candidate)
        if stage == GEOMETRY
        else _write_valid_pair(tmp_path, candidate)
    )
    if corruption == "invalid_pair_json":
        (pair_dir / "pair_qc.json").write_text("{", encoding="utf-8")
    elif corruption == "empty_mapping":
        (pair_dir / "residue_mapping.parquet").write_bytes(b"")
    elif corruption == "wrong_candidate":
        report = json.loads((pair_dir / "pair_qc.json").read_text(encoding="utf-8"))
        report["uniprot_id"] = "WRONG"
        _write_json(pair_dir / "pair_qc.json", report)
    elif corruption == "missing_geometry_field":
        _write_json(
            pair_dir / "pair_geometry_qc.json",
            {
                "pdb_id": candidate["pdb_id"],
                "chain_id": candidate["chain_id"],
                "uniprot_id": candidate["uniprot_id"],
            },
        )
    elif corruption == "invalid_geometry_parquet":
        (pair_dir / "residue_geometry.parquet").write_text(
            "not parquet", encoding="utf-8"
        )

    validation = validate_stage_outputs(stage, candidate, tmp_path)

    assert not validation.valid
    plan = build_stage_plan(
        candidate,
        [],
        {
            **_output_state(pair=True),
            stage: validation,
        },
        resume=True,
        force_stages=set(),
        base_seed=1,
    )
    assert next(item for item in plan if item.stage == stage).action == "run"


def test_malformed_numeric_report_field_is_invalid_not_an_exception(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    pair_dir = _write_valid_pair(tmp_path, candidate)
    report = json.loads((pair_dir / "pair_qc.json").read_text(encoding="utf-8"))
    report["mapped_residue_count"] = "not-an-integer"
    _write_json(pair_dir / "pair_qc.json", report)

    validation = validate_stage_outputs(PAIR, candidate, tmp_path)

    assert not validation.valid
    assert str(pair_dir / "pair_qc.json") in validation.invalid_paths


def test_geometry_failure_blocks_robust_and_segment() -> None:
    history = [
        {
            "screening_index": 6,
            "stage": GEOMETRY,
            "attempt": 1,
            "status": "failed_geometry",
        }
    ]
    plan = build_stage_plan(
        _candidate(),
        history,
        _output_state(pair=True, robust=True, segment=True),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    assert next(item for item in plan if item.stage == GEOMETRY).action == "run"
    assert next(item for item in plan if item.stage == ROBUST).action == "blocked_upstream"
    assert (
        next(item for item in plan if item.stage == SEGMENT_CONTEXT).action
        == "blocked_upstream"
    )


def test_verified_no_segments_supersedes_old_segment_failure() -> None:
    plan = build_stage_plan(
        _candidate(),
        [
            {
                "screening_index": 6,
                "stage": SEGMENT_CONTEXT,
                "attempt": 1,
                "status": "failed_segment_context",
            }
        ],
        {
            **_output_state(pair=True, geometry=True, robust=True),
            SEGMENT_CONTEXT: _validation(True, "successful_no_segments"),
        },
        resume=True,
        force_stages={SEGMENT_CONTEXT},
        base_seed=1,
    )

    segment = next(item for item in plan if item.stage == SEGMENT_CONTEXT)
    assert segment.action == "skip_complete"
    assert segment.reason == "successful_no_segments"


def test_stale_no_segments_is_blocked_while_robust_reruns() -> None:
    plan = build_stage_plan(
        _candidate(),
        [],
        {
            **_output_state(pair=True, geometry=True),
            ROBUST: _validation(False),
            SEGMENT_CONTEXT: _validation(True, "successful_no_segments"),
        },
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    assert next(item for item in plan if item.stage == ROBUST).action == "run"
    assert (
        next(item for item in plan if item.stage == SEGMENT_CONTEXT).action
        == "blocked_upstream"
    )


def test_persisted_downstream_invalidation_forces_stale_cache_rerun() -> None:
    history = [
        {
            "screening_index": 6,
            "stage": GEOMETRY,
            "attempt": 2,
            "status": "complete",
        },
        {
            "screening_index": 6,
            "stage": ROBUST,
            "attempt": 2,
            "status": "blocked_upstream",
        },
        {
            "screening_index": 6,
            "stage": SEGMENT_CONTEXT,
            "attempt": 2,
            "status": "blocked_upstream",
        },
    ]
    plan = build_stage_plan(
        _candidate(),
        history,
        _output_state(pair=True, geometry=True, robust=True, segment=True),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    assert next(item for item in plan if item.stage == ROBUST).action == "run"
    assert (
        next(item for item in plan if item.stage == SEGMENT_CONTEXT).action
        == "blocked_upstream"
    )


def test_interrupted_running_attempt_is_recoverable() -> None:
    history = [
        {
            "screening_index": 6,
            "stage": GEOMETRY,
            "attempt": 2,
            "status": "running",
        }
    ]
    plan = build_stage_plan(
        _candidate(),
        history,
        _output_state(pair=True),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )

    geometry = next(item for item in plan if item.stage == GEOMETRY)
    assert geometry.action == "run"
    assert geometry.reason == "interrupted_attempt"


def test_stage_seed_is_stable_and_distinct() -> None:
    first = derive_stage_seed(20260730, 6, GEOMETRY)

    assert first == derive_stage_seed(20260730, 6, GEOMETRY)
    assert first != derive_stage_seed(20260730, 7, GEOMETRY)
    assert first != derive_stage_seed(20260730, 6, ROBUST)
    assert 0 <= first <= (2**32 - 1)


def test_corrupt_segment_csv_is_invalid_not_no_segments(tmp_path: Path) -> None:
    candidate = _candidate()
    pair_dir = _write_valid_robust(tmp_path, candidate, with_segments=False)
    (pair_dir / "disagreement_segments.csv").write_bytes(b"\xff\xfe\x00")

    validation = validate_stage_outputs(SEGMENT_CONTEXT, candidate, tmp_path)

    assert not validation.valid
    assert str(pair_dir / "disagreement_segments.csv") in validation.invalid_paths


def test_no_segments_is_a_successful_terminal_state(tmp_path: Path) -> None:
    candidate = _candidate()
    _write_valid_robust(tmp_path, candidate, with_segments=False)

    validation = validate_stage_outputs(SEGMENT_CONTEXT, candidate, tmp_path)

    assert validation.valid
    assert validation.status == "successful_no_segments"
    assert all(Path(path).exists() for path in validation.checked_paths)


def test_invalid_pairwise_npz_invalidates_geometry(tmp_path: Path) -> None:
    candidate = _candidate()
    pair_dir = _write_valid_geometry(tmp_path, candidate)
    (pair_dir / "pairwise_geometry.npz").write_bytes(b"not-an-npz")

    validation = validate_stage_outputs(GEOMETRY, candidate, tmp_path)

    assert not validation.valid
    assert str(pair_dir / "pairwise_geometry.npz") in validation.invalid_paths


def test_corrupt_segments_invalidates_robust_stage(tmp_path: Path) -> None:
    candidate = _candidate()
    pair_dir = _write_valid_robust(tmp_path, candidate)
    (pair_dir / "disagreement_segments.csv").write_bytes(b"\xff\xfe\x00")

    validation = validate_stage_outputs(ROBUST, candidate, tmp_path)

    assert not validation.valid
    assert str(pair_dir / "disagreement_segments.csv") in validation.invalid_paths


def test_zero_return_code_still_requires_valid_outputs() -> None:
    plan = build_stage_plan(
        _candidate(),
        [],
        _output_state(),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )[0]

    record = make_terminal_status_record(
        plan,
        run_id="run",
        attempt=1,
        return_code=0,
        runtime_seconds=0.5,
        validation=_validation(False),
    )

    assert record.status == "failed_pair"
    assert record.error_type == "OutputValidationError"


def test_validator_exception_is_written_as_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_stage_plan(
        {
            **_candidate(),
            "commands": {PAIR: [sys.executable, "-c", "pass"]},
        },
        [],
        _output_state(),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )[0]
    monkeypatch.setattr(
        "dual_uq.screening_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "dual_uq.screening_runner.validate_stage_outputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("validator crashed")),
    )
    status_path = tmp_path / "status.jsonl"

    terminal = execute_stage(
        plan,
        project_root=tmp_path,
        log_path=tmp_path / "stage.log",
        status_path=status_path,
        run_id="run",
        attempt=1,
    )

    records = [
        json.loads(line)
        for line in status_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == ["running", "failed_pair"]
    assert terminal.error_type == "RuntimeError"
    assert terminal.error_message == "validator crashed"


def test_subprocess_adapter_exception_is_written_as_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_stage_plan(
        {
            **_candidate(),
            "commands": {PAIR: [sys.executable, "-c", "pass"]},
        },
        [],
        _output_state(),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )[0]
    monkeypatch.setattr(
        "dual_uq.screening_runner.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("adapter rejected command")
        ),
    )
    status_path = tmp_path / "status.jsonl"

    terminal = execute_stage(
        plan,
        project_root=tmp_path,
        log_path=tmp_path / "stage.log",
        status_path=status_path,
        run_id="run",
        attempt=1,
    )

    assert terminal.status == "failed_pair"
    assert terminal.error_type == "ValueError"
    assert terminal.error_message == "adapter rejected command"


def test_failed_stage_record_contains_structured_error() -> None:
    plan = build_stage_plan(
        _candidate(),
        [],
        _output_state(),
        resume=True,
        force_stages=set(),
        base_seed=1,
    )[0]

    record = make_terminal_status_record(
        plan,
        run_id="run",
        attempt=3,
        return_code=17,
        runtime_seconds=1.25,
        validation=_validation(False),
        error_type="CalledProcessError",
        error_message="stage exited nonzero",
    )

    assert record.status == "failed_pair"
    assert record.error_type == "CalledProcessError"
    assert record.error_message == "stage exited nonzero"
    assert record.return_code == 17
    assert record.runtime_seconds == 1.25


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        (
            [
                {
                    "screening_index": 6,
                    "stage": PAIR,
                    "attempt": 1,
                    "status": "complete",
                },
                {
                    "screening_index": 6,
                    "stage": PAIR,
                    "attempt": 2,
                    "status": "failed_pair",
                },
            ],
            "failed_pair",
        ),
        (
            [
                {
                    "screening_index": 6,
                    "stage": PAIR,
                    "attempt": 1,
                    "status": "failed_pair",
                },
                {
                    "screening_index": 6,
                    "stage": PAIR,
                    "attempt": 2,
                    "status": "complete",
                },
            ],
            "complete",
        ),
    ],
)
def test_history_reduction_uses_latest_attempt_and_preserves_history(
    records: list[dict[str, object]],
    expected: str,
) -> None:
    reduced = reduce_status_history(records)
    state = reduced[(6, PAIR)]

    assert state.status == expected
    assert state.attempt == 2
    assert len(state.history) == 2


def test_pending_and_unknown_status_do_not_supersede_valid_terminal() -> None:
    records = [
        {
            "screening_index": 6,
            "stage": PAIR,
            "attempt": 1,
            "status": "complete",
        },
        {
            "screening_index": 6,
            "stage": PAIR,
            "attempt": 2,
            "status": "pending",
        },
        {
            "screening_index": 6,
            "stage": PAIR,
            "attempt": 3,
            "status": "typo_status",
        },
    ]

    state = reduce_status_history(records)[(6, PAIR)]

    assert state.attempt == 1
    assert state.status == "complete"


def test_pending_is_supported_when_no_attempt_has_started() -> None:
    state = reduce_status_history(
        [
            {
                "screening_index": 6,
                "stage": PAIR,
                "attempt": 1,
                "status": "pending",
            }
        ]
    )[(6, PAIR)]

    assert state.status == "pending"


def test_force_stage_invalidates_that_stage_and_all_downstream() -> None:
    plan = build_stage_plan(
        _candidate(),
        [],
        _output_state(pair=True, geometry=True, robust=True, segment=True),
        resume=True,
        force_stages={GEOMETRY},
        base_seed=1,
    )

    assert next(item for item in plan if item.stage == PAIR).action == "skip_complete"
    assert next(item for item in plan if item.stage == GEOMETRY).action == "run"
    assert next(item for item in plan if item.stage == ROBUST).action == "blocked_upstream"
    assert (
        next(item for item in plan if item.stage == SEGMENT_CONTEXT).action
        == "blocked_upstream"
    )


def test_structured_status_summary_uses_latest_stage_attempts() -> None:
    common = {
        "run_id": "run",
        "screening_index": 24,
        "pdb_id": "3o4p",
        "chain_id": "A",
        "uniprot_id": "Q7SIG4",
        "pair_name": "3o4p_A__Q7SIG4",
        "preflight_status": "pass_full_length",
        "provisional_stratum": "long_backbone_proxy",
    }
    records = [
        {**common, "stage": PAIR, "attempt": 1, "status": "failed_pair", "runtime_seconds": 1.0},
        {**common, "stage": PAIR, "attempt": 2, "status": "complete", "runtime_seconds": 2.0},
        {**common, "stage": GEOMETRY, "attempt": 1, "status": "complete", "runtime_seconds": 3.0},
        {**common, "stage": ROBUST, "attempt": 1, "status": "complete", "runtime_seconds": 4.0},
        {
            **common,
            "stage": SEGMENT_CONTEXT,
            "attempt": 1,
            "status": "successful_no_segments",
            "runtime_seconds": 0.0,
        },
    ]

    summary, statistics = summarize_status_records(records)

    assert summary.loc[0, "pair_status"] == "complete"
    assert summary.loc[0, "segment_context_status"] == "successful_no_segments"
    assert summary.loc[0, "candidate_terminal_status"] == "complete"
    assert summary.loc[0, "total_runtime_seconds"] == 10.0
    assert statistics["complete_candidates"] == 1
    assert statistics["successful_no_segment_cases"] == 1


def test_summary_counts_recovered_interruption_and_latest_error_chronologically() -> None:
    common = {
        "run_id": "run",
        "screening_index": 6,
        "pdb_id": "1gci",
        "chain_id": "A",
        "uniprot_id": "P29600",
        "pair_name": "1gci_A__P29600",
        "preflight_status": "pass_full_length",
    }
    records = [
        {**common, "stage": GEOMETRY, "attempt": 1, "status": "running"},
        {
            **common,
            "stage": ROBUST,
            "attempt": 1,
            "status": "failed_robust",
            "error_type": "OldError",
            "error_message": "older downstream",
        },
        {**common, "stage": GEOMETRY, "attempt": 2, "status": "complete"},
        {
            **common,
            "stage": PAIR,
            "attempt": 2,
            "status": "failed_pair",
            "error_type": "NewError",
            "error_message": "newer upstream",
        },
    ]

    summary, statistics = summarize_status_records(records)

    assert statistics["interrupted_attempts"] == 1
    assert summary.loc[0, "last_error_type"] == "NewError"
    assert summary.loc[0, "last_error_message"] == "newer upstream"


def test_default_skipped_warning_is_not_counted_as_eligible() -> None:
    common = {
        "run_id": "run",
        "screening_index": 35,
        "pdb_id": "1x8p",
        "chain_id": "A",
        "uniprot_id": "Q94734",
        "pair_name": "1x8p_A__Q94734",
        "preflight_status": "warn_construct_difference",
    }
    records = [
        {
            **common,
            "stage": stage,
            "attempt": 1,
            "status": "skipped_preflight",
        }
        for stage in (PAIR, GEOMETRY, ROBUST, SEGMENT_CONTEXT)
    ]

    _, statistics = summarize_status_records(records)

    assert statistics["eligible_candidates"] == 0
