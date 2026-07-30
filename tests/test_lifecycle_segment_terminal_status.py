from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.lifecycle import build_candidate_lifecycle


def _inputs(index: int = 101) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.DataFrame(
        [
            {
                "screening_index": index,
                "pdb_id": f"{index}abc",
                "chain_id": "A",
                "uniprot_id": f"P{index:05d}",
                "provisional_stratum": "lower_global_confidence_replacement",
            }
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": index,
                "preflight_status": "pass_full_length",
                "preflight_reason": "pass",
                "full_length_mapping_coverage": 1.0,
                "entity_mapping_coverage": 1.0,
                "sequence_identity": 1.0,
                "observed_ca_fraction_of_mapped": 1.0,
            }
        ]
    )
    return pool, preflight


def _pair_dir(root: Path, index: int) -> Path:
    return root / f"{index}abc_A__P{index:05d}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_upstream(root: Path, index: int) -> Path:
    pair_dir = _pair_dir(root, index)
    _write_json(
        pair_dir / "pair_qc.json",
        {
            "mapping_coverage": 1.0,
            "sequence_identity": 1.0,
            "quality_flag": "pass",
        },
    )
    _write_json(
        pair_dir / "pair_geometry_qc.json",
        {
            "pdb_afdb_aa_match_fraction": 1.0,
            "mapped_ca_coverage": 1.0,
        },
    )
    _write_json(
        pair_dir / "robust_pair_diagnostics.json",
        {"residue_count": 100},
    )
    return pair_dir


def _build(
    root: Path,
    status: pd.DataFrame,
    *,
    index: int = 101,
) -> pd.Series:
    pool, preflight = _inputs(index)
    lifecycle, _ = build_candidate_lifecycle(
        pool=pool,
        preflight=preflight,
        status=status,
        pair_root=root,
    )
    return lifecycle.iloc[0]


def test_explicit_successful_no_segments_overrides_nonempty_robust_csv(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    pd.DataFrame(
        [
            {
                "threshold": 1.0,
                "residue_count": 1,
                "max_disagreement": 1.2,
            }
        ]
    ).to_csv(pair_dir / "disagreement_segments.csv", index=False)
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "status": "complete",
                "segment_context_status": "successful_no_segments",
            }
        ]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "successful_no_segments"
    assert bool(row["complete_diagnostics"]) is True
    assert bool(row["quality_pass"]) is True


def test_explicit_complete_with_valid_artifact_remains_complete(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    _write_json(pair_dir / "segment_context.json", [{"start_position": 10}])
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "status": "complete",
                "segment_context_status": "complete",
            }
        ]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "complete"
    assert bool(row["complete_diagnostics"]) is True


def test_explicit_skipped_complete_preserves_completed_lifecycle_semantics(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    _write_json(pair_dir / "segment_context.json", [{"start_position": 10}])
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "status": "complete",
                "segment_context_status": "skipped_complete",
            }
        ]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "complete"
    assert bool(row["complete_diagnostics"]) is True


def test_explicit_failure_is_not_overridden_by_stale_complete_artifact(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    _write_json(pair_dir / "segment_context.json", [{"start_position": 10}])
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "status": "failed_segment_context",
                "segment_context_status": "failed_segment_context",
            }
        ]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "failed_segment_context"
    assert bool(row["complete_diagnostics"]) is False


def test_latest_attempt_terminal_wins_without_mutating_history(
    tmp_path: Path,
) -> None:
    _write_complete_upstream(tmp_path, 101)
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "stage": "segment_context",
                "attempt": 1,
                "status": "failed_segment_context",
            },
            {
                "screening_index": 101,
                "stage": "segment_context",
                "attempt": 2,
                "status": "successful_no_segments",
            },
        ]
    )
    original = status.copy(deep=True)

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "successful_no_segments"
    pd.testing.assert_frame_equal(status, original)


def test_interrupted_latest_attempt_is_not_inferred_from_artifacts(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    (pair_dir / "disagreement_segments.csv").write_text(
        "threshold,residue_count\n",
        encoding="utf-8",
    )
    status = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "stage": "segment_context",
                "attempt": 1,
                "status": "successful_no_segments",
            },
            {
                "screening_index": 101,
                "stage": "segment_context",
                "attempt": 2,
                "status": "running",
            },
        ]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "in_progress"
    assert bool(row["complete_diagnostics"]) is False


def test_legacy_artifact_fallback_applies_without_stage_status(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 101)
    (pair_dir / "disagreement_segments.csv").write_text("\n", encoding="utf-8")
    status = pd.DataFrame(
        [{"screening_index": 101, "status": "complete"}]
    )

    row = _build(tmp_path, status)

    assert row["segment_context_status"] == "successful_no_segments"
    assert bool(row["complete_diagnostics"]) is True


@pytest.mark.parametrize("index", [103, 105, 106])
def test_replacement_no_segment_regression_does_not_become_not_started(
    tmp_path: Path,
    index: int,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, index)
    pd.DataFrame(
        [{"threshold": 1.0, "residue_count": 1}]
    ).to_csv(pair_dir / "disagreement_segments.csv", index=False)
    status = pd.DataFrame(
        [
            {
                "screening_index": index,
                "status": "complete",
                "segment_context_status": "successful_no_segments",
            }
        ]
    )

    row = _build(tmp_path, status, index=index)

    assert row["segment_context_status"] == "successful_no_segments"
    assert bool(row["complete_diagnostics"]) is True


def test_original_legacy_complete_artifact_semantics_are_unchanged(
    tmp_path: Path,
) -> None:
    pair_dir = _write_complete_upstream(tmp_path, 6)
    _write_json(pair_dir / "segment_context.json", [{"start_position": 42}])
    status = pd.DataFrame(
        [{"screening_index": 6, "status": "complete"}]
    )

    row = _build(tmp_path, status, index=6)

    assert row["pair_status"] == "complete"
    assert row["geometry_status"] == "complete"
    assert row["robust_status"] == "complete"
    assert row["segment_context_status"] == "complete"
    assert bool(row["complete_diagnostics"]) is True
