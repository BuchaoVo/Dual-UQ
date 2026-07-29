from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dual_uq.mapped_confidence import (
    build_mapped_confidence_residue_table,
    find_low_confidence_segments,
    summarize_mapped_confidence,
    validate_mapped_confidence_input,
)

MODEL_ID = "AF-PTEST-F2"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "22_score_mapped_confidence.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "score_mapped_confidence",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCORE_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCORE_SCRIPT)


def _residue_table(
    positions: list[int] | range | None = None,
    *,
    low_positions: set[int] | None = None,
    low_value: float = 75.0,
    observed_missing: set[int] | None = None,
    mapped_false: set[int] | None = None,
) -> pd.DataFrame:
    low_positions = low_positions or set()
    observed_missing = observed_missing or set()
    mapped_false = mapped_false or set()
    values = list(range(1, 21) if positions is None else positions)
    return pd.DataFrame(
        {
            "uniprot_residue_number": values,
            "mapped": [position not in mapped_false for position in values],
            "observed_ca": [
                position not in observed_missing for position in values
            ],
            "fragment_covered": True,
            "plddt": [
                low_value if position in low_positions else 95.0
                for position in values
            ],
            "auth_asym_id": "A",
            "auth_seq_id": values,
            "insertion_code": "",
        }
    )


def _validate(
    table: pd.DataFrame,
    *,
    selected_model_entity_id: str = MODEL_ID,
    plddt_model_entity_id: str = MODEL_ID,
    fragment_start: int = 1,
    fragment_end: int = 20,
):
    return validate_mapped_confidence_input(
        table,
        selected_model_entity_id=selected_model_entity_id,
        plddt_model_entity_id=plddt_model_entity_id,
        fragment_start=fragment_start,
        fragment_end=fragment_end,
    )


def test_internal_below_80_segment_is_local_low_confidence() -> None:
    table = _residue_table(low_positions=set(range(8, 14)))

    summary = summarize_mapped_confidence(table)

    assert summary.longest_below_80_length == 6
    assert summary.longest_internal_below_80_length == 6
    assert summary.longest_below_80_segment is not None
    assert summary.longest_below_80_segment.is_internal
    assert summary.is_low_conf_local


def test_internal_below_70_segment_is_strong_local_evidence() -> None:
    table = _residue_table(
        low_positions=set(range(8, 14)),
        low_value=65.0,
    )

    summary = summarize_mapped_confidence(table)

    assert summary.longest_below_70_length == 6
    assert summary.longest_below_80_length == 6
    assert summary.is_low_conf_local


@pytest.mark.parametrize(
    "low_positions",
    [set(range(1, 7)), set(range(15, 21))],
)
def test_terminal_only_segment_is_not_local_low_confidence(
    low_positions: set[int],
) -> None:
    summary = summarize_mapped_confidence(
        _residue_table(low_positions=low_positions)
    )

    assert summary.longest_below_80_length == 6
    assert summary.longest_below_80_segment is not None
    assert summary.longest_below_80_segment.is_terminal
    assert not summary.longest_below_80_segment.is_internal
    assert not summary.is_low_conf_local


def test_unmapped_position_breaks_low_confidence_run() -> None:
    positions = [position for position in range(1, 21) if position != 11]
    low = {8, 9, 10, 12, 13, 14}

    summary = summarize_mapped_confidence(
        _residue_table(positions, low_positions=low)
    )

    assert summary.longest_below_80_length == 3
    assert summary.below_80_segment_count == 2
    assert summary.unmapped_position_interruption_count == 1


def test_missing_ca_breaks_run_and_is_reported() -> None:
    summary = summarize_mapped_confidence(
        _residue_table(
            low_positions=set(range(8, 14)),
            observed_missing={11},
        )
    )

    assert summary.longest_below_80_length == 3
    assert summary.missing_ca_interruption_count == 1
    assert summary.scorable_position_count == 19
    assert not summary.is_low_conf_local


def test_nonconsecutive_uniprot_numbers_form_separate_runs() -> None:
    table = _residue_table(
        [8, 9, 10, 20, 21, 22],
        low_positions={8, 9, 10, 20, 21, 22},
    )

    segments = find_low_confidence_segments(table, threshold=80.0)

    assert [segment.positions for segment in segments] == [
        (8, 9, 10),
        (20, 21, 22),
    ]


def test_threshold_comparison_is_strictly_below() -> None:
    table = _residue_table([1, 2, 3, 4])
    table["plddt"] = [69.99, 70.0, 79.99, 80.0]

    summary = summarize_mapped_confidence(table)

    assert summary.below_70_position_count == 1
    assert summary.below_80_position_count == 3


@pytest.mark.parametrize("invalid", [np.nan, -0.01, 100.01])
def test_invalid_mapped_plddt_is_structured_failure(invalid: float) -> None:
    table = _residue_table()
    table.loc[7, "plddt"] = invalid

    validation = _validate(table)

    assert not validation.valid
    assert validation.reason == "invalid_mapped_plddt"


def test_identical_duplicate_uniprot_positions_are_deduplicated() -> None:
    table = _residue_table()
    table = pd.concat([table, table.iloc[[7]]], ignore_index=True)

    validation = _validate(table)

    assert validation.valid
    assert validation.residue_table is not None
    assert len(validation.residue_table) == 20


@pytest.mark.parametrize(
    ("column", "value"),
    [("plddt", 60.0), ("observed_ca", False)],
)
def test_conflicting_duplicate_uniprot_positions_fail(
    column: str,
    value: object,
) -> None:
    table = _residue_table()
    duplicate = table.iloc[[7]].copy()
    duplicate[column] = value
    table = pd.concat([table, duplicate], ignore_index=True)

    validation = _validate(table)

    assert not validation.valid
    assert validation.reason == "conflicting_duplicate_uniprot_position"


def test_selected_fragment_must_cover_every_mapped_position() -> None:
    table = _residue_table(range(1, 22))

    validation = _validate(table, fragment_start=1, fragment_end=20)

    assert not validation.valid
    assert validation.reason == "afdb_fragment_coverage_mismatch"


def test_plddt_model_must_match_selected_model() -> None:
    validation = _validate(
        _residue_table(),
        plddt_model_entity_id="AF-PTEST-F1",
    )

    assert not validation.valid
    assert validation.reason == "afdb_model_entity_id_mismatch"


def test_missing_model_identifiers_cannot_match() -> None:
    validation = _validate(
        _residue_table(),
        selected_model_entity_id=None,
        plddt_model_entity_id=None,
    )

    assert not validation.valid
    assert validation.reason == "afdb_model_entity_id_mismatch"


@pytest.mark.parametrize(
    ("column", "value"),
    [("auth_asym_id", ""), ("auth_seq_id", 1.5)],
)
def test_invalid_explicit_auth_key_is_rejected(
    column: str,
    value: object,
) -> None:
    table = _residue_table()
    if column == "auth_seq_id":
        table[column] = table[column].astype(float)
    table.loc[0, column] = value

    validation = _validate(table)

    assert not validation.valid
    assert validation.reason == "missing_explicit_auth_residue_key"


def test_mapped_statistics_use_only_observed_mapped_residues() -> None:
    table = _residue_table([1, 2, 3, 4, 5])
    table["plddt"] = [10.0, 20.0, 30.0, 40.0, 100.0]
    table.loc[0, "mapped"] = False
    table.loc[1, "observed_ca"] = False

    summary = summarize_mapped_confidence(table)

    assert summary.mapped_position_count == 4
    assert summary.scorable_position_count == 3
    assert summary.mapped_plddt_min == 30.0
    assert summary.mapped_plddt_q10 == pytest.approx(32.0)
    assert summary.mapped_plddt_median == 40.0


@pytest.mark.parametrize(
    ("run_length", "expected"),
    [(4, False), (5, True)],
)
def test_minimum_internal_run_length_boundary(
    run_length: int,
    expected: bool,
) -> None:
    low_positions = set(range(8, 8 + run_length))

    summary = summarize_mapped_confidence(
        _residue_table(low_positions=low_positions)
    )

    assert summary.is_low_conf_local is expected


def test_multiple_segments_count_all_and_choose_earliest_tied_longest() -> None:
    table = _residue_table(
        range(1, 31),
        low_positions={7, 8, 9, 10, 11, 20, 21, 22, 23, 24},
    )

    summary = summarize_mapped_confidence(table)

    assert summary.below_80_segment_count == 2
    assert summary.longest_below_80_segment is not None
    assert summary.longest_below_80_segment.start_uniprot == 7
    assert summary.longest_below_80_segment.end_uniprot == 11


def test_empty_input_is_structured_failure() -> None:
    validation = _validate(_residue_table().iloc[0:0])

    assert not validation.valid
    assert validation.reason == "no_mapped_residues"


def test_explicit_t4_numbering_does_not_require_legacy_column() -> None:
    table = _residue_table(low_positions=set(range(8, 13)))

    assert "pdb_residue_number" not in table
    validation = _validate(table)
    summary = summarize_mapped_confidence(table)

    assert validation.valid
    assert summary.is_low_conf_local


def test_fragment_offset_and_observed_ca_use_semantic_residue_keys() -> None:
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [101, 102, 103],
            "auth_asym_id": ["A", "A", "A"],
            "label_asym_id": [pd.NA, pd.NA, pd.NA],
            "auth_seq_id": [10, 11, 12],
            "label_seq_id": [pd.NA, pd.NA, pd.NA],
            "insertion_code": ["", "", ""],
        }
    )
    pdb_ca = pd.DataFrame(
        {
            "auth_asym_id": ["A", "A"],
            "label_asym_id": ["X", "X"],
            "auth_seq_id": [10, 12],
            "label_seq_id": [1, 3],
            "insertion_code": ["", ""],
            "x": [0.0, 2.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )

    result = build_mapped_confidence_residue_table(
        mapping,
        pdb_ca,
        np.array([61.0, 72.0, 83.0]),
        fragment_start=101,
        fragment_end=103,
    )

    assert result["uniprot_residue_number"].tolist() == [101, 102, 103]
    assert result["plddt"].tolist() == [61.0, 72.0, 83.0]
    assert result["observed_ca"].tolist() == [True, False, True]
    assert "pdb_residue_number" not in result


def test_script_rejects_duplicate_pass_candidate_keys() -> None:
    duplicated = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "PTEST",
            },
            {
                "screening_index": 101,
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "PTEST",
            },
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        SCORE_SCRIPT._validate_candidate_keys(duplicated)


def test_script_artifact_paths_are_bound_to_selected_version(tmp_path) -> None:
    prediction = {
        "cifUrl": (
            "https://example.test/files/"
            "AF-PTEST-F2-model_v6.cif"
        ),
        "plddtDocUrl": (
            "https://example.test/files/"
            "AF-PTEST-F2-confidence_v6.json"
        ),
    }

    model_path, plddt_path = SCORE_SCRIPT._selected_artifact_paths(
        tmp_path,
        uniprot_id="PTEST",
        model_entity_id=MODEL_ID,
        version=6,
        prediction=prediction,
    )

    assert model_path.name == "AF-PTEST-F2-model_v6.cif"
    assert plddt_path.name == "AF-PTEST-F2-confidence_v6.json"


def test_script_malformed_candidate_becomes_failure_row(tmp_path) -> None:
    row = SimpleNamespace(
        screening_index=101,
        pdb_id="1abc",
        chain_id="A",
        uniprot_id="PTEST",
        preflight_status="pass_full_length",
        selected_afdb_model_entity_id=MODEL_ID,
        afdb_version=np.nan,
        afdb_fragment_start=1,
        afdb_fragment_end=20,
    )

    result = SCORE_SCRIPT._score_candidate(
        row,
        root=tmp_path,
        metadata=pd.DataFrame(),
        terminal_buffer=5,
        minimum_run_length=5,
    )

    assert result["screening_index"] == 101
    assert result["scoring_status"] == "failed"
    assert result["is_low_conf_local"] is None


def test_script_malformed_metadata_record_becomes_failure_row(tmp_path) -> None:
    row = SimpleNamespace(
        screening_index=101,
        pdb_id="1abc",
        chain_id="A",
        uniprot_id="PTEST",
        preflight_status="pass_full_length",
        selected_afdb_model_entity_id=MODEL_ID,
        afdb_version=6,
        afdb_fragment_start=1,
        afdb_fragment_end=20,
    )
    metadata = pd.DataFrame(
        {
            "uniprot_id": ["PTEST"],
            "prediction_records_json": [json.dumps([None])],
        }
    )

    result = SCORE_SCRIPT._score_candidate(
        row,
        root=tmp_path,
        metadata=metadata,
        terminal_buffer=5,
        minimum_run_length=5,
    )

    assert result["scoring_status"] == "failed"
    assert "malformed_local_afdb_metadata" in result["scoring_reason"]


def test_script_cif_identity_must_match_selected_model(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        SCORE_SCRIPT,
        "MMCIF2Dict",
        lambda _: {"_entry.id": ["AF-WRONG-F1"]},
    )
    monkeypatch.setattr(
        SCORE_SCRIPT,
        "load_chain_ca_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "label_seq_id": [1, 2, 3],
                "bfactor": [90.0, 91.0, 92.0],
            }
        ),
    )

    with pytest.raises(ValueError, match="identity"):
        SCORE_SCRIPT._validate_model_bfactor(
            tmp_path / "model.cif",
            np.array([90.0, 91.0, 92.0]),
            model_entity_id=MODEL_ID,
            fragment_start=101,
            fragment_end=103,
        )


def test_script_bfactor_mismatch_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        SCORE_SCRIPT,
        "MMCIF2Dict",
        lambda _: {"_entry.id": [MODEL_ID]},
    )
    monkeypatch.setattr(
        SCORE_SCRIPT,
        "load_chain_ca_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "label_seq_id": [1, 2, 3],
                "bfactor": [90.0, 91.0, 50.0],
            }
        ),
    )

    with pytest.raises(ValueError, match="bfactor_mismatch"):
        SCORE_SCRIPT._validate_model_bfactor(
            tmp_path / "model.cif",
            np.array([90.0, 91.0, 92.0]),
            model_entity_id=MODEL_ID,
            fragment_start=101,
            fragment_end=103,
        )


def test_stats_are_finite_for_successful_summary() -> None:
    summary = summarize_mapped_confidence(_residue_table())

    assert summary.scoring_status == "success"
    assert summary.scoring_reason == "scored"
    assert math.isfinite(summary.mapped_plddt_min)
    assert math.isfinite(summary.mapped_plddt_q10)
    assert math.isfinite(summary.mapped_plddt_median)
