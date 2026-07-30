from __future__ import annotations

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from dual_uq.schema import AmbiguousLegacyResidueIdentifier
from dual_uq.screening_runner import SEGMENT_CONTEXT, validate_stage_outputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def segment_script() -> ModuleType:
    path = ROOT / "scripts" / "11_characterize_disagreement_segments.py"
    spec = spec_from_file_location("segment_context_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_schema() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uniprot_residue_number": [102, 100, 101],
            "auth_asym_id": ["A", "A", "A"],
            "auth_seq_id": [42, 42, 42],
            "insertion_code": ["B", "", "A"],
            "label_asym_id": ["X", "X", "X"],
            "label_seq_id": [12, 10, 11],
            "plddt": [95.0, 96.0, 97.0],
            "aligned_ca_distance": [1.2, 0.2, 1.4],
        }
    )


def test_t4_schema_without_legacy_column_prepares_by_uniprot(
    segment_script: ModuleType,
) -> None:
    prepared, provenance = segment_script.prepare_residue_geometry(
        _new_schema()
    )

    assert "pdb_residue_number" not in prepared
    assert prepared["uniprot_residue_number"].tolist() == [100, 101, 102]
    assert provenance["mode"] == "explicit_auth_label"


def test_auth_label_and_insertion_provenance_remain_distinct(
    segment_script: ModuleType,
) -> None:
    prepared, _ = segment_script.prepare_residue_geometry(_new_schema())

    assert list(
        prepared[
            [
                "auth_asym_id",
                "auth_seq_id",
                "insertion_code",
                "label_asym_id",
                "label_seq_id",
            ]
        ].itertuples(index=False, name=None)
    ) == [
        ("A", 42, "", "X", 10),
        ("A", 42, "A", "X", 11),
        ("A", 42, "B", "X", 12),
    ]
    assert segment_script.author_residue_keys(prepared) == {
        ("A", 42, ""),
        ("A", 42, "A"),
        ("A", 42, "B"),
    }


def test_author_gaps_and_insertion_codes_do_not_define_continuity(
    segment_script: ModuleType,
) -> None:
    table = _new_schema()
    table["uniprot_residue_number"] = [12, 10, 11]
    table["auth_seq_id"] = [900, -7, 42]
    table["insertion_code"] = ["B", "", "A"]

    prepared, _ = segment_script.prepare_residue_geometry(table)
    segment = segment_script.segment_rows_for_interval(
        prepared,
        start_position=10,
        end_position=12,
    )

    assert segment["uniprot_residue_number"].tolist() == [10, 11, 12]
    assert segment["auth_seq_id"].tolist() == [-7, 42, 900]
    assert segment["insertion_code"].tolist() == ["", "A", "B"]


def test_unambiguous_legacy_artifact_migrates_and_records_alias(
    segment_script: ModuleType,
) -> None:
    legacy = pd.DataFrame(
        {
            "uniprot_residue_number": [2, 1],
            "pdb_chain_id": ["A", "A"],
            "pdb_residue_number": ["42A", "42"],
            "plddt": [90.0, 95.0],
            "aligned_ca_distance": [1.2, 0.2],
        }
    )

    prepared, provenance = segment_script.prepare_residue_geometry(legacy)

    assert list(
        prepared[
            ["auth_asym_id", "auth_seq_id", "insertion_code"]
        ].itertuples(index=False, name=None)
    ) == [("A", 42, ""), ("A", 42, "A")]
    assert set(prepared["residue_mapping_provenance"]) == {"legacy_alias"}
    assert provenance["mode"] == "legacy_alias"


@pytest.mark.parametrize(
    "legacy",
    [
        pd.DataFrame(
            {
                "uniprot_residue_number": [1],
                "pdb_chain_id": ["A"],
                "pdb_residue_number": ["A42"],
                "plddt": [95.0],
                "aligned_ca_distance": [0.2],
            }
        ),
        pd.DataFrame(
            {
                "uniprot_residue_number": [1],
                "pdb_residue_number": ["42"],
                "plddt": [95.0],
                "aligned_ca_distance": [0.2],
            }
        ),
        pd.DataFrame(
            {
                "uniprot_residue_number": [1],
                "auth_asym_id": ["B"],
                "pdb_chain_id": ["A"],
                "pdb_residue_number": ["42"],
                "plddt": [95.0],
                "aligned_ca_distance": [0.2],
            }
        ),
        pd.DataFrame(
            {
                "uniprot_residue_number": [1, 2],
                "pdb_chain_id": ["A", "B"],
                "pdb_residue_number": ["42", "43"],
                "plddt": [95.0, 94.0],
                "aligned_ca_distance": [0.2, 0.3],
            }
        ),
    ],
)
def test_ambiguous_legacy_or_missing_chain_fails_structurally(
    segment_script: ModuleType,
    legacy: pd.DataFrame,
) -> None:
    with pytest.raises(
        AmbiguousLegacyResidueIdentifier,
        match="Legacy|chain",
    ):
        segment_script.prepare_residue_geometry(legacy)


@pytest.mark.parametrize(
    "pair_name",
    [
        "8pb5_A__P0DPA9",
        "8c3x_A__A0A7I9C8Z1",
        "3ip0_A__P26281",
    ],
)
def test_replacement_t4_fixtures_prepare_without_legacy_column(
    segment_script: ModuleType,
    pair_name: str,
) -> None:
    source = ROOT / "data/processed/pairs" / pair_name
    raw = pd.read_parquet(source / "residue_geometry.parquet")
    assert "pdb_residue_number" not in raw

    prepared, provenance = segment_script.prepare_residue_geometry(raw)

    assert len(prepared) == len(raw)
    assert prepared["uniprot_residue_number"].is_monotonic_increasing
    assert provenance["mode"] == "explicit_auth_label"


def test_1ake_numeric_and_validator_regression(
    segment_script: ModuleType,
    tmp_path: Path,
) -> None:
    source = ROOT / "data/processed/pairs/1ake_A__P69441"
    pair_dir = (
        tmp_path
        / "data"
        / "processed"
        / "pairs"
        / "1ake_A__P69441"
    )
    pair_dir.mkdir(parents=True)
    for filename in (
        "pair_qc.json",
        "residue_geometry.parquet",
        "disagreement_segments.csv",
        "pairwise_geometry.npz",
    ):
        shutil.copy2(source / filename, pair_dir / filename)
    baseline = pd.read_csv(source / "segment_context.csv")

    result = segment_script.run_segment_context(
        pair_dir,
        threshold=1.0,
        min_length=3,
        flank_size=10,
    )

    numeric_columns = [
        "local_fit_rmsd",
        "flank_fit_segment_rmsd",
        "internal_distance_mae",
        "segment_to_rest_pae_median",
        "segment_to_rest_pae_q90",
        "median_pdb_residue_sasa",
    ]
    pd.testing.assert_frame_equal(
        result[numeric_columns],
        baseline[numeric_columns],
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )
    assert {
        "start_position",
        "end_position",
        "residue_count",
        "segment_classification",
        "classification_explanation",
        "nearest_hetero_distance",
    }.issubset(result.columns)

    payload = json.loads(
        (pair_dir / "segment_context.json").read_text(encoding="utf-8")
    )
    assert payload
    assert payload[0]["residue_numbering_mode"] == "explicit_auth_label"

    validation = validate_stage_outputs(
        SEGMENT_CONTEXT,
        {"pair_name": "1ake_A__P69441"},
        tmp_path,
    )
    assert validation.valid is True
    assert validation.status == "complete"
