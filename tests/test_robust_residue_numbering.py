from __future__ import annotations

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from dual_uq.schema import AmbiguousLegacyResidueIdentifier
from dual_uq.screening_runner import ROBUST, validate_stage_outputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def robust_script() -> ModuleType:
    path = ROOT / "scripts" / "analysis" / "diagnose_pair_robustness.py"
    spec = spec_from_file_location("robust_pair_diagnostics_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_schema() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uniprot_residue_number": [102, 100, 101],
            "auth_asym_id": ["A", "A", "A"],
            "auth_seq_id": [90, 42, 42],
            "insertion_code": ["", "", "A"],
            "label_asym_id": ["B", "B", "B"],
            "label_seq_id": [12, 10, 11],
            "plddt": [95.0, 96.0, 97.0],
            "ca_disagreement": [1.2, 0.2, 1.4],
        }
    )


def test_new_schema_without_legacy_column_prepares_successfully(
    robust_script: ModuleType,
) -> None:
    prepared, provenance = robust_script.prepare_residue_geometry(
        _new_schema()
    )

    assert "pdb_residue_number" not in prepared
    assert prepared["uniprot_residue_number"].tolist() == [100, 101, 102]
    assert prepared["aligned_ca_distance"].tolist() == [0.2, 1.4, 1.2]
    assert provenance["mode"] == "explicit_auth_label"


def test_auth_and_label_numbering_remain_distinct(
    robust_script: ModuleType,
) -> None:
    prepared, _ = robust_script.prepare_residue_geometry(_new_schema())

    assert prepared["auth_seq_id"].tolist() == [42, 42, 90]
    assert prepared["label_seq_id"].tolist() == [10, 11, 12]
    assert not prepared["auth_seq_id"].equals(prepared["label_seq_id"])


def test_insertion_codes_preserve_three_distinct_author_residues(
    robust_script: ModuleType,
) -> None:
    table = _new_schema()
    table["uniprot_residue_number"] = [102, 100, 101]
    table["auth_seq_id"] = [42, 42, 42]
    table["insertion_code"] = ["B", "", "A"]

    prepared, _ = robust_script.prepare_residue_geometry(table)

    assert list(
        prepared[
            ["auth_asym_id", "auth_seq_id", "insertion_code"]
        ].itertuples(index=False, name=None)
    ) == [("A", 42, ""), ("A", 42, "A"), ("A", 42, "B")]


def test_noncontiguous_author_ids_sort_by_uniprot_position(
    robust_script: ModuleType,
) -> None:
    table = _new_schema()
    table["auth_seq_id"] = [500, -2, 77]

    prepared, _ = robust_script.prepare_residue_geometry(table)

    assert prepared["uniprot_residue_number"].tolist() == [100, 101, 102]
    assert prepared["auth_seq_id"].tolist() == [-2, 77, 500]


def test_unambiguous_legacy_numbering_is_migrated_with_provenance(
    robust_script: ModuleType,
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

    prepared, provenance = robust_script.prepare_residue_geometry(legacy)

    assert prepared["auth_asym_id"].tolist() == ["A", "A"]
    assert prepared["auth_seq_id"].tolist() == [42, 42]
    assert prepared["insertion_code"].tolist() == ["", "A"]
    assert prepared["label_seq_id"].isna().all()
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
    ],
)
def test_ambiguous_legacy_numbering_fails_structurally(
    robust_script: ModuleType,
    legacy: pd.DataFrame,
) -> None:
    with pytest.raises(
        AmbiguousLegacyResidueIdentifier,
        match="Legacy|chain",
    ):
        robust_script.prepare_residue_geometry(legacy)


def test_duplicate_uniprot_or_author_keys_fail_structurally(
    robust_script: ModuleType,
) -> None:
    duplicate_uniprot = _new_schema()
    duplicate_uniprot.loc[1, "uniprot_residue_number"] = 102
    with pytest.raises(ValueError, match="duplicate UniProt"):
        robust_script.prepare_residue_geometry(duplicate_uniprot)

    duplicate_author = _new_schema()
    duplicate_author.loc[1, "auth_seq_id"] = 90
    duplicate_author.loc[1, "insertion_code"] = ""
    with pytest.raises(ValueError, match="duplicate author"):
        robust_script.prepare_residue_geometry(duplicate_author)


def test_1ake_numeric_and_output_schema_regression(
    robust_script: ModuleType,
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
    for filename in ("residue_geometry.parquet", "pairwise_geometry.npz"):
        shutil.copy2(source / filename, pair_dir / filename)
    baseline = json.loads(
        (source / "robust_pair_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )

    result = robust_script.run_robust_diagnostics(
        pair_dir,
        local_permutations=1000,
        pair_permutations=500,
        min_separation=6,
        seed=20260729,
    )

    for section in (
        "local_plddt_disagreement_test",
        "pae_pairwise_error_test",
    ):
        for field in ("rho", "permutation_pvalue"):
            assert result[section][field] == pytest.approx(
                baseline[section][field],
                abs=1e-6,
            )
    assert (
        result["largest_segment_residue_count_at_1A"]
        == baseline["largest_segment_residue_count_at_1A"]
    )
    assert result["residue_numbering_provenance"]["mode"] == (
        "explicit_auth_label"
    )

    high_confidence = pd.read_csv(
        pair_dir / "high_confidence_disagreement.csv"
    )
    assert {
        "uniprot_residue_number",
        "auth_asym_id",
        "auth_seq_id",
        "insertion_code",
        "label_asym_id",
        "label_seq_id",
        "residue_mapping_provenance",
        "plddt",
        "aligned_ca_distance",
    }.issubset(high_confidence.columns)
    assert "pdb_residue_number" not in high_confidence

    candidate = {"pair_name": "1ake_A__P69441"}
    validation = validate_stage_outputs(
        ROBUST,
        candidate,
        tmp_path,
    )
    assert validation.valid is True
    assert validation.status == "complete"
