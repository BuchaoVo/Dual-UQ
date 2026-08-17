import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.evaluation import pair_validity


def _fixtures() -> pair_validity.PairValidityInputs:
    cohort = pd.DataFrame(
        [
            {
                "protein_id": "p1",
                "candidate_id": "p1",
                "pair_id": "1abc_A__P00001",
                "canonical_accession": "P00001",
                "pdb_id": "1abc",
                "pdb_chain": "A",
                "afdb_model_id": "AF-P00001-F1",
                "canonical_sequence_length": 2,
                "canonical_sequence": "A" * 2,
                "pdb_structure_ref": "data/raw/pdb/1abc.cif",
                "afdb_structure_ref": "data/raw/afdb/P00001/AF-P00001-F1/model.cif",
            },
            {
                "protein_id": "p2",
                "candidate_id": "p2",
                "pair_id": "2abc_A__P00002",
                "canonical_accession": "P00002",
                "pdb_id": "2abc",
                "pdb_chain": "A",
                "afdb_model_id": "AF-P00002-F1",
                "canonical_sequence_length": 50,
                "canonical_sequence": "A" * 50,
                "pdb_structure_ref": "data/raw/pdb/2abc.cif",
                "afdb_structure_ref": "data/raw/afdb/P00002/AF-P00002-F1/model.cif",
            },
        ]
    )
    masks = pd.DataFrame(
        [
            {"protein_id": "p1", "canonical_position": 1, "mapping_present": True, "pdb_mapped": True, "afdb_mapped": True, "pdb_backbone_complete": True, "afdb_backbone_complete": True, "common_mask": True, "observability_evaluation_status": "EVALUATED"},
            {"protein_id": "p1", "canonical_position": 2, "mapping_present": True, "pdb_mapped": True, "afdb_mapped": True, "pdb_backbone_complete": True, "afdb_backbone_complete": False, "common_mask": False, "observability_evaluation_status": "EVALUATED"},
            {"protein_id": "p2", "canonical_position": 1, "mapping_present": True, "pdb_mapped": True, "afdb_mapped": True, "pdb_backbone_complete": True, "afdb_backbone_complete": True, "common_mask": True, "observability_evaluation_status": "EVALUATED"},
        ]
    )
    admissions = pd.DataFrame(
        [
            {
                "candidate_id": "p1",
                "admission_status": "FORMALLY_ADMITTED",
                "mapping_valid": True,
                "paired_sequence_identity": 1.0,
                "mismatch_count": 0,
                "mapped_residue_count": 2,
                "coordinate_nonobservable_count": 0,
                "common_mask_count": 1,
                "common_mask_fraction_of_mapped": 0.5,
                "exact_variant_authorized": False,
            }
        ]
    )
    response = pd.DataFrame(
        [
            {"protein_id": "p1", "median_abs_d": 0.2, "mean_abs_d": 0.2},
            {"protein_id": "p2", "median_abs_d": 0.8, "mean_abs_d": 0.8},
        ]
    )
    return pair_validity.PairValidityInputs(
        project_root=Path("."),
        cohort=cohort,
        common_masks=masks,
        admissions=admissions,
        protein_response=response,
        pdb_metadata=pd.DataFrame(),
        afdb_confidence=pd.DataFrame(),
        input_provenance={},
    )


def test_pair_table_keeps_provenance_dimensions_without_weighted_score() -> None:
    table = pair_validity.build_pair_validity_table(_fixtures())

    assert len(table) == 2
    assert table.loc[table.protein_id == "p1", "identity_status"].item() == "exact"
    assert table.loc[table.protein_id == "p1", "mask_row_count"].item() == 2
    assert table.loc[table.protein_id == "p1", "common_mask_count_observed"].item() == 1
    assert table.loc[table.protein_id == "p2", "identity_status"].item() == "unresolved"
    assert "pair_validity_score" not in table


def test_high_comparability_requires_explicit_evidence_and_does_not_infer_missing() -> None:
    table = pair_validity.build_pair_validity_table(_fixtures())
    selected = pair_validity.select_high_comparability(table)

    assert selected.protein_id.tolist() == ["p1"]
    assert table.loc[table.protein_id == "p2", "comparability_group"].item() == "unresolved"
    assert "missing_explicit_admission_evidence" in table.loc[
        table.protein_id == "p2", "high_comparability_exclusion_reason"
    ].item()


def test_response_comparison_aggregates_at_protein_unit() -> None:
    table = pair_validity.build_pair_validity_table(_fixtures())
    comparison = pair_validity.build_response_comparison(table)
    summary = pair_validity.summarize_pair_validity(table, comparison)

    assert len(comparison) == 2
    assert summary["protein_count"] == 2
    assert summary["high_comparability_count"] == 1
    assert summary["response_full_median"] == pytest.approx(0.5)
    assert summary["response_clean_median"] == pytest.approx(0.2)


def test_conflicting_admission_sources_remain_unresolved() -> None:
    first = pd.DataFrame(
        [{"candidate_id": "p", "admission_status": "FORMALLY_ADMITTED"}]
    )
    second = pd.DataFrame(
        [{"candidate_id": "p", "admission_status": "MAPPING_FAIL"}]
    )

    merged = pair_validity._merge_admission_sources([first, second], {"p"})

    assert bool(merged.loc[0, "admission_evidence_conflict"])
    assert merged.loc[0, "admission_conflict_fields"] == "admission_status"
    assert pd.isna(merged.loc[0, "admission_status"])


def test_pair_validity_cli_is_thin_and_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analysis/analyze_pair_validity.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
    assert "--output-dir" in completed.stdout
