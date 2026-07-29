from __future__ import annotations

import pandas as pd
import pytest

from dual_uq.indexing_diagnostics import diagnose_numbering_intersections


def _mapping(pdb_start: int, uniprot_start: int, count: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb_chain_id": ["A"] * count,
            "pdb_residue_number": range(pdb_start, pdb_start + count),
            "uniprot_residue_number": range(uniprot_start, uniprot_start + count),
        }
    )


def _atoms(
    auth_start: int,
    label_start: int,
    count: int = 10,
    *,
    auth_chain: str = "A",
    label_chain: str = "A",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_pdb": ["ATOM"] * count,
            "atom_name": ["CA"] * count,
            "auth_asym_id": [auth_chain] * count,
            "label_asym_id": [label_chain] * count,
            "auth_seq_id": range(auth_start, auth_start + count),
            "label_seq_id": range(label_start, label_start + count),
            "insertion_code": [""] * count,
        }
    )


def _record(model_id: str, start: int, end: int) -> dict[str, object]:
    return {
        "modelEntityId": model_id,
        "entryId": model_id,
        "latestVersion": 1,
        "sequenceStart": start,
        "sequenceEnd": end,
        "cifUrl": f"https://example.test/{model_id}.cif",
        "isComplex": False,
    }


@pytest.mark.parametrize(
    ("mapping", "pdb_atoms", "afdb_atoms", "records", "selected", "expected"),
    [
        (
            _mapping(1, 10),
            _atoms(101, 1),
            _atoms(10, 10),
            [_record("AF-label", 10, 19)],
            "AF-label",
            "auth_label_numbering_mismatch",
        ),
        (
            _mapping(1, 100),
            _atoms(1, 1),
            _atoms(1, 1),
            [_record("AF-wrong", 200, 209), _record("AF-cover", 100, 109)],
            "AF-wrong",
            "afdb_fragment_selection_mismatch",
        ),
        (
            _mapping(1, 100),
            _atoms(1, 1),
            _atoms(1, 1),
            [_record("AF-offset", 100, 109)],
            "AF-offset",
            "afdb_residue_offset_mismatch",
        ),
        (
            _mapping(1, 100),
            _atoms(1, 1),
            _atoms(1, 1),
            [_record("AF-left", 1, 50), _record("AF-right", 151, 200)],
            "AF-left",
            "unsupported_afdb_coverage",
        ),
    ],
)
def test_diagnoses_distinct_numbering_and_fragment_failures(
    mapping: pd.DataFrame,
    pdb_atoms: pd.DataFrame,
    afdb_atoms: pd.DataFrame,
    records: list[dict[str, object]],
    selected: str,
    expected: str,
) -> None:
    result = diagnose_numbering_intersections(
        mapping=mapping,
        pdb_atom_site=pdb_atoms,
        afdb_atom_site=afdb_atoms,
        prediction_records=records,
        selected_model_entity_id=selected,
        chain_id="A",
    )

    assert result["categorical_root_cause"] == expected
    assert result["supporting_evidence"]
    assert result["recommended_next_action"]
    assert result["numbering_sets"]["sifts_uniprot"]["count"] == 10
    assert "auth_label_intersections" in result
    assert "uniprot_afdb_intersections" in result
    assert len(result["afdb_fragment_intervals"]) == len(records)


def test_preserves_insertion_code_in_auth_residue_keys() -> None:
    mapping = pd.DataFrame(
        {
            "pdb_chain_id": ["A"],
            "pdb_residue_number": ["10A"],
            "uniprot_residue_number": [100],
        }
    )
    pdb_atoms = _atoms(10, 1, count=1)
    pdb_atoms.loc[0, "insertion_code"] = "A"

    result = diagnose_numbering_intersections(
        mapping=mapping,
        pdb_atom_site=pdb_atoms,
        afdb_atom_site=_atoms(1, 1, count=1),
        prediction_records=[_record("AF-offset", 100, 100)],
        selected_model_entity_id="AF-offset",
        chain_id="A",
    )

    assert result["numbering_sets"]["pdb_auth_residue_keys"]["preview"] == ["10A"]
    assert result["auth_label_intersections"]["sifts_pdb_vs_auth"] == 1
