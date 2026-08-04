from __future__ import annotations

import math

import pytest

from dual_uq.dataset.models.structure import (
    AtomRecord,
    ResidueKey,
    ResidueProvenance,
    canonical_amino_acid,
    group_residue_records,
    normalize_insertion_code,
    select_backbone_atoms,
)


def _residue(
    *,
    auth_chain: str = "A",
    auth_seq: int = 42,
    insertion: str | None = "",
    label_chain: str | None = "X",
    label_seq: int | None = 7,
    raw_resname: str = "ALA",
    record_type: str = "ATOM",
) -> ResidueProvenance:
    return ResidueProvenance(
        key=ResidueKey("pdb:fixture", auth_chain, auth_seq, insertion),
        label_chain_id=label_chain,
        label_seq_id=label_seq,
        raw_resname=raw_resname,
        record_type=record_type,
    )


def _atom(
    atom_name: str,
    *,
    residue: ResidueProvenance | None = None,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    altloc: str | None = "",
    occupancy: float | None = 1.0,
) -> AtomRecord:
    return AtomRecord(
        residue=residue or _residue(),
        atom_name=atom_name,
        element=atom_name[0],
        altloc=altloc,
        occupancy=occupancy,
        x=x,
        y=y,
        z=z,
    )


def test_residue_identity_distinguishes_insertion_codes() -> None:
    keys = {
        ResidueKey("pdb:fixture", "A", 42, ""),
        ResidueKey("pdb:fixture", "A", 42, "A"),
        ResidueKey("pdb:fixture", "A", 42, "B"),
    }
    assert len(keys) == 3


def test_auth_and_label_numbering_are_preserved_independently() -> None:
    residue = _residue(auth_seq=101, label_seq=1)
    assert residue.key.auth_seq_id == 101
    assert residue.label_seq_id == 1


def test_missing_label_numbering_is_not_filled_from_auth() -> None:
    residue = _residue(label_chain=None, label_seq=None)
    assert residue.label_chain_id is None
    assert residue.label_seq_id is None


def test_auth_and_label_chain_ids_are_preserved_independently() -> None:
    residue = _residue(auth_chain="AUTH", label_chain="LABEL")
    assert residue.key.auth_chain_id == "AUTH"
    assert residue.label_chain_id == "LABEL"


@pytest.mark.parametrize("token", [None, "", " ", ".", "?"])
def test_missing_insertion_code_is_canonicalized(token: str | None) -> None:
    assert normalize_insertion_code(token) == ""
    assert ResidueKey("pdb:fixture", "A", 42, token).insertion_code == ""


def test_insertion_code_preserves_case_and_rejects_multiple_characters() -> None:
    assert normalize_insertion_code("a") == "a"
    with pytest.raises(ValueError, match="insertion_code"):
        normalize_insertion_code("AB")


def test_modified_residue_preserves_raw_resname() -> None:
    residue = _residue(raw_resname="MSE", record_type="HETATM")
    assert residue.raw_resname == "MSE"
    assert residue.record_type == "HETATM"


def test_mse_canonicalization_preserves_source_provenance() -> None:
    residue = _residue(raw_resname="MSE", record_type="HETATM")
    assert canonical_amino_acid("MSE") == "M"
    assert residue.canonical_aa == "M"
    assert residue.raw_resname == "MSE"
    assert residue.record_type == "HETATM"


def test_unsupported_residue_is_not_silently_mapped_to_x() -> None:
    residue = _residue(raw_resname="ATP", record_type="HETATM")
    assert canonical_amino_acid("ATP") is None
    assert residue.canonical_aa is None
    assert residue.raw_resname == "ATP"


def test_atom_record_preserves_atom_and_residue_identity() -> None:
    residue = _residue(auth_seq=-1, insertion="B")
    atom = _atom("CA", residue=residue, altloc="A", occupancy=0.75)
    assert atom.residue == residue
    assert atom.atom_name == "CA"
    assert atom.altloc == "A"
    assert atom.occupancy == 0.75
    assert atom.coordinates == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("field", ["x", "y", "z"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_are_rejected(field: str, value: float) -> None:
    arguments = {"x": 1.0, "y": 2.0, "z": 3.0, field: value}
    with pytest.raises(ValueError, match="finite"):
        _atom("CA", **arguments)


def test_bool_is_not_accepted_as_sequence_number() -> None:
    with pytest.raises(TypeError, match="auth_seq_id"):
        ResidueKey("pdb:fixture", "A", True, "")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="label_seq_id"):
        _residue(label_seq=False)  # type: ignore[arg-type]


def test_blank_author_chain_is_distinct_from_missing_author_chain() -> None:
    assert ResidueKey("pdb:fixture", "", 1, "").auth_chain_id == ""
    with pytest.raises(TypeError, match="auth_chain_id"):
        ResidueKey("pdb:fixture", None, 1, "")  # type: ignore[arg-type]


def test_duplicate_residue_identity_with_conflicting_resname_is_rejected() -> None:
    ala = _atom("CA", residue=_residue(raw_resname="ALA"))
    gly = _atom("N", residue=_residue(raw_resname="GLY"))
    with pytest.raises(ValueError, match="conflicting residue provenance"):
        group_residue_records((ala, gly))


def test_incomplete_backbone_is_explicit_not_silently_dropped() -> None:
    selection = select_backbone_atoms((_atom("N"), _atom("CA"), _atom("C")))
    assert tuple(atom.atom_name for atom in selection.selected_atoms) == (
        "N",
        "CA",
        "C",
    )
    assert selection.missing_atoms == ("O",)


def test_residue_groups_are_stably_ordered_and_preserve_42_variants() -> None:
    residues = [
        _residue(insertion="B", label_seq=9),
        _residue(insertion="", label_seq=7),
        _residue(insertion="A", label_seq=8, raw_resname="MSE", record_type="HETATM"),
    ]
    groups = group_residue_records(tuple(_atom("CA", residue=item) for item in residues))
    assert [group.residue.key.insertion_code for group in groups] == ["", "A", "B"]
    assert groups[1].residue.raw_resname == "MSE"
    assert groups[1].residue.record_type == "HETATM"
