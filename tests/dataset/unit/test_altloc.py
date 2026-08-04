from __future__ import annotations

import math
from itertools import permutations

import pytest

from dual_uq.dataset.models.structure import (
    AtomRecord,
    ResidueKey,
    ResidueProvenance,
    altloc_rank,
    normalize_altloc,
    select_backbone_atoms,
    select_preferred_atom,
)

RESIDUE = ResidueProvenance(
    key=ResidueKey("pdb:index36-fixture", "A", 36, ""),
    label_chain_id="X",
    label_seq_id=31,
    raw_resname="ALA",
    record_type="ATOM",
)


def _atom(
    atom_name: str = "CA",
    *,
    altloc: str | None = "",
    occupancy: float | None = 1.0,
    x: float = 1.0,
    residue: ResidueProvenance = RESIDUE,
) -> AtomRecord:
    return AtomRecord(
        residue=residue,
        atom_name=atom_name,
        element=atom_name[0],
        altloc=altloc,
        occupancy=occupancy,
        x=x,
        y=2.0,
        z=3.0,
    )


def test_altloc_blank_and_a_follow_verified_priority() -> None:
    blank = _atom(altloc="", occupancy=0.5, x=1.0)
    alt_a = _atom(altloc="A", occupancy=1.0, x=2.0)
    assert select_preferred_atom((blank, alt_a)) == alt_a

    tied_blank = _atom(altloc="", occupancy=1.0, x=3.0)
    assert select_preferred_atom((tied_blank, alt_a)) == tied_blank


def test_altloc_a_and_b_follow_verified_priority() -> None:
    alt_a = _atom(altloc="A", occupancy=0.8, x=1.0)
    alt_b = _atom(altloc="B", occupancy=0.8, x=2.0)
    assert select_preferred_atom((alt_b, alt_a)) == alt_a


def test_other_altlocs_use_verified_lexical_tie_break() -> None:
    alt_b = _atom(altloc="B", occupancy=0.8, x=1.0)
    alt_c = _atom(altloc="C", occupancy=0.8, x=2.0)
    assert altloc_rank("B") == altloc_rank("C")
    assert select_preferred_atom((alt_b, alt_c)) == alt_c


def test_same_altloc_prefers_verified_occupancy_rule() -> None:
    lower = _atom(altloc="A", occupancy=0.25, x=1.0)
    higher = _atom(altloc="A", occupancy=0.75, x=2.0)
    assert select_preferred_atom((higher, lower)) == higher


def test_altloc_selection_is_independent_of_input_order() -> None:
    records = (
        _atom(altloc="B", occupancy=0.7, x=1.0),
        _atom(altloc="", occupancy=0.7, x=2.0),
        _atom(altloc="A", occupancy=0.9, x=3.0),
    )
    assert select_preferred_atom(records) == select_preferred_atom(tuple(reversed(records)))


def test_altloc_selection_is_stable_across_permutations() -> None:
    records = (
        _atom(altloc="B", occupancy=0.8, x=1.0),
        _atom(altloc="A", occupancy=0.8, x=2.0),
        _atom(altloc="", occupancy=0.7, x=3.0),
        _atom(altloc="C", occupancy=0.6, x=4.0),
    )
    winners = {select_preferred_atom(order) for order in permutations(records)}
    assert winners == {records[1]}


@pytest.mark.parametrize("token", [None, "", " ", ".", "?"])
def test_missing_altloc_tokens_normalize_consistently(token: str | None) -> None:
    assert normalize_altloc(token) == ""
    assert _atom(altloc=token).altloc == ""


def test_missing_occupancy_follows_explicit_rule() -> None:
    missing = _atom(altloc="", occupancy=None, x=1.0)
    present = _atom(altloc="B", occupancy=0.0, x=2.0)
    assert select_preferred_atom((missing, present)) == present

    missing_a = _atom(altloc="A", occupancy=None, x=3.0)
    assert select_preferred_atom((missing, missing_a)) == missing


def test_exact_duplicate_atom_can_be_deduplicated() -> None:
    atom = _atom(altloc="A", occupancy=0.5)
    duplicate = _atom(altloc="A", occupancy=0.5)
    assert select_preferred_atom((atom, duplicate)) == atom


def test_ambiguous_duplicate_atom_is_rejected() -> None:
    left = _atom(altloc="A", occupancy=0.5, x=1.0)
    right = _atom(altloc="A", occupancy=0.5, x=9.0)
    with pytest.raises(ValueError, match="ambiguous duplicate atom"):
        select_preferred_atom((left, right))


def test_selected_atom_preserves_altloc_and_occupancy_provenance() -> None:
    selected = select_preferred_atom(
        (_atom(altloc="B", occupancy=0.4), _atom(altloc="A", occupancy=0.9))
    )
    assert selected.residue == RESIDUE
    assert selected.altloc == "A"
    assert selected.occupancy == 0.9


def test_backbone_extractor_selects_at_most_one_n_ca_c_o() -> None:
    records = (
        _atom("N"),
        _atom("CA", altloc="B", occupancy=0.4, x=1.0),
        _atom("CA", altloc="A", occupancy=0.9, x=2.0),
        _atom("C"),
        _atom("O"),
    )
    selection = select_backbone_atoms(records)
    assert tuple(atom.atom_name for atom in selection.selected_atoms) == (
        "N",
        "CA",
        "C",
        "O",
    )
    assert selection.atom("CA").altloc == "A"


def test_backbone_extractor_reports_missing_atoms() -> None:
    selection = select_backbone_atoms((_atom("N"), _atom("CA")))
    assert selection.missing_atoms == ("C", "O")


def test_backbone_selection_does_not_depend_on_record_order() -> None:
    records = (
        _atom("O"),
        _atom("CA", altloc="B", occupancy=0.2),
        _atom("N"),
        _atom("C"),
        _atom("CA", altloc="A", occupancy=0.8),
    )
    assert select_backbone_atoms(records) == select_backbone_atoms(tuple(reversed(records)))


def test_non_backbone_atoms_do_not_replace_backbone_atoms() -> None:
    selection = select_backbone_atoms((_atom("CB"), _atom("CA", x=7.0)))
    assert tuple(atom.atom_name for atom in selection.selected_atoms) == ("CA",)
    assert selection.atom("CA").x == 7.0


@pytest.mark.parametrize("occupancy", [True, -0.1, 1.1, math.nan, math.inf])
def test_invalid_occupancy_is_rejected(occupancy: object) -> None:
    with pytest.raises((TypeError, ValueError), match="occupancy"):
        _atom(occupancy=occupancy)  # type: ignore[arg-type]


def test_multi_character_altloc_is_rejected() -> None:
    with pytest.raises(ValueError, match="altloc"):
        _atom(altloc="AB")


def test_preferred_atom_candidates_must_share_atom_and_residue_identity() -> None:
    with pytest.raises(ValueError, match="same residue and atom"):
        select_preferred_atom((_atom("CA"), _atom("N")))
