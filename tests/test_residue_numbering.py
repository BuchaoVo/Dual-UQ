from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.schema import (
    AmbiguousLegacyResidueIdentifier,
    normalize_residue_mapping,
)
from dual_uq.sifts import parse_sifts_residue_mapping
from dual_uq.structure_io import (
    ResidueJoinError,
    join_residue_mapping_to_ca,
    load_chain_ca_table,
)


def _mapping(**overrides: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "auth_asym_id": ["A"],
        "label_asym_id": ["X"],
        "auth_seq_id": [42],
        "label_seq_id": [7],
        "insertion_code": [""],
        "uniprot_residue_number": [101],
    }
    values.update(overrides)
    size = max(len(value) for value in values.values() if isinstance(value, list))
    values = {
        key: value * size if isinstance(value, list) and len(value) == 1 else value
        for key, value in values.items()
    }
    return pd.DataFrame(values)


def _ca(**overrides: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "auth_asym_id": ["A"],
        "label_asym_id": ["X"],
        "auth_seq_id": [42],
        "label_seq_id": [7],
        "insertion_code": [""],
        "x": [1.0],
        "y": [2.0],
        "z": [3.0],
    }
    values.update(overrides)
    size = max(len(value) for value in values.values() if isinstance(value, list))
    values = {
        key: value * size if isinstance(value, list) and len(value) == 1 else value
        for key, value in values.items()
    }
    return pd.DataFrame(values)


def _write_atom_site(path: Path, rows: list[tuple[object, ...]]) -> None:
    columns = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    body = ["data_test", "#", "loop_", *columns]
    body.extend(" ".join(map(str, row)) for row in rows)
    body.append("#")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _atom(
    atom_id: int,
    *,
    group: str = "ATOM",
    atom: str = "CA",
    residue: str = "ALA",
    auth_chain: str = "A",
    label_chain: str = "X",
    auth_seq: int = 42,
    label_seq: int = 7,
    insertion: str = "?",
) -> tuple[object, ...]:
    return (
        group,
        atom_id,
        "C",
        atom,
        ".",
        residue,
        label_chain,
        1,
        label_seq,
        insertion,
        float(atom_id),
        2.0,
        3.0,
        1.0,
        80.0,
        auth_seq,
        residue,
        auth_chain,
        atom,
        1,
    )


def test_auth_chain_is_used_when_auth_and_label_chains_differ() -> None:
    joined, diagnostics = join_residue_mapping_to_ca(_mapping(), _ca())

    assert len(joined) == 1
    assert joined.loc[0, "residue_join_mode"] == "auth"
    assert diagnostics["auth_match_count"] == 1
    assert diagnostics["label_fallback_match_count"] == 0


def test_auth_sequence_is_used_when_auth_and_label_sequences_differ() -> None:
    joined, diagnostics = join_residue_mapping_to_ca(
        _mapping(auth_seq_id=[-3], label_seq_id=[19]),
        _ca(auth_seq_id=[-3], label_seq_id=[19]),
    )

    assert joined.loc[0, "auth_seq_id"] == -3
    assert diagnostics["auth_match_count"] == 1


def test_insertion_codes_distinguish_42a_and_42b() -> None:
    mapping = _mapping(
        auth_seq_id=[42, 42, 42],
        label_seq_id=[6, 7, 8],
        insertion_code=["", "A", "B"],
        uniprot_residue_number=[100, 101, 102],
    )
    ca = _ca(
        auth_seq_id=[42, 42, 42],
        label_seq_id=[6, 7, 8],
        insertion_code=["", "A", "B"],
        x=[0.0, 1.0, 2.0],
        y=[0.0, 1.0, 2.0],
        z=[0.0, 1.0, 2.0],
    )

    joined, diagnostics = join_residue_mapping_to_ca(mapping, ca)

    assert list(joined["uniprot_residue_number"]) == [100, 101, 102]
    assert diagnostics["auth_match_count"] == 3


@pytest.mark.parametrize("auth_ids", [[5, 9, 20], [-2, 0, 3]])
def test_noncontinuous_zero_and_negative_author_numbering(auth_ids: list[int]) -> None:
    mapping = _mapping(
        auth_seq_id=auth_ids,
        label_seq_id=[1, 2, 3],
        insertion_code=["", "", ""],
        uniprot_residue_number=[11, 12, 13],
    )
    ca = _ca(
        auth_seq_id=auth_ids,
        label_seq_id=[1, 2, 3],
        insertion_code=["", "", ""],
        x=[1.0, 2.0, 3.0],
        y=[1.0, 2.0, 3.0],
        z=[1.0, 2.0, 3.0],
    )

    joined, diagnostics = join_residue_mapping_to_ca(mapping, ca)

    assert list(joined["auth_seq_id"]) == auth_ids
    assert diagnostics["auth_match_count"] == 3


def test_multiple_atom_rows_collapse_to_one_residue(tmp_path: Path) -> None:
    cif = tmp_path / "atoms.cif"
    _write_atom_site(cif, [_atom(1, atom="N"), _atom(2), _atom(3, atom="C")])

    table = load_chain_ca_table(cif, "A")

    assert len(table) == 1
    assert table.loc[0, "auth_seq_id"] == 42


def test_auth_success_never_uses_label_fallback() -> None:
    mapping = _mapping(label_asym_id=["WRONG"], label_seq_id=[999])
    joined, diagnostics = join_residue_mapping_to_ca(mapping, _ca())

    assert joined.loc[0, "residue_join_mode"] == "auth"
    assert diagnostics["label_fallback_match_count"] == 0


def test_unique_verified_label_fallback_when_auth_is_unavailable() -> None:
    mapping = _mapping(
        auth_asym_id=[pd.NA],
        auth_seq_id=[pd.NA],
        insertion_code=[pd.NA],
    )
    joined, diagnostics = join_residue_mapping_to_ca(
        mapping,
        _ca(insertion_code=["B"]),
    )

    assert joined.loc[0, "residue_join_mode"] == "label_fallback"
    assert joined.loc[0, "insertion_code"] == "B"
    assert diagnostics["label_fallback_match_count"] == 1


def test_present_but_wrong_auth_id_is_not_hidden_by_label_fallback() -> None:
    mapping = _mapping(auth_asym_id=["WRONG"], auth_seq_id=[999])

    joined, diagnostics = join_residue_mapping_to_ca(mapping, _ca())

    assert joined.empty
    assert diagnostics["label_fallback_match_count"] == 0
    assert diagnostics["unmatched_mapping_count"] == 1
    assert diagnostics["residue_join_mode"] == "none"


def test_ambiguous_auth_and_label_keys_fail_structurally() -> None:
    duplicate_ca = pd.concat([_ca(), _ca(x=[9.0])], ignore_index=True)

    with pytest.raises(ResidueJoinError, match="duplicate"):
        join_residue_mapping_to_ca(_mapping(), duplicate_ca)


def test_duplicate_mapping_author_key_fails_before_join() -> None:
    mapping = _mapping(
        auth_seq_id=[42, 42],
        label_seq_id=[7, 8],
        uniprot_residue_number=[101, 102],
    )

    with pytest.raises(ResidueJoinError, match="mapping.*duplicate"):
        join_residue_mapping_to_ca(mapping, _ca())


def test_duplicate_unobserved_author_key_is_retained_without_coordinates() -> None:
    mapping = _mapping(
        auth_seq_id=[66, 66, 66],
        label_seq_id=[65, 66, 67],
        uniprot_residue_number=[65, 66, 67],
    )

    joined, diagnostics = join_residue_mapping_to_ca(
        mapping,
        _ca(auth_seq_id=[68], label_seq_id=[68]),
    )

    assert joined.empty
    assert diagnostics["auth_match_count"] == 0
    assert diagnostics["unmatched_mapping_count"] == 3
    assert diagnostics["residue_join_mode"] == "none"


def test_unused_duplicate_label_keys_do_not_block_unique_auth_join() -> None:
    ca = _ca(
        auth_seq_id=[41, 42],
        label_seq_id=[7, 7],
        x=[1.0, 2.0],
        y=[1.0, 2.0],
        z=[1.0, 2.0],
    )

    joined, diagnostics = join_residue_mapping_to_ca(_mapping(), ca)

    assert len(joined) == 1
    assert diagnostics["auth_match_count"] == 1


def test_legacy_mapping_aliases_are_migrated_at_io_boundary() -> None:
    legacy = pd.DataFrame(
        {
            "pdb_chain_id": ["A", "A"],
            "pdb_residue_number": ["42A", "-1"],
            "uniprot_residue_number": [101, 102],
        }
    )

    normalized = normalize_residue_mapping(legacy)

    assert list(normalized["auth_asym_id"]) == ["A", "A"]
    assert list(normalized["auth_seq_id"]) == [42, -1]
    assert list(normalized["insertion_code"]) == ["A", ""]
    assert normalized["label_seq_id"].isna().all()
    assert set(normalized["residue_mapping_provenance"]) == {"legacy_alias"}


def test_ambiguous_legacy_residue_number_fails_structurally() -> None:
    legacy = pd.DataFrame(
        {
            "pdb_chain_id": ["A"],
            "pdb_residue_number": ["A42"],
            "uniprot_residue_number": [101],
        }
    )

    with pytest.raises(AmbiguousLegacyResidueIdentifier):
        normalize_residue_mapping(legacy)


def test_conflicting_legacy_chain_aliases_fail_structurally() -> None:
    legacy = pd.DataFrame(
        {
            "pdb_chain_id": ["A"],
            "chain_id": ["B"],
            "pdb_residue_number": ["42"],
            "uniprot_residue_number": [101],
        }
    )

    with pytest.raises(AmbiguousLegacyResidueIdentifier, match="chain aliases"):
        normalize_residue_mapping(legacy)


def test_complementary_legacy_chain_aliases_are_coalesced() -> None:
    legacy = pd.DataFrame(
        {
            "pdb_chain_id": [pd.NA, "B"],
            "chain_id": ["A", pd.NA],
            "pdb_residue_number": ["1", "2"],
        }
    )

    normalized = normalize_residue_mapping(legacy)

    assert list(normalized["auth_asym_id"]) == ["A", "B"]


@pytest.mark.parametrize("version", ["bogus", "", 2.5])
def test_malformed_declared_schema_version_fails(version: object) -> None:
    mapping = _mapping(residue_mapping_schema_version=[version])

    with pytest.raises(AmbiguousLegacyResidueIdentifier, match="schema version"):
        normalize_residue_mapping(mapping)


def test_new_residue_table_preserves_all_five_numbering_fields(
    tmp_path: Path,
) -> None:
    cif = tmp_path / "numbering.cif"
    _write_atom_site(
        cif,
        [
            _atom(
                1,
                auth_chain="AUTH",
                label_chain="LAB",
                auth_seq=0,
                label_seq=17,
                insertion="B",
            )
        ],
    )

    table = load_chain_ca_table(cif, "AUTH")

    assert {
        "auth_asym_id",
        "label_asym_id",
        "auth_seq_id",
        "label_seq_id",
        "insertion_code",
    }.issubset(table.columns)
    assert table.loc[0, "insertion_code"] == "B"


def test_modified_amino_acid_hetatm_ca_is_retained(tmp_path: Path) -> None:
    cif = tmp_path / "mse.cif"
    _write_atom_site(
        cif,
        [_atom(1, group="HETATM", residue="MSE")],
    )

    table = load_chain_ca_table(cif, "A")

    assert len(table) == 1
    assert table.loc[0, "residue_name"] == "MSE"


def test_inconsistent_atom_row_namespaces_fail(tmp_path: Path) -> None:
    cif = tmp_path / "inconsistent.cif"
    _write_atom_site(
        cif,
        [
            _atom(1, atom="N", auth_seq=41, label_seq=7),
            _atom(2, atom="CA", auth_seq=42, label_seq=7),
        ],
    )

    with pytest.raises(ResidueJoinError, match="inconsistent"):
        load_chain_ca_table(cif, "A")


def test_sifts_mapping_records_author_ids_without_fabricating_label_ids(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "mapping.xml.gz"
    xml = b"""\
<entry>
  <entity>
    <segment>
      <listResidue>
        <residue>
          <crossRefDb dbSource="PDB" dbChainId="AUTH"
                      dbResNum="42A" dbResName="ALA"/>
          <crossRefDb dbSource="UniProt" dbAccessionId="P00001"
                      dbResNum="101" dbResName="A"/>
        </residue>
      </listResidue>
    </segment>
  </entity>
</entry>
"""
    with gzip.open(xml_path, "wb") as handle:
        handle.write(xml)

    mapping = parse_sifts_residue_mapping(
        xml_path,
        chain_id="AUTH",
        uniprot_id="P00001",
    )

    assert mapping.loc[0, "auth_asym_id"] == "AUTH"
    assert mapping.loc[0, "auth_seq_id"] == 42
    assert mapping.loc[0, "insertion_code"] == "A"
    assert pd.isna(mapping.loc[0, "label_asym_id"])
    assert pd.isna(mapping.loc[0, "label_seq_id"])
    assert mapping.loc[0, "residue_mapping_provenance"] == "sifts_auth"
