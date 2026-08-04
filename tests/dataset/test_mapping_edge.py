from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset.audits.observability import mechanism_evidence
from dual_uq.dataset.models import AFDBFragment, DerivationError
from dual_uq.dataset.services import mapping as mapping_module
from dual_uq.mapped_confidence import build_mapped_confidence_residue_table
from dual_uq.schema import AmbiguousLegacyResidueIdentifier
from dual_uq.sifts import parse_sifts_residue_mapping


def _write_sifts(
    path: Path,
    *,
    pdb_id: str,
    accession: str,
    label_positions: list[int],
    uniprot_positions: list[int],
    author_positions: list[str],
    pdbe_coordinate_system: str = "PDBe",
) -> None:
    residues = []
    for label, uniprot, author in zip(
        label_positions, uniprot_positions, author_positions, strict=True
    ):
        residues.append(
            f"""
        <residue dbSource="PDBe" dbCoordSys="{pdbe_coordinate_system}"
                 dbResNum="{label}" dbResName="ALA">
          <crossRefDb dbSource="PDB" dbAccessionId="{pdb_id}"
                      dbChainId="A" dbResNum="{author}" dbResName="ALA"/>
          <crossRefDb dbSource="UniProt" dbAccessionId="{accession}"
                      dbResNum="{uniprot}" dbResName="A"/>
          <residueDetail dbSource="PDBe" property="Annotation">
            Not_Observed
          </residueDetail>
        </residue>
"""
        )
    xml = (
        "<entry><entity entityId=\"A\"><segment><listResidue>"
        + "".join(residues)
        + "</listResidue></segment></entity></entry>"
    )
    with gzip.open(path, "wb") as handle:
        handle.write(xml.encode("utf-8"))


def _write_scheme(
    path: Path,
    *,
    label_positions: list[int],
    duplicate_first: bool = False,
) -> None:
    rows = [
        f"A 1 {position} ALA ? ? {position} A ."
        for position in label_positions
    ]
    if duplicate_first:
        rows.append(rows[0])
    path.write_text(
        "\n".join(
            [
                "data_test",
                "#",
                "loop_",
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.entity_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
                "_pdbx_poly_seq_scheme.auth_mon_id",
                "_pdbx_poly_seq_scheme.auth_seq_num",
                "_pdbx_poly_seq_scheme.pdb_seq_num",
                "_pdbx_poly_seq_scheme.pdb_strand_id",
                "_pdbx_poly_seq_scheme.pdb_ins_code",
                *rows,
                "#",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("pdb_id", "accession", "label_positions", "uniprot_positions"),
    [
        ("2wfi", "Q13427", [3, 4, 5, 6, 7], [1, 2, 3, 4, 5]),
        (
            "4uyr",
            "P08640",
            [22, 23, 24, 25, 26, 27, 28, 29, 207, 208, 209, 210, 211],
            [22, 23, 24, 25, 26, 27, 28, 29, 207, 208, 209, 210, 211],
        ),
    ],
)
def test_legacy_parser_reproduces_mapping_edge_author_ambiguity(
    tmp_path: Path,
    pdb_id: str,
    accession: str,
    label_positions: list[int],
    uniprot_positions: list[int],
) -> None:
    sifts = tmp_path / f"{pdb_id}.xml.gz"
    _write_sifts(
        sifts,
        pdb_id=pdb_id,
        accession=accession,
        label_positions=label_positions,
        uniprot_positions=uniprot_positions,
        author_positions=["null"] * len(label_positions),
    )

    with pytest.raises(
        AmbiguousLegacyResidueIdentifier,
        match="SIFTS PDB residue identifier 'null' is ambiguous",
    ):
        parse_sifts_residue_mapping(sifts, chain_id="A", uniprot_id=accession)


@pytest.mark.parametrize(
    ("pdb_id", "accession", "label_positions", "uniprot_positions"),
    [
        ("2wfi", "Q13427", [3, 4, 5, 6, 7], [1, 2, 3, 4, 5]),
        (
            "4uyr",
            "P08640",
            [22, 23, 24, 25, 26, 27, 28, 29, 207, 208, 209, 210, 211],
            [22, 23, 24, 25, 26, 27, 28, 29, 207, 208, 209, 210, 211],
        ),
    ],
)
def test_shared_adapter_uses_explicit_label_relation_when_author_is_null(
    tmp_path: Path,
    pdb_id: str,
    accession: str,
    label_positions: list[int],
    uniprot_positions: list[int],
) -> None:
    sifts = tmp_path / f"{pdb_id}.xml.gz"
    cif = tmp_path / f"{pdb_id}.cif"
    _write_sifts(
        sifts,
        pdb_id=pdb_id,
        accession=accession,
        label_positions=label_positions,
        uniprot_positions=uniprot_positions,
        author_positions=["null"] * len(label_positions),
    )
    _write_scheme(cif, label_positions=label_positions)

    result = mapping_module.parse_sifts_mapping_with_explicit_labels(
        sifts,
        cif,
        chain_id="A",
        uniprot_id=accession,
    )

    assert len(result) == len(label_positions)
    assert result["auth_asym_id"].tolist() == ["A"] * len(label_positions)
    assert result["auth_seq_id"].isna().all()
    assert result["label_asym_id"].tolist() == ["A"] * len(label_positions)
    assert result["label_seq_id"].tolist() == label_positions
    assert result["uniprot_residue_number"].tolist() == uniprot_positions
    assert result["insertion_code"].tolist() == [""] * len(label_positions)
    assert set(result["residue_mapping_provenance"]) == {
        "sifts_pdbe_mmcif_explicit_label"
    }


def test_shared_adapter_preserves_successful_author_mapping_control(
    tmp_path: Path,
) -> None:
    sifts = tmp_path / "2vb1.xml.gz"
    cif = tmp_path / "2vb1.cif"
    _write_sifts(
        sifts,
        pdb_id="2vb1",
        accession="P00698",
        label_positions=[1],
        uniprot_positions=[1],
        author_positions=["1"],
    )
    _write_scheme(cif, label_positions=[1])

    result = mapping_module.parse_sifts_mapping_with_explicit_labels(
        sifts,
        cif,
        chain_id="A",
        uniprot_id="P00698",
    )

    assert result.loc[0, "auth_asym_id"] == "A"
    assert result.loc[0, "auth_seq_id"] == 1
    assert pd.isna(result.loc[0, "label_asym_id"])
    assert pd.isna(result.loc[0, "label_seq_id"])
    assert result.loc[0, "residue_mapping_provenance"] == "sifts_auth"


def test_shared_adapter_rejects_nonunique_explicit_label_relation(
    tmp_path: Path,
) -> None:
    sifts = tmp_path / "edge.xml.gz"
    cif = tmp_path / "edge.cif"
    _write_sifts(
        sifts,
        pdb_id="2wfi",
        accession="Q13427",
        label_positions=[3],
        uniprot_positions=[1],
        author_positions=["null"],
    )
    _write_scheme(cif, label_positions=[3], duplicate_first=True)

    with pytest.raises(
        AmbiguousLegacyResidueIdentifier, match="explicit mmCIF label relation"
    ):
        mapping_module.parse_sifts_mapping_with_explicit_labels(
            sifts,
            cif,
            chain_id="A",
            uniprot_id="Q13427",
        )


def test_shared_adapter_rejects_non_pdbe_parent_coordinate_system(
    tmp_path: Path,
) -> None:
    sifts = tmp_path / "edge.xml.gz"
    cif = tmp_path / "edge.cif"
    _write_sifts(
        sifts,
        pdb_id="2wfi",
        accession="Q13427",
        label_positions=[3],
        uniprot_positions=[1],
        author_positions=["null"],
        pdbe_coordinate_system="PDBresnum",
    )
    _write_scheme(cif, label_positions=[3])

    with pytest.raises(
        AmbiguousLegacyResidueIdentifier, match="explicit PDBe coordinate identity"
    ):
        mapping_module.parse_sifts_mapping_with_explicit_labels(
            sifts,
            cif,
            chain_id="A",
            uniprot_id="Q13427",
        )


def test_mapped_confidence_preserves_multiple_label_only_unobserved_rows() -> None:
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [1, 2, 3],
            "auth_asym_id": ["A", "A", "A"],
            "auth_seq_id": [pd.NA, pd.NA, 3],
            "insertion_code": ["", "", ""],
            "label_asym_id": ["A", "A", "A"],
            "label_seq_id": [1, 2, 3],
        }
    )
    pdb_ca = pd.DataFrame(
        {
            "auth_asym_id": ["A"],
            "auth_seq_id": [3],
            "insertion_code": [""],
            "label_asym_id": ["A"],
            "label_seq_id": [3],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    )

    result = build_mapped_confidence_residue_table(
        mapping,
        pdb_ca,
        pd.Series([91.0, 92.0, 93.0]).to_numpy(),
        fragment_start=1,
        fragment_end=3,
    )

    assert result["observed_ca"].tolist() == [False, False, True]
    assert result["plddt"].tolist() == [91.0, 92.0, 93.0]


def test_mapping_provenance_keeps_one_to_many_unobserved_source_residue() -> None:
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [65, 66, 67],
            "uniprot_residue_name": ["S", "Y", "G"],
            "pdb_residue_name": ["GYS", "GYS", "GYS"],
            "auth_asym_id": ["A", "A", "A"],
            "auth_seq_id": [66, 66, 66],
            "insertion_code": ["", "", ""],
            "label_asym_id": ["A", "A", "A"],
            "label_seq_id": [65, 66, 67],
        }
    )
    pdb_ca = pd.DataFrame(
        {
            "auth_asym_id": ["A"],
            "auth_seq_id": [68],
            "insertion_code": [""],
            "label_asym_id": ["A"],
            "label_seq_id": [68],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    )

    result, gaps, diagnostics = mapping_module.mapping_with_provenance(
        mapping, pdb_ca
    )

    assert result["uniprot_position"].tolist() == [65, 66, 67]
    assert result["output_position"].tolist() == [1, 2, 3]
    assert result["observed_ca"].tolist() == [False, False, False]
    assert result["auth_seq_id"].tolist() == [66, 66, 66]
    assert result["label_seq_id"].tolist() == [65, 66, 67]
    assert gaps["gap_count"] == 0
    assert diagnostics["unmatched_mapping_count"] == 3


def test_mechanism_adapter_structures_missing_author_provenance_failure() -> None:
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [1, 2, 3, 4],
            "uniprot_position": [1, 2, 3, 4],
            "segment_id": [1, 1, 1, 1],
            "auth_asym_id": ["A", "A", "A", "A"],
            "auth_seq_id": [pd.NA, 2, 3, 4],
            "insertion_code": ["", "", "", ""],
            "label_asym_id": ["A", "A", "A", "A"],
            "label_seq_id": [1, 2, 3, 4],
        }
    )
    pdb_ca = pd.DataFrame(
        {
            "auth_asym_id": ["A", "A", "A"],
            "auth_seq_id": [2, 3, 4],
            "insertion_code": ["", "", ""],
            "label_asym_id": ["A", "A", "A"],
            "label_seq_id": [2, 3, 4],
            "x": [0.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0],
            "z": [0.0, 0.0, 0.0],
        }
    )
    afdb_ca = pd.DataFrame(
        {
            "uniprot_position": [1, 2, 3, 4],
            "x": [0.0, 0.0, 1.0, 0.0],
            "y": [0.0, 0.0, 0.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(DerivationError) as error:
        mechanism_evidence(
            mapping,
            pdb_ca,
            afdb_ca,
            np.zeros((4, 4)),
            np.full(4, 90.0),
            AFDBFragment("AF-PTEST-F1", 1, 4, 4),
            {
                "thresholds": {
                    "easy_control": {
                        "mapped_plddt_median_min": 90.0,
                        "ca_disagreement_p90_max": 1.0,
                        "high_conf_segment_max_length": 2,
                    },
                    "high_pae_long_range": {
                        "q90_threshold": 10.0,
                        "above_10_fraction_threshold": 0.5,
                        "above_15_fraction_threshold": 0.5,
                    },
                    "high_confidence_state_disagreement": {
                        "minimum_segment_disagreement": 1.0,
                        "minimum_segment_length": 3,
                        "minimum_segment_median_plddt": 90.0,
                    },
                }
            },
            quality_pass=False,
        )

    assert error.value.code == "missing_explicit_auth_residue_key"
