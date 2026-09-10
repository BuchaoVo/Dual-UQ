from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dual_uq.evaluation.structcal_local_response_cases import (
    StructCalLocalResponseCasesError,
    build_structcal_local_cases,
    load_structcal_local_cohort,
)


def _write_cif(path: Path, *, offset: float = 0.0) -> None:
    rows = []
    serial = 1
    for seq_id, residue in ((1, "ALA"), (2, "CYS"), (3, "ASP")):
        for atom_name, dx in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            rows.append(
                f"ATOM {serial} {atom_name} {residue} A A {seq_id} {seq_id} . "
                f"{offset + dx:.3f} 0.000 0.000 1.00 10.0 1"
            )
            serial += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "data_fixture\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.label_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _write_release(tmp_path: Path, *, missing_second: bool = False, bad_aa: bool = False) -> Path:
    root = tmp_path / "project"
    release = root / "artifacts/releases/structcal_v1"
    core = release / "core"
    tracks = release / "tracks"
    (root / "data/raw").mkdir(parents=True)
    _write_cif(root / "data/raw/a.cif")
    if not missing_second:
        _write_cif(root / "data/raw/b.cif", offset=0.5)
    core.mkdir(parents=True)
    tracks.mkdir(parents=True)
    pd.DataFrame(
        [{"protein_id": "P1", "canonical_sequence": "ACD"}]
    ).to_parquet(core / "proteins.parquet", index=False)
    pd.DataFrame(
        [
            {"structure_id": "s1", "protein_id": "P1", "chain_id": "A", "source_file_ref": "data/raw/a.cif"},
            {"structure_id": "s2", "protein_id": "P1", "chain_id": "A", "source_file_ref": "data/raw/b.cif"},
        ]
    ).to_parquet(core / "structures.parquet", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "protein_id": "P1",
                "arm": "representation_variation",
                "condition_1_structure_id": "s1",
                "condition_2_structure_id": "s2",
                "condition_1_label": "PDB",
                "condition_2_label": "AFDB",
                "state_family": None,
            }
        ]
    ).to_parquet(core / "condition_pairs.parquet", index=False)
    mapping = []
    for position, aa in enumerate("ACD", start=1):
        mapping.append(
            {
                "pair_id": "pair-1",
                "protein_id": "P1",
                "canonical_position": position,
                "canonical_aa": aa,
                "condition_1_residue_id": f"A:{position}",
                "condition_2_residue_id": f"A:{position}",
                "condition_1_aa": "X" if bad_aa and position == 2 else aa,
                "condition_2_aa": aa,
                "condition_1_mapped": True,
                "condition_2_mapped": True,
                "condition_1_coordinate_visible": True,
                "condition_2_coordinate_visible": True,
                "common_mapped": True,
                "common_coordinate_visible": True,
            }
        )
    pd.DataFrame(mapping).to_parquet(core / "residue_mappings.parquet", index=False)
    pd.DataFrame(
        [{"protein_id": "P1", "identity_cluster_id": "cluster-1", "split": "VALIDATION"}]
    ).to_parquet(core / "splits.parquet", index=False)
    pd.DataFrame(
        [{
            "pair_id": "pair-1", "protein_id": "P1", "identity_cluster_id": "cluster-1",
            "split": "VALIDATION", "benchmark_role": "INVARIANCE_NATURALISTIC",
            "arm": "representation_variation", "state_family": None,
        }]
    ).to_parquet(tracks / "track_i_invariance.parquet", index=False)
    return root


def test_loader_selects_frozen_track_and_preserves_orientation(tmp_path: Path) -> None:
    root = _write_release(tmp_path)

    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    assert cohort[["condition_1_label", "condition_2_label"]].iloc[0].tolist() == ["PDB", "AFDB"]
    assert cohort.loc[0, "identity_cluster_id"] == "cluster-1"
    assert cohort.loc[0, "track_or_diagnostic"] == "TRACK_I"


def test_case_builder_uses_frozen_mapping_and_reports_pair_local_missing_asset(tmp_path: Path) -> None:
    root = _write_release(tmp_path, missing_second=True)
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    cases, exclusions = build_structcal_local_cases(root, cohort, atom_names=("N", "CA", "C", "O"))

    assert cases.empty
    assert exclusions.loc[0, "status"] == "UNRESOLVED"
    assert exclusions.loc[0, "reason"] == "STRUCTURAL_ASSET_MISSING"


def test_condition_identity_mismatch_is_pair_local_unresolved(tmp_path: Path) -> None:
    root = _write_release(tmp_path, bad_aa=True)
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    cases, exclusions = build_structcal_local_cases(root, cohort, atom_names=("N", "CA", "C", "O"))

    assert cases.empty
    assert exclusions.loc[0, "reason"] == "RESIDUE_IDENTITY_MISMATCH"


def test_controlled_pair_consensus_opt_in_still_rejects_disagreeing_view_identities(
    tmp_path: Path,
) -> None:
    root = _write_release(tmp_path, bad_aa=True)
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")
    cohort["track_or_diagnostic"] = "CONTROLLED_DIAGNOSTIC"

    cases, exclusions = build_structcal_local_cases(
        root,
        cohort,
        atom_names=("N", "CA", "C", "O"),
        allow_controlled_pair_consensus=True,
    )

    assert cases.empty
    assert exclusions.loc[0, "reason"] == "RESIDUE_IDENTITY_MISMATCH"


def test_canonical_mapping_identity_mismatch_fails_hard(tmp_path: Path) -> None:
    root = _write_release(tmp_path)
    path = root / "artifacts/releases/structcal_v1/core/residue_mappings.parquet"
    mapping = pd.read_parquet(path)
    mapping.loc[mapping.index[1], "canonical_aa"] = "X"
    mapping.to_parquet(path, index=False)
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    with pytest.raises(StructCalLocalResponseCasesError, match="canonical sequence"):
        build_structcal_local_cases(root, cohort, atom_names=("N", "CA", "C", "O"))


def test_sequence_incomparable_pair_is_explicitly_unresolved(tmp_path: Path) -> None:
    root = _write_release(tmp_path)
    path = root / "artifacts/releases/structcal_v1/core/condition_pairs.parquet"
    pairs = pd.read_parquet(path)
    pairs["sequence_comparable"] = False
    pairs.to_parquet(path, index=False)
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    cases, exclusions = build_structcal_local_cases(root, cohort, atom_names=("N", "CA", "C", "O"))

    assert cases.empty
    assert exclusions.loc[0, "reason"] == "SEQUENCE_NOT_COMPARABLE"


def test_structure_identity_mismatch_is_pair_local_unresolved(tmp_path: Path) -> None:
    root = _write_release(tmp_path)
    path = root / "data/raw/a.cif"
    path.write_text(path.read_text(encoding="utf-8").replace("ALA A A 1", "SER A A 1"), encoding="utf-8")
    cohort = load_structcal_local_cohort(root, cohort="track_i", split="VALIDATION")

    cases, exclusions = build_structcal_local_cases(root, cohort, atom_names=("N", "CA", "C", "O"))

    assert cases.empty
    assert exclusions.loc[0, "reason"] == "STRUCTURE_IDENTITY_MISMATCH"
