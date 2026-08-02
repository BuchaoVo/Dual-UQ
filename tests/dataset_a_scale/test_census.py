from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dual_uq.dataset_a_scale.census import (
    CENSUS_CONFIG,
    CensusSkip,
    IdentityRecord,
    IdentitySources,
    build_manifest_row,
    count_all_sequence_mismatches,
    discover_local_pairs,
    evaluate_protein,
    load_identity_sources,
    run_round1_census,
    write_round1_report,
)
from dual_uq.dataset_a_scale.stages.p0 import P0_STAGE_NAME, run_p0

# ---------------------------------------------------------------------------
# load_identity_sources
# ---------------------------------------------------------------------------


def _write_lifecycle_csv(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_identity_sources_joins_primary_and_replacement_pools(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    _write_lifecycle_csv(
        primary,
        [{"pair_name": "1gci_A__P29600", "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(
        replacement,
        [
            {
                "pair_name": "5avd_A__P00772",
                "screening_index": "103",
                "primary_category": "easy_control",
            }
        ],
    )

    sources = load_identity_sources(primary, replacement)

    assert sources.records["1gci_A__P29600"] == IdentityRecord(
        screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
    )
    assert sources.records["5avd_A__P00772"] == IdentityRecord(
        screening_index=103,
        mechanism_label_prior="easy_control",
        source="replacement_candidate_lifecycle",
    )
    assert sources.conflicts == {}


def test_load_identity_sources_agreeing_duplicate_is_not_a_conflict(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    row = {"pair_name": "3zoj_A__F2QVG4", "screening_index": "36", "primary_category": "high_pae_long_range"}
    _write_lifecycle_csv(primary, [row])
    _write_lifecycle_csv(replacement, [row])

    sources = load_identity_sources(primary, replacement)

    assert sources.records["3zoj_A__F2QVG4"].screening_index == 36
    assert sources.conflicts == {}


def test_load_identity_sources_flags_conflicting_screening_index(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    _write_lifecycle_csv(
        primary,
        [{"pair_name": "1abc_A__P1", "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(
        replacement,
        [{"pair_name": "1abc_A__P1", "screening_index": "106", "primary_category": "easy_control"}],
    )

    sources = load_identity_sources(primary, replacement)

    assert "1abc_A__P1" not in sources.records
    assert len(sources.conflicts["1abc_A__P1"]) == 2


# ---------------------------------------------------------------------------
# discover_local_pairs
# ---------------------------------------------------------------------------


def test_discover_local_pairs_lists_only_directories_with_pair_qc(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    (pairs_root / "1abc_A__P1").mkdir(parents=True)
    (pairs_root / "1abc_A__P1" / "pair_qc.json").write_text("{}", encoding="utf-8")
    (pairs_root / "incomplete_pair").mkdir(parents=True)

    assert discover_local_pairs(pairs_root) == ["1abc_A__P1"]


# ---------------------------------------------------------------------------
# build_manifest_row
# ---------------------------------------------------------------------------


def _write_local_pair(
    project_root: Path,
    pair_id: str,
    *,
    pdb_id: str = "1abc",
    uniprot_id: str = "P12345",
    model_entity_id: str = "AF-P12345-F1",
) -> dict[str, Path]:
    pair_dir = project_root / "data/processed/pairs" / pair_id
    pair_dir.mkdir(parents=True)
    afdb_dir = project_root / "data/raw/afdb" / uniprot_id / model_entity_id
    afdb_dir.mkdir(parents=True)
    pdb_path = project_root / "data/raw/pdb" / f"{pdb_id}.cif"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text("data_fixture\n#\n", encoding="utf-8")
    mapping_path = pair_dir / "residue_mapping.parquet"
    mapping_path.write_text("", encoding="utf-8")
    model_path = afdb_dir / "model.cif"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    metadata_path = afdb_dir / "metadata.json"
    for path in (model_path, plddt_path, pae_path, metadata_path):
        path.write_text("{}", encoding="utf-8")
    pair_qc_path = pair_dir / "pair_qc.json"
    pair_qc_path.write_text(
        json.dumps(
            {
                "pdb_id": pdb_id,
                "chain_id": "A",
                "uniprot_id": uniprot_id,
                "quality_flag": "pass",
                "pdb_path": str(pdb_path),
                "mapping_path": str(mapping_path),
                "afdb_model_path": str(model_path),
                "plddt_path": str(plddt_path),
                "pae_path": str(pae_path),
            }
        ),
        encoding="utf-8",
    )
    return {
        "pair_dir": pair_dir,
        "pair_qc_path": pair_qc_path,
        "pdb_path": pdb_path,
        "mapping_path": mapping_path,
        "afdb_dir": afdb_dir,
        "metadata_path": metadata_path,
    }


def test_build_manifest_row_success(tmp_path: Path) -> None:
    project_root = tmp_path
    paths = _write_local_pair(project_root, "1abc_A__P12345")
    sources = IdentitySources(
        records={
            "1abc_A__P12345": IdentityRecord(
                screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, dict)
    assert result["protein_id"] == "index6"
    assert result["screening_index"] == 6
    assert result["pair_id"] == "1abc_A__P12345"
    assert result["mechanism_label"] == "easy_control"
    assert result["tier"] == 1
    assert result["pair_qc_path"] == str(paths["pair_qc_path"])
    assert result["residue_mapping_path"] == str(paths["mapping_path"])
    assert result["afdb_metadata_path"] == str(paths["metadata_path"])


def test_build_manifest_row_skips_when_no_identity_source(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "1abc_A__P12345")
    sources = IdentitySources(records={}, conflicts={})

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert result == CensusSkip(pair_id="1abc_A__P12345", code="no_identity_source", details={})


def test_build_manifest_row_skips_on_identity_conflict(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "1abc_A__P12345")
    conflicting = [
        IdentityRecord(screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"),
        IdentityRecord(
            screening_index=106,
            mechanism_label_prior="easy_control",
            source="replacement_candidate_lifecycle",
        ),
    ]
    sources = IdentitySources(records={}, conflicts={"1abc_A__P12345": conflicting})

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "identity_source_conflict"


def test_build_manifest_row_skips_when_declared_file_missing(tmp_path: Path) -> None:
    project_root = tmp_path
    paths = _write_local_pair(project_root, "1abc_A__P12345")
    paths["pdb_path"].unlink()
    sources = IdentitySources(
        records={
            "1abc_A__P12345": IdentityRecord(
                screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "declared_file_missing"
    assert result.details["field"] == "pdb_structure_path"


# ---------------------------------------------------------------------------
# Real P0/P1 fixture shared by count_all_sequence_mismatches / evaluate_protein tests
# ---------------------------------------------------------------------------

_ATOM_COLUMNS = (
    "group_PDB",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "auth_asym_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "pdbx_PDB_model_num",
)

_ONE_TO_THREE = {"M": "MET", "A": "ALA", "G": "GLY", "S": "SER"}


def _write_mmcif(path: Path, entry_id: str, residues: list[tuple[int, str]], *, offset: float) -> None:
    values = []
    serial = 0
    for auth_seq_id, resname in residues:
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            serial += 1
            values.append(
                " ".join(
                    str(value)
                    for value in (
                        "ATOM",
                        atom_name[0],
                        atom_name,
                        ".",
                        resname,
                        "A",
                        auth_seq_id,
                        "A",
                        auth_seq_id,
                        "?",
                        offset + serial,
                        offset + atom_index + 0.25,
                        offset + auth_seq_id / 10.0,
                        1.0,
                        1,
                    )
                )
            )
    path.write_text(
        f"data_{entry_id}\n_entry.id {entry_id}\nloop_\n"
        + "\n".join(f"_atom_site.{column}" for column in _ATOM_COLUMNS)
        + "\n"
        + "\n".join(values)
        + "\n#\n",
        encoding="utf-8",
    )


def _build_real_fixture(root: Path, *, pdb_sequence: str) -> dict[str, object]:
    """Build one real, valid-except-for-pdb_sequence P0-ready pair fixture.

    mapping canonical sequence is fixed at 'MAG'; pdb_sequence controls what
    residue identities are actually written into the PDB mmCIF, so callers can
    induce zero, one, or several isolated amino-acid mismatches.
    """
    pair_id = "1abc_A__P12345"
    model_id = "AF-P12345-F1"
    mapping_seq = "MAG"
    assert len(pdb_sequence) == len(mapping_seq)

    pair_dir = root / "data/processed/pairs" / pair_id
    afdb_dir = root / "data/raw/afdb/P12345" / model_id
    pdb_path = root / "data/raw/pdb/1abc.cif"
    pair_dir.mkdir(parents=True)
    afdb_dir.mkdir(parents=True)
    pdb_path.parent.mkdir(parents=True)

    auth_positions = [42, 43, 44]
    uniprot_positions = [100, 101, 102]
    mapping = pd.DataFrame(
        {
            "output_position": [1, 2, 3],
            "uniprot_id": ["P12345"] * 3,
            "uniprot_residue_number": uniprot_positions,
            "uniprot_residue_name": list(mapping_seq),
            "pdb_residue_name": [_ONE_TO_THREE[aa] for aa in mapping_seq],
            "auth_asym_id": ["A"] * 3,
            "auth_seq_id": auth_positions,
            "insertion_code": [""] * 3,
            "label_asym_id": ["A"] * 3,
            "label_seq_id": auth_positions,
        }
    )
    mapping_path = pair_dir / "residue_mapping.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    pdb_residues = [
        (auth_seq, _ONE_TO_THREE[aa]) for auth_seq, aa in zip(auth_positions, pdb_sequence, strict=True)
    ]
    afdb_residues = [(index + 1, _ONE_TO_THREE[aa]) for index, aa in enumerate(mapping_seq)]
    _write_mmcif(pdb_path, "1ABC", pdb_residues, offset=0.0)
    model_path = afdb_dir / "model.cif"
    _write_mmcif(model_path, model_id, afdb_residues, offset=100.0)

    metadata_path = afdb_dir / "metadata.json"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    metadata_path.write_text(
        json.dumps(
            {
                "modelEntityId": model_id,
                "entryId": model_id,
                "latestVersion": 6,
                "uniprotStart": 100,
                "uniprotEnd": 102,
                "sequenceStart": 100,
                "sequenceEnd": 102,
                "cifUrl": f"https://example.test/{model_id}-model_v6.cif",
                "plddtDocUrl": f"https://example.test/{model_id}-confidence_v6.json",
                "paeDocUrl": f"https://example.test/{model_id}-predicted_aligned_error_v6.json",
            }
        ),
        encoding="utf-8",
    )
    plddt_path.write_text(
        json.dumps({"residueNumber": [1, 2, 3], "confidenceScore": [95.0, 96.0, 97.0]}),
        encoding="utf-8",
    )
    pae_path.write_text(
        json.dumps(
            [{"predicted_aligned_error": [[0.0] * 3 for _ in range(3)], "max_predicted_aligned_error": 31.75}]
        ),
        encoding="utf-8",
    )
    pair_qc_path = pair_dir / "pair_qc.json"
    pair_qc_path.write_text(
        json.dumps(
            {
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "P12345",
                "quality_flag": "pass",
                "mapped_residue_count": 3,
                "mapped_uniprot_start": 100,
                "mapped_uniprot_end": 102,
                "pdb_path": str(pdb_path),
                "mapping_path": str(mapping_path),
                "afdb_model_path": str(model_path),
                "plddt_path": str(plddt_path),
                "pae_path": str(pae_path),
                "afdb_model_entity_id": model_id,
                "afdb_version": 6,
                "afdb_fragment_start": 100,
                "afdb_fragment_end": 102,
                "afdb_fragment_length": 3,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "protein_id": "index6",
        "screening_index": 6,
        "pair_id": pair_id,
        "mechanism_label": "easy_control",
        "tier": 1,
        "pair_qc_path": str(pair_qc_path),
        "residue_mapping_path": str(mapping_path),
        "pdb_structure_path": str(pdb_path),
        "afdb_metadata_path": str(metadata_path),
        "afdb_model_path": str(model_path),
        "afdb_plddt_path": str(plddt_path),
        "afdb_pae_path": str(pae_path),
    }
    return {"row": row, "pair_id": pair_id, "pdb_path": pdb_path}


# ---------------------------------------------------------------------------
# count_all_sequence_mismatches
# ---------------------------------------------------------------------------


def test_count_all_sequence_mismatches_counts_every_position_not_just_first(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAS")  # mismatches at position 1 and 3
    stage_dir = tmp_path / "census" / "index6" / P0_STAGE_NAME
    result = run_p0(
        manifest_row=fixture["row"],
        project_root=tmp_path,
        stage_dir=stage_dir,
        config=CENSUS_CONFIG,
        pipeline_version="dataset-a.v1",
        run_id="census-test",
    )
    assert result.validation.validation_pass is True

    count = count_all_sequence_mismatches(
        pdb_path=fixture["pdb_path"], p0_stage_dir=stage_dir, pair_id=fixture["pair_id"]
    )

    assert count == 2


def test_count_all_sequence_mismatches_is_zero_for_matching_sequence(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")
    stage_dir = tmp_path / "census" / "index6" / P0_STAGE_NAME
    run_p0(
        manifest_row=fixture["row"],
        project_root=tmp_path,
        stage_dir=stage_dir,
        config=CENSUS_CONFIG,
        pipeline_version="dataset-a.v1",
        run_id="census-test",
    )

    count = count_all_sequence_mismatches(
        pdb_path=fixture["pdb_path"], p0_stage_dir=stage_dir, pair_id=fixture["pair_id"]
    )

    assert count == 0


# ---------------------------------------------------------------------------
# evaluate_protein / run_round1_census
# ---------------------------------------------------------------------------


def test_evaluate_protein_reports_ok_outcome_for_valid_pair(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")

    record = evaluate_protein(
        fixture["row"],
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p1_pairing_ok"
    assert record.failure_code is None


def test_evaluate_protein_reports_p1_failure_with_mismatch_count(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAG")  # single isolated mismatch at position 1

    record = evaluate_protein(
        fixture["row"],
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p1_failure"
    assert record.failure_code == "pdb_amino_acid_mismatch"
    assert record.mismatch_count == 1


def test_evaluate_protein_reports_p0_failure_without_writing_stage(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")
    bad_row = {**fixture["row"], "pair_qc_path": str(tmp_path / "does_not_exist.json")}

    record = evaluate_protein(
        bad_row,
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p0_failure"
    assert record.failure_code == "missing_required_file"
    assert not (tmp_path / "census").exists()


def test_run_round1_census_end_to_end_produces_report_with_summary(tmp_path: Path) -> None:
    project_root = tmp_path
    ok_fixture = _build_real_fixture(project_root, pdb_sequence="MAG")
    (project_root / "reports").mkdir()
    _write_lifecycle_csv(
        project_root / "reports/candidate_lifecycle.csv",
        [{"pair_name": ok_fixture["pair_id"], "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(project_root / "reports/replacement_candidate_lifecycle.csv", [])

    report = run_round1_census(
        pairs_root=project_root / "data/processed/pairs",
        candidate_lifecycle_path=project_root / "reports/candidate_lifecycle.csv",
        replacement_lifecycle_path=project_root / "reports/replacement_candidate_lifecycle.csv",
        project_root=project_root,
        census_stage_root=project_root / "census",
        config=CENSUS_CONFIG,
    )

    assert len(report.records) == 1
    assert report.records[0].outcome == "p1_pairing_ok"
    assert report.summary["total_local_pairs"] == 1
    assert report.summary["outcome_counts"] == {"p1_pairing_ok": 1}

    out_path = project_root / "reports/dataset_a_census/round1_report.json"
    write_round1_report(report, out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["outcome_counts"] == {"p1_pairing_ok": 1}
    assert payload["records"][0]["pair_id"] == ok_fixture["pair_id"]
