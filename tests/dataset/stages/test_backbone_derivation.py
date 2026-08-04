from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset.pipeline.status import LifecycleStatus
from dual_uq.dataset.stages.derivation import (
    P1_STAGE_NAME,
    resolve_p1_inputs,
    run_p1,
)
from dual_uq.dataset.stages.resolution import run_p0

P0_CONFIG = {
    "mapping_policy": "explicit_auth_label",
    "fragment_policy": "full_coverage",
}
P1_CONFIG = {
    "pairing_policy": "frozen_mapping_auth_only",
    "output_chain": "A",
}


def _atom_rows(
    residues: list[dict[str, object]],
    *,
    source_offset: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    serial = 0
    for residue in residues:
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            serial += 1
            rows.append(
                {
                    **residue,
                    "atom_name": atom_name,
                    "element": atom_name[0],
                    "altloc": ".",
                    "occupancy": 1.0,
                    "x": source_offset + serial,
                    "y": source_offset + atom_index + 0.25,
                    "z": source_offset + int(residue["auth_seq_id"]) / 10.0,
                }
            )
    return rows


def _write_mmcif(path: Path, entry_id: str, rows: list[dict[str, object]]) -> None:
    columns = (
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
    values = []
    for row in rows:
        values.append(
            " ".join(
                str(value)
                for value in (
                    row["record_type"],
                    row["element"],
                    row["atom_name"],
                    row["altloc"],
                    row["resname"],
                    row["label_chain_id"],
                    row["label_seq_id"],
                    row["auth_chain_id"],
                    row["auth_seq_id"],
                    row["insertion_code"] or "?",
                    row["x"],
                    row["y"],
                    row["z"],
                    row["occupancy"],
                    1,
                )
            )
        )
    path.write_text(
        f"data_{entry_id}\n_entry.id {entry_id}\nloop_\n"
        + "\n".join(f"_atom_site.{column}" for column in columns)
        + "\n"
        + "\n".join(values)
        + "\n#\n",
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    mutate_mapping=None,
    mutate_pdb=None,
    mutate_afdb=None,
) -> dict[str, Path | dict[str, object]]:
    pair_id = "1abc_A__P12345"
    model_id = "AF-P12345-F2"
    pair_dir = root / "pairs" / pair_id
    afdb_dir = root / "raw/afdb/P12345" / model_id
    pdb_path = root / "raw/pdb/1abc.cif"
    pair_dir.mkdir(parents=True)
    afdb_dir.mkdir(parents=True)
    pdb_path.parent.mkdir(parents=True)

    mapping = pd.DataFrame(
        {
            "output_position": [1, 2, 3],
            "pdb_residue_name": ["MSE", "ALA", "GLY"],
            "uniprot_id": ["P12345"] * 3,
            "uniprot_residue_number": [100, 101, 102],
            "uniprot_residue_name": ["M", "A", "G"],
            "auth_asym_id": ["A"] * 3,
            "auth_seq_id": [42, 42, 42],
            "insertion_code": ["", "A", "B"],
            "label_asym_id": ["X"] * 3,
            "label_seq_id": [7, 8, 9],
        }
    )
    pdb_residues = [
        {
            "record_type": "HETATM" if row.pdb_residue_name == "MSE" else "ATOM",
            "resname": row.pdb_residue_name,
            "auth_chain_id": "A",
            "auth_seq_id": int(row.auth_seq_id),
            "insertion_code": row.insertion_code,
            "label_chain_id": "X",
            "label_seq_id": int(row.label_seq_id),
        }
        for row in mapping.itertuples()
    ]
    afdb_residues = [
        {
            "record_type": "ATOM",
            "resname": name,
            "auth_chain_id": "A",
            "auth_seq_id": position,
            "insertion_code": "",
            "label_chain_id": "A",
            "label_seq_id": position,
        }
        for position, name in enumerate(("MET", "ALA", "GLY"), start=1)
    ]
    if mutate_mapping is not None:
        mutate_mapping(mapping)
    pdb_rows = _atom_rows(pdb_residues, source_offset=0.0)
    afdb_rows = _atom_rows(afdb_residues, source_offset=100.0)
    if mutate_pdb is not None:
        mutate_pdb(pdb_rows)
    if mutate_afdb is not None:
        mutate_afdb(afdb_rows)

    model_path = afdb_dir / "model.cif"
    metadata_path = afdb_dir / "metadata.json"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    mapping_path = pair_dir / "residue_mapping.tsv"
    pair_qc_path = pair_dir / "pair_qc.json"
    _write_mmcif(pdb_path, "1ABC", pdb_rows)
    _write_mmcif(model_path, model_id, afdb_rows)
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")
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
                "paeDocUrl": (
                    f"https://example.test/{model_id}-predicted_aligned_error_v6.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    plddt_path.write_text(
        json.dumps(
            {
                "residueNumber": [1, 2, 3],
                "confidenceScore": [95.0, 96.0, 97.0],
            }
        ),
        encoding="utf-8",
    )
    pae_path.write_text(
        json.dumps(
            [
                {
                    "predicted_aligned_error": np.zeros((3, 3)).tolist(),
                    "max_predicted_aligned_error": 31.75,
                }
            ]
        ),
        encoding="utf-8",
    )
    pair_qc_path.write_text(
        json.dumps(
            {
                "protein_id": "fixture-protein",
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
    manifest_row = {
        "protein_id": "index103",
        "screening_index": 103,
        "pair_id": pair_id,
        "mechanism_label": "easy_control",
        "tier": 2,
        "pair_qc_path": str(pair_qc_path.relative_to(root)),
        "residue_mapping_path": str(mapping_path.relative_to(root)),
        "pdb_structure_path": str(pdb_path.relative_to(root)),
        "afdb_metadata_path": str(metadata_path.relative_to(root)),
        "afdb_model_path": str(model_path.relative_to(root)),
        "afdb_plddt_path": str(plddt_path.relative_to(root)),
        "afdb_pae_path": str(pae_path.relative_to(root)),
    }
    p0_dir = root / "run/P0_resolve_and_freeze_inputs"
    p0_result = run_p0(
        manifest_row=manifest_row,
        project_root=root,
        stage_dir=p0_dir,
        config=P0_CONFIG,
        pipeline_version="dataset-a.v1",
        run_id="p0-run",
    )
    assert p0_result.status is LifecycleStatus.COMPLETE
    return {
        "p0_dir": p0_dir,
        "p1_dir": root / f"run/{P1_STAGE_NAME}",
        "pdb_path": pdb_path,
        "model_path": model_path,
        "manifest_row": manifest_row,
    }


def _run(root: Path, fixture: dict[str, object], **kwargs: object):
    options: dict[str, object] = {
        "project_root": root,
        "p0_stage_dir": fixture["p0_dir"],
        "stage_dir": fixture["p1_dir"],
        "config": P1_CONFIG,
        "pipeline_version": "dataset-a.v1",
        "run_id": "p1-run",
    }
    options.update(kwargs)
    return run_p1(**options)  # type: ignore[arg-type]


def test_valid_paired_backbone_has_identical_sequence_and_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.COMPLETE
    assert result.validation.validation_pass is True
    assert result.residue_count == 3
    assert result.canonical_sequence == "MAG"
    stage_dir = fixture["p1_dir"]
    pdb_lines = (stage_dir / "outputs/pdb_backbone.pdb").read_text().splitlines()
    afdb_lines = (stage_dir / "outputs/afdb_backbone.pdb").read_text().splitlines()
    assert len([line for line in pdb_lines if line.startswith("ATOM")]) == 12
    assert [line[17:20] for line in pdb_lines if line[12:16].strip() == "CA"] == [
        "MET",
        "ALA",
        "GLY",
    ]
    assert [line[17:20] for line in afdb_lines if line[12:16].strip() == "CA"] == [
        "MET",
        "ALA",
        "GLY",
    ]


def test_outputs_are_deterministic_and_unchanged_run_skips(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _run(tmp_path, fixture)
    stage_dir = fixture["p1_dir"]
    before = {path: path.read_bytes() for path in stage_dir.rglob("*") if path.is_file()}

    second = _run(tmp_path, fixture)

    assert first.status is LifecycleStatus.COMPLETE
    assert second.status is LifecycleStatus.SKIPPED_VALIDATED
    assert {path: path.read_bytes() for path in before} == before


def test_auth_label_and_42_insertion_variants_are_preserved(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(tmp_path, fixture)

    provenance = pd.read_csv(
        fixture["p1_dir"] / "outputs/backbone_residue_provenance.tsv",
        sep="\t",
        keep_default_na=False,
    )
    pdb = provenance.loc[provenance["source"] == "pdb"]
    residues = pdb.drop_duplicates("output_position")
    assert list(residues["auth_seq_id"]) == [42, 42, 42]
    assert list(residues["insertion_code"]) == ["", "A", "B"]
    assert list(residues["label_seq_id"]) == [7, 8, 9]
    assert set(residues["auth_chain_id"]) == {"A"}
    assert set(residues["label_chain_id"]) == {"X"}


def test_missing_output_positions_are_generated_from_frozen_uniprot_order(
    tmp_path: Path,
) -> None:
    def remove_and_shuffle(mapping: pd.DataFrame) -> None:
        mapping.drop(columns="output_position", inplace=True)
        mapping.sort_values("uniprot_residue_number", ascending=False, inplace=True)

    fixture = _fixture(tmp_path, mutate_mapping=remove_and_shuffle)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.COMPLETE
    assert result.canonical_sequence == "MAG"
    provenance = pd.read_csv(
        fixture["p1_dir"] / "outputs/backbone_residue_provenance.tsv", sep="\t"
    )
    residues = provenance.drop_duplicates(["source", "output_position"])
    assert list(residues.loc[residues["source"] == "pdb", "output_position"]) == [
        1,
        2,
        3,
    ]
    assert list(residues.loc[residues["source"] == "pdb", "uniprot_position"]) == [
        100,
        101,
        102,
    ]


def test_missing_mapping_label_is_not_used_for_join_and_structure_label_is_preserved(
    tmp_path: Path,
) -> None:
    def remove_mapping_label(mapping: pd.DataFrame) -> None:
        mapping["label_asym_id"] = ""
        mapping["label_seq_id"] = ""

    fixture = _fixture(tmp_path, mutate_mapping=remove_mapping_label)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.COMPLETE
    provenance = pd.read_csv(
        fixture["p1_dir"] / "outputs/backbone_residue_provenance.tsv", sep="\t"
    )
    pdb = provenance.loc[provenance["source"] == "pdb"]
    assert set(pdb["label_chain_id"]) == {"X"}
    assert set(pdb["label_seq_id"]) == {7, 8, 9}


def test_label_identity_is_not_used_as_auth_fallback(tmp_path: Path) -> None:
    def change_auth(mapping: pd.DataFrame) -> None:
        mapping.loc[1, "auth_seq_id"] = 999

    fixture = _fixture(tmp_path, mutate_mapping=change_auth)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "residue_count_mismatch"
    assert result.validation.details["missing_pdb_output_positions"] == [2]


def test_afdb_model_position_does_not_guess_offset(tmp_path: Path) -> None:
    def shift_model(rows: list[dict[str, object]]) -> None:
        for row in rows:
            row["auth_seq_id"] = int(row["auth_seq_id"]) + 10

    fixture = _fixture(tmp_path, mutate_afdb=shift_model)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "afdb_model_position_mismatch"


def test_deterministic_altloc_selection_reuses_a4(tmp_path: Path) -> None:
    def add_altloc(rows: list[dict[str, object]]) -> None:
        ca = next(
            row
            for row in rows
            if row["auth_seq_id"] == 42
            and row["insertion_code"] == "A"
            and row["atom_name"] == "CA"
        )
        ca["altloc"] = "A"
        ca["occupancy"] = 0.8
        alternative = {**ca, "altloc": "B", "occupancy": 0.2, "x": 999.0}
        rows.insert(0, alternative)

    fixture = _fixture(tmp_path, mutate_pdb=add_altloc)
    _run(tmp_path, fixture)

    provenance = pd.read_csv(
        fixture["p1_dir"] / "outputs/backbone_residue_provenance.tsv", sep="\t"
    )
    selected = provenance.loc[
        (provenance["source"] == "pdb")
        & (provenance["output_position"] == 2)
        & (provenance["atom_name"] == "CA")
    ].iloc[0]
    assert selected["altloc"] == "A"
    assert selected["occupancy"] == pytest.approx(0.8)
    assert selected["x"] != 999.0


def test_missing_backbone_atom_fails_without_dropping_residue(tmp_path: Path) -> None:
    def remove_oxygen(rows: list[dict[str, object]]) -> None:
        rows[:] = [
            row
            for row in rows
            if not (
                row["auth_seq_id"] == 42
                and row["insertion_code"] == "B"
                and row["atom_name"] == "O"
            )
        ]

    fixture = _fixture(tmp_path, mutate_pdb=remove_oxygen)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "missing_backbone_atom"
    assert result.validation.details["output_position"] == 3
    assert result.validation.details["missing_atoms"] == ["O"]
    assert not (fixture["p1_dir"] / "outputs/pdb_backbone.pdb").exists()


def test_ambiguous_atom_fails_structurally(tmp_path: Path) -> None:
    def duplicate_atom(rows: list[dict[str, object]]) -> None:
        atom = next(row for row in rows if row["atom_name"] == "N")
        rows.append({**atom, "x": 999.0})

    fixture = _fixture(tmp_path, mutate_pdb=duplicate_atom)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "ambiguous_backbone_atom"


@pytest.mark.parametrize("coordinate", ["nan", "inf", "-inf"])
def test_nonfinite_coordinate_fails_validation(tmp_path: Path, coordinate: str) -> None:
    def change_coordinate(rows: list[dict[str, object]]) -> None:
        rows[0]["x"] = coordinate

    fixture = _fixture(tmp_path, mutate_pdb=change_coordinate)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "invalid_atom_record"


def test_mse_source_provenance_is_preserved_while_output_is_canonical(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(tmp_path, fixture)

    provenance = pd.read_csv(
        fixture["p1_dir"] / "outputs/backbone_residue_provenance.tsv", sep="\t"
    )
    mse = provenance.loc[
        (provenance["source"] == "pdb")
        & (provenance["output_position"] == 1)
    ]
    assert set(mse["raw_resname"]) == {"MSE"}
    assert set(mse["record_type"]) == {"HETATM"}
    assert set(mse["canonical_aa"]) == {"M"}


@pytest.mark.parametrize(
    ("source", "failure_code"),
    [("pdb", "pdb_amino_acid_mismatch"), ("afdb", "afdb_amino_acid_mismatch")],
)
def test_source_amino_acid_mismatch_is_not_trimmed(
    tmp_path: Path, source: str, failure_code: str
) -> None:
    def mismatch(rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row["atom_name"] == "CA":
                target_auth = 42 if source == "pdb" else 1
                if row["auth_seq_id"] == target_auth:
                    for candidate in rows:
                        if (
                            candidate["auth_seq_id"] == target_auth
                            and candidate["insertion_code"] == row["insertion_code"]
                        ):
                            candidate["resname"] = "GLY"
                    break

    fixture = _fixture(
        tmp_path,
        mutate_pdb=mismatch if source == "pdb" else None,
        mutate_afdb=mismatch if source == "afdb" else None,
    )

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == failure_code
    assert not (fixture["p1_dir"] / "outputs/pdb_backbone.pdb").exists()


def test_nonmonotonic_uniprot_positions_fail_explicitly(tmp_path: Path) -> None:
    def reorder(mapping: pd.DataFrame) -> None:
        mapping["uniprot_residue_number"] = [100, 102, 101]

    fixture = _fixture(tmp_path, mutate_mapping=reorder)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "uniprot_position_order_mismatch"


def test_missing_source_residue_reports_count_mismatch_without_intersection_trim(
    tmp_path: Path,
) -> None:
    def remove_residue(rows: list[dict[str, object]]) -> None:
        rows[:] = [
            row
            for row in rows
            if not (row["auth_seq_id"] == 42 and row["insertion_code"] == "B")
        ]

    fixture = _fixture(tmp_path, mutate_pdb=remove_residue)

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "residue_count_mismatch"
    assert result.validation.details["missing_pdb_output_positions"] == [3]


def test_input_and_config_drift_are_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(tmp_path, fixture)
    config_drift = _run(tmp_path, fixture, config={**P1_CONFIG, "output_chain": "B"})
    pdb_path = fixture["pdb_path"]
    pdb_path.write_bytes(pdb_path.read_bytes() + b"# drift\n")

    input_drift = _run(tmp_path, fixture)

    assert input_drift.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert input_drift.failure_code == "input_digest_mismatch"
    assert config_drift.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert config_drift.failure_code == "config_digest_mismatch"


def test_corrupted_existing_output_cannot_skip(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(tmp_path, fixture)
    output = fixture["p1_dir"] / "outputs/pdb_backbone.pdb"
    output.write_text("corrupt\n", encoding="utf-8")

    result = _run(tmp_path, fixture)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "output_hash_mismatch"


def test_read_only_resolution_does_not_write_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    resolution = resolve_p1_inputs(
        project_root=tmp_path,
        p0_stage_dir=fixture["p0_dir"],
        config=P1_CONFIG,
        pipeline_version="dataset-a.v1",
    )

    assert resolution.residue_count == 3
    assert resolution.canonical_sequence == "MAG"
    assert not fixture["p1_dir"].exists()
