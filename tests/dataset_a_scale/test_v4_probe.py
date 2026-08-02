"""Tests for scripts/dataset_a_scale/v4_variant_recovery_probe.py (TASK-B).

The probe is a standalone script (not a library module) per TASK-B's declared
scope, so it is loaded directly from its file path rather than imported as a
package -- this repo's `scripts/` tree has no __init__.py and no existing
precedent for tests importing from it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from dual_uq.dataset_a_scale.stages.p0 import P0_STAGE_NAME, run_p0

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dataset_a_scale"
    / "v4_variant_recovery_probe.py"
)
_spec = importlib.util.spec_from_file_location("v4_variant_recovery_probe", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
v4_probe = importlib.util.module_from_spec(_spec)
# Dataclasses with `from __future__ import annotations` resolve string type hints via
# sys.modules[cls.__module__], so the module must be registered before exec_module runs.
sys.modules[_spec.name] = v4_probe
_spec.loader.exec_module(v4_probe)

CENSUS_CONFIG = {
    "purpose": "dataset_a_census_round1",
    "mapping_policy": "explicit_auth_label",
    "fragment_policy": "full_coverage",
}

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


def _write_mmcif(
    path: Path,
    entry_id: str,
    residues: list[tuple[int, str]],
    *,
    offset: float,
    omit_atom_names: dict[int, set[str]] | None = None,
) -> None:
    omit_atom_names = omit_atom_names or {}
    values = []
    serial = 0
    for auth_seq_id, resname in residues:
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            if atom_name in omit_atom_names.get(auth_seq_id, set()):
                continue
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


def _build_fixture(
    root: Path,
    *,
    pdb_sequence: str,
    afdb_sequence: str | None = None,
    omit_pdb_output_positions: frozenset[int] = frozenset(),
    omit_atom_names: dict[int, set[str]] | None = None,
) -> dict[str, object]:
    """3-residue P0-ready pair; mapping sequence fixed at 'MAG'."""
    mapping_sequence = "MAG"
    afdb_sequence = mapping_sequence if afdb_sequence is None else afdb_sequence
    assert len(pdb_sequence) == len(mapping_sequence)
    assert len(afdb_sequence) == len(mapping_sequence)
    pair_id = "1abc_A__P12345"
    model_id = "AF-P12345-F1"

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
            "uniprot_residue_name": list(mapping_sequence),
            "pdb_residue_name": [_ONE_TO_THREE[aa] for aa in mapping_sequence],
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
        (auth_seq, _ONE_TO_THREE[aa])
        for output_position, (auth_seq, aa) in enumerate(
            zip(auth_positions, pdb_sequence, strict=True), start=1
        )
        if output_position not in omit_pdb_output_positions
    ]
    afdb_residues = [(index + 1, _ONE_TO_THREE[aa]) for index, aa in enumerate(afdb_sequence)]
    _write_mmcif(pdb_path, "1ABC", pdb_residues, offset=0.0, omit_atom_names=omit_atom_names)
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
                "uniprot_length": 500,
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
    return {"row": row, "pair_id": pair_id, "pdb_path": pdb_path, "afdb_path": model_path}


def _run_p0(tmp_path: Path, fixture: dict[str, object]) -> Path:
    stage_dir = tmp_path / "stage" / "index6" / P0_STAGE_NAME
    result = run_p0(
        manifest_row=fixture["row"],
        project_root=tmp_path,
        stage_dir=stage_dir,
        config=CENSUS_CONFIG,
        pipeline_version="dataset-a.v1",
        run_id="v4-probe-test",
    )
    assert result.validation.validation_pass is True
    return stage_dir


# ---------------------------------------------------------------------------
# load_mismatch_proteins
# ---------------------------------------------------------------------------


def test_load_mismatch_proteins_filters_to_pdb_amino_acid_mismatch(tmp_path: Path) -> None:
    report_path = tmp_path / "round1_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {},
                "records": [
                    {"pair_id": "a", "failure_code": "pdb_amino_acid_mismatch", "mismatches": []},
                    {"pair_id": "b", "failure_code": "unsupported_afdb_fragment", "mismatches": []},
                    {"pair_id": "c", "failure_code": "pdb_amino_acid_mismatch", "mismatches": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    records = v4_probe.load_mismatch_proteins(report_path)

    assert [r["pair_id"] for r in records] == ["a", "c"]


# ---------------------------------------------------------------------------
# probe_protein
# ---------------------------------------------------------------------------


def test_probe_protein_recovered_when_only_known_mismatch_present(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, pdb_sequence="GAG")  # mismatch only at position 1
    stage_dir = _run_p0(tmp_path, fixture)

    result = v4_probe.probe_protein(
        pair_id=fixture["pair_id"],
        protein_id="index6",
        known_mismatch_output_positions={1},
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        fragment_start=100,
        model_length=3,
    )

    assert result.bucket == "recovered"
    assert result.unexpected_mismatches == ()
    assert result.missing_pdb_positions == ()
    assert result.other_issues == ()


def test_probe_protein_blocked_downstream_missing_pdb(tmp_path: Path) -> None:
    # known mismatch at position 1; position 3 has no PDB coverage at all
    fixture = _build_fixture(
        tmp_path, pdb_sequence="GAG", omit_pdb_output_positions=frozenset({3})
    )
    stage_dir = _run_p0(tmp_path, fixture)

    result = v4_probe.probe_protein(
        pair_id=fixture["pair_id"],
        protein_id="index6",
        known_mismatch_output_positions={1},
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        fragment_start=100,
        model_length=3,
    )

    assert result.bucket == "blocked_downstream_missing_pdb"
    assert result.missing_pdb_positions == (3,)
    assert result.unexpected_mismatches == ()


def test_probe_protein_blocked_downstream_other_for_missing_backbone_atom(tmp_path: Path) -> None:
    # known mismatch at position 1; position 3 is present but missing its O atom
    fixture = _build_fixture(
        tmp_path, pdb_sequence="GAG", omit_atom_names={44: {"O"}}
    )
    stage_dir = _run_p0(tmp_path, fixture)

    result = v4_probe.probe_protein(
        pair_id=fixture["pair_id"],
        protein_id="index6",
        known_mismatch_output_positions={1},
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        fragment_start=100,
        model_length=3,
    )

    assert result.bucket == "blocked_downstream_other"
    assert result.missing_pdb_positions == ()
    assert result.unexpected_mismatches == ()
    assert any(issue.code == "missing_backbone_atom" for issue in result.other_issues)


def test_probe_protein_still_mismatching_when_known_set_is_incomplete(tmp_path: Path) -> None:
    # two real mismatches (positions 1 and 3), but the site table only knows about position 1
    fixture = _build_fixture(tmp_path, pdb_sequence="GAS")
    stage_dir = _run_p0(tmp_path, fixture)

    result = v4_probe.probe_protein(
        pair_id=fixture["pair_id"],
        protein_id="index6",
        known_mismatch_output_positions={1},  # deliberately incomplete
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        fragment_start=100,
        model_length=3,
    )

    assert result.bucket == "still_mismatching"
    assert result.unexpected_mismatches == (3,)


def test_probe_protein_afdb_mismatch_is_never_exempted(tmp_path: Path) -> None:
    """AFDB-side AA mismatches are not exempted by the known-PDB-mismatch set (PDR-01 §1.5)."""
    fixture = _build_fixture(tmp_path, pdb_sequence="GAG", afdb_sequence="MSG")  # afdb position 2 differs

    stage_dir = _run_p0(tmp_path, fixture)

    result = v4_probe.probe_protein(
        pair_id=fixture["pair_id"],
        protein_id="index6",
        known_mismatch_output_positions={1},
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        fragment_start=100,
        model_length=3,
    )

    assert result.bucket == "blocked_downstream_other"
    assert any(issue.code == "afdb_amino_acid_mismatch" for issue in result.other_issues)
