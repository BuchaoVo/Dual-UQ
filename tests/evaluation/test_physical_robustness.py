from pathlib import Path

import pandas as pd
import pytest

from dual_uq.evaluation.physical_robustness import (
    ESMFOLD_SAMPLE_INDICES,
    EvoEF2Protocol,
    PhysicalEvaluationError,
    _mutation_spec_from_pdb,
    _pdb_atom_line,
    manifest_payload,
    parse_energy_terms,
    paired_sequence_effects,
    preference_effects,
    prepare_inputs,
    score_prepared_inputs,
    screen_prepared_inputs,
    summarize_protein_effects,
)


def test_parse_energy_terms_requires_total_and_preserves_components() -> None:
    parsed = parse_energy_terms("reference_ALA = -1.25\nintraR_vdwatt = 2\nTotal = 0.75\n")
    assert parsed == {"reference_ALA": -1.25, "intraR_vdwatt": 2.0, "Total": 0.75}


def test_parse_energy_terms_rejects_missing_total() -> None:
    with pytest.raises(PhysicalEvaluationError, match="Total"):
        parse_energy_terms("reference_ALA = -1.25\n")


def test_preference_effects_are_within_protein_and_wt_adjusted() -> None:
    rows = pd.DataFrame(
        [
            {"protein_id": "p", "generated_condition": "WT", "evaluated_condition": "PDB", "sample_index": None, "total_energy": 10.0},
            {"protein_id": "p", "generated_condition": "WT", "evaluated_condition": "AFDB", "sample_index": None, "total_energy": 13.0},
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "PDB", "sample_index": 0, "total_energy": 8.0},
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "AFDB", "sample_index": 0, "total_energy": 14.0},
            {"protein_id": "p", "generated_condition": "AFDB", "evaluated_condition": "PDB", "sample_index": 1, "total_energy": 9.0},
            {"protein_id": "p", "generated_condition": "AFDB", "evaluated_condition": "AFDB", "sample_index": 1, "total_energy": 16.0},
        ]
    )
    result = preference_effects(rows).set_index("generated_condition")
    assert result.loc["PDB", "generated_preference_median"] == 6.0
    assert result.loc["AFDB", "generated_preference_median"] == 7.0
    assert result.loc["PDB", "wt_preference"] == 3.0
    assert result.loc["PDB", "baseline_adjusted_effect"] == 3.0


def test_manifest_binds_evaluator_revision_and_protocol(tmp_path: Path) -> None:
    executable = tmp_path / "EvoEF2"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o755)
    protocol = EvoEF2Protocol(executable)
    payload = manifest_payload(protocol, cohort="pilot", row_count=40)
    assert payload["source_revision"] == "38df01d305ed728ef067c3e0d22072058f33e255"
    assert payload["repair_runs"] == 3
    assert len(payload["executable_sha256"]) == 64


def test_pdb_atom_line_preserves_insertion_code() -> None:
    line = _pdb_atom_line(1, "CA", "ALA", 59, "A", 1.0, 2.0, 3.0, "C")
    assert line[22:26] == "  59"
    assert line[26] == "A"


def test_summarize_protein_effects_keeps_conditions_and_wt_baseline() -> None:
    records = pd.DataFrame(
        [
            {"protein_id": "p", "generated_condition": "WT", "evaluated_condition": "PDB", "sample_index": -1, "total_energy": 10.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "WT", "evaluated_condition": "AFDB", "sample_index": -1, "total_energy": 13.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "PDB", "sample_index": 0, "total_energy": 8.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "AFDB", "sample_index": 0, "total_energy": 14.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "AFDB", "evaluated_condition": "PDB", "sample_index": 1, "total_energy": 9.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "AFDB", "evaluated_condition": "AFDB", "sample_index": 1, "total_energy": 16.0, "status": "success"},
        ]
    )
    summary = summarize_protein_effects(records).iloc[0]
    assert summary["wt_preference"] == 3.0
    assert summary["pdb_baseline_adjusted_preference_median"] == 3.0
    assert summary["afdb_baseline_adjusted_preference_median"] == 4.0
    assert summary["generated_condition_difference"] == 1.0


def test_paired_sequence_effects_uses_afdb_minus_pdb_orientation() -> None:
    rows = pd.DataFrame(
        [
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "PDB", "sample_index": 0, "sequence_hash": "h", "total_energy": 4.0, "status": "success"},
            {"protein_id": "p", "generated_condition": "PDB", "evaluated_condition": "AFDB", "sample_index": 0, "sequence_hash": "h", "total_energy": 7.0, "status": "success"},
        ]
    )
    result = paired_sequence_effects(rows).iloc[0]
    assert result["E_P"] == 4.0
    assert result["E_A"] == 7.0
    assert result["preference_e"] == 3.0


def test_formal_subset_uses_the_frozen_esmfold_indices() -> None:
    assert ESMFOLD_SAMPLE_INDICES == (0, 32, 64, 96, 128, 160, 192, 224)


def test_mutation_spec_uses_evoef2_reference_chain_position_format(tmp_path: Path) -> None:
    pdb = tmp_path / "wt.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   7       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  CYS A   8       1.000   0.000   0.000  1.00  0.00           C\n",
        encoding="ascii",
    )
    assert _mutation_spec_from_pdb(pdb, "AD") == "CA8D;"


def test_mutation_spec_rejects_insertion_code_without_fallback(tmp_path: Path) -> None:
    pdb = tmp_path / "wt.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   7A      0.000   0.000   0.000  1.00  0.00           C\n",
        encoding="ascii",
    )
    with pytest.raises(PhysicalEvaluationError, match="insertion"):
        _mutation_spec_from_pdb(pdb, "D")


def test_score_prepared_inputs_records_tool_failure_without_scientific_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "EvoEF2"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o755)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    input_path = case_dir / "input.pdb"
    input_path.write_text("ATOM\n", encoding="ascii")
    from dual_uq.evaluation import physical_robustness as module

    def fail(*args: object, **kwargs: object) -> object:
        raise PhysicalEvaluationError("tool unavailable")

    monkeypatch.setattr(module, "evaluate_structure", fail)
    manifest = tmp_path / "input_manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "schema": "evoef2_physical_input_manifest_v1",
                "protein_ids": ["p"],
                "sample_indices": [0],
                "cases": [
                    {
                        "protein_id": "p",
                        "generated_condition": "PDB",
                        "evaluated_condition": "PDB",
                        "sample_index": 0,
                        "sample_class": "paired",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "s",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                    {
                        "protein_id": "p",
                        "generated_condition": "PDB",
                        "evaluated_condition": "AFDB",
                        "sample_index": 0,
                        "sample_class": "paired",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "a",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                    {
                        "protein_id": "p",
                        "generated_condition": "WT",
                        "evaluated_condition": "PDB",
                        "sample_index": -1,
                        "sample_class": "reference",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "s",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                    {
                        "protein_id": "p",
                        "generated_condition": "WT",
                        "evaluated_condition": "AFDB",
                        "sample_index": -1,
                        "sample_class": "reference",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "a",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                    {
                        "protein_id": "p",
                        "generated_condition": "AFDB",
                        "evaluated_condition": "PDB",
                        "sample_index": 0,
                        "sample_class": "paired",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "s",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                    {
                        "protein_id": "p",
                        "generated_condition": "AFDB",
                        "evaluated_condition": "AFDB",
                        "sample_index": 0,
                        "sample_class": "paired",
                        "sequence": "A",
                        "sequence_hash": __import__("hashlib").sha256(b"A").hexdigest(),
                        "structure_sha256": "a",
                        "source_chain_id": "A",
                        "input_path": "case/input.pdb",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = score_prepared_inputs(EvoEF2Protocol(executable), manifest)
    assert len(records) == 6
    assert records["status"].tolist() == ["failed"] * 6
    assert records["failure_type"].tolist() == [
        "BuildMutant", "BuildMutant", "PhysicalEvaluationError",
        "PhysicalEvaluationError", "BuildMutant", "BuildMutant",
    ]
    assert not list(tmp_path.glob(".evoef2-case-*"))


def test_screen_marks_build_failure_without_running_energy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dual_uq.evaluation import physical_robustness as module

    executable = tmp_path / "EvoEF2"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o755)
    input_path = tmp_path / "input.pdb"
    input_path.write_text("ATOM\n", encoding="ascii")
    sequence_hash = __import__("hashlib").sha256(b"A").hexdigest()
    cases = []
    for generated_condition in ("PDB", "AFDB"):
        for evaluated_condition in ("PDB", "AFDB"):
            cases.append({
                "protein_id": "p", "generated_condition": generated_condition,
                "evaluated_condition": evaluated_condition, "sample_index": 0,
                "sample_class": "paired", "sequence": "A", "sequence_hash": sequence_hash,
                "structure_sha256": evaluated_condition, "source_chain_id": "A",
                "input_path": "input.pdb", "wt_input_path": "input.pdb",
            })
    for evaluated_condition in ("PDB", "AFDB"):
        cases.append({
            "protein_id": "p", "generated_condition": "WT",
            "evaluated_condition": evaluated_condition, "sample_index": -1,
            "sample_class": "reference", "sequence": "A", "sequence_hash": sequence_hash,
            "structure_sha256": evaluated_condition, "source_chain_id": "A",
            "input_path": "input.pdb",
        })
    manifest = tmp_path / "input_manifest.json"
    manifest.write_text(__import__("json").dumps({
        "schema": "evoef2_physical_input_manifest_v1", "protein_ids": ["p"],
        "sample_indices": [0], "cases": cases,
    }), encoding="utf-8")

    def fail_build(*args: object, **kwargs: object) -> object:
        raise PhysicalEvaluationError("build unavailable")

    monkeypatch.setattr(module, "build_mutant", fail_build)
    monkeypatch.setattr(module, "repair_structure", lambda *args, **kwargs: input_path)
    monkeypatch.setattr(module, "_repaired_sequence", lambda path: "A")
    result = screen_prepared_inputs(EvoEF2Protocol(executable), manifest)
    assert result["status"].tolist() == ["failed"] * 4 + ["success"] * 2
    assert result["failure_type"].iloc[:4].tolist() == ["BuildMutant"] * 4
    assert result["failure_type"].iloc[4:].isna().all()
    assert not list(tmp_path.glob(".evoef2-screen-*"))


def test_prepare_inputs_materializes_only_reusable_wt_structures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace
    from dual_uq.evaluation import physical_robustness as module

    rows = []
    for condition in ("PDB", "AFDB"):
        sequence = "A" if condition == "PDB" else "C"
        rows.append(
            {
                "protein_id": "p",
                "backbone_condition": condition,
                "sample_index": 0,
                "sample_class": "paired",
                "sequence": sequence,
                "sequence_hash": __import__("hashlib").sha256(sequence.encode()).hexdigest(),
            }
        )
    conditions = tuple(
        SimpleNamespace(
            protein_id="p",
            condition=condition,
            source_path=tmp_path / f"{condition}.cif",
            source_sha256=condition,
            source_chain_id="A",
            projection_coordinates=(),
            wt_sequence="G",
        )
        for condition in ("PDB", "AFDB")
    )
    monkeypatch.setattr(module, "load_physical_conditions", lambda *args, **kwargs: conditions)
    monkeypatch.setattr(module.pd, "read_parquet", lambda *args, **kwargs: pd.DataFrame(rows))
    monkeypatch.setattr(module, "render_common_mask_pdb", lambda *args, **kwargs: b"ATOM\n")

    output = tmp_path / "prepared"
    manifest = prepare_inputs(
        project_root=tmp_path,
        protein_ids=("p",),
        sample_indices=(0,),
        output_root=output,
    )

    payload = __import__("json").loads(manifest.read_text())
    assert len(payload["cases"]) == 6
    assert len(list(output.rglob("input.pdb"))) == 2
    assert {
        case["input_path"] for case in payload["cases"]
    } == {"p/wt_pdb/input.pdb", "p/wt_afdb/input.pdb"}
