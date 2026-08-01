from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset_a_scale.hashing import sha256_file
from dual_uq.dataset_a_scale.schema import LifecycleStatus
from dual_uq.dataset_a_scale.stages.p0 import (
    P0_INPUT_PATH_FIELDS,
    resolve_p0_inputs,
    run_p0,
)

CONFIG = {"mapping_policy": "explicit_auth_label", "fragment_policy": "full_coverage"}


def _write_mmcif(path: Path, entry_id: str, length: int) -> None:
    rows = "\n".join(f"ATOM CA A {position} ?" for position in range(1, length + 1))
    path.write_text(
        f"data_{entry_id}\n"
        f"_entry.id {entry_id}\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        f"{rows}\n#\n",
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    model_id: str = "AF-P12345-F2",
    fragment_start: int = 100,
    fragment_end: int = 103,
    mapped_positions: list[int] | None = None,
) -> dict[str, object]:
    mapped_positions = mapped_positions or [101, 102, 103]
    pair_id = "1abc_A__P12345"
    pair_dir = root / "pairs" / pair_id
    afdb_dir = root / "raw" / "afdb" / "P12345" / model_id
    pdb_path = root / "raw" / "pdb" / "1abc.cif"
    pair_dir.mkdir(parents=True)
    afdb_dir.mkdir(parents=True)
    pdb_path.parent.mkdir(parents=True)

    model_length = fragment_end - fragment_start + 1
    model_path = afdb_dir / "model.cif"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    metadata_path = afdb_dir / "metadata.json"
    mapping_path = pair_dir / "residue_mapping.tsv"
    pair_qc_path = pair_dir / "pair_qc.json"

    _write_mmcif(pdb_path, "1ABC", len(mapped_positions))
    _write_mmcif(model_path, model_id, model_length)
    plddt_path.write_text(
        json.dumps(
            {
                "residueNumber": list(range(1, model_length + 1)),
                "confidenceScore": [95.0] * model_length,
                "confidenceCategory": ["H"] * model_length,
            }
        ),
        encoding="utf-8",
    )
    pae_path.write_text(
        json.dumps(
            [
                {
                    "predicted_aligned_error": np.zeros(
                        (model_length, model_length)
                    ).tolist(),
                    "max_predicted_aligned_error": 31.75,
                }
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "modelEntityId": model_id,
                "entryId": model_id,
                "latestVersion": 6,
                "uniprotStart": fragment_start,
                "uniprotEnd": fragment_end,
                "sequenceStart": fragment_start,
                "sequenceEnd": fragment_end,
                "cifUrl": f"https://example.test/{model_id}-model_v6.cif",
                "plddtDocUrl": (
                    f"https://example.test/{model_id}-confidence_v6.json"
                ),
                "paeDocUrl": (
                    "https://example.test/"
                    f"{model_id}-predicted_aligned_error_v6.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    mapping = pd.DataFrame(
        {
            "output_position": range(1, len(mapped_positions) + 1),
            "pdb_residue_name": ["ALA"] * len(mapped_positions),
            "uniprot_id": ["P12345"] * len(mapped_positions),
            "uniprot_residue_number": mapped_positions,
            "auth_asym_id": ["A"] * len(mapped_positions),
            "auth_seq_id": range(10, 10 + len(mapped_positions)),
            "insertion_code": [""] * len(mapped_positions),
            "label_asym_id": ["A"] * len(mapped_positions),
            "label_seq_id": range(1, len(mapped_positions) + 1),
        }
    )
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")
    pair_qc_path.write_text(
        json.dumps(
            {
                "protein_id": "legacy-protein-id",
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "P12345",
                "quality_flag": "pass",
                "mapped_residue_count": len(mapped_positions),
                "mapped_uniprot_start": min(mapped_positions),
                "mapped_uniprot_end": max(mapped_positions),
                "mapping_coverage": 1.0,
                "sequence_identity": 1.0,
                "pdb_path": str(pdb_path),
                "mapping_path": str(mapping_path),
                "afdb_model_path": str(model_path),
                "plddt_path": str(plddt_path),
                "pae_path": str(pae_path),
                "afdb_model_entity_id": model_id,
                "afdb_version": 6,
                "afdb_fragment_start": fragment_start,
                "afdb_fragment_end": fragment_end,
                "afdb_fragment_length": model_length,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "pair_qc_path": pair_qc_path,
        "residue_mapping_path": mapping_path,
        "pdb_structure_path": pdb_path,
        "afdb_metadata_path": metadata_path,
        "afdb_model_path": model_path,
        "afdb_plddt_path": plddt_path,
        "afdb_pae_path": pae_path,
    }
    return {
        "protein_id": "index103",
        "screening_index": 103,
        "pair_id": pair_id,
        "mechanism_label": "easy_control",
        "tier": 2,
        **{name: str(path.relative_to(root)) for name, path in paths.items()},
    }


def _run(root: Path, row: dict[str, object], **kwargs: object):
    options: dict[str, object] = {
        "manifest_row": row,
        "project_root": root,
        "stage_dir": root / "run" / "P0_resolve_and_freeze_inputs",
        "config": CONFIG,
        "pipeline_version": "dataset-a.v1",
        "run_id": "run-001",
    }
    options.update(kwargs)
    return run_p0(**options)  # type: ignore[arg-type]


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_valid_p0_fixture_writes_frozen_lock(tmp_path: Path) -> None:
    row = _fixture(tmp_path)

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.COMPLETE
    assert result.validation.validation_pass is True
    assert result.input_digest == result.input_lock["input_digest"]
    assert result.input_lock["validation_pass"] is True
    assert result.input_lock["selected_model_identity"] == "AF-P12345-F2"
    assert result.input_lock["fragment_interval"] == {
        "uniprot_start": 100,
        "uniprot_end": 103,
        "model_length": 4,
    }
    stage_dir = tmp_path / "run" / "P0_resolve_and_freeze_inputs"
    for relative in (
        "stage_manifest.json",
        "validation.json",
        "outputs/frozen_inputs.json",
        "outputs/residue_mapping.tsv",
        "outputs/input_lock.json",
    ):
        assert (stage_dir / relative).is_file()
    assert set(result.input_files) == set(P0_INPUT_PATH_FIELDS)
    assert all(record.sha256 == sha256_file(record.resolved_path) for record in result.input_files.values())


def test_input_digest_is_deterministic_and_independent_of_row_key_order(
    tmp_path: Path,
) -> None:
    row = _fixture(tmp_path)

    first = resolve_p0_inputs(row, project_root=tmp_path, config=CONFIG)
    second = resolve_p0_inputs(
        dict(reversed(list(row.items()))), project_root=tmp_path, config=CONFIG
    )

    assert first.input_digest == second.input_digest
    assert first.config_digest == second.config_digest


def test_explicit_path_change_with_identical_bytes_is_input_drift(
    tmp_path: Path,
) -> None:
    row = _fixture(tmp_path)
    first = _run(tmp_path, row)
    original = tmp_path / str(row["afdb_metadata_path"])
    replacement = original.with_name("metadata-copy.json")
    replacement.write_bytes(original.read_bytes())
    changed = {**row, "afdb_metadata_path": str(replacement.relative_to(tmp_path))}

    resolved = resolve_p0_inputs(changed, project_root=tmp_path, config=CONFIG)
    rerun = _run(tmp_path, changed)

    assert resolved.input_files["afdb_metadata_path"].sha256 == first.input_files[
        "afdb_metadata_path"
    ].sha256
    assert resolved.input_digest != first.input_digest
    assert rerun.status is LifecycleStatus.BLOCKED_INPUT_DRIFT


def test_repeated_unchanged_validation_is_skipped_without_writes(
    tmp_path: Path,
) -> None:
    row = _fixture(tmp_path)
    first = _run(tmp_path, row)
    stage_dir = tmp_path / "run" / "P0_resolve_and_freeze_inputs"
    before = {path: path.read_bytes() for path in stage_dir.rglob("*") if path.is_file()}

    second = _run(tmp_path, row)

    assert first.status is LifecycleStatus.COMPLETE
    assert second.status is LifecycleStatus.SKIPPED_VALIDATED
    assert {path: path.read_bytes() for path in before} == before


def test_missing_required_file_is_structured_failure(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    (tmp_path / str(row["afdb_pae_path"])).unlink()

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.validation.validation_pass is False
    assert result.failure_code == "missing_required_file"
    assert result.validation.details["logical_name"] == "afdb_pae_path"


def test_raw_sha_and_input_digest_change_with_file_content(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    before = resolve_p0_inputs(row, project_root=tmp_path, config=CONFIG)
    pdb_path = tmp_path / str(row["pdb_structure_path"])
    pdb_path.write_bytes(pdb_path.read_bytes() + b"# changed bytes\n")

    after = resolve_p0_inputs(row, project_root=tmp_path, config=CONFIG)

    assert before.input_files["pdb_structure_path"].sha256 != after.input_files[
        "pdb_structure_path"
    ].sha256
    assert before.input_digest != after.input_digest


def test_file_change_after_lock_blocks_input_drift_without_overwrite(
    tmp_path: Path,
) -> None:
    row = _fixture(tmp_path)
    _run(tmp_path, row)
    lock_path = tmp_path / "run/P0_resolve_and_freeze_inputs/outputs/input_lock.json"
    frozen_lock = lock_path.read_bytes()
    pdb_path = tmp_path / str(row["pdb_structure_path"])
    pdb_path.write_bytes(pdb_path.read_bytes() + b"# drift\n")

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert result.failure_code == "input_digest_mismatch"
    assert lock_path.read_bytes() == frozen_lock


def test_invalid_file_change_after_lock_is_still_input_drift(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    _run(tmp_path, row)
    plddt_path = tmp_path / str(row["afdb_plddt_path"])
    plddt_path.write_text("{}", encoding="utf-8")

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert result.failure_code == "input_digest_mismatch"


def test_deleted_file_after_lock_is_blocked_input_drift(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    _run(tmp_path, row)
    (tmp_path / str(row["afdb_pae_path"])).unlink()

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert result.failure_code == "input_unresolvable_drift"
    assert result.validation.details["cause_code"] == "missing_required_file"


def test_config_drift_blocks_without_overwrite(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    _run(tmp_path, row)
    lock_path = tmp_path / "run/P0_resolve_and_freeze_inputs/outputs/input_lock.json"
    frozen_lock = lock_path.read_bytes()

    result = _run(
        tmp_path,
        row,
        config={**CONFIG, "fragment_policy": "different"},
    )

    assert result.status is LifecycleStatus.BLOCKED_INPUT_DRIFT
    assert result.failure_code == "config_digest_mismatch"
    assert lock_path.read_bytes() == frozen_lock


def test_invalid_existing_output_cannot_skip_or_be_overwritten(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    _run(tmp_path, row)
    frozen_path = tmp_path / "run/P0_resolve_and_freeze_inputs/outputs/frozen_inputs.json"
    frozen_path.write_bytes(b"corrupt historical output")

    result = _run(tmp_path, row)

    assert result.status is LifecycleStatus.FAILED_VALIDATION
    assert result.failure_code == "output_hash_mismatch"
    assert frozen_path.read_bytes() == b"corrupt historical output"


def test_model_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    metadata = tmp_path / str(row["afdb_metadata_path"])
    _rewrite_json(metadata, lambda payload: payload.update(modelEntityId="AF-WRONG-F2"))

    result = _run(tmp_path, row)

    assert result.failure_code == "afdb_model_identity_mismatch"


def test_fragment_length_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    metadata = tmp_path / str(row["afdb_metadata_path"])
    _rewrite_json(metadata, lambda payload: payload.update(uniprotEnd=104, sequenceEnd=104))

    result = _run(tmp_path, row)

    assert result.failure_code == "afdb_fragment_length_mismatch"


@pytest.mark.parametrize(
    ("start", "end", "positions"),
    [
        (100, 102, [101, 102, 103]),
        (200, 203, [101, 102, 103]),
    ],
)
def test_partial_or_no_fragment_coverage_is_unsupported(
    tmp_path: Path, start: int, end: int, positions: list[int]
) -> None:
    row = _fixture(
        tmp_path,
        fragment_start=start,
        fragment_end=end,
        mapped_positions=positions,
    )

    result = _run(tmp_path, row)

    assert result.failure_code == "unsupported_afdb_fragment"


def test_index9_regression_is_unsupported_before_numbering_validation(
    tmp_path: Path,
) -> None:
    row = _fixture(
        tmp_path,
        model_id="AF-0000000365840311",
        fragment_start=1368,
        fragment_end=1493,
        mapped_positions=list(range(1024, 1193)),
    )
    mapping_path = tmp_path / str(row["residue_mapping_path"])
    mapping = pd.read_csv(mapping_path, sep="\t").drop(
        columns=["auth_asym_id", "label_asym_id"]
    )
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    result = _run(tmp_path, row)

    assert result.failure_code == "unsupported_afdb_fragment"
    assert "number" not in result.failure_message.lower()


def test_selected_f1_does_not_fallback_to_available_covering_f2(tmp_path: Path) -> None:
    row = _fixture(
        tmp_path,
        model_id="AF-P12345-F1",
        fragment_start=1,
        fragment_end=50,
        mapped_positions=[101, 102, 103],
    )
    unused = tmp_path / "raw/afdb/P12345/AF-P12345-F2"
    unused.mkdir(parents=True)
    (unused / "metadata.json").write_text(
        json.dumps({"modelEntityId": "AF-P12345-F2", "uniprotStart": 100, "uniprotEnd": 103}),
        encoding="utf-8",
    )

    result = _run(tmp_path, row)

    assert result.failure_code == "unsupported_afdb_fragment"
    assert result.selected_model_entity_id == "AF-P12345-F1"


def test_missing_mapping_required_column_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    mapping_path = tmp_path / str(row["residue_mapping_path"])
    mapping = pd.read_csv(mapping_path, sep="\t").drop(columns="insertion_code")
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    result = _run(tmp_path, row)

    assert result.failure_code == "missing_mapping_column"


def test_duplicate_uniprot_position_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    mapping_path = tmp_path / str(row["residue_mapping_path"])
    mapping = pd.read_csv(mapping_path, sep="\t", keep_default_na=False)
    mapping.loc[1, "uniprot_residue_number"] = mapping.loc[0, "uniprot_residue_number"]
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    result = _run(tmp_path, row)

    assert result.failure_code == "ambiguous_uniprot_mapping"


def test_label_numbering_cannot_replace_missing_auth_numbering(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    mapping_path = tmp_path / str(row["residue_mapping_path"])
    mapping = pd.read_csv(mapping_path, sep="\t", keep_default_na=False)
    mapping["auth_seq_id"] = mapping["auth_seq_id"].astype(object)
    mapping.loc[0, "auth_seq_id"] = ""
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    result = _run(tmp_path, row)

    assert result.failure_code == "invalid_auth_residue_identity"


def test_mapping_does_not_guess_fragment_offset(tmp_path: Path) -> None:
    row = _fixture(tmp_path, mapped_positions=[1, 2, 3])

    result = _run(tmp_path, row)

    assert result.failure_code == "unsupported_afdb_fragment"


def test_plddt_length_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    plddt = tmp_path / str(row["afdb_plddt_path"])
    _rewrite_json(
        plddt,
        lambda payload: (
            payload["residueNumber"].pop(),
            payload["confidenceScore"].pop(),
            payload["confidenceCategory"].pop(),
        ),
    )

    result = _run(tmp_path, row)

    assert result.failure_code == "plddt_length_mismatch"


def test_pae_length_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    pae = tmp_path / str(row["afdb_pae_path"])
    pae.write_text(
        json.dumps([{"predicted_aligned_error": np.zeros((3, 3)).tolist()}]),
        encoding="utf-8",
    )

    result = _run(tmp_path, row)

    assert result.failure_code == "pae_fragment_length_mismatch"


def test_pae_metadata_model_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    metadata = tmp_path / str(row["afdb_metadata_path"])
    _rewrite_json(
        metadata,
        lambda payload: payload.update(
            paeDocUrl=(
                "https://example.test/AF-WRONG-F2-"
                "predicted_aligned_error_v6.json"
            )
        ),
    )

    result = _run(tmp_path, row)

    assert result.failure_code == "pae_model_identity_mismatch"


def test_empty_path_and_wrong_file_type_are_structured_failures(
    tmp_path: Path,
) -> None:
    empty = _fixture(tmp_path / "empty")
    empty["pair_qc_path"] = ""
    wrong_type = _fixture(tmp_path / "wrong")
    wrong_type["afdb_pae_path"] = wrong_type["afdb_model_path"]

    empty_result = _run(tmp_path / "empty", empty)
    wrong_result = _run(tmp_path / "wrong", wrong_type)

    assert empty_result.failure_code == "invalid_input_path"
    assert wrong_result.failure_code == "invalid_input_file_type"


def test_invalid_pdb_mmcif_content_is_rejected(tmp_path: Path) -> None:
    row = _fixture(tmp_path)
    pdb_path = tmp_path / str(row["pdb_structure_path"])
    pdb_path.write_text("not an mmCIF structure", encoding="utf-8")

    result = _run(tmp_path, row)

    assert result.failure_code == "invalid_pdb_structure"
