from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def summary_script() -> ModuleType:
    path = ROOT / "scripts" / "12_build_a0_candidate_summary.py"
    spec = spec_from_file_location("a0_confidence_evidence_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(
    *,
    index: int = 2,
    pdb_id: str = "4i8h",
    uniprot_id: str = "P00760",
) -> dict[str, object]:
    return {
        "screening_index": index,
        "source": "screening_pool",
        "pdb_id": pdb_id,
        "chain_id": "A",
        "uniprot_id": uniprot_id,
        "pair_name": f"{pdb_id}_A__{uniprot_id}",
        "provisional_stratum": "easy_control",
    }


def _lifecycle(identity: dict[str, object], **updates: object) -> dict[str, object]:
    result = {
        **identity,
        "preflight_status": "pass_full_length",
        "full_length_mapping_coverage": 1.0,
        "entity_mapping_coverage": 1.0,
        "sequence_identity": 1.0,
        "observed_ca_fraction_of_mapped": 1.0,
        "pair_status": "complete",
        "geometry_status": "complete",
        "robust_status": "complete",
        "segment_context_status": "successful_no_segments",
    }
    result.update(updates)
    return result


def _write_pair(
    root: Path,
    identity: dict[str, object],
    *,
    model_id: str | None = None,
    version: int = 6,
    fragment_start: int = 1,
    scores: list[float] | None = None,
    mapped_positions: list[int] | None = None,
    observed_positions: list[int] | None = None,
    metadata_model_id: str | None = None,
    plddt_url_model_id: str | None = None,
    pae_url_model_id: str | None = None,
    pae: np.ndarray | None = None,
) -> Path:
    uniprot = str(identity["uniprot_id"])
    selected = model_id or f"AF-{uniprot}-F2"
    metadata_model = metadata_model_id or selected
    plddt_model = plddt_url_model_id or metadata_model
    pae_model = pae_url_model_id or metadata_model
    scores = scores or [95.0] * 8
    fragment_end = fragment_start + len(scores) - 1
    mapped_positions = mapped_positions or list(
        range(fragment_start, fragment_end + 1)
    )
    observed_positions = observed_positions or mapped_positions

    pair_dir = (
        root / "data/processed/pairs" / str(identity["pair_name"])
    )
    pair_dir.mkdir(parents=True)
    model_dir = root / "data/raw/afdb" / uniprot / selected
    model_dir.mkdir(parents=True)
    paths = {
        "afdb_model_path": model_dir / "model.cif",
        "plddt_path": model_dir / "plddt.json",
        "pae_path": model_dir / "pae.json",
    }
    paths["afdb_model_path"].write_text("model", encoding="utf-8")
    paths["plddt_path"].write_text(
        json.dumps({"confidenceScore": scores}),
        encoding="utf-8",
    )
    paths["pae_path"].write_text("[]", encoding="utf-8")
    metadata = {
        "modelEntityId": metadata_model,
        "latestVersion": version,
        "uniprotStart": fragment_start,
        "uniprotEnd": fragment_end,
        "cifUrl": f"https://example/{metadata_model}-model_v{version}.cif",
        "plddtDocUrl": (
            f"https://example/{plddt_model}-confidence_v{version}.json"
        ),
        "paeDocUrl": (
            f"https://example/{pae_model}"
            f"-predicted_aligned_error_v{version}.json"
        ),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    qc = {
        "pdb_id": identity["pdb_id"],
        "chain_id": identity["chain_id"],
        "uniprot_id": uniprot,
        "afdb_model_entity_id": selected,
        "afdb_version": version,
        "afdb_fragment_start": fragment_start,
        "afdb_fragment_end": fragment_end,
        **{key: str(value) for key, value in paths.items()},
    }
    (pair_dir / "pair_qc.json").write_text(
        json.dumps(qc),
        encoding="utf-8",
    )

    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": mapped_positions,
            "auth_asym_id": ["A"] * len(mapped_positions),
            "auth_seq_id": mapped_positions,
            "insertion_code": [""] * len(mapped_positions),
        }
    )
    mapping.to_parquet(pair_dir / "residue_mapping.parquet", index=False)
    score_by_position = {
        fragment_start + offset: score
        for offset, score in enumerate(scores)
    }
    geometry = pd.DataFrame(
        {
            "uniprot_residue_number": observed_positions,
            "auth_asym_id": ["A"] * len(observed_positions),
            "auth_seq_id": observed_positions,
            "insertion_code": [""] * len(observed_positions),
            "plddt": [score_by_position[pos] for pos in observed_positions],
            "aligned_ca_distance": [0.2] * len(observed_positions),
        }
    )
    geometry.to_parquet(
        pair_dir / "residue_geometry.parquet",
        index=False,
    )
    (pair_dir / "pair_geometry_qc.json").write_text(
        json.dumps(
            {
                "pdb_id": identity["pdb_id"],
                "chain_id": "A",
                "uniprot_id": uniprot,
                "mapped_ca_count": len(observed_positions),
                "median_aligned_ca_distance": 0.2,
                "p90_aligned_ca_distance": 0.2,
                "max_aligned_ca_distance": 0.2,
            }
        ),
        encoding="utf-8",
    )
    positions = np.asarray(observed_positions)
    matrix = (
        np.full((len(positions), len(positions)), 3.0)
        if pae is None
        else np.asarray(pae)
    )
    np.fill_diagonal(matrix, 0.0)
    np.savez(
        pair_dir / "pairwise_geometry.npz",
        uniprot_positions=positions,
        symmetric_pae=matrix,
    )
    return pair_dir


def _assemble(
    module: ModuleType,
    root: Path,
    identity: dict[str, object],
    *,
    lifecycle: dict[str, object] | None = None,
    report: dict[str, object] | None = None,
    strict: bool = True,
) -> dict[str, object]:
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )
    return module._assemble_candidate(
        identity,
        root=root,
        config=config,
        preflight={},
        lifecycle=lifecycle or _lifecycle(identity),
        replacement={},
        pilot={},
        mechanism={},
        mapped_confidence=report or {},
        strict=strict,
    )


@pytest.mark.parametrize(
    ("index", "pdb_id", "uniprot_id"),
    [
        (2, "4i8h", "P00760"),
        (5, "6s2m", "P02689"),
        (10, "6zm8", "A0A7S6K8E4"),
    ],
)
def test_original_nonpilot_assembles_canonical_confidence_and_pae(
    summary_script: ModuleType,
    tmp_path: Path,
    index: int,
    pdb_id: str,
    uniprot_id: str,
) -> None:
    identity = _identity(
        index=index,
        pdb_id=pdb_id,
        uniprot_id=uniprot_id,
    )
    _write_pair(tmp_path, identity, scores=[95.0] * 100)

    row = _assemble(summary_script, tmp_path, identity)

    assert row["confidence_model_match"] is True
    assert row["selected_afdb_model_entity_id"] == f"AF-{uniprot_id}-F2"
    assert row["selected_afdb_version"] == 6
    assert row["afdb_fragment_start"] == 1
    assert row["mapped_plddt_median"] == pytest.approx(95.0)
    assert row["pae_48_95_pair_count"] > 0
    assert row["classification_status"] == "complete_classification"


def test_canonical_assembly_is_independent_of_candidate_input_order(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identities = [
        _identity(index=10, pdb_id="6zm8", uniprot_id="A0A7S6K8E4"),
        _identity(index=2, pdb_id="4i8h", uniprot_id="P00760"),
    ]
    for identity in identities:
        _write_pair(tmp_path, identity, scores=[95.0] * 100)

    def assemble(items: list[dict[str, object]]) -> list[tuple[int, str]]:
        rows = [
            _assemble(summary_script, tmp_path, identity)
            for identity in items
        ]
        return sorted(
            (
                int(row["screening_index"]),
                str(row["selected_afdb_model_entity_id"]),
            )
            for row in rows
        )

    assert assemble(identities) == assemble(list(reversed(identities)))


def test_non_f1_fragment_uses_actual_uniprot_interval(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity(uniprot_id="PTEST")
    _write_pair(
        tmp_path,
        identity,
        fragment_start=101,
        scores=[99] * 6 + [40] * 5 + [99] * 6,
    )

    row = _assemble(summary_script, tmp_path, identity)

    assert row["selected_afdb_model_entity_id"] == "AF-PTEST-F2"
    assert row["afdb_fragment_start"] == 101
    assert row["mapped_plddt_min"] == pytest.approx(40.0)
    assert row["longest_internal_below_80_length"] == 5


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("metadata_model_id", "AF-P00760-F3", "model identity"),
        ("plddt_url_model_id", "AF-P00760-F3", "pLDDT model identity"),
        ("pae_url_model_id", "AF-P00760-F3", "PAE model identity"),
    ],
)
def test_model_identity_conflicts_fail_strictly(
    summary_script: ModuleType,
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    identity = _identity()
    _write_pair(tmp_path, identity, **{field: value})

    with pytest.raises(ValueError, match=match):
        _assemble(summary_script, tmp_path, identity)


def test_mapped_confidence_reuses_run_interruption_semantics(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity(uniprot_id="PGAPS")
    _write_pair(
        tmp_path,
        identity,
        fragment_start=101,
        scores=[95] * 6 + [60] * 8 + [95] * 6,
        mapped_positions=[
            *range(101, 111),
            *range(112, 121),
        ],
        observed_positions=[
            *range(101, 109),
            110,
            *range(112, 121),
        ],
    )

    row = _assemble(summary_script, tmp_path, identity)

    assert row["longest_internal_below_80_length"] == 3
    assert row["is_low_conf_local"] is False


def test_long_range_pae_uses_sequence_separation_not_protein_length(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity(uniprot_id="PPAE")
    positions = [1, 49, 97]
    high = np.full((3, 3), 20.0)
    _write_pair(
        tmp_path,
        identity,
        scores=[95.0] * 100,
        mapped_positions=positions,
        observed_positions=positions,
        pae=high,
    )

    row = _assemble(summary_script, tmp_path, identity)

    assert row["pae_48_95_pair_count"] == 2
    assert row["pae_96_plus_pair_count"] == 1
    assert row["long_range_48_95_pae_q90"] == pytest.approx(20.0)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_invalid_pairwise_pae_is_rejected(
    summary_script: ModuleType,
    tmp_path: Path,
    invalid: str,
) -> None:
    identity = _identity(uniprot_id="PBAD")
    pair_dir = _write_pair(tmp_path, identity)
    matrix = np.ones((7, 7))
    if invalid == "nonfinite":
        matrix = np.ones((8, 8))
        matrix[0, 7] = np.nan
    np.savez(
        pair_dir / "pairwise_geometry.npz",
        uniprot_positions=np.arange(1, 9),
        symmetric_pae=matrix,
    )

    with pytest.raises(ValueError):
        _assemble(summary_script, tmp_path, identity)


def test_incomplete_geometry_does_not_use_stale_confidence_artifacts(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_pair(tmp_path, identity)

    row = _assemble(
        summary_script,
        tmp_path,
        identity,
        lifecycle=_lifecycle(identity, geometry_status="failed_geometry"),
        strict=False,
    )

    assert row["confidence_model_match"] is None
    assert row["mapped_plddt_median"] is None
    assert row["classification_status"] == "insufficient_evidence"


def test_complete_geometry_missing_declared_artifact_fails_strictly(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    pair_dir = _write_pair(tmp_path, identity)
    (pair_dir / "pairwise_geometry.npz").unlink()

    with pytest.raises(FileNotFoundError, match="pairwise"):
        _assemble(summary_script, tmp_path, identity)


def test_special_report_is_cross_check_not_canonical_override(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_pair(tmp_path, identity)
    report = {
        "scoring_status": "success",
        "confidence_model_match": True,
        "plddt_bfactor_match": True,
        "mapped_plddt_min": 1.0,
        "mapped_plddt_q10": 1.0,
        "mapped_plddt_median": 1.0,
        "longest_internal_below_70_length": 99,
        "longest_internal_below_80_length": 99,
        "terminal_only_below_80": False,
    }

    with pytest.raises(ValueError, match="mapped confidence"):
        _assemble(
            summary_script,
            tmp_path,
            identity,
            report=report,
        )


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("pair_status", "failed_pair"),
        ("geometry_status", "failed_geometry"),
    ],
)
def test_failed_candidates_are_not_repaired_by_stale_artifacts(
    summary_script: ModuleType,
    tmp_path: Path,
    stage: str,
    expected: str,
) -> None:
    identity = _identity()
    _write_pair(tmp_path, identity)

    row = _assemble(
        summary_script,
        tmp_path,
        identity,
        lifecycle=_lifecycle(identity, **{stage: expected}),
        strict=False,
    )

    assert row["classification_status"] == "insufficient_evidence"
    assert row["selection_eligible"] is False
