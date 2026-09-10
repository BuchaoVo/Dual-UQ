from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import dual_uq.dataset.stages.candidate_derivation as derivation_module
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset import run_derivation
from dual_uq.dataset.models import (
    AFDBFragment,
    BiologicalIdentity,
    CandidateContext,
    CandidateDerivationResult,
    DerivationConfig,
    DerivationError,
    LogicalAssetRef,
    StageResult,
)
from dual_uq.dataset.stages.candidate_derivation import derive_candidate, verify_bound_hash

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "src/dual_uq").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return ProjectPaths.discover(project_root=tmp_path)


def _context(index: int) -> CandidateContext:
    accession = f"P{index:05d}"
    return CandidateContext(
        candidate_index=index,
        identity=BiologicalIdentity(
            pair_id=f"1abc_A__{accession}",
            pdb_id="1abc",
            chain_id="A",
            uniprot_accession=accession,
            polymer_entity_id=f"1ABC_{index}",
        ),
        exact_afdb_accession=accession,
        expected_afdb_model_identity=f"AF-{accession}-F1",
        assets=(
            LogicalAssetRef(
                asset_type="metadata",
                logical_path=f"data/raw/afdb/{accession}/metadata.json",
                sha256=None,
                provenance="fixture",
            ),
        ),
        source_bindings=(),
        protocol_binding="fixture-binding",
    )


def _config() -> DerivationConfig:
    return DerivationConfig(
        protocol_version="v1",
        protocol_binding="fixture-binding",
        preflight_thresholds={},
        observability_thresholds={},
        preflight={"preflight_pass": True},
        scope={"candidate_admission_changed": False},
    )


def test_fragment_only_metadata_is_a_candidate_level_canonical_failure(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    accession = "P0DTD1"
    model_id = "AF-0000000365840311"
    assets = []
    for asset_type, suffix in (
        ("afdb_metadata", "metadata.json"),
        ("pdb_mmcif", "pdb.cif"),
        ("sifts", "sifts.xml"),
        ("afdb_structure", "model.cif"),
        ("afdb_pae", "pae.json"),
        ("afdb_confidence", "plddt.json"),
    ):
        logical_path = f"data/raw/fixture/{accession}/{suffix}"
        path = paths.resolve_logical(logical_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if asset_type == "afdb_metadata":
            path.write_text(
                json.dumps(
                    [
                        {
                            "uniprotAccession": accession,
                            "uniprotSequence": "A" * 126,
                            "modelEntityId": model_id,
                            "sequenceStart": 1368,
                            "sequenceEnd": 1493,
                        }
                    ]
                )
            )
        else:
            path.write_text("fixture")
        assets.append(
            LogicalAssetRef(asset_type, logical_path, None, "fixture")
        )
    context = CandidateContext(
        candidate_index=46,
        identity=BiologicalIdentity(
            pair_id="7kr0_A__P0DTD1",
            pdb_id="7kr0",
            chain_id="A",
            uniprot_accession=accession,
            polymer_entity_id="7KR0_1",
        ),
        exact_afdb_accession=accession,
        expected_afdb_model_identity=model_id,
        assets=tuple(assets),
        source_bindings=(),
        protocol_binding="fixture-binding",
    )

    result = derive_candidate(context, _config(), paths)

    assert result.report_record["raw_complete"] is True
    assert result.report_record["canonical_sequence_complete"] is False
    assert result.report_record["prediction_record_sequence_length"] == 126
    assert result.report_record["prediction_record_interval"] == (1368, 1493)
    assert result.report_record["primary_failure_stage"] == "canonical_sequence"
    assert (
        result.report_record["primary_failure_code"]
        == "missing_canonical_sequence_provenance"
    )
    assert result.report_record["attrition_class"] == "identity_or_provenance_issue"


@pytest.mark.parametrize("size", [1, 8, 48, 213])
def test_same_runner_handles_all_panel_sizes_without_count_switches(
    size: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def controlled_derivation(context, config, paths):
        del config, paths
        record = {
            "candidate_index": context.candidate_index,
            "raw_complete": True,
            "canonical_sequence_complete": True,
            "pair_qc_complete": True,
            "mapping_complete": True,
            "fragment_resolved": True,
            "pae_bound": True,
            "confidence_bound": True,
            "mechanism_observable": True,
            "primary_failure_stage": None,
            "attrition_class": None,
        }
        return CandidateDerivationResult(
            context=context,
            stages=(StageResult(stage="observability", status="complete"),),
            evidence={},
            report_record=record,
        )

    monkeypatch.setattr(
        "dual_uq.dataset.pipeline.runner.derive_candidate", controlled_derivation
    )
    panel = [_context(index) for index in reversed(range(1, size + 1))]

    result = run_derivation(panel=panel, config=_config(), paths=_paths(tmp_path))

    assert [row.context.candidate_index for row in result.candidates] == list(
        range(1, size + 1)
    )
    assert result.summary["candidate_count"] == size
    assert f"N={size}" in result.attrition_bias_probe["scope"]


def test_duplicate_candidate_identity_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate candidate_index"):
        run_derivation(
            panel=[_context(1), _context(1)],
            config=_config(),
            paths=_paths(tmp_path),
        )


def test_duplicate_biological_tuple_is_rejected_even_if_pair_id_differs(
    tmp_path: Path,
) -> None:
    first = _context(1)
    duplicate = CandidateContext(
        candidate_index=2,
        identity=BiologicalIdentity(
            pair_id="alias_A__P00001",
            pdb_id=first.identity.pdb_id,
            chain_id=first.identity.chain_id,
            uniprot_accession=first.identity.uniprot_accession,
            polymer_entity_id=first.identity.polymer_entity_id,
        ),
        exact_afdb_accession=first.exact_afdb_accession,
        expected_afdb_model_identity=first.expected_afdb_model_identity,
        assets=first.assets,
        source_bindings=first.source_bindings,
        protocol_binding=first.protocol_binding,
    )

    with pytest.raises(ValueError, match="duplicate biological identity"):
        run_derivation(
            panel=[first, duplicate],
            config=_config(),
            paths=_paths(tmp_path),
        )


def test_noncontiguous_subset_uses_candidate_identity_order_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def controlled_derivation(context, config, paths):
        del config, paths
        record = {
            "candidate_index": context.candidate_index,
            "raw_complete": True,
            "canonical_sequence_complete": True,
            "pair_qc_complete": True,
            "mapping_complete": True,
            "fragment_resolved": True,
            "pae_bound": True,
            "confidence_bound": True,
            "mechanism_observable": True,
            "primary_failure_stage": None,
            "attrition_class": None,
        }
        return CandidateDerivationResult(
            context=context,
            stages=(StageResult(stage="observability", status="complete"),),
            evidence={},
            report_record=record,
        )

    monkeypatch.setattr(
        "dual_uq.dataset.pipeline.runner.derive_candidate", controlled_derivation
    )

    result = run_derivation(
        panel=[_context(101), _context(3), _context(19)],
        config=_config(),
        paths=_paths(tmp_path),
    )

    assert [candidate.context.candidate_index for candidate in result.candidates] == [
        3,
        19,
        101,
    ]


def test_missing_raw_asset_becomes_one_structured_failure_with_dependency_skips(
    tmp_path: Path,
) -> None:
    context = _context(1)

    result = derive_candidate(context, _config(), _paths(tmp_path))

    assert result.report_record["primary_failure_stage"] == "raw"
    assert result.report_record["primary_failure_code"] == "missing_raw_asset"
    assert result.report_record["raw_complete"] is False
    assert next(stage.status for stage in result.stages) == "failed"
    assert all(stage.status == "skipped_dependency" for stage in result.stages[1:])


def test_bound_hash_drift_is_structured_and_never_rebaselined(tmp_path: Path) -> None:
    path = tmp_path / "raw.bin"
    path.write_bytes(b"changed")

    with pytest.raises(DerivationError, match="raw hash drift") as error:
        verify_bound_hash(path, hashlib.sha256(b"original").hexdigest())

    assert error.value.code == "raw_hash_drift"


def test_unexpected_raw_read_error_is_attributed_to_raw_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args, **kwargs):
        del args, kwargs
        raise OSError("read denied")

    monkeypatch.setattr(derivation_module, "_verify_asset", unreadable)

    result = derive_candidate(_context(1), _config(), _paths(tmp_path))

    assert result.report_record["raw_complete"] is False
    assert result.report_record["primary_failure_stage"] == "raw"
    assert result.report_record["primary_failure_code"] == "unexpected_OSError"


def test_model_artifact_validation_failure_is_fragment_causal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    asset_types = (
        "afdb_metadata",
        "pdb_mmcif",
        "sifts",
        "afdb_structure",
        "afdb_pae",
        "afdb_confidence",
    )
    assets = []
    for asset_type in asset_types:
        relative = f"data/raw/{asset_type}.bin"
        path = paths.resolve_logical(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        assets.append(LogicalAssetRef(asset_type, relative, None, "fixture"))
    context = CandidateContext(
        candidate_index=1,
        identity=BiologicalIdentity(
            pair_id="1abc_A__P00001",
            pdb_id="1abc",
            chain_id="A",
            uniprot_accession="P00001",
            polymer_entity_id="1ABC_1",
        ),
        exact_afdb_accession="P00001",
        expected_afdb_model_identity="AF-P00001-F1",
        assets=tuple(assets),
        source_bindings=(),
        protocol_binding="fixture-binding",
        prederivation_evidence=(("pdb_entity_length", 3),),
    )
    exact_record = {
        "uniprotAccession": "P00001",
        "modelEntityId": "AF-P00001-F1",
        "latestVersion": 6,
    }
    monkeypatch.setattr(
        derivation_module,
        "extract_prediction_record_sequence",
        lambda *args: {
            "prediction_sequence_source_field": "uniprotSequence",
            "prediction_sequence_length": 3,
            "prediction_sequence_sha256": "a" * 64,
            "prediction_interval": [1, 3],
            "metadata_record_count": 1,
            "prediction_record_count": 1,
            "model_entity_id": "AF-P00001-F1",
            "nonselected_sibling_record_count": 0,
            "exact_record": exact_record,
        },
    )
    monkeypatch.setattr(
        derivation_module,
        "extract_canonical_sequence",
        lambda *args: {
            "sequence_source_field": "uniprotSequence",
            "sequence_length": 3,
            "sequence_sha256": "a" * 64,
            "canonical_sequence_provenance": "full_span_exact_prediction_record",
            "metadata_record_count": 1,
            "prediction_record_count": 1,
            "model_entity_id": "AF-P00001-F1",
            "nonselected_sibling_record_count": 0,
            "exact_record": exact_record,
        },
    )
    monkeypatch.setattr(derivation_module, "metadata_records", lambda *args: [exact_record])
    monkeypatch.setattr(
        derivation_module, "parse_sifts_residue_mapping", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        derivation_module, "load_chain_ca_table", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        derivation_module,
        "compute_pair_quality",
        lambda *args, **kwargs: {
            "pair_qc_status": "pair_qc_pass",
            "preflight_status": "pass_full_length",
            "preflight_reason": "fixture",
            "mapped_interval": [1, 3],
            "mapping_coverage": 1.0,
        },
    )
    mapped = pd.DataFrame({"observed_ca": [True, True, True]})
    monkeypatch.setattr(
        derivation_module,
        "mapping_with_provenance",
        lambda *args: (
            mapped,
            {
                "gap_count": 0,
                "segment_count": 1,
                "largest_uniprot_gap": 0,
                "gap_boundaries": [],
                "nearest_gap_metadata_available": True,
            },
            {},
        ),
    )
    monkeypatch.setattr(
        derivation_module, "residue_mapping_audit_records", lambda *args: []
    )
    fragment = AFDBFragment("AF-P00001-F1", 1, 3, 3)
    monkeypatch.setattr(
        derivation_module,
        "resolve_exact_fragment",
        lambda *args: {
            "fragment_resolution_status": "fragment_resolved",
            "exact_accession_prediction_record_count": 1,
            "mapped_interval": [1, 3],
            "fragment_candidates": [
                {
                    "model_entity_id": "AF-P00001-F1",
                    "uniprot_start": 1,
                    "uniprot_end": 3,
                    "model_residue_count": 3,
                }
            ],
            "full_cover_count": 1,
            "selected_model_entity_id": "AF-P00001-F1",
            "selected_fragment": fragment,
        },
    )
    monkeypatch.setattr(
        derivation_module,
        "validate_frozen_model_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DerivationError("afdb_model_identity_mismatch", "wrong model")
        ),
    )

    result = derive_candidate(context, _config(), paths)

    assert result.report_record["fragment_resolved"] is False
    assert result.report_record["primary_failure_stage"] == "fragment"
    assert result.report_record["primary_failure_code"] == "afdb_model_identity_mismatch"
    assert next(
        stage.status for stage in result.stages if stage.stage == "fragment"
    ) == "failed"


def test_public_api_and_source_layout_have_one_portable_generic_runner() -> None:
    source_root = REPOSITORY_ROOT / "src/dual_uq/dataset"
    source = "\n".join(path.read_text() for path in source_root.rglob("*.py"))
    source_paths = tuple(path.relative_to(source_root) for path in source_root.rglob("*.py"))

    assert callable(run_derivation)
    assert not (REPOSITORY_ROOT / "src/dual_uq/dataset_a").exists()
    assert "/home/" + "zbc/" not in source
    assert "/mnt/data/users/" not in source
    assert all("derive_48" not in path.as_posix().lower() for path in source_paths)
    assert "def derive_48" not in source.lower()
    assert "class derive_48" not in source.lower()
    assert "derive_213" not in source.lower()


def test_thin_pilot_has_no_embedded_scientific_implementations() -> None:
    path = REPOSITORY_ROOT / "scripts/dataset/derive.py"
    tree = ast.parse(path.read_text())
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_derive_one" not in definitions
    assert "_mechanism_evidence" not in definitions
    assert "_mapping_with_provenance" not in definitions
    assert "run_derivation" not in definitions


def test_frozen_scientific_contracts_have_one_canonical_implementation() -> None:
    derivation = (
        REPOSITORY_ROOT / "src/dual_uq/dataset/stages/candidate_derivation.py"
    ).read_text()
    fragments = (
        REPOSITORY_ROOT / "src/dual_uq/dataset/policies/fragments.py"
    ).read_text()
    mapping = (
        REPOSITORY_ROOT / "src/dual_uq/dataset/services/mapping.py"
    ).read_text()

    assert "from dual_uq.core.hashing import sha256_file" in derivation
    assert "from ..stages.resolution import" in fragments
    assert "def compute_mapping_quality_metrics(" in mapping
    assert "def classify_mapping_quality(" in mapping
