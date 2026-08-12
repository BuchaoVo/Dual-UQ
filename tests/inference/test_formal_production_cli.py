from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.dataset.services.proteinmpnn_scoring import adapt_formal_runtime_requests
from dual_uq.inference.formal import FormalRequestDefinition, derive_orchestration_id
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    make_decoding_realization,
)
from dual_uq.models.scoring import (
    CandidateCollection,
    ScorerBinding,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
)
from dual_uq.structure import StructureCondition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts/dataset/run_formal_scoring.py"
SCORE_CONTRACT = "stage0_fixed_sequence_autoregressive_mask_logp_v1"


def _module():
    spec = importlib.util.spec_from_file_location("run_formal_scoring", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _definition(condition: str) -> FormalRequestDefinition:
    wt = "ACD"
    probe = "AED"
    wt_hash = sha256_bytes(wt.encode("ascii"))
    probe_hash = sha256_bytes(probe.encode("ascii"))
    realization = make_decoding_realization(
        protein_id="fixture_A__P00001",
        mask_length=3,
        repeat_index=0,
        seed=0,
        protocol_version=SCORE_CONTRACT,
    )
    request = ScoreRequest(
        condition=StructureCondition(
            protein_id="fixture_A__P00001",
            condition_id=condition,
            source=condition,
            structure_sha256=("1" if condition == "PDB" else "2") * 64,
        ),
        scoring_domain_id=sha256_canonical(
            {
                "protein_id": "fixture_A__P00001",
                "canonical_positions": [1, 2, 3],
                "canonical_sequence_sha256": wt_hash,
            }
        ),
        canonical_positions=(1, 2, 3),
        candidate_collection=CandidateCollection(
            collection_id=sha256_canonical({"sequence_hashes": [probe_hash]}),
            wt=ScoringVariant(
                variant_kind=VariantKind.WT,
                variant_id=None,
                sequence_hash=wt_hash,
                sequence=wt,
                position=None,
                wt_aa=None,
                mut_aa=None,
            ),
            probes=(
                ScoringVariant(
                    variant_kind=VariantKind.PROBE,
                    variant_id=probe_hash,
                    sequence_hash=probe_hash,
                    sequence=probe,
                    position=2,
                    wt_aa="C",
                    mut_aa="E",
                ),
            ),
        ),
        repeat_index=0,
        seed=0,
        realization_id=realization.fingerprint,
        realization_algorithm=realization.algorithm,
        score_contract_id=SCORE_CONTRACT,
    )
    fingerprint = sha256_canonical({"condition": condition, "fixture": True})
    return FormalRequestDefinition(
        scientific_fingerprint=fingerprint,
        orchestration_id=derive_orchestration_id(fingerprint),
        workflow_request_id=f"fixture::{condition}::r00",
        request=request,
        scorer_binding=ScorerBinding(
            scorer_id="ProteinMPNN",
            implementation_id=(
                "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
            ),
            checkpoint_id=(
                "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
            ),
            score_contract_id=SCORE_CONTRACT,
        ),
        workflow_metadata={"cluster_id_30": "fixture-cluster"},
    )


def test_cli_modes_keep_inventory_default_and_freeze_batch_size() -> None:
    module = _module()
    default = module.parse_args(
        ["--project-root", str(REPOSITORY_ROOT.resolve())]
    )
    worker = module.parse_args(
        [
            "--project-root",
            str(REPOSITORY_ROOT.resolve()),
            "--mode",
            "worker",
            "--device",
            "cuda:3",
            "--worker-count",
            "8",
            "--worker-index",
            "3",
        ]
    )

    assert default.mode == "inventory"
    assert worker.mode == "worker"
    assert worker.device == "cuda:3"
    assert worker.worker_count == 8
    assert worker.worker_index == 3
    assert worker.batch_size == 128


def test_execution_modes_reject_nonfrozen_batch_size() -> None:
    module = _module()
    args = module.parse_args(
        [
            "--project-root",
            str(REPOSITORY_ROOT.resolve()),
            "--mode",
            "preflight",
            "--device",
            "cuda:0",
            "--batch-size",
            "64",
        ]
    )

    with pytest.raises(module.FormalInventoryError, match="batch size"):
        module.validate_execution_arguments(args)


def test_execution_modes_require_explicit_model_python() -> None:
    module = _module()
    args = module.parse_args(
        [
            "--project-root",
            str(REPOSITORY_ROOT.resolve()),
            "--mode",
            "preflight",
            "--device",
            "cuda:0",
        ]
    )

    with pytest.raises(module.FormalInventoryError, match="model Python"):
        module.validate_execution_arguments(args)


def test_code_provenance_covers_loaded_dual_uq_modules() -> None:
    module = _module()

    provenance = module._code_provenance(REPOSITORY_ROOT)
    paths = {row["path"] for row in provenance["source_files"]}

    assert "src/dual_uq/inference/formal.py" in paths
    assert "src/dual_uq/inference/materialization.py" in paths
    assert "src/dual_uq/models/proteinmpnn.py" in paths
    assert "scripts/dataset/run_formal_scoring.py" in paths
    assert len(provenance["effective_execution_code_sha256"]) == 64


def test_worker_payload_adapts_normalized_requests_without_changing_binding() -> None:
    module = _module()
    selected = (_definition("PDB"), _definition("AFDB"))

    def resolve(request: ScoreRequest) -> ProteinMPNNStructureInput:
        return ProteinMPNNStructureInput(
            protein_id=request.protein_id,
            backbone_condition=request.condition.condition_id,
            uniprot_positions=request.canonical_positions,
            wt_sequence_projection="ACD",
            coordinates=np.zeros((3, 4, 3), dtype=np.float32),
            structure_sha256=request.condition.structure_sha256,
        )

    payload = module._build_formal_worker_request(
        selected,
        context_definitions=selected,
        projection_resolver=resolve,
    )

    assert payload["schema_version"] == "stage0_formal_protein_request_v2"
    assert payload["protein_id"] == "fixture_A__P00001"
    assert len(payload["tasks"]) == 2
    assert {row["binding"]["orchestration_id"] for row in payload["tasks"]} == {
        row.orchestration_id for row in selected
    }
    assert {row["binding"]["backbone_condition"] for row in payload["tasks"]} == {
        "PDB",
        "AFDB",
    }
    assert payload["pdb_coordinates"] == np.zeros((3, 4, 3)).tolist()
    assert payload["afdb_coordinates"] == np.zeros((3, 4, 3)).tolist()
    assert len(adapt_formal_runtime_requests(payload)) == 2


def test_worker_bridge_materializes_canonical_shards_without_direct_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    selected = (_definition("PDB"), _definition("AFDB"))

    def resolve(request: ScoreRequest) -> ProteinMPNNStructureInput:
        return ProteinMPNNStructureInput(
            protein_id=request.protein_id,
            backbone_condition=request.condition.condition_id,
            uniprot_positions=request.canonical_positions,
            wt_sequence_projection="ACD",
            coordinates=np.zeros((3, 4, 3), dtype=np.float32),
            structure_sha256=request.condition.structure_sha256,
        )

    monkeypatch.setattr(
        module, "build_final_confirmatory_projection_resolver", lambda _root: resolve
    )

    def fake_run(command, **_kwargs):
        response = Path(command[command.index("--response") + 1])
        request_path = Path(command[command.index("--formal-request") + 1])
        shard_directory = Path(command[command.index("--shard-directory") + 1])
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        for task in payload["tasks"]:
            binding = task["binding"]
            shard = {
                "schema_version": "stage0_fixed_probe_scoring_shard_v1",
                "binding": binding,
                "wt_score": {
                    "score_sum_logp_mask": -3.0,
                    "score_mean_logp_mask": -1.0,
                    "scored_residue_count": 3,
                },
                "candidate_scores": [
                    {
                        "sequence_hash": payload["candidate_sequence_hashes"][0],
                        "score_sum_logp_mask": -4.0,
                        "score_mean_logp_mask": -4.0 / 3.0,
                        "scored_residue_count": 3,
                    }
                ],
                "execution_environment": {"device": "fake"},
            }
            (shard_directory / task["output_filename"]).write_text(
                json.dumps(shard), encoding="utf-8"
            )
        response.write_text(
            json.dumps(
                {
                    "schema_version": "stage0_formal_runtime_response_v1",
                    "request_sha256": module.sha256_file(request_path),
                    "status": "complete",
                    "completed_shards": len(payload["tasks"]),
                    "execution_environment": {"device": "fake"},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    artifact_root = tmp_path / "artifacts"
    result = module._execute_via_model_worker(
        selected,
        context_definitions=selected,
        project_root=REPOSITORY_ROOT,
        artifact_root=artifact_root,
        staging_root=tmp_path / "staging",
        model_python=Path(sys.executable),
        device="cuda:0",
        batch_size=128,
        execution_provenance={"effective_code": "a" * 64},
    )

    assert result["executed"] == 2
    assert all(row.artifact.resolve(artifact_root).is_file() for row in selected)
