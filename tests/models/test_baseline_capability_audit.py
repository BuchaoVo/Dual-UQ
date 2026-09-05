from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.models.baseline_capability_audit import (
    CANONICAL_AA_ORDER,
    CapabilityCard,
    guard_train_only_inference,
    validate_capability_registry,
    validate_local_preferences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_ROOT = REPO_ROOT / "artifacts" / "baselines" / "structcal_v1" / "capabilities"
EXPECTED_MODEL_IDS = {
    "proteinmpnn_v48_020",
    "esm_if1_gvp4_t16_142M_UR50",
    "pifold_official_checkpoint_pth",
    "lm_design_esm2_650m",
    "dplm2_650m",
    "esm3_sm_open_v1",
    "carbondesign_params_tar",
}


def _card(model_id: str = "pifold_official_checkpoint") -> dict[str, object]:
    return {
        "schema_version": "structcal_baseline_capability_v1",
        "model_id": model_id,
        "display_name": "PiFold",
        "tier": "B",
        "architecture_class": "non_autoregressive_geometry_gnn",
        "provenance": {
            "official_repository": "https://github.com/A4Bio/PiFold",
            "source_revision": "b" * 40,
            "source_revision_type": "git_commit",
            "checkpoint_id": "checkpoint.pth",
            "checkpoint_source": "https://github.com/A4Bio/PiFold/releases",
            "checkpoint_configuration": "official CATH 4.2 checkpoint",
            "checkpoint_sha256": "a" * 64,
            "license": "MIT",
            "access_constraints": "none",
            "implementation_language": "Python",
            "primary_framework": "PyTorch",
            "accelerator_assumptions": "CUDA optional; audited on CUDA",
        },
        "environment": {
            "environment_id": "structcal-pifold",
            "environment_path": "/example/conda/envs/structcal-pifold",
            "status": "SUPPORTED",
            "python_version": "3.10.20",
            "pytorch_version": "2.12.1+cu130",
            "framework_cuda": "13.0",
            "gpu_detected": True,
            "official_model_import": "SUPPORTED",
            "checkpoint_load": "SUPPORTED",
            "minimal_inference": "SUPPORTED",
            "critical_packages": {"torch_scatter": "2.1.2+pt212cu130"},
            "limitations": [],
        },
        "structure_conditioning": {
            "status": "SUPPORTED",
            "explanation": "Official geometry-only graph encoder accepts backbone atoms.",
            "type": "raw_backbone_geometry_graph",
            "required_atoms": ["N", "CA", "C", "O"],
            "tokenization": "none",
            "supports_missing_coordinates": False,
            "supports_chain_breaks": False,
            "supports_multi_chain": False,
            "supports_partial_structure": False,
            "uses_sequence_context_during_structure_encoding": False,
            "uses_sidechain_information": False,
            "structure_tokenization_required": False,
            "limitations": ["single-chain official checkpoint"],
        },
        "local_response": {
            "status": "SUPPORTED",
            "explanation": "A single official forward pass returns geometry-only logits.",
            "semantic_class": "L0",
            "probe_state": "single geometry-only forward pass",
            "sequence_context_type": "none",
            "iteration_index": 0,
            "decoding_context": "none",
            "temperature_if_relevant": None,
            "native_sequence_leakage": False,
            "output_shape": "[L,20]",
            "aa_order": list(CANONICAL_AA_ORDER),
            "limitations": [],
        },
        "generation": {
            "status": "SUPPORTED",
            "explanation": "Official one-shot categorical decoding is preserved.",
            "semantics": "NON_AUTOREGRESSIVE_ONE_SHOT",
            "native_decoding_steps": "1",
            "requires_initial_sequence": False,
            "supports_all_mask_initialization": False,
            "supports_fixed_positions": False,
            "supports_variable_length": True,
            "supports_exact_structural_length": True,
            "outputs_canonical_amino_acids_only": True,
            "temperature_semantics": "NATIVE_SEQUENCE_SAMPLING_TEMPERATURE",
            "temperature_acts_on": "one-shot categorical logits",
            "temperature_stage": "single decoding step",
            "temperature_applied_every_iterative_step": False,
            "official_inference_exposes_temperature": True,
            "seed_reproducibility": "DETERMINISTIC_EXACT",
            "rng_sources": ["PyTorch CPU RNG", "PyTorch CUDA RNG"],
            "native_decoding_preserved": True,
            "limitations": [],
        },
        "sequence_scoring": {
            "status": "SUPPORTED",
            "explanation": "The one-shot factorization yields an additive model-native log probability.",
            "semantics": "EXACT_NATIVE_LOG_LIKELIHOOD",
            "score_direction": "higher_is_better_log_probability",
            "length_normalization": "mean_over_modeled_residues",
            "uses_native_sequence_context": False,
            "decoding_order_matters": False,
            "additive_across_residues": True,
            "comparable_across_proteins": True,
            "comparable_across_models": False,
            "limitations": ["raw scores are not cross-model comparable"],
        },
        "cross_evaluator": {
            "proteinmpnn": "SUPPORTED",
            "proteinmpnn_explanation": "Canonical sequence and exact target length are preserved.",
            "esm_if1": "SUPPORTED",
            "esm_if1_explanation": "Canonical sequence and exact target length are preserved.",
        },
        "historical_reuse": {
            "local": {"status": "NOT_REUSABLE", "explanation": "No historical PiFold output exists."},
            "generation": {"status": "NOT_REUSABLE", "explanation": "No historical PiFold output exists."},
            "scoring": {"status": "NOT_REUSABLE", "explanation": "No historical PiFold output exists."},
        },
        "structcal_tasks": {
            "track_i_local": {"status": "SUPPORTED", "explanation": "L0 local semantics."},
            "track_ii_local": {"status": "SUPPORTED", "explanation": "L0 local semantics."},
            "controlled_local": {"status": "SUPPORTED", "explanation": "L0 local semantics."},
            "generation": {"status": "SUPPORTED", "explanation": "Official native generation."},
            "sequence_distribution": {"status": "SUPPORTED", "explanation": "Native full sequences."},
            "sequence_scoring": {"status": "SUPPORTED", "explanation": "Native factorized score."},
            "cross_evaluation": {"status": "SUPPORTED", "explanation": "Canonical exact-length output."},
            "quality_guardrail": {"status": "SUPPORTED", "explanation": "Canonical exact-length output."},
        },
        "overall_v1_class": "FULL_PRIMARY",
        "limitations": [],
    }


def test_capability_card_rejects_unknown_capability_status() -> None:
    payload = _card()
    payload["local_response"]["status"] = "PROBABLY_SUPPORTED"  # type: ignore[index]
    with pytest.raises(ValueError, match="PROBABLY_SUPPORTED"):
        CapabilityCard.model_validate(payload)


def test_l0_l1_card_forbids_native_sequence_leakage() -> None:
    payload = _card()
    payload["local_response"]["native_sequence_leakage"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="native-sequence leakage"):
        CapabilityCard.model_validate(payload)


def test_capability_card_rejects_performance_fields() -> None:
    payload = _card()
    payload["accuracy"] = 0.99
    with pytest.raises(ValueError, match="accuracy"):
        CapabilityCard.model_validate(payload)


def test_registry_requires_unique_model_ids() -> None:
    card = CapabilityCard.model_validate(_card())
    with pytest.raises(ValueError, match="duplicate model_id"):
        validate_capability_registry((card, card))


def test_local_preferences_require_canonical_order_and_normalization() -> None:
    probabilities = np.full((3, 20), 0.05, dtype=np.float64)
    validate_local_preferences(probabilities, CANONICAL_AA_ORDER)

    with pytest.raises(ValueError, match="amino-acid order"):
        validate_local_preferences(probabilities, tuple(reversed(CANONICAL_AA_ORDER)))
    invalid = probabilities.copy()
    invalid[0, 0] = 0.2
    with pytest.raises(ValueError, match="normalized"):
        validate_local_preferences(invalid, CANONICAL_AA_ORDER)


def test_train_only_guard_rejects_locked_test_and_validation() -> None:
    guard_train_only_inference({"P02619": "TRAIN"}, ("P02619",))
    with pytest.raises(ValueError, match="LOCKED_TEST inference is forbidden"):
        guard_train_only_inference({"P02619": "LOCKED_TEST"}, ("P02619",))
    with pytest.raises(ValueError, match="TRAIN-only"):
        guard_train_only_inference({"P02619": "VALIDATION"}, ("P02619",))


def test_persisted_capability_registry_is_complete_and_performance_free() -> None:
    registry = json.loads((CAPABILITY_ROOT / "registry.json").read_text())
    cards = []
    for relative_path in registry["card_paths"]:
        payload = json.loads((CAPABILITY_ROOT / relative_path).read_text())
        cards.append(CapabilityCard.model_validate(payload))

        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                forbidden = {
                    "accuracy",
                    "recovery",
                    "perplexity",
                    "nll",
                    "r_local",
                    "ranking",
                }
                assert forbidden.isdisjoint({str(key).lower() for key in node})
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    validated = validate_capability_registry(cards)
    assert {card.model_id for card in validated} == EXPECTED_MODEL_IDS
    assert registry["performance_results_complete"] is False


def test_smoke_manifest_is_train_only() -> None:
    manifest = json.loads((CAPABILITY_ROOT / "smoke_tests.json").read_text())
    assert manifest["locked_test_inference_occurred"] is False
    assert manifest["validation_inference_occurred"] is False
    assert {case["split"] for case in manifest["selected_cases"]} == {"TRAIN"}
    assert {case["protein_id"] for case in manifest["selected_cases"]} == {"P02619"}


def test_capability_matrix_matches_cards() -> None:
    matrix = pd.read_csv(CAPABILITY_ROOT / "capability_matrix.tsv", sep="\t")
    assert set(matrix["model_id"]) == EXPECTED_MODEL_IDS

    for row in matrix.itertuples(index=False):
        payload = json.loads((CAPABILITY_ROOT / "cards" / f"{row.model_id}.json").read_text())
        card = CapabilityCard.model_validate(payload)
        assert row.environment_status == card.environment.status
        assert row.local_semantic_class == card.local_response.semantic_class
        assert row.local_response_status == card.local_response.status
        assert row.generation_status == card.generation.status
        assert row.overall_v1_class == card.overall_v1_class


def test_frozen_integrity_and_execution_protocol_remain_model_independent() -> None:
    integrity = json.loads((CAPABILITY_ROOT / "frozen_benchmark_integrity.json").read_text())
    assert integrity["release_file_count"] == len(integrity["release_file_list"]) == 19
    assert integrity["git_status_paths_under_release"] == []
    assert integrity["locked_test_inference_occurred"] is False

    protocol = json.loads((CAPABILITY_ROOT / "execution_protocol.json").read_text())
    assert protocol["benchmark_definition_mutable"] is False
    assert protocol["split_guard"]["capability_audit_inference_split"] == "TRAIN"
    assert protocol["split_guard"]["locked_test_inference_used"] is False
    assert protocol["performance_outputs_present"] is False
