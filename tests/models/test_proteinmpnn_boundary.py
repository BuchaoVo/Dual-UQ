from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq import schema as legacy_schema
from dual_uq.dataset.services import proteinmpnn_scoring as historical
from dual_uq.models import proteinmpnn as canonical
from dual_uq.models.scoring import ScoreRecord, VariantKind

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPOSITORY_ROOT / "experiments/p2_design_baseline/scale1/scale1b_v2"
STAGE0_ROOT = REPOSITORY_ROOT / "experiments/p2_design_baseline/stage0"


def test_historical_service_delegates_model_semantics_to_canonical_adapter() -> None:
    assert historical.DecodingRealization is canonical.DecodingRealization
    assert historical.ProteinMPNNScore is canonical.ProteinMPNNScore
    assert historical.ProteinMPNNRuntime is canonical.ProteinMPNNAdapter
    assert historical.ScoringBackboneProjection is canonical.ProteinMPNNStructureInput
    assert historical.make_decoding_realization is canonical.make_decoding_realization
    assert historical.tile_decoding_order is canonical.tile_decoding_order
    assert historical.score_target_log_probs is canonical.score_target_log_probs
    assert (
        historical.load_authorized_proteinmpnn_runtime
        is canonical.load_authorized_proteinmpnn_adapter
    )

    historical_source = inspect.getsource(historical)
    canonical_source = inspect.getsource(canonical)
    assert "protein_mpnn_utils.py" not in historical_source
    assert "protein_mpnn_utils.py" in canonical_source
    assert "module.ProteinMPNN(" not in historical_source
    assert "module.ProteinMPNN(" in canonical_source


def test_canonical_score_extraction_matches_independent_reference_values() -> None:
    log_probs = np.asarray(
        [
            [[-4.0, -1.0, -9.0], [-2.0, -7.0, -3.0]],
            [[-0.5, -8.0, -6.0], [-5.0, -0.25, -4.0]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([[1, 2], [0, 1]], dtype=np.int64)

    scores = canonical.score_target_log_probs(log_probs, targets)

    assert scores == (
        canonical.ProteinMPNNScore(
            score_sum_logp_mask=-4.0,
            score_mean_logp_mask=-2.0,
        ),
        canonical.ProteinMPNNScore(
            score_sum_logp_mask=-0.75,
            score_mean_logp_mask=-0.375,
        ),
    )


def test_canonical_realization_matches_frozen_scoring_plan() -> None:
    plan = pd.read_parquet(V2_ROOT / "scale1b_v2_proteinmpnn_scoring_plan.parquet")
    masks = pd.read_parquet(V2_ROOT / "scale1b_v2_primary_common_masks.parquet")
    row = plan.iloc[0]
    mask_length = int((masks["protein_id"] == row["protein_id"]).sum())

    realization = canonical.make_decoding_realization(
        protein_id=str(row["protein_id"]),
        mask_length=mask_length,
        repeat_index=int(row["repeat"]),
        seed=int(row["seed"]),
        protocol_version=str(row["score_contract_id"]),
    )

    assert realization.fingerprint == row["explicit_realization_id"]
    assert realization.algorithm == row["realization_algorithm"]
    assert sorted(realization.order) == list(range(mask_length))


def test_normalized_wt_record_preserves_authoritative_identity_without_sentinel() -> None:
    row = pd.read_parquet(STAGE0_ROOT / "fixed_probe_wt_scores.parquet").iloc[0]

    record = historical.normalized_wt_score_record(row.to_dict())

    assert record.variant_kind is VariantKind.WT
    assert record.variant_id is None
    assert record.sequence_hash is None
    assert record.position is None
    assert record.wt_aa is None
    assert record.mut_aa is None
    assert record.protein_id == row["protein_id"]
    assert record.condition_id == row["backbone_condition"]
    assert record.structure_sha256 == row["backbone_sha256"]
    assert record.repeat_index == row["repeat_index"]
    assert record.seed == row["seed"]
    assert record.realization_id == row["decoding_realization_sha256"]
    assert record.scorer_id == canonical.PROTEINMPNN_SCORER_ID
    assert record.implementation_id == canonical.AUTHORIZED_IMPLEMENTATION_COMMIT
    assert record.checkpoint_id == row["model_checkpoint_sha256"]
    assert record.score_contract_id == row["scoring_protocol"]
    assert record.score_sum_logp_mask == row["score_sum_logp_mask"]
    assert record.score_mean_logp_mask == row["score_mean_logp_mask"]
    assert record.scored_residue_count == row["scored_residue_count"]


def test_normalized_probe_record_preserves_authoritative_mutation_identity() -> None:
    row = pd.read_parquet(STAGE0_ROOT / "fixed_probe_scores.parquet").iloc[0]

    record = historical.normalized_probe_score_record(row.to_dict())

    assert record.variant_kind is VariantKind.PROBE
    assert record.variant_id == row["sequence_hash"]
    assert record.sequence_hash == row["sequence_hash"]
    assert record.position == row["position"]
    assert record.wt_aa == row["wt_aa"]
    assert record.mut_aa == row["mut_aa"]
    assert record.condition_id == row["backbone_condition"]
    assert record.structure_sha256 == row["backbone_sha256"]
    assert record.realization_id == row["decoding_realization_sha256"]
    assert record.score_sum_logp_mask == row["score_sum_logp_mask"]
    assert record.score_mean_logp_mask == row["score_mean_logp_mask"]
    assert record.scored_residue_count == row["scored_residue_count"]


def _score_record(**changes: object) -> ScoreRecord:
    values: dict[str, object] = {
        "protein_id": "fixture_A__P00001",
        "condition_id": "PDB",
        "structure_sha256": "1" * 64,
        "variant_kind": VariantKind.WT,
        "variant_id": None,
        "sequence_hash": None,
        "position": None,
        "wt_aa": None,
        "mut_aa": None,
        "repeat_index": 0,
        "seed": 0,
        "realization_id": "2" * 64,
        "scorer_id": "ProteinMPNN",
        "implementation_id": "3" * 40,
        "checkpoint_id": "4" * 64,
        "score_contract_id": "fixture_score_contract_v1",
        "score_sum_logp_mask": -6.0,
        "score_mean_logp_mask": -2.0,
        "scored_residue_count": 3,
    }
    values.update(changes)
    return ScoreRecord(**values)  # type: ignore[arg-type]


def test_score_record_is_frozen_and_root_schema_is_compatibility_alias() -> None:
    record = _score_record()

    assert legacy_schema.ScoreRecord is ScoreRecord
    with pytest.raises(FrozenInstanceError):
        record.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"position": -1}, "WT measurements cannot carry probe identity"),
        (
            {
                "variant_kind": VariantKind.PROBE,
                "variant_id": None,
                "sequence_hash": None,
                "position": None,
                "wt_aa": None,
                "mut_aa": None,
            },
            "PROBE measurements require complete probe identity",
        ),
        ({"score_mean_logp_mask": -1.0}, "mean score must equal sum / residue count"),
        ({"score_sum_logp_mask": float("nan")}, "scores must be finite"),
    ],
)
def test_score_record_rejects_invalid_variant_or_score_semantics(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _score_record(**changes)


def test_probe_record_requires_real_substitution_and_stable_hash_identity() -> None:
    with pytest.raises(ValueError, match="valid single substitution"):
        _score_record(
            variant_kind=VariantKind.PROBE,
            variant_id="5" * 64,
            sequence_hash="5" * 64,
            position=10,
            wt_aa="A",
            mut_aa="A",
        )


def test_proteinmpnn_adapter_has_no_dataset_or_intervention_dependency() -> None:
    source = inspect.getsource(canonical)

    assert "dual_uq.dataset" not in source
    assert "StructureCondition" not in source
    assert "StructuralIntervention" not in source
    assert "PDB" not in source
    assert "AFDB" not in source
